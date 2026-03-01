import os
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from projectaria_tools.core import data_provider, calibration, mps

def main():
    # ----------------------------------------------------------------
    # 1. Argument Parser 설정
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Convert Aria Data to 3DGS (COLMAP) format")
    parser.add_argument("--aria_dir", type=str, required=True, help="Aria 데이터 폴더 경로 (예: .../301_1253)")
    parser.add_argument("--output_dir", type=str, required=True, help="변환된 3DGS 데이터가 저장될 경로")
    parser.add_argument("--width", type=int, default=1024, help="출력 이미지 가로 해상도")
    parser.add_argument("--height", type=int, default=1024, help="출력 이미지 세로 해상도")
    parser.add_argument("--alpha_mask", action="store_true", help="Fisheye 밖의 빈 공간을 투명 마스킹 처리 (PNG 저장)")
    parser.add_argument("--rotate", action="store_true", help="이미지를 시계 방향으로 90도 회전 및 좌표계 동기화")
    args = parser.parse_args()

    # 절대 경로로 변환
    aria_dir = os.path.abspath(args.aria_dir)
    out_dir = os.path.abspath(args.output_dir)

    # 폴더 이름을 기반으로 .vrs 파일과 mps 폴더 이름 자동 추론
    base_name = os.path.basename(os.path.normpath(aria_dir))
    
    vrs_file = os.path.join(aria_dir, f"{base_name}.vrs")
    traj_csv = os.path.join(aria_dir, f"mps_{base_name}_vrs", "slam", "closed_loop_trajectory.csv")
    points_csv = os.path.join(aria_dir, f"mps_{base_name}_vrs", "slam", "semidense_points.csv.gz")

    # 3DGS가 요구하는 COLMAP 형태의 출력 폴더 구조 세팅
    img_out_dir = os.path.join(out_dir, "images")
    sparse_out_dir = os.path.join(out_dir, "sparse", "0")
    
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(sparse_out_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 2. VRS 및 MPS 데이터 프로바이더 로드
    # ----------------------------------------------------------------
    provider = data_provider.create_vrs_data_provider(vrs_file)
    rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")
    
    mps_paths = mps.MpsDataPaths()
    mps_paths.slam.closed_loop_trajectory = traj_csv
    mps_data_provider = mps.MpsDataProvider(mps_paths)
    
    # ----------------------------------------------------------------
    # 3. 카메라 캘리브레이션 (Fisheye -> Pinhole 변환 세팅)
    # ----------------------------------------------------------------
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib("camera-rgb")
    T_device_camera = device_calib.get_transform_device_sensor("camera-rgb")
    
    # 3DGS 학습을 위해 이미지를 Pinhole 형태로 쫙 펴줍니다. (해상도 및 화각 설정)
    width, height = args.width, args.height
    focal_length = 500.0  
    dst_calib = calibration.get_linear_camera_calibration(width, height, focal_length, "camera-rgb")
    
    # cameras.txt 작성
    f = dst_calib.get_focal_lengths() # [fx, fy]
    c = dst_calib.get_principal_point() # [cx, cy]

    out_width, out_height = width, height
    out_f, out_c = list(f), list(c)
    if args.rotate:
        out_width, out_height = height, width
        out_f = [f[1], f[0]]  # fx와 fy 스왑
        out_c = [height - c[1], c[0]]  # 회전된 중심점 재계산

    with open(os.path.join(sparse_out_dir, "cameras.txt"), "w") as f_cam:
        # COLMAP PINHOLE 규격: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        f_cam.write(f"1 PINHOLE {out_width} {out_height} {out_f[0]} {out_f[1]} {out_c[0]} {out_c[1]}\n")
    # ----------------------------------------------------------------
    # 4. 이미지 Undistort 및 Pose 추출 (images.txt 작성)
    # ----------------------------------------------------------------
    images_txt_lines = []
    num_images = provider.get_num_data(rgb_stream_id)
    
    print("이미지 Undistortion 및 Pose 계산 중...")
    valid_img_id = 1
    for i in tqdm(range(num_images)):
        image_data, record = provider.get_image_data_by_index(rgb_stream_id, i)
        image_time_ns = record.capture_timestamp_ns
        
        # 해당 시간의 Device Pose 가져오기
        pose_info = mps_data_provider.get_closed_loop_pose(image_time_ns)
        if not pose_info:
            continue # SLAM Pose 매칭이 안되는 프레임은 버림
            
        # 좌표계 연산 (Aria -> COLMAP)
        T_world_device = pose_info.transform_world_device
        T_world_camera = T_world_device @ T_device_camera
        T_camera_world = T_world_camera.inverse() # COLMAP은 World -> Camera 방향의 외부에 파라미터 사용
        matrix = T_camera_world.to_matrix()

        # 회전 적용 시 카메라 외부 파라미터(Extrinsics) 변경
        if args.rotate:
            # 시계방향 90도 회전을 의미하는 변환 행렬 (X' = -Y, Y' = X)
            R_90cw = np.array([
                [0, -1,  0,  0],
                [1,  0,  0,  0],
                [0,  0,  1,  0],
                [0,  0,  0,  1]
            ])
            matrix = R_90cw @ matrix

        # 회전행렬 -> 쿼터니언 변환
        rot = R.from_matrix(matrix[:3, :3]).as_quat() # Scipy는 [x, y, z, w] 순서로 반환
        qw, qx, qy, qz = rot[3], rot[0], rot[1], rot[2]
        tx, ty, tz = matrix[:3, 3]
        
        # 이미지 렌즈 왜곡 펴기 (Fisheye -> Pinhole)
        img_array = image_data.to_numpy_array()
        undistorted_img = calibration.distort_by_calibration(img_array, dst_calib, src_calib)

        # 회전 및 알파 마스크 적용 (이미지)
        if args.rotate:
            undistorted_img = np.rot90(undistorted_img, k=-1) # 시계 방향 90도 회전
            
        if args.alpha_mask:
            # np.any보다 np.max가 연산이 훨씬 빠릅니다.
            mask = (np.max(undistorted_img, axis=-1) > 0).astype(np.uint8) * 255
            undistorted_img = np.dstack((undistorted_img, mask))
            ext = "png"
        else:
            ext = "jpg"

        # [핵심 속도 최적화] PIL이 빠르게 처리할 수 있도록 메모리 연속성(Contiguous) 강제 정렬
        undistorted_img = np.ascontiguousarray(undistorted_img)

        # 저장
        img_name = f"frame_{valid_img_id:05d}.{ext}"
        save_path = os.path.join(img_out_dir, img_name)
        
        if ext == "png":
            # PNG 압축 옵션을 최소화(compress_level=1)하여 저장 속도를 극대화
            Image.fromarray(undistorted_img).save(save_path, optimize=False, compress_level=1)
        else:
            Image.fromarray(undistorted_img).save(save_path, quality=95)

        # images.txt 양식: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
        images_txt_lines.append(f"{valid_img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_name}")
        images_txt_lines.append("") # 3DGS는 2D 포인트 정보가 필요 없으므로 빈 줄 추가
        
        valid_img_id += 1

    with open(os.path.join(sparse_out_dir, "images.txt"), "w") as f_img:
        f_img.write("\n".join(images_txt_lines))

    # ----------------------------------------------------------------
    # 5. 초기 3D Point Cloud 추출 (points3D.txt 작성)
    # ----------------------------------------------------------------
    print("초기 3D 포인트 클라우드 추출 중...")
    points_df = pd.read_csv(points_csv)
    with open(os.path.join(sparse_out_dir, "points3D.txt"), "w") as f_pts:
        for _, row in points_df.iterrows():
            pid = int(row['uid'])
            x, y, z = row['px_world'], row['py_world'], row['pz_world']
            # 포맷: POINT3D_ID X Y Z R G B ERROR TRACK[]
            f_pts.write(f"{pid} {x} {y} {z} 128 128 128 0\n")
            
    print(f"변환 완료! 학습 폴더: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()