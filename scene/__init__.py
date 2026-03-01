#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param args: ModelParams (데이터셋 경로, 모델 저장 경로 등)
        :param gaussians: 앞서 생성한 빈 GaussianModel 객체
        :param load_iteration: 이어서 학습(resume)할 때 불러올 특정 이터레이션 번호
        :param shuffle: 학습 시 카메라(뷰포인트) 순서를 랜덤으로 섞을지 여부
        :param resolution_scales: 다중 해상도 학습/테스트를 위한 스케일 리스트 (기본은 원본 해상도 1.0)
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # -------------------------------------------------------------------------
        # REVIEW: 1. 체크포인트(이전 학습 기록) 확인
        # -------------------------------------------------------------------------
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # -------------------------------------------------------------------------
        # REVIEW: 2. 데이터셋 타입 자동 인식 및 로드
        # 데이터셋 폴더의 구조를 보고 COLMAP 데이터인지 Blender(합성) 데이터인지 파악합니다.
        # -------------------------------------------------------------------------
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # -------------------------------------------------------------------------
        # REVIEW: 3. 초기 메타데이터 백업 (처음 학습하는 경우만)
        # 나중에 뷰어를 띄우거나 결과를 디버깅할 때 사용할 수 있도록,
        # 원본 포인트 클라우드(input.ply)와 카메라 정보(cameras.json)를 출력 폴더에 복사해둡니다.
        # -------------------------------------------------------------------------
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # -------------------------------------------------------------------------
        # REVIEW: 4. 카메라 셔플 및 초기화
        # -------------------------------------------------------------------------
        if shuffle:
            # 딥러닝 학습 시 데이터 순서가 고정되어 있으면 과적합(Overfitting)이 올 수 있으므로 순서를 섞습니다.
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        # 씬(Scene)의 크기를 계산합니다. (가우시안의 초기 크기나 densify 기준을 잡을 때 쓰임)
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 지정된 해상도(resolution_scales)별로 실제 PyTorch 텐서가 포함된 Camera 객체 리스트를 만듭니다.
        # (이미지를 GPU에 올리거나, 종횡비/FOV 등을 계산하는 무거운 작업이 여기서 일어남)
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # -------------------------------------------------------------------------
        # REVIEW: 5. Gaussian 모델 파라미터 초기화
        # -------------------------------------------------------------------------
        if self.loaded_iter:    
            # 이어서 학습하는 경우: 저장해둔 point_cloud.ply 파일을 불러와서 가우시안들을 복원합니다.
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # 처음 학습하는 경우: COLMAP(또는 Blender)에서 얻은 듬성듬성한 초기 포인트 클라우드를
            # 바탕으로 가우시안들의 위치, 크기, 색상 등을 최초로 세팅합니다.
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    # -------------------------------------------------------------------------
    # REVIEW: 6. 모델 저장 (Save) 및 데이터 반환 유틸리티
    # -------------------------------------------------------------------------
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        # 가우시안들의 위치, SH(색상), 크기, 회전, 투명도 정보를 묶어 .ply 파일로 저장합니다.
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # 각 카메라 뷰마다 노출(Exposure) 값이 다를 수 있으므로 이를 보정하는 파라미터도 JSON으로 함께 저장합니다.
        # (밝기가 들쭉날쭉한 야외 데이터셋 등에서 유용한 기능)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        # 학습 루프(train.py)에서 카메라 뷰포인트를 뽑아갈 때 호출합니다.
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        # 평가/테스트 렌더링 시 호출합니다.
        return self.test_cameras[scale]
