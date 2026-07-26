#
# train_incremental.py
# -----------------------------------------------------------------------------
# SLAM keyframe 이벤트 기반 incremental 3DGS 학습 진입점.
#
# train.py(배치 트레이너, exp01~47이 검증한 경로)는 건드리지 않는다 — 이 파일이
# incremental 전용 오케스트레이션을 전부 소유한다. GaussianModel/render/loss 등
# 재사용 가능한 조각만 3dgs-custom에서 import해서 쓴다.
#
# 설계 근거 (VINGS-Mono_custom 분석, gs_floaterLab exp48 Phase 0b 실패 원인 분석):
#   exp48 Phase 0(1 keyframe = 1 이미지 = 1 서브프로세스)은 각 영역이 "딱 한 번,
#   200 iter 학습받고 영원히 방치"되는 구조라 held-out PSNR 15.8dB로 사실상
#   미학습이었다. 이 스크립트는 두 가지로 그 문제를 구조적으로 없앤다:
#     1) 로컬 윈도우 샘플링: 매 iteration 카메라를 "그 청크 하나"가 아니라
#        "최근 window_size개 keyframe의 카메라 풀"에서 무작위로 뽑는다.
#        (VINGS-Mono의 train_once_gaussian이 tracker local window에서
#         random.randint로 뽑는 것과 동일한 패턴)
#     2) freeze-when-stable: 최근 이벤트 동안 accum_rgb_grad가 거의 안 늘어난
#        gaussian은 "stable"로 표시하고 이후 gradient를 0으로 밀어 업데이트에서
#        제외한다. (VINGS-Mono의 stablemask_control을 우리 accum_rgb_grad
#        버퍼로 이식 — 새 buffer를 gaussian_model.py에 추가하지 않고 이 스크립트가
#        로컬로 관리한다)
#
# v1 스코프 (의도적 단순화, 미포함):
#   - densify/prune 없음 (add_extra_points로만 점이 늘어남) — cameras_extent=0
#     버그·densify_until_iter 이슈 계열을 이번 v1에서는 아예 우회한다.
#   - carve loss는 옵션(--carve_loss_config)으로만 붙어있고 기본 off — 먼저
#     "로컬 윈도우 + freeze"만으로 배관이 실제로 고쳐지는지 격리해서 검증한다.
#   - VINGS-Mono의 render-residual 기반 init 정리(add_new_frame의 delete-then-add)와
#     주기적 전체 재검증(storage_control)은 다음 단계 후보로 남겨둔다.
# -----------------------------------------------------------------------------

import os
import sys
import json
import collections
import random
import argparse
from pathlib import Path
from collections import deque

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from gaussian_renderer import render
from eval.carve_loss import CarveLoss, CarveLossConfig
from scene.dataset_readers import read_points3D_text
from scene.gaussian_model import BasicPointCloud

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


def load_chunk_cameras_and_pcd(chunk_dir: Path, cam_args):
    """청크 폴더(COLMAP 텍스트 포맷)에서 카메라 전부(dense 버전은 keyframe 구간의
    조밀한 프레임 수십 장, keyframe-only 버전은 1장)와 그 청크의 point cloud를 읽어온다.
    Scene()을 통째로 안 쓰는 이유: Scene은 호출 즉시 GaussianModel.create_from_pcd를
    불러버려서(냉시작 가정) incremental 루프의 "이미 존재하는 gaussians에 이어붙이기"와
    맞지 않는다 — 그래서 하위 레벨 함수(sceneLoadTypeCallbacks/cameraList_from_camInfos)만
    직접 호출해 카메라·pcd만 뽑는다.
    """
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(chunk_dir), "images", "", False, False,
        init_pcd_filter=False, init_pcd_expand_factor=3.0,
    )
    cams = cameraList_from_camInfos(scene_info.train_cameras, 1.0, cam_args, False, False)
    return cams, scene_info.point_cloud, scene_info.nerf_normalization["radius"]


def main():
    parser = argparse.ArgumentParser(description="exp48 incremental v2 — local window + freeze-when-stable")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--manifest", type=str, required=True,
                         help="build_incremental_chunks.py가 만든 manifest.json 경로")
    parser.add_argument("--dataset_root", type=str, required=True,
                         help="chunk_NNN/ 폴더들이 들어있는 04_incremental 디렉토리")
    parser.add_argument("--window_size", type=int, default=8,
                         help="로컬 윈도우에 유지할 최근 keyframe '구간'의 개수 "
                              "(dense 버전에서는 구간 하나당 프레임 수십 장이 들어있을 수 있음)")
    parser.add_argument("--iters_per_event", type=int, default=60,
                         help="keyframe 이벤트 하나당 학습 iteration 수")
    parser.add_argument("--stable_grad_thresh", type=float, default=5e-4,
                         help="이 이벤트 동안 accum_rgb_grad 증가분이 이 값 미만이면 stable로 승격")
    parser.add_argument("--save_every", type=int, default=10,
                         help="몇 이벤트마다 체크포인트/ply를 저장할지")
    parser.add_argument("--max_events", type=int, default=-1,
                         help="디버그용 — 처음 N개 keyframe만 처리 (-1이면 전부)")
    parser.add_argument("--carve_loss_config", type=str, default=None)
    parser.add_argument("--constant_lr", action="store_true",
                         help="xyz LR 감쇠 스케줄 끄고 초기값 고정(VINGS-Mono_custom 방식) — "
                              "position_lr_max_steps=30000 기준 감쇠가 우리 총 iteration(수천)엔 "
                              "안 맞을 수 있다는 가설 검증용")
    parser.add_argument("--init_source", type=str, default="slam", choices=["slam", "ppm", "roma", "both", "all", "hybrid"],
                         help="chunk_idx>=1의 신규 init 점 소스: slam(원래 SLAM map point) / "
                              "ppm(depth-mono+PPM) / roma(RoMA dense correspondence) / both / all / hybrid")
    parser.add_argument("--size_threshold_from_iter", type=int, default=3000,
                         help="이 iteration 이후로 화면상 너무 큰 gaussian도 pruning 대상에 포함 "
                              "(opacity_reset_interval과 분리된 별도 값)")
    parser.add_argument("--trace_event", type=int, default=-1,
                         help="이 이벤트에서 새로 추가된 gaussian들을 ancestor_idx로 표식해두고 "
                              "이후 매 이벤트마다 생존/opacity/scale을 추적 출력 (-1이면 끔)")
    parser.add_argument("--cameras_extent_source", type=str, default=None,
                         help="densify_and_prune의 scene extent를 이 데이터셋(예: data/03_rgb_3dgs_full)의 "
                              "전체 카메라 분포로 고정해서 씀 — replay라 최종 궤적을 미리 아는 걸 활용. "
                              "생략 시 chunk_000의 로컬 extent(부정확) 사용")

    args = parser.parse_args()
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    safe_state(False)

    out_dir = Path(dataset.model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "point_cloud").mkdir(exist_ok=True)
    # render.py/get_combined_args가 model_path/cfg_args를 하드코딩해서 읽으므로 미리 써둔다
    # (train.py의 prepare_output_and_logger와 동일 포맷).
    from argparse import Namespace as _NS
    with open(out_dir / "cfg_args", "w") as _f:
        _f.write(str(_NS(**vars(dataset))))

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.max_events > 0:
        manifest = manifest[: args.max_events]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    fixed_cameras_extent = None
    if args.cameras_extent_source:
        _full_info = sceneLoadTypeCallbacks["Colmap"](args.cameras_extent_source, "images", "", False, False)
        fixed_cameras_extent = _full_info.nerf_normalization["radius"]
        print(f"[incremental] fixed cameras_extent = {fixed_cameras_extent:.3f} (from {args.cameras_extent_source})")

    carve_loss = CarveLoss(CarveLossConfig.from_yaml(args.carve_loss_config), args.dataset_root) \
        if args.carve_loss_config else CarveLoss(CarveLossConfig(enabled=False), args.dataset_root)

    window = deque(maxlen=args.window_size)
    stable_mask = None          # (N,) bool — True면 이 gaussian은 gradient를 0으로 밀어 freeze
    prev_accum_rgb_grad = None  # (N,) float — 직전 이벤트 종료 시점의 accum_rgb_grad 스냅샷

    draw_counts = collections.Counter()  # image_name -> 학습 중 실제로 뽑힌 횟수 (진단용)
    global_iter = 0
    event_log = []
    trace_ancestor_ids = None  # 추적 대상 이벤트에서 태어난 gaussian들의 ancestor_idx 값(고정 ID)
    event_ranges = {}          # event_idx -> (start_idx, end_idx) of ancestor_idx

    for event_idx, chunk in enumerate(manifest):
        chunk_dir = Path(args.dataset_root) / f"chunk_{chunk['chunk_idx']:03d}"
        cams, pcd, radius = load_chunk_cameras_and_pcd(chunk_dir, dataset)

        n_before = gaussians._xyz.shape[0] if gaussians._xyz.numel() > 0 else 0

        # scene extent는 학습 내내 고정값이어야 함(배치 트레이너의 scene.cameras_extent와 동일 의미).
        # event_idx==0에서 한 번만 정해서 이후 이벤트에서도 그대로 씀.
        if event_idx == 0:
            cameras_extent = fixed_cameras_extent if fixed_cameras_extent else (radius if radius > 0 else 3.0)

        if event_idx == 0:
            # 첫 keyframe: 냉시작. Scene을 안 쓰므로 create_from_pcd를 직접 호출.
            # (points3D.txt는 chunk_000에 한해 real point — build_incremental_chunks.py 참조)
            gaussians.create_from_pcd(pcd, cams, cameras_extent)
            gaussians.training_setup(opt)
            event_ranges[0] = (0, gaussians._xyz.shape[0])
            stable_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
            prev_accum_rgb_grad = gaussians.accum_rgb_grad.clone()
        else:
            # chunk_idx>=1은 points3D.txt가 3점짜리 더미(Scene 냉시작용)이고,
            # 진짜 신규 map point는 extra_points3D.txt(SLAM)와/또는 ppm_points3D.txt
            # (depth-mono+PPM, build_depthmono_ppm_chunks.py 산출물, --init_source로 선택)에 있음.
            xyz_parts, rgb_parts = [], []
            if args.init_source in ("slam", "both", "all", "hybrid"):
                p = chunk_dir / "sparse" / "0" / "extra_points3D.txt"
                if p.exists() and p.stat().st_size > 0:
                    xyz, rgb, _ = read_points3D_text(str(p))
                    xyz_parts.append(xyz); rgb_parts.append(rgb)
            if args.init_source in ("ppm", "both", "all", "hybrid"):
                p = chunk_dir / "sparse" / "0" / "ppm_points3D.txt"
                if p.exists() and p.stat().st_size > 0:
                    xyz, rgb, _ = read_points3D_text(str(p))
                    xyz_parts.append(xyz); rgb_parts.append(rgb)
            if args.init_source in ("roma", "all", "hybrid"):
                p = chunk_dir / "sparse" / "0" / "roma_points3D.txt"
                if p.exists() and p.stat().st_size > 0:
                    xyz, rgb, _ = read_points3D_text(str(p))
                    xyz_parts.append(xyz); rgb_parts.append(rgb)
            if xyz_parts:
                xyz = np.concatenate(xyz_parts, axis=0)
                rgb = np.concatenate(rgb_parts, axis=0)
                extra_pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
                gaussians.add_extra_points(extra_pcd)
            n_new = gaussians._xyz.shape[0] - n_before
            event_ranges[event_idx] = (n_before, gaussians._xyz.shape[0])
            stable_mask = torch.cat([stable_mask, torch.zeros(n_new, dtype=torch.bool, device="cuda")])
            prev_accum_rgb_grad = torch.cat([prev_accum_rgb_grad, torch.zeros(n_new, device="cuda")])

        if event_idx == args.trace_event:
            # 이 이벤트에서 막 태어난 gaussian들의 ancestor_idx를 표식 — 이후 split/clone으로
            # 자식이 생겨도 ancestor_idx는 부모 것을 물려받으므로(gaussian_model.py 확인됨)
            # 이 값들로 "이 keyframe 혈통"을 계속 추적할 수 있다.
            trace_ancestor_ids = gaussians.ancestor_idx[n_before:].clone()
            print(f"[trace] event {event_idx} 추적 시작: gaussian {len(trace_ancestor_ids)}개 표식")

        # dense 버전은 한 keyframe 구간에 프레임이 수십 장 있을 수 있어 윈도우는
        # "구간(bucket)의 리스트"를 들고 있다가 샘플링 시 펼쳐서(flatten) 고른다.
        window.append(cams)
        pooled_cams = [c for bucket in window for c in bucket]

        for _ in range(args.iters_per_event):
            global_iter += 1
            # VINGS-Mono_custom은 xyz LR 감쇠 스케줄이 없음(고정 LR) — "총 iteration을 미리 안다"는
            # 가정이 온라인/incremental과 안 맞기 때문으로 추정, --constant_lr로 그 방식 테스트 가능.
            if not args.constant_lr:
                gaussians.update_learning_rate(global_iter)
            if global_iter % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cam = random.choice(pooled_cams)
            draw_counts[viewpoint_cam.image_name] += 1

            render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                 use_trained_exp=dataset.train_test_exp)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            loss.backward(retain_graph=carve_loss.cfg.enabled)

            if gaussians._xyz.grad is not None:
                gaussians.accum_rgb_grad += gaussians._xyz.grad.norm(dim=-1)

            if carve_loss.cfg.enabled:
                L_carve, _ = carve_loss.compute_loss(gaussians, global_iter)
                if L_carve is not None:
                    L_carve.backward()

            # freeze-when-stable: stable gaussian의 gradient를 0으로 밀어 optimizer.step에서 제외
            with torch.no_grad():
                for pname in ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest"):
                    g = getattr(gaussians, pname).grad
                    if g is not None:
                        g[stable_mask] = 0

            # --- Densification (train.py의 densify_and_prune 구간과 동일 로직, 절대 iteration 기준) ---
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            with torch.no_grad():
                if global_iter < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                    gaussians.accum_visibility[visibility_filter] += 1

                    if global_iter > opt.densify_from_iter and global_iter % opt.densification_interval == 0:
                        # train.py 원본은 size_threshold 발동 시점을 opacity_reset_interval로 재활용하는데
                        # (reset 직후부터 큰 gaussian 정리를 시작한다는 의미), 이 스크립트에서 opacity_reset을
                        # 끄고 테스트하면 size_threshold까지 같이 꺼져버려 변수가 섞인다 — 별도 고정값으로 분리.
                        size_threshold = 20 if global_iter > args.size_threshold_from_iter else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, opt.min_opacity_prune_threshold,
                            cameras_extent, size_threshold, radii, iteration=global_iter,
                        )
                        # densify_and_prune이 N을 바꾸므로 이 스크립트가 로컬로 관리하는
                        # stable_mask/prev_accum_rgb_grad도 다시 맞춰야 함. GaussianModel 내부
                        # 9종 버퍼(ancestor_idx 등)는 densify_and_split/clone/prune_points가
                        # 이미 정합적으로 처리함 — 여기선 우리 것만 재정렬.
                        _n = gaussians._xyz.shape[0]
                        if stable_mask.shape[0] != _n:
                            stable_mask = torch.zeros(_n, dtype=torch.bool, device="cuda")
                            prev_accum_rgb_grad = gaussians.accum_rgb_grad.clone()

                    if global_iter % opt.opacity_reset_interval == 0:
                        # Selective opacity reset: only reset Gaussians visible/active in current window.
                        active_events = range(max(0, event_idx - len(window) + 1), event_idx + 1)
                        active_ancestors = []
                        for e in active_events:
                            if e in event_ranges:
                                start_idx, end_idx = event_ranges[e]
                                active_ancestors.append((start_idx, end_idx))
                        
                        active_mask = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.bool, device="cuda")
                        for start_idx, end_idx in active_ancestors:
                            active_mask = active_mask | ((gaussians.ancestor_idx >= start_idx) & (gaussians.ancestor_idx < end_idx))
                        
                        opacities = gaussians.get_opacity.clone()
                        opacities_reset = gaussians.inverse_opacity_activation(
                            torch.min(opacities, torch.ones_like(opacities) * 0.01)
                        )
                        
                        current_opacities_raw = gaussians._opacity.clone()
                        current_opacities_raw[active_mask] = opacities_reset[active_mask]
                        
                        optimizable_tensors = gaussians.replace_tensor_to_optimizer(current_opacities_raw, "opacity")
                        gaussians._opacity = optimizable_tensors["opacity"]

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # 이벤트 종료: stable-mask 갱신 (이번 이벤트 동안 거의 안 움직인 gaussian → stable 승격)
        with torch.no_grad():
            delta = gaussians.accum_rgb_grad - prev_accum_rgb_grad
            newly_stable = (~stable_mask) & (delta < args.stable_grad_thresh)
            stable_mask = stable_mask | newly_stable
            prev_accum_rgb_grad = gaussians.accum_rgb_grad.clone()

        n_total = gaussians._xyz.shape[0]
        n_stable = int(stable_mask.sum().item())
        print(f"[event {event_idx:03d}] kf_id={chunk['kf_id']} window_kfs={len(window)} "
              f"pool={len(pooled_cams)} N={n_total} (+{n_total - n_before}) stable={n_stable} loss={loss.item():.4f}")
        event_log.append({"event_idx": event_idx, "kf_id": chunk["kf_id"], "N": n_total,
                           "pool_size": len(pooled_cams),
                           "n_stable": n_stable, "global_iter": global_iter, "loss": float(loss.item())})

        if trace_ancestor_ids is not None:
            with torch.no_grad():
                alive_mask = torch.isin(gaussians.ancestor_idx, trace_ancestor_ids)
                n_alive = int(alive_mask.sum().item())
                in_window = event_idx - args.trace_event < args.window_size
                if n_alive > 0:
                    op = gaussians.get_opacity[alive_mask]
                    sc = gaussians.get_scaling[alive_mask].max(dim=1).values
                    print(f"[trace] event {event_idx:03d} (윈도우 {'안' if in_window else '밖'}) "
                          f"생존 {n_alive}/{len(trace_ancestor_ids)} "
                          f"opacity(mean/min/max)={op.mean().item():.3f}/{op.min().item():.3f}/{op.max().item():.3f} "
                          f"scale_max(mean/max)={sc.mean().item():.4f}/{sc.max().item():.4f}")
                else:
                    print(f"[trace] event {event_idx:03d} (윈도우 {'안' if in_window else '밖'}) "
                          f"생존 0/{len(trace_ancestor_ids)} — 전멸")

        if (event_idx + 1) % args.save_every == 0 or event_idx == len(manifest) - 1:
            # render.py/Scene()가 "point_cloud/iteration_{N}/point_cloud.ply"를 하드코딩해서 찾으므로
            # 기존 render/eval 툴체인과 바로 호환되도록 event가 아니라 global_iter로 이름 붙인다.
            ply_dir = out_dir / "point_cloud" / f"iteration_{global_iter}"
            ply_dir.mkdir(parents=True, exist_ok=True)
            gaussians.save_ply(str(ply_dir / "point_cloud.ply"))
            torch.save((gaussians.capture(), global_iter), str(out_dir / f"chkpnt{global_iter}.pth"))
            (out_dir / "event_log.json").write_text(json.dumps(event_log, indent=2), encoding="utf-8")

    (out_dir / "draw_counts.json").write_text(json.dumps(dict(draw_counts), indent=2), encoding="utf-8")
    print(f"\nDone. {len(manifest)} events, {global_iter} total iterations, "
          f"final N={gaussians._xyz.shape[0]}, stable={int(stable_mask.sum().item())}.")


if __name__ == "__main__":
    main()
