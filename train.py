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
# train.py
# -----------------------------------------------------------------------------
# REVIEW (이 파일 한눈에 보기)
# -----------------------------------------------------------------------------
# 이 스크립트는 3D Gaussian Splatting의 "학습 루프"를 담당합니다.
#
# 큰 흐름:
#   1) Scene / GaussianModel 생성 및 optimizer 세팅
#   2) 매 iteration마다 랜덤 카메라(viewpoint) 하나 뽑아서 render
#   3) render 결과 vs GT 이미지로 loss 계산 (L1 + DSSIM, + optional depth reg)
#   4) backward
#   5) (densification 구간이면) gaussians를 늘리거나(prune/densify) opacity reset
#   6) optimizer step + 주기적 저장/테스트
#
# 핵심 키워드:
#   - render() 가 미분 가능(differentiable)하게 2D rasterization을 수행
#   - densify_and_prune() 가 포인트 수를 "학습 중"에 조절(성능의 핵심)
# -----------------------------------------------------------------------------

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# REVIEW: loss_utils의 ssim은 보통 "SSIM" 반환(높을수록 유사).
# 여기선 (1-SSIM)을 loss로 씁니다. 즉 SSIM을 최대화하는 것과 동일.

from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    # REVIEW: TensorBoard는 "있으면 쓰고 없으면 안 씀" (optional dependency)
    # 따라서 서버 환경/conda 환경에 따라 로깅이 조용히 꺼질 수 있음.
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    # REVIEW: fused_ssim이 있으면 ssim 계산을 더 빠르게(커스텀 CUDA/확장) 수행.
    # 없으면 파이썬 구현/일반 구현 ssim() 사용.
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    # REVIEW: SparseGaussianAdam은 "보이는 가우시안만" 업데이트해서 속도/메모리 최적화.
    # diff_gaussian_rasterization(가속 rasterizer) 설치가 되어 있어야 함.
    # opt.optimizer_type == "sparse_adam" 일 때만 사용.
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # -------------------------------------------------------------------------
    # REVIEW: training() 인자 의미
    #   dataset : 데이터/카메라/이미지/옵션(white_background 등) 포함
    #   opt     : OptimizationParams (iterations, lr schedule, densify 설정 등)
    #   pipe    : PipelineParams (SH 계산 방식, covariance 계산 방식, debug 등)
    #   testing_iterations : 테스트/리포트 수행 iteration 리스트
    #   saving_iterations  : 결과 저장 iteration 리스트
    #   checkpoint_iterations : 체크포인트 저장 iteration 리스트
    #   checkpoint : 재개(restore)할 체크포인트 경로
    #   debug_from : 특정 iteration부터 pipe.debug 켜서 디버깅
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # REVIEW: 1. 초기화 및 준비 단계
    # 데이터셋을 불러오고, 가우시안 모델을 세팅하며, 최적화(Optimizer)를 준비합니다.
    # -------------------------------------------------------------------------

    # Sparse Adam 옵션을 켰으나 설치되지 않은 경우 에러를 띄우고 종료합니다.
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    # 텐서보드 로거나 출력 폴더를 준비합니다.
    tb_writer = prepare_output_and_logger(dataset)

    # 가우시안 포인트들을 관리하는 핵심 클래스입니다. (SH degree, Optimizer 종류 설정)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    # Scene 클래스는 데이터셋(카메라, 이미지, 초기 포인트 클라우드)을 로드하고 가우시안 모델에 바인딩합니다.
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # 이전에 멈춘 체크포인트(.pth)가 있다면 로드하여 학습 상태(iter, 파라미터)를 복원합니다.
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 배경색 설정 (데이터셋 옵션에 따라 흰색 또는 검은색)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # GPU 연산 시간을 측정하기 위한 CUDA Event 객체
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # Depth 기반의 정규화(Regularization) loss를 사용할 경우, 학습 진행에 따른 가중치 감소 함수를 설정합니다.
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 학습에 사용할 카메라 뷰포인트들을 리스트(스택)로 복사해옵니다.
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # 로깅을 위한 지수 이동 평균(EMA) 변수
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 학습 진행률을 보여주는 tqdm 프로그레스 바
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # -------------------------------------------------------------------------
    # REVIEW: 2. 메인 학습 루프 시작
    # 매 이터레이션마다 카메라 1개를 뽑아 렌더링하고, 실제 이미지와 비교해 Loss를 구합니다.
    # -------------------------------------------------------------------------
    for iteration in range(first_iter, opt.iterations + 1):
        # [GUI 처리 로직] SIBR 실시간 뷰어와 통신하여 현재 화면을 렌더링해서 보내거나 조작을 받습니다.
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # 설정된 스케줄에 따라 가우시안 중심점(XYZ)의 Learning Rate를 점진적으로 줄입니다.
        gaussians.update_learning_rate(iteration)

        # 1000 이터레이션마다 Spherical Harmonics(SH, 보는 각도에 따른 색상 변화 표현)의 차수를 높입니다.
        # 처음부터 고차원 SH를 학습하면 불안정하므로 점진적으로 디테일을 올리는 기법입니다.
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 스택에서 무작위로 카메라 뷰포인트 하나를 뽑습니다. (비어있으면 다시 채움)
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 배경을 랜덤으로 할지 고정색으로 할지 결정합니다.
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # -------------------------------------------------------------------------
        # REVIEW: 3. 렌더링 (Forward Pass)
        # 선택된 카메라 위치에서 현재 가우시안들을 2D 이미지로 Rasterization 합니다.
        # 이 함수는 미분 가능(differentiable)하므로 역전파가 가능합니다.
        # -------------------------------------------------------------------------
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        # 렌더링된 2D 이미지
        image = render_pkg["render"]                                # 렌더링된 2D 이미지
        viewspace_point_tensor = render_pkg["viewspace_points"]     # 2D 화면상 가우시안 중심 좌표 (기울기 누적용)
        visibility_filter = render_pkg["visibility_filter"]         # 현재 뷰에서 보이는 가우시안 마스크
        radii = render_pkg["radii"]                                 # 화면에 투영된 가우시안의 2D 반지름

        # 객체 마스킹(배경 제거 등) 옵션이 있다면 적용합니다. -> 자동으로 RGBA로 들어오면 alpha mask적용해서 학습하는 듯?
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        # -------------------------------------------------------------------------
        # REVIEW: 4. Loss 계산 및 역전파 (Backward Pass)
        # 렌더링 이미지와 Ground Truth(실제 정답) 이미지를 비교합니다.
        # -------------------------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()

        # 픽셀 단위의 L1 Loss
        Ll1 = l1_loss(image, gt_image)
        
        # 구조적 유사도(SSIM) Loss 계산 (FUSED_SSIM이 빠름)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 최종 Loss = L1 Loss와 DSSIM(1-SSIM)의 가중합
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # (선택적) Depth 정규화: 단안(Monocular) Depth 맵이 있다면 3D 구조를 더 잘 잡도록 도와줍니다.
        # ---------------------------------------------------------------------
        # Depth regularization (깊이 정규화 / 선택적)
        #
        # 목적:
        #   RGB 재구성(loss: L1 + SSIM)만으로도 학습은 되지만,
        #   텍스처가 약하거나 반복무늬/무텍스처 영역에서는 깊이(3D 구조)가 흔들릴 수 있음.
        #   그래서 "외부에서 얻은 단안 depth(prior)"와 렌더된 depth를 맞추는 보조 loss를 추가함.
        #
        # 왜 depth가 아니라 inverse depth(1/depth)를 쓰나?
        #   - 원근 카메라에서는 가까운 곳의 depth 변화가 더 중요하고 민감함
        #   - inverse depth는 가까운 영역의 차이를 더 크게 반영해 최적화가 안정적인 편
        #
        # depth_l1_weight(iteration):
        #   - iteration에 따라 depth loss 가중치를 스케줄링(초반/후반 영향 조절)
        #
        # viewpoint_cam.depth_reliable:
        #   - 이 뷰의 depth prior가 믿을만한지(깨짐/결측/노이즈 심함 등) 표시
        #   - False면 depth loss를 아예 건너뜀
        # ---------------------------------------------------------------------
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            # render_pkg["depth"]:
            #   - 현재 Gaussian 장면을 이 카메라에서 렌더링했을 때의 "예측 inverse depth" (모델 출력)
            invDepth = render_pkg["depth"]

            # viewpoint_cam.invdepthmap:
            #   - 데이터셋에서 제공되는 "단안 inverse depth prior" (정답이라기보단 힌트/규제항)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()

            # depth_mask:
            #   - depth prior가 유효한 픽셀만 1, 나머지는 0 (결측/무효 영역 제외)
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # 픽셀 단위 L1: |invDepth - mono_invdepth|
            # 마스크를 곱해서 유효한 픽셀만 남기고,
            # mean()으로 전체 평균을 내서 스칼라 loss로 만듦
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()

            # 스케줄 가중치 적용 (iteration에 따라 depth 규제가 강해지거나 약해짐)
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            # 최종 loss에 더해줌 (RGB loss + depth loss)
            loss += Ll1depth
            # 로그 출력을 위해 파이썬 float로 변환
            Ll1depth = Ll1depth.item()
        else:
            # depth loss를 사용하지 않는 경우(가중치=0 or depth가 unreliable)
            Ll1depth = 0

        loss.backward()
        iter_end.record()
        # -------------------------------------------------------------------------
        # REVIEW: 5. 최적화 및 구조 변경 (No Grad 블록)
        # 기울기를 반영하고, 필요에 따라 가우시안을 쪼개거나 삭제합니다.
        # -------------------------------------------------------------------------
        with torch.no_grad():
            # 진행 바 업데이트용 Loss 기록
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 정해진 주기마다 TensorBoard 로깅 및 PSNR 평가, 그리고 .ply 모델 저장을 수행합니다.
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # --- 핵심: Densification (밀도 제어) 구간 ---
            # 특정 이터레이션(보통 15,000)까지만 포인트 개수를 늘리거나 줄입니다.
            if iteration < opt.densify_until_iter:
                # 가지치기(Pruning)를 위해 각 가우시안이 2D 화면상에서 가졌던 최대 반지름을 기록합니다.
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 위치(XYZ) 변화에 대한 기울기(Gradient)를 누적합니다. (어느 부분이 부족한지 파악)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 일정 주기(densification_interval)마다 실제로 가우시안을 쪼개거나(Split/Clone) 지웁니다(Prune).
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 기울기가 큰(복잡한) 곳은 나누고, 투명도(Opacity)가 너무 낮거나 너무 커진 가우시안은 제거합니다.
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                # 가우시안이 구름처럼 퍼지는 현상(Floaters)을 막기 위해 주기적으로 투명도를 리셋(낮춤)합니다. -> 0으로 낮춰서 floater가 이미지에 큰 영향을 주도록...
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer Step (실제 파라미터 업데이트)
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                # Sparse Adam이면 이번 뷰에서 화면에 보인(visible) 가우시안들만 업데이트하여 속도를 높입니다.
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 나중에 이어서 학습할 수 있도록 정해진 이터레이션에 체크포인트(.pth)를 저장합니다.
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    # -------------------------------------------------------------------------
    # REVIEW: 커맨드라인 인자(Argument) 설정 부분
    # 3DGS 학습을 실행할 때 터미널에서 입력받는 다양한 옵션들을 정의합니다.
    # -------------------------------------------------------------------------
    parser = ArgumentParser(description="Training script parameters")
    
    # 1. 외부 파일(arguments.py)에서 정의된 파라미터 그룹을 불러옵니다.
    lp = ModelParams(parser)            # Model: 데이터셋 경로, SH degree, 배경색 등
    op = OptimizationParams(parser)     # Optimization: Learning Rate, densify 시작/종료 시점 등
    pp = PipelineParams(parser)         # Pipeline: 렌더링 시 파이썬 구현체 사용 여부, 디버그 옵션 등

    # 2. SIBR 실시간 뷰어(네트워크 GUI) 연동을 위한 네트워크 설정
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)

    # 3. 디버깅 및 에러 추적 옵션
    parser.add_argument('--debug_from', type=int, default=-1)
    # 지정한 이터레이션부터 렌더링 파이프라인의 디버그 모드를 켭니다. (-1은 사용 안 함)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # PyTorch의 autograd anomaly detection을 켭니다. Loss가 NaN이 뜨는 등 
    # 그래디언트 역전파 중 문제가 생겼을 때 원인을 찾기 좋습니다. (단, 학습 속도는 느려집니다.)

    # 4. 평가 및 저장 주기 설정
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # 지정한 이터레이션(기본 7000, 30000)에서 테스트 셋을 렌더링하여 PSNR / L1 loss 등을 평가합니다.
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # 지정한 이터레이션에 학습된 가우시안 포인트 클라우드(.ply)를 저장합니다.
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # 모델의 전체 상태(Optimizer 상태 등 포함)를 저장하여 나중에 이어서 학습할 수 있게 합니다.
    
    # 5. 기타 편의 옵션
    parser.add_argument("--quiet", action="store_true")
    # 터미널에 출력되는 로그(진행률 등)를 최소화합니다.
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    # 실시간 GUI 뷰어 서버를 켭니다. SSH 환경 등 뷰어가 필요 없는 서버 환경에서는 이 옵션으로 끌 수 있습니다.
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # 이전에 저장해둔 .pth 체크포인트 파일 경로를 입력하면, 해당 시점부터 이어서 학습(Resume)합니다.

    # -------------------------------------------------------------------------
    # REVIEW: 인자 파싱 및 학습 준비
    # -------------------------------------------------------------------------
    args = parser.parse_args(sys.argv[1:])
    # 사용자가 따로 지정하지 않았더라도, 전체 학습이 끝나는 시점(args.iterations)에는 무조건 한 번 저장하도록 추가합니다.
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # 난수 시드(seed)를 고정하여 학습 결과의 재현성(reproducibility)을 확보합니다.
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # 뷰어를 비활성화하지 않았다면, 위에서 설정한 IP/PORT로 뷰어와 통신할 서버를 엽니다.
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 실제 학습을 수행하는 메인 함수 호출. 
    # 파서에서 그룹별로 추출한(extract) 파라미터 묶음들과 리스트 형태의 반복 주기들을 넘겨줍니다.
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
