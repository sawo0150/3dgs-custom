#!/usr/bin/env python3
"""render_view_subset.py — render.py의 최소 변형: 전체 뷰가 아니라 지정한 인덱스만 렌더링.
gs_floaterLab PPT용 before/after 렌더 생성을 위해 신설 (render.py는 건드리지 않음).

사용: python render_view_subset.py -m <model_path> --iteration 30000 --indices 100 500 900
출력: <model_path>/ppt_views/ours_<iter>/{renders,gt}/<idx>.png
"""
import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def main(dataset, iteration, pipeline, indices, separate_sh):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTrainCameras()
        out_dir = os.path.join(dataset.model_path, "ppt_views", f"ours_{scene.loaded_iter}")
        makedirs(os.path.join(out_dir, "renders"), exist_ok=True)
        makedirs(os.path.join(out_dir, "gt"), exist_ok=True)

        for idx in indices:
            if idx >= len(views):
                print(f"[skip] idx {idx} >= {len(views)} views")
                continue
            view = views[idx]
            rendering = render(view, gaussians, pipeline, background,
                                use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(out_dir, "renders", f"{idx:05d}.png"))
            torchvision.utils.save_image(gt, os.path.join(out_dir, "gt", f"{idx:05d}.png"))
            print(f"[saved] idx={idx} name={view.image_name}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a subset of views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering subset " + args.model_path)
    safe_state(args.quiet)
    main(model.extract(args), args.iteration, pipeline.extract(args), args.indices, SPARSE_ADAM_AVAILABLE)
