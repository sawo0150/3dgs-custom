from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, List

from omegaconf import DictConfig, OmegaConf


MODEL_KEYS = [
    "sh_degree", "source_path", "model_path", "images", "depths", "resolution",
    "white_background", "train_test_exp", "data_device", "eval",
    "init_pcd_filter", "init_pcd_expand_factor",
]
PIPELINE_KEYS = [
    "convert_SHs_python", "compute_cov3D_python", "debug", "antialiasing", "beta",
]
OPT_KEYS = [
    "iterations", "position_lr_init", "position_lr_final", "position_lr_delay_mult",
    "position_lr_max_steps", "feature_lr", "opacity_lr", "scaling_lr", "rotation_lr",
    "exposure_lr_init", "exposure_lr_final", "exposure_lr_delay_steps",
    "exposure_lr_delay_mult", "percent_dense", "lambda_dssim", "densification_interval",
    "opacity_reset_interval", "densify_from_iter", "densify_until_iter",
    "densify_grad_threshold", "min_opacity_prune_threshold", "depth_l1_weight_init",
    "depth_l1_weight_final", "sparse_depth_weight_init", "sparse_depth_weight_final",
    "sparse_depth_max_points", "sparse_depth_global_max_points", "sparse_depth_min_depth",
    "sparse_depth_require_rendered", "random_background", "optimizer_type",
    "optimizer_beta1", "optimizer_beta2",
]


def _namespace_from_keys(src: dict, keys: list[str]) -> SimpleNamespace:
    payload = {k: src[k] for k in keys if k in src}
    return SimpleNamespace(**payload)


def build_legacy_groups(cfg: DictConfig) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    dataset = OmegaConf.to_container(cfg.dataset, resolve=True)
    train = OmegaConf.to_container(cfg.train, resolve=True)

    model_src = {
        "sh_degree": dataset.get("sh_degree", 3),
        "source_path": dataset.get("source_path"),
        "model_path": dataset.get("model_path"),
        "images": dataset.get("images", "images"),
        "depths": dataset.get("depths", ""),
        "resolution": dataset.get("resolution", -1),
        "white_background": dataset.get("white_background", False),
        "train_test_exp": dataset.get("train_test_exp", False),
        "data_device": dataset.get("data_device", "cuda"),
        "eval": dataset.get("eval", False),
        "init_pcd_filter": dataset.get("init_pcd_filter", False),
        "init_pcd_expand_factor": float(dataset.get("init_pcd_expand_factor", 3.0)),
    }

    pipe_src = {
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
        "antialiasing": False,
        "beta": float(train.get("softmax_depth_beta", 5.0)),
    }

    model_args = _namespace_from_keys(model_src, MODEL_KEYS)
    opt_args = _namespace_from_keys(train, OPT_KEYS)
    pipe_args = _namespace_from_keys(pipe_src, PIPELINE_KEYS)
    return model_args, opt_args, pipe_args


def build_schedule_lists(cfg: DictConfig) -> Tuple[List[int], List[int], List[int]]:
    train = OmegaConf.to_container(cfg.train, resolve=True)
    testing_iterations = list(train.get("testing_iterations", []))
    save_iterations = list(train.get("save_iterations", []))
    checkpoint_iterations = list(train.get("checkpoint_iterations", []))
    iterations = int(train["iterations"])
    if iterations not in save_iterations:
        save_iterations.append(iterations)
    return testing_iterations, save_iterations, checkpoint_iterations


def inject_runtime_args(cfg: DictConfig, model_args: SimpleNamespace, opt_args: SimpleNamespace,
                        pipe_args: SimpleNamespace, run_dir: Path) -> None:
    model_args.model_path = str(run_dir)
    setattr(model_args, "keyframe_list_path", cfg.selection.get("keyframe_list_path", None))
    setattr(model_args, "selection_keep_test", cfg.selection.get("selection_keep_test", True))
    if cfg.pose.get("use_pose_override", False):
        setattr(model_args, "pose_override_path", cfg.pose.get("pose_path", None))
        setattr(model_args, "pose_override_format", cfg.pose.get("format", "auto"))
    else:
        setattr(model_args, "pose_override_path", None)
        setattr(model_args, "pose_override_format", "auto")

    setattr(opt_args, "debug_from", int(cfg.train.get("debug_from", -1)))
    setattr(opt_args, "detect_anomaly", bool(cfg.train.get("detect_anomaly", False)))
    setattr(opt_args, "quiet", bool(cfg.train.get("quiet", False)))
    setattr(opt_args, "viewer_enabled", bool(cfg.viewer.get("enabled", False)))
    setattr(opt_args, "viewer_ip", str(cfg.viewer.get("ip", "127.0.0.1")))
    setattr(opt_args, "viewer_port", int(cfg.viewer.get("port", 6009)))
    setattr(opt_args, "ambiguity_log_interval", int(cfg.logging.get("ambiguity_log_interval", 2000)))
    setattr(opt_args, "gaussian_metrics_log_interval", int(cfg.logging.get("gaussian_metrics_log_interval", 2000)))
    setattr(opt_args, "low_opacity_threshold", float(cfg.logging.get("low_opacity_threshold", 0.1)))
    setattr(opt_args, "large_scale_threshold", float(cfg.logging.get("large_scale_threshold", 0.1)))
    setattr(opt_args, "diag_grad_interval", int(cfg.logging.get("diag_grad_interval", 500)))
