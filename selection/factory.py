from __future__ import annotations

from types import SimpleNamespace

from selection.uniform import UniformSelector
from selection.pose_threshold import PoseThresholdSelector
from selection.hybrid import HybridSelector
from selection.parallax import ParallaxSelector


class AllSelector(UniformSelector):
    def __init__(self):
        super().__init__(stride=1, offset=0, max_gap=10**9)


def build_selector(name: str, cfg) -> object:
    if name == "all":
        return AllSelector()
    if name == "uniform":
        return UniformSelector(
            stride=int(cfg.get("stride", 10)),
            offset=int(cfg.get("offset", 0)),
            max_gap=int(cfg.get("max_gap", 10**9)),
        )
    if name == "pose_threshold":
        return PoseThresholdSelector(
            trans_thresh_m=float(cfg.get("trans_thresh_m", 0.15)),
            rot_thresh_deg=float(cfg.get("rot_thresh_deg", 10.0)),
            max_gap=int(cfg.get("max_gap", 20)),
            min_gap=int(cfg.get("min_gap", 0)),
        )
    if name == "parallax":
        return ParallaxSelector(
            parallax_thresh_px=float(cfg.get("parallax_thresh_px", 8.0)),
            max_gap=int(cfg.get("max_gap", 20)),
            min_gap=int(cfg.get("min_gap", 0)),
        )
    if name == "hybrid":
        return HybridSelector(
            trans_thresh_m=float(cfg.get("trans_thresh_m", 0.15)),
            rot_thresh_deg=float(cfg.get("rot_thresh_deg", 10.0)),
            max_gap=int(cfg.get("max_gap", 20)),
            min_gap=int(cfg.get("min_gap", 0)),
            blur_max=float(cfg.get("blur_max", 1e9)),
            brightness_min=float(cfg.get("brightness_min", 0.0)),
            brightness_max=float(cfg.get("brightness_max", 255.0)),
        )
    raise ValueError(f"Unknown selector: {name}")


def build_selector_from_flat_args(args) -> object:
    cfg = vars(args)
    return build_selector(args.selector, cfg)
