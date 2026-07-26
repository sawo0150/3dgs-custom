from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def calc_depth_ambiguity(alpha_depth: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """SparseGS-style per-pixel depth ambiguity map."""
    alpha_depth = alpha_depth.squeeze()
    modes = modes.squeeze()
    mean = torch.mean(alpha_depth)
    std = torch.std(alpha_depth)
    denom_floor = torch.maximum(alpha_depth.min(), mean - 3 * std)
    denom = alpha_depth + denom_floor
    denom = torch.clamp(denom, min=1e-8)
    return (alpha_depth - modes) / denom


def _dip_stat(values: torch.Tensor) -> float:
    try:
        import diptest
    except Exception:
        return math.nan

    if values.numel() < 4:
        return math.nan
    return float(diptest.dipstat(values.detach().float().cpu().numpy()))


def summarize_depth_ambiguity(alpha_depth: torch.Tensor, modes: torch.Tensor) -> Dict[str, float]:
    diff = calc_depth_ambiguity(alpha_depth, modes)
    finite = torch.isfinite(diff)
    positive = diff[finite & (diff > 0)]
    if positive.numel() == 0:
        return {
            "ambiguity/mean_positive": 0.0,
            "ambiguity/p95": 0.0,
            "ambiguity/max": 0.0,
            "ambiguity/positive_ratio": 0.0,
            "ambiguity/dip": math.nan,
        }

    return {
        "ambiguity/mean_positive": float(positive.mean().item()),
        "ambiguity/p95": float(torch.quantile(positive, 0.95).item()),
        "ambiguity/max": float(positive.max().item()),
        "ambiguity/positive_ratio": float((positive.numel() / max(1, finite.sum().item()))),
        "ambiguity/dip": _dip_stat(positive),
    }


def normalized_ambiguity_image(alpha_depth: torch.Tensor, modes: torch.Tensor) -> Optional[torch.Tensor]:
    diff = calc_depth_ambiguity(alpha_depth, modes).detach()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    max_val = diff.max()
    if max_val <= 0:
        return None
    return (diff / max_val).unsqueeze(0)
