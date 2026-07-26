from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from plyfile import PlyData

from scene.colmap_loader import read_points3D_binary, read_points3D_text


def load_sparse_points(source_path: str) -> Optional[np.ndarray]:
    sparse_dir = Path(source_path) / "sparse" / "0"
    ply_path = sparse_dir / "points3D.ply"
    txt_path = sparse_dir / "points3D.txt"
    bin_path = sparse_dir / "points3D.bin"

    if ply_path.exists():
        vertices = PlyData.read(str(ply_path))["vertex"]
        return np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    if bin_path.exists():
        xyz, _, _ = read_points3D_binary(str(bin_path))
        return xyz.astype(np.float32)
    if txt_path.exists():
        xyz, _, _ = read_points3D_text(str(txt_path))
        return xyz.astype(np.float32)
    return None


def nearest_sparse_median(xyz: np.ndarray, sparse_points: Optional[np.ndarray]) -> float:
    if sparse_points is None or len(sparse_points) == 0 or len(xyz) == 0:
        return float("nan")
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(sparse_points).query(xyz, k=1, workers=-1)
        return float(np.median(distances))
    except Exception:
        # Fallback for environments without a working SciPy KD-tree.
        mins = []
        chunk = 8192
        sparse = torch.from_numpy(sparse_points).cuda()
        for start in range(0, len(xyz), chunk):
            q = torch.from_numpy(xyz[start:start + chunk]).cuda()
            mins.append(torch.cdist(q, sparse).min(dim=1).values.cpu())
        return float(torch.cat(mins).median().item())


def summarize_gaussians(
    gaussians,
    sparse_points: Optional[np.ndarray] = None,
    low_opacity_threshold: float = 0.1,
    large_scale_threshold: float = 0.1,
) -> Dict[str, float]:
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().float().cpu().numpy()
        opacity = gaussians.get_opacity.detach().float().flatten()
        scale = gaussians.get_scaling.detach().float()
        scale_max = scale.max(dim=1).values

        gaussian_count = int(opacity.numel())
        low_opacity_count = int((opacity < low_opacity_threshold).sum().item())
        large_scale_count = int((scale_max > large_scale_threshold).sum().item())

        denom = max(1, gaussian_count)
        return {
            "gaussian/count": gaussian_count,
            "gaussian/low_opacity_count": low_opacity_count,
            "gaussian/low_opacity_ratio": low_opacity_count / denom,
            "gaussian/large_scale_count": large_scale_count,
            "gaussian/large_scale_ratio": large_scale_count / denom,
            "gaussian/opacity_median": float(opacity.median().item()) if gaussian_count else float("nan"),
            "gaussian/scale_median": float(scale_max.median().item()) if gaussian_count else float("nan"),
            "gaussian/nearest_orb_median": nearest_sparse_median(xyz, sparse_points),
        }


def save_gaussian_summary(summary: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
