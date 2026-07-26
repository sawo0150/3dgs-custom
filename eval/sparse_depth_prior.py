from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.graphics_utils import geom_transform_points


class SparseDepthPrior:
    """Sparse inverse-depth supervision from projected COLMAP/OpenMAVIS points."""

    def __init__(
        self,
        sparse_points: Optional[np.ndarray],
        max_points_per_view: int = 2048,
        global_max_points: int = 100000,
        min_depth: float = 0.2,
        require_rendered: bool = True,
    ) -> None:
        self.max_points_per_view = int(max_points_per_view)
        self.min_depth = float(min_depth)
        self.require_rendered = bool(require_rendered)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._points_cuda: Optional[torch.Tensor] = None

        if sparse_points is None or len(sparse_points) == 0:
            self.points = None
            return

        points = np.asarray(sparse_points, dtype=np.float32)
        if global_max_points > 0 and len(points) > global_max_points:
            indices = np.linspace(0, len(points) - 1, int(global_max_points), dtype=np.int64)
            points = points[indices]
        self.points = torch.from_numpy(points)

    @property
    def available(self) -> bool:
        return self.points is not None and self.max_points_per_view > 0

    def _points(self) -> torch.Tensor:
        if self._points_cuda is None:
            self._points_cuda = self.points.cuda(non_blocking=True)
        return self._points_cuda

    def project(self, viewpoint_cam) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(getattr(viewpoint_cam, "uid", getattr(viewpoint_cam, "image_name", "camera")))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if not self.available:
            empty_i = torch.empty(0, dtype=torch.long)
            empty_f = torch.empty(0, dtype=torch.float32)
            return empty_i, empty_i, empty_f

        with torch.no_grad():
            points = self._points()
            view = geom_transform_points(points, viewpoint_cam.world_view_transform)
            proj = geom_transform_points(points, viewpoint_cam.full_proj_transform)

            z = view[:, 2]
            valid = (
                (z > self.min_depth)
                & (proj[:, 0] >= -1.0)
                & (proj[:, 0] <= 1.0)
                & (proj[:, 1] >= -1.0)
                & (proj[:, 1] <= 1.0)
            )
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                empty_i = torch.empty(0, dtype=torch.long)
                empty_f = torch.empty(0, dtype=torch.float32)
                self._cache[key] = (empty_i, empty_i, empty_f)
                return self._cache[key]

            if valid_idx.numel() > self.max_points_per_view:
                pick = torch.linspace(
                    0,
                    valid_idx.numel() - 1,
                    self.max_points_per_view,
                    device=valid_idx.device,
                ).long()
                valid_idx = valid_idx[pick]

            h = int(viewpoint_cam.image_height)
            w = int(viewpoint_cam.image_width)
            xs = torch.round(((proj[valid_idx, 0] + 1.0) * w - 1.0) * 0.5).long().clamp(0, w - 1)
            ys = torch.round(((proj[valid_idx, 1] + 1.0) * h - 1.0) * 0.5).long().clamp(0, h - 1)
            inv_depth = (1.0 / z[valid_idx].clamp_min(self.min_depth)).float()

            self._cache[key] = (ys.cpu(), xs.cpu(), inv_depth.cpu())
            return self._cache[key]

    def loss(self, rendered_invdepth: torch.Tensor, viewpoint_cam) -> Tuple[torch.Tensor, int, float]:
        ys_cpu, xs_cpu, target_cpu = self.project(viewpoint_cam)
        if target_cpu.numel() == 0:
            zero = rendered_invdepth.sum() * 0.0
            return zero, 0, 0.0

        ys = ys_cpu.to(rendered_invdepth.device, non_blocking=True)
        xs = xs_cpu.to(rendered_invdepth.device, non_blocking=True)
        target = target_cpu.to(rendered_invdepth.device, non_blocking=True)
        pred = rendered_invdepth[0, ys, xs]

        if self.require_rendered:
            valid = pred > 0
            if valid.sum().item() == 0:
                zero = rendered_invdepth.sum() * 0.0
                return zero, 0, 0.0
            pred = pred[valid]
            target = target[valid]

        raw = F.smooth_l1_loss(pred, target, beta=0.05, reduction="mean")
        mean_abs = torch.abs(pred.detach() - target.detach()).mean().item()
        return raw, int(target.numel()), float(mean_abs)
