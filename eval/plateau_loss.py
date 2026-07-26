"""
Sparse Support Plateau Loss
============================
3D regulariser that penalises Gaussians lying outside the plateau of their
nearest SLAM (or depth-augmented) anchor point.

Maths:
    Spherical:
        D_min(x) = min_j  ||x - p_j|| / tau_j
        L = mean_n  max(D_min(x_n) - 1, 0)^2   (quadratic hinge)

    Ellipsoidal:
        D_aniso_j(x) = sqrt( ((x-p)·u_t1/tau_t)^2
                           + ((x-p)·u_t2/tau_t)^2
                           + ((x-p)·u_n /tau_n )^2 )
        L = mean_n  max(min_j D_aniso_j(x_n) - 1, 0)^2

Usage (train.py):
    # init once before loop
    pl_cfg = PlateauLossConfig.from_yaml(path)
    pl     = PlateauLoss(pl_cfg, dataset.source_path)

    # --- inside training loop ---
    # BEFORE loss.backward():
    L_p, pl_metrics = pl.compute_loss(gaussians, iteration)
    if L_p is not None:
        loss = loss + pl_cfg.lambda_plateau * L_p

    # INSIDE torch.no_grad() AFTER backward():
    pl_metrics.update(pl.post_backward(gaussians, iteration))

    # AFTER densify_and_prune():
    pl.reset_sampler()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

try:
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# Anchor-chunk sizes control intermediate tensor memory
_CHUNK_SPHER = 2048   # 8192 gaussians × 2048 anchors × 4 B ≈ 67 MB
_CHUNK_ELLIP = 512    # 8192 × 512 × 3 × 4 B ≈ 50 MB per delta/c tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlateauLossConfig:
    enabled: bool = False
    type: str = "spherical"          # "spherical" | "ellipsoidal"

    # Anchor source.  None → auto-load from source_path/sparse/0/points3D.txt
    anchor_path: Optional[str] = None
    obs_min: int = 3                 # min track length (ignored when anchor_path set)

    # kNN params (used for isolation filter AND spacing h_j)
    knn_k: int = 5
    knn_iso_mult: float = 3.0        # set 0 to disable isolation filter

    # Tau — spherical
    alpha: float = 0.6
    tau_min: float = 0.05
    tau_max: float = 0.60

    # Tau — ellipsoidal
    alpha_n: float = 0.4
    alpha_t: float = 0.9
    tau_n_max: float = 0.30
    tau_t_max: float = 0.60

    # Schedule
    start_iter: int = 5000
    lambda_plateau: float = 0.01
    # lambda_schedule: step-function override.  List of [iter, lambda] pairs,
    # sorted ascending by iter.  At each iteration, uses the lambda of the last
    # breakpoint whose iter <= current iter.  Overrides lambda_plateau when set.
    # Example: [[1000, 0.1], [7000, 0.03], [15000, 0.0]]
    lambda_schedule: Optional[list] = None

    # Opacity weighting: multiply hinge by per-Gaussian opacity.
    # Effect: high-opacity floaters get strong gradient on BOTH xyz (move toward
    # anchor) AND opacity (decrease → pruned). Low-opacity surface Gaussians are
    # largely unaffected even when outside plateau.
    opacity_weight: bool = False

    # Exponential loss kernel: exp(clamp(D-1, 0, 8)) - 1  instead of (D-1)^2.
    # Gives dramatically stronger gradient for distant floaters:
    #   D=2: 1.7×,  D=5: 6.7×,  D=10: clamped → 2980 vs 18 (quadratic)
    # Use a smaller lambda when enabling (e.g. 0.05 vs 0.10 for quadratic).
    exp_loss: bool = False

    # Adaptive floater pruning (applied in post_backward).
    # Directly prunes Gaussians that are clearly floaters:
    #   d_euc > adaptive_prune_d_euc  AND
    #   (opacity > adaptive_prune_opacity  OR  max_scale > adaptive_prune_scale)
    # Low-opacity / small distant Gaussians are left for the loss to handle.
    adaptive_prune: bool = False
    adaptive_prune_d_euc: float = 1.5      # Euclidean distance threshold (m)
    adaptive_prune_opacity: float = 0.5    # sigmoid-opacity threshold
    adaptive_prune_scale: float = 0.15     # max axis scale threshold (m)
    adaptive_prune_interval: int = 500     # prune every N iterations
    adaptive_prune_start_iter: int = 7000  # first prune iteration

    # Cyclic sampler
    sample_size: int = 8192          # 4096 | 8192 | 16384

    # Pop-2 Z-clip (applied in post_backward, independent of loss schedule)
    pop2_zclip: bool = True
    pop2_z_threshold: float = 2.0
    pop2_start_iter: int = 5000
    pop2_interval: int = 1000

    @staticmethod
    def from_yaml(path: str) -> "PlateauLossConfig":
        assert _YAML_OK, "PyYAML required: pip install pyyaml"
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = PlateauLossConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PlateauLoss:
    """Plateau-field regulariser.  Thread-unsafe (designed for single-GPU training)."""

    def __init__(self, cfg: PlateauLossConfig, source_path: str):
        self.cfg = cfg
        self._ready = False
        if not cfg.enabled:
            return

        assert _SKLEARN_OK, "scikit-learn required for PlateauLoss (pip install scikit-learn)"

        # 1. Load anchor xyz
        if cfg.anchor_path is not None:
            pts = np.load(cfg.anchor_path).astype(np.float32)
        else:
            pts = _load_slam_anchors(source_path, cfg.obs_min)

        # 2. kNN isolation filter (skip if mult == 0)
        if cfg.knn_iso_mult > 0 and len(pts) > cfg.knn_k + 1:
            pts = _knn_isolation(pts, cfg.knn_k, cfg.knn_iso_mult)

        print(f"[PlateauLoss] type={cfg.type!r}  anchors={len(pts)}"
              f"  sample_size={cfg.sample_size}  start_iter={cfg.start_iter}"
              f"  lambda={cfg.lambda_plateau}")

        # 3. kNN for spacing h_j (run once on CPU)
        nbrs = NearestNeighbors(n_neighbors=cfg.knn_k + 1,
                                algorithm="ball_tree").fit(pts)
        dists, inds = nbrs.kneighbors(pts)
        h_j = dists[:, cfg.knn_k].astype(np.float32)

        # 4. Build GPU tensors
        self._anchors = torch.tensor(pts, dtype=torch.float32, device="cuda")

        if cfg.type == "spherical":
            tau = np.clip(cfg.alpha * h_j, cfg.tau_min, cfg.tau_max).astype(np.float32)
            self._tau = torch.tensor(tau, device="cuda")

        elif cfg.type == "ellipsoidal":
            tau_n, tau_t, frames = _compute_ellipsoid(pts, inds, h_j, cfg)
            self._tau_n  = torch.tensor(tau_n,  device="cuda")
            self._tau_t  = torch.tensor(tau_t,  device="cuda")
            self._frames = torch.tensor(frames, device="cuda")   # (M, 3, 3)
        else:
            raise ValueError(f"Unknown plateau type: {cfg.type!r}")

        # 5. Cyclic sampler state
        self._n_gauss_cached: int = -1
        self._perm:  Optional[torch.Tensor] = None  # CPU permutation
        self._cursor: int = 0

        self._ready = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _lambda_at(self, iteration: int) -> float:
        """Return effective lambda for this iteration (respects lambda_schedule)."""
        sched = self.cfg.lambda_schedule
        if not sched:
            return self.cfg.lambda_plateau
        lam = self.cfg.lambda_plateau
        for breakpoint_iter, breakpoint_lam in sched:
            if iteration >= breakpoint_iter:
                lam = breakpoint_lam
            else:
                break
        return lam

    def compute_loss(self, gaussians, iteration: int
                     ) -> Tuple[Optional[torch.Tensor], Dict]:
        """Compute plateau loss over a cyclic Gaussian subset.

        Call BEFORE loss.backward() so xyz gradients attach correctly.
        Returns (loss_tensor, metrics).  loss_tensor is None before start_iter
        or when effective lambda == 0.
        """
        if not self._ready or iteration < self.cfg.start_iter:
            return None, {}

        lam = self._lambda_at(iteration)
        if lam == 0.0:
            return None, {"plateau/lambda": 0.0}

        xyz_all = gaussians.get_xyz                     # (N,3), grad-attached
        idx = self._sample_cyclic_idx(xyz_all.shape[0])
        idx_gpu = idx.to("cuda")
        xyz_sub = xyz_all[idx_gpu]                      # (S,3), grad flows back

        # Opacity weighting: weight each Gaussian's hinge^2 by its opacity.
        # Floaters (high opacity, far from anchor) get ∂L/∂xyz AND ∂L/∂opacity.
        weights = None
        if self.cfg.opacity_weight:
            op_all = gaussians.get_opacity              # (N,1), sigmoid-activated
            weights = op_all[idx_gpu, 0]               # (S,), grad flows to opacity

        if self.cfg.type == "spherical":
            loss = _spherical_loss(xyz_sub, self._anchors, self._tau, weights, self.cfg.exp_loss)
        else:
            loss = _ellipsoidal_loss(
                xyz_sub, self._anchors,
                self._frames, self._tau_n, self._tau_t, weights, self.cfg.exp_loss,
            )

        return loss, {"plateau/loss_raw": float(loss.detach()), "plateau/lambda": lam}

    def post_backward(self, gaussians, iteration: int) -> Dict:
        """Pop-2 Z-clip + adaptive floater pruning.  Call inside torch.no_grad() after backward()."""
        if not self._ready:
            return {}
        metrics: Dict = {}
        cfg = self.cfg

        if (cfg.pop2_zclip
                and iteration >= cfg.pop2_start_iter
                and iteration % cfg.pop2_interval == 0):
            z = gaussians.get_xyz[:, 2]
            mask_remove = z >= cfg.pop2_z_threshold
            n_pruned = int(mask_remove.sum().item())
            if n_pruned > 0:
                gaussians.prune_points(mask_remove)
                metrics["plateau/pop2_pruned"] = n_pruned

        if (cfg.adaptive_prune
                and iteration >= cfg.adaptive_prune_start_iter
                and iteration % cfg.adaptive_prune_interval == 0):
            metrics.update(self._adaptive_prune(gaussians))

        return metrics

    def _adaptive_prune(self, gaussians) -> Dict:
        """Prune floaters with large Euclidean distance AND high opacity or large scale."""
        cfg = self.cfg
        xyz = gaussians.get_xyz.detach()   # (N, 3)
        N = xyz.shape[0]

        # Min Euclidean distance to any anchor (chunked to bound VRAM)
        d_min = xyz.new_full((N,), float("inf"))
        chunk = 2048
        for s in range(0, self._anchors.shape[0], chunk):
            a = self._anchors[s:s + chunk]
            d = torch.cdist(xyz, a)        # (N, chunk)
            d_min = torch.minimum(d_min, d.min(dim=1).values)

        far = d_min > cfg.adaptive_prune_d_euc

        opacity   = gaussians.get_opacity.detach()[:, 0]              # (N,)
        max_scale = gaussians.get_scaling.detach().max(dim=1).values   # (N,)

        confident = (opacity > cfg.adaptive_prune_opacity) | (max_scale > cfg.adaptive_prune_scale)
        prune_mask = far & confident
        n_pruned = int(prune_mask.sum().item())
        if n_pruned > 0:
            gaussians.prune_points(prune_mask)
        return {"plateau/adaptive_pruned": n_pruned}

    def reset_sampler(self):
        """Invalidate cyclic permutation.  Call after densify_and_prune() changes N."""
        self._n_gauss_cached = -1

    # ------------------------------------------------------------------
    # Cyclic sampler
    # ------------------------------------------------------------------

    def _sample_cyclic_idx(self, N: int) -> torch.Tensor:
        """Return a CPU index tensor of length min(sample_size, N)."""
        S = min(self.cfg.sample_size, N)

        if N != self._n_gauss_cached:
            self._perm = torch.randperm(N)
            self._cursor = 0
            self._n_gauss_cached = N

        end = self._cursor + S
        if end <= N:
            idx = self._perm[self._cursor:end]
            self._cursor = end % N
        else:
            tail = self._perm[self._cursor:]
            self._perm = torch.randperm(N)
            need = S - len(tail)
            head = self._perm[:need]
            idx = torch.cat([tail, head])
            self._cursor = need

        return idx

    def _sample_cyclic(self, xyz_all: torch.Tensor) -> torch.Tensor:
        idx = self._sample_cyclic_idx(xyz_all.shape[0])
        return xyz_all[idx.to("cuda")]


# ---------------------------------------------------------------------------
# Loss kernels (free functions — testable independently)
# ---------------------------------------------------------------------------

def _hinge_to_loss(hinge: torch.Tensor, exp_loss: bool) -> torch.Tensor:
    """Apply loss kernel to hinge = clamp(D-1, 0).
    exp_loss=False: quadratic  (D-1)^2
    exp_loss=True:  exponential  exp(clamp(D-1, 0, 8)) - 1  (stronger for distant floaters)
    """
    if exp_loss:
        return torch.exp(hinge.clamp(max=8.0)) - 1.0
    return hinge * hinge


def _spherical_loss(xyz_sub: torch.Tensor,
                    anchors: torch.Tensor,
                    tau: torch.Tensor,
                    weights: Optional[torch.Tensor] = None,
                    exp_loss: bool = False) -> torch.Tensor:
    """Hinge loss on min normalised Euclidean distance to any anchor.
    weights: per-Gaussian scalar (e.g. opacity). If given, loss = mean(w * kernel(hinge)).
    """
    M = anchors.shape[0]
    D_min = xyz_sub.new_full((xyz_sub.shape[0],), float("inf"))

    for s in range(0, M, _CHUNK_SPHER):
        a = anchors[s:s + _CHUNK_SPHER]
        t = tau[s:s + _CHUNK_SPHER]
        d_norm = torch.cdist(xyz_sub, a) / t
        D_min = torch.minimum(D_min, d_norm.min(dim=1).values)

    hinge = torch.clamp(D_min - 1.0, min=0.0)
    sq = _hinge_to_loss(hinge, exp_loss)
    if weights is not None:
        sq = sq * weights
    return sq.mean()


def _ellipsoidal_loss(xyz_sub: torch.Tensor,
                      anchors: torch.Tensor,
                      frames: torch.Tensor,
                      tau_n: torch.Tensor,
                      tau_t: torch.Tensor,
                      weights: Optional[torch.Tensor] = None,
                      exp_loss: bool = False) -> torch.Tensor:
    """Hinge loss on min anisotropic distance to any anchor.
    weights: per-Gaussian scalar (e.g. opacity). If given, loss = mean(w * kernel(hinge)).
    """
    M = anchors.shape[0]
    D_min = xyz_sub.new_full((xyz_sub.shape[0],), float("inf"))

    for s in range(0, M, _CHUNK_ELLIP):
        a  = anchors[s:s + _CHUNK_ELLIP]
        fr = frames[s:s + _CHUNK_ELLIP]
        tn = tau_n[s:s + _CHUNK_ELLIP]
        tt = tau_t[s:s + _CHUNK_ELLIP]

        delta = xyz_sub[:, None, :] - a[None, :, :]
        c = torch.einsum("cjk,scj->sck", fr, delta)
        d2 = (c[..., 0] / tt) ** 2 + (c[..., 1] / tt) ** 2 + (c[..., 2] / tn) ** 2
        D_min = torch.minimum(D_min, d2.sqrt().min(dim=1).values)

    hinge = torch.clamp(D_min - 1.0, min=0.0)
    sq = _hinge_to_loss(hinge, exp_loss)
    if weights is not None:
        sq = sq * weights
    return sq.mean()


# ---------------------------------------------------------------------------
# Anchor loading / preprocessing (CPU, called once at init)
# ---------------------------------------------------------------------------

def _load_slam_anchors(source_path: str, obs_min: int) -> np.ndarray:
    """Parse COLMAP points3D.txt, return xyz filtered by track length >= obs_min."""
    p = Path(source_path) / "sparse" / "0" / "points3D.txt"
    if not p.exists():
        raise FileNotFoundError(f"[PlateauLoss] points3D.txt not found: {p}")
    pts: list = []
    with open(p) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            if len(tok) < 8:
                continue
            n_track = (len(tok) - 8) // 2      # COLMAP: 2 ints per track entry
            if n_track >= obs_min:
                pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
    if not pts:
        raise ValueError(f"[PlateauLoss] No points pass obs_min={obs_min} in {p}")
    return np.array(pts, dtype=np.float32)


def _knn_isolation(pts: np.ndarray, k: int, mult: float) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    knn_dist = dists[:, k]
    return pts[knn_dist <= mult * np.median(knn_dist)]


def _compute_ellipsoid(
        pts: np.ndarray,
        inds: np.ndarray,
        h_j: np.ndarray,
        cfg: PlateauLossConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-anchor (tau_n, tau_t, frame) via kNN PCA.  O(N) Python loop."""
    N = len(pts)
    k = cfg.knn_k
    tau_n = np.clip(cfg.alpha_n * h_j, cfg.tau_min, cfg.tau_n_max).astype(np.float32)
    tau_t = np.clip(cfg.alpha_t * h_j, cfg.tau_min, cfg.tau_t_max).astype(np.float32)
    frames = np.empty((N, 3, 3), dtype=np.float32)

    for i in range(N):
        neigh = pts[inds[i, 1:k + 1]]              # (k, 3) — exclude self
        X = neigh - neigh.mean(0)
        cov = (X.T @ X) / max(k - 1, 1)
        _, evec = np.linalg.eigh(cov)              # ascending eigenvalues
        # evec[:,0] = surface normal (min variance = tight direction)
        # evec[:,1], evec[:,2] = tangents (loose directions)
        frames[i] = np.stack(
            [evec[:, 1], evec[:, 2], evec[:, 0]], axis=1
        )   # columns: [u_t1, u_t2, u_n]  — matches loss kernel

    return tau_n, tau_t, frames
