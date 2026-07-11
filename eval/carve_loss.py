"""
Free-Space Carve Loss
=====================
3D regulariser targeting densification floaters via observed-free-space
evidence.  Designed from label-validated analysis (gs_floaterLab
context/experiments/carve_loss_design.md, AUC 0.98 vs human-deleted floaters).

Signals (precomputed once from cameras + SLAM points, no extra inputs):
    transit(v)  = #rays passing through voxel v strictly in front of a SLAM
                  point observed by that camera (frustum approximation)
    terminal(v) = #rays terminating near voxel v (surface evidence)
    rho(x)      = transit / (transit + term_k * terminal)   in [0, 1]
    w(x)        = rho(x) * min(d5nn_slam(x) / d5_tau, 1)
    score_i     = w(x_i) * (1 - max opacity within maxop_radius of x_i)

Components (all optional, share the same field):
    soft   : L = lambda * mean(score * opacity) over score > score_min subset.
             Gradient path to opacity only (score is detached) — floaters
             cannot regrow opacity, surface with real photometric support
             resists.  Start early (densification phase) to suppress dust.
    prune  : every prune_interval iters, delete highest-score Gaussians until
             the cumulative *visual contribution* (opacity x area x accum
             visibility) of deleted non-floaters reaches budget_per_cycle.
             Self-limiting: total harm bounded by budget_total.
    gate   : right after each densify step, delete newborns (birth_step ==
             current iteration) whose score > gate_score — blocks floater
             births in confirmed-empty space.

Usage (train.py):
    carve = CarveLoss(CarveLossConfig.from_yaml(path), dataset.source_path)
    # inside training loop, after plateau backward (opacity-only gradient):
    L_c, c_metrics = carve.compute_loss(gaussians, iteration)
    if L_c is not None:
        L_c.backward()          # adds opacity grads only
        loss = loss + L_c.detach()
    # inside torch.no_grad(), after densification block:
    c_metrics.update(carve.post_backward(gaussians, iteration,
                                         densify_happened))
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
    from scipy.spatial import cKDTree
    from scipy.ndimage import uniform_filter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CarveLossConfig:
    enabled: bool = False

    # Field build
    voxel: float = 0.10
    ray_step: float = 0.06
    surf_margin: float = 0.20
    min_t: float = 0.15
    max_depth: float = 15.0
    term_k: float = 3.0
    smooth: int = 3
    d5_tau: float = 0.25
    cam_stride: int = 1          # 대형 anchor셋(MPS)에서 field 빌드 가속용
    target_voxel: float = 0.0    # >0이면 ray 타깃을 voxel 다운샘플 (MPS 626k → ~190k)
    anchor_term_min: float = 0.0 # >0이면 terminal 증거가 이 값 이하인 anchor를 d5에서 제외
                                 # (MPS semidense의 outlier 7.6%가 허공을 합법화하는 것 차단)
    img_w: int = 1024
    img_h: int = 1024
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 511.5
    cy: float = 511.5

    # Score refresh
    maxop_radius: float = 0.05
    refresh_interval: int = 200      # score staleness bound (also refreshed on N change)

    # Soft opacity loss
    soft_enabled: bool = True
    lambda_carve: float = 0.05
    soft_start_iter: int = 1000
    score_min: float = 0.3

    # Budget prune
    prune_enabled: bool = True
    prune_start_iter: int = 7000
    prune_interval: int = 1000
    budget_per_cycle: float = 0.0005   # fraction of total visual contribution
    budget_total: float = 0.005
    prune_score_min: float = 0.3       # never prune below this score

    # Birth gate
    gate_enabled: bool = True
    gate_score: float = 0.95

    # Carve-potential force (exp40 A안): 빈 공간 점을 표면 attractor까지 연속 견인.
    # phi = attractor(terminal 증거 or 가시 gaussian 점유 voxel)까지의 distance transform.
    # L = lambda_force * mean_{score>force_score_min}( w_i * phi(x_i) ) — xyz gradient 경로.
    force_enabled: bool = False
    lambda_force: float = 0.05
    force_score_min: float = 0.5
    force_start_iter: int = 1000

    @staticmethod
    def from_yaml(path: str) -> "CarveLossConfig":
        assert _YAML_OK, "PyYAML required"
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = CarveLossConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CarveLoss:
    def __init__(self, cfg: CarveLossConfig, source_path: str):
        self.cfg = cfg
        self._ready = False
        if not cfg.enabled:
            return
        assert _SCIPY_OK, "scipy required for CarveLoss"

        slam = _load_points3d(source_path)
        Rs, ts = _load_cameras(source_path)
        targets = slam
        if cfg.target_voxel > 0:
            key = np.floor(slam / cfg.target_voxel).astype(np.int64)
            _, uidx = np.unique(key, axis=0, return_index=True)
            targets = slam[uidx]
        print(f"[CarveLoss] building field: {len(slam)} SLAM pts "
              f"(ray targets {len(targets)}), {len(Rs)} cams stride {cfg.cam_stride} ...")
        centers = np.stack([-R.T @ t for R, t in zip(Rs, ts)]).astype(np.float32)
        lo = centers.min(0) - np.maximum(centers.max(0) - centers.min(0), [2., 2., 3.])
        hi = centers.max(0) + np.maximum(centers.max(0) - centers.min(0), [2., 2., 3.])
        dims = np.ceil((hi - lo) / cfg.voxel).astype(int) + 1
        transit, terminal = _build_fields(Rs[::cfg.cam_stride], ts[::cfg.cam_stride],
                                          targets, lo, dims, cfg)
        transit = uniform_filter(transit, size=cfg.smooth)
        terminal = uniform_filter(terminal, size=cfg.smooth)
        rho = transit / (transit + cfg.term_k * terminal + 1e-6)

        if cfg.anchor_term_min > 0:
            gi = np.floor((slam - lo) / cfg.voxel).astype(np.int64)
            inb = ((gi >= 0) & (gi < dims[None, :])).all(1)
            ev = np.zeros(len(slam), np.float32)
            ev[inb] = terminal[gi[inb, 0], gi[inb, 1], gi[inb, 2]]
            n_before = len(slam)
            slam = slam[ev > cfg.anchor_term_min]
            print(f"[CarveLoss] anchor terminal filter: {n_before} → {len(slam)}")

        self._rho = torch.tensor(rho, dtype=torch.float32, device="cuda")
        self._terminal_np = terminal   # phi attractor 구성용 (CPU 보관)
        self._lo = torch.tensor(lo, dtype=torch.float32, device="cuda")
        self._dims = torch.tensor(np.asarray(dims), dtype=torch.long, device="cuda")
        self._slam_tree = cKDTree(slam)

        # cached per-Gaussian score
        self._score: Optional[torch.Tensor] = None
        self._phi: Optional[torch.Tensor] = None
        self._score_iter = -10**9
        self._score_n = -1
        self._harm_spent = 0.0    # cumulative pruned contribution fraction
        print(f"[CarveLoss] field ready: grid {tuple(dims)}, "
              f"soft={cfg.soft_enabled} prune={cfg.prune_enabled} gate={cfg.gate_enabled}")
        self._ready = True

    # ------------------------------------------------------------------

    def _rho_at(self, xyz: torch.Tensor) -> torch.Tensor:
        gi = torch.floor((xyz - self._lo) / self.cfg.voxel).long()
        inb = ((gi >= 0) & (gi < self._dims)).all(dim=1)
        out = torch.zeros(xyz.shape[0], device="cuda")
        g = gi[inb]
        out[inb] = self._rho[g[:, 0], g[:, 1], g[:, 2]]
        return out

    def _refresh_score(self, gaussians, iteration: int, force: bool = False):
        N = gaussians.get_xyz.shape[0]
        if (not force and self._score is not None and N == self._score_n
                and iteration - self._score_iter < self.cfg.refresh_interval):
            return
        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            opac = gaussians.get_opacity.detach()[:, 0]
            rho = self._rho_at(xyz)
            xyz_np = xyz.cpu().numpy()
            d5, _ = self._slam_tree.query(xyz_np, k=5, workers=-1)
            d5 = torch.tensor(d5.mean(1), dtype=torch.float32, device="cuda")
            w = rho * (d5 / self.cfg.d5_tau).clamp(max=1.0)
            # neighbourhood max opacity (CPU KDTree, ~1-2 s at 150k pts)
            tree = cKDTree(xyz_np)
            pairs = tree.query_ball_point(xyz_np, self.cfg.maxop_radius,
                                          workers=-1, return_sorted=False)
            op_np = opac.cpu().numpy()
            maxop = np.fromiter((op_np[p].max() for p in pairs),
                                dtype=np.float32, count=N)
            self._score = w * (1.0 - torch.tensor(maxop, device="cuda"))
            self._score_iter = iteration
            self._score_n = N
            if self.cfg.force_enabled:
                self._refresh_phi(xyz_np, op_np)

    # ------------------------------------------------------------------

    def _refresh_phi(self, xyz_np, op_np):
        """phi = attractor(terminal 증거 or 가시 gaussian 점유)까지 EDT. score 갱신 시 함께."""
        from scipy.ndimage import distance_transform_edt
        lo = self._lo.cpu().numpy()
        dims = self._dims.cpu().numpy()
        attract = self._terminal_np > 1.0
        vis = xyz_np[op_np > 0.3]
        gi = np.floor((vis - lo) / self.cfg.voxel).astype(np.int64)
        inb = ((gi >= 0) & (gi < dims[None, :])).all(1)
        occ = np.zeros(tuple(dims), bool)
        occ[gi[inb, 0], gi[inb, 1], gi[inb, 2]] = True
        attract = attract | occ
        phi = distance_transform_edt(~attract) * self.cfg.voxel
        self._phi = torch.tensor(phi, dtype=torch.float32, device="cuda")

    def _phi_at(self, xyz: torch.Tensor) -> torch.Tensor:
        """trilinear phi(x) — xyz로 gradient 흐름."""
        g = (xyz - self._lo) / self.cfg.voxel - 0.5
        d = self._dims
        g0 = g.floor().long()
        f = (g - g0.float()).clamp(0, 1)
        vals = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = torch.stack([(g0[:, 0] + dx).clamp(0, d[0] - 1),
                                       (g0[:, 1] + dy).clamp(0, d[1] - 1),
                                       (g0[:, 2] + dz).clamp(0, d[2] - 1)], 1)
                    wgt = ((f[:, 0] if dx else 1 - f[:, 0])
                           * (f[:, 1] if dy else 1 - f[:, 1])
                           * (f[:, 2] if dz else 1 - f[:, 2]))
                    vals = vals + wgt * self._phi[idx[:, 0], idx[:, 1], idx[:, 2]]
        return vals

    def compute_loss(self, gaussians, iteration: int
                     ) -> Tuple[Optional[torch.Tensor], Dict]:
        """Soft opacity loss.  Call after other backwards; touches opacity only."""
        if not self._ready:
            return None, {}
        soft_on = self.cfg.soft_enabled and iteration >= self.cfg.soft_start_iter
        force_on = self.cfg.force_enabled and iteration >= self.cfg.force_start_iter
        if not soft_on and not force_on:
            return None, {}
        self._refresh_score(gaussians, iteration)
        loss = None
        metrics: Dict = {}
        if soft_on:
            sel = self._score > self.cfg.score_min
            if int(sel.sum()) > 0:
                op = gaussians.get_opacity[sel, 0]        # grad flows to opacity
                loss = self.cfg.lambda_carve * (self._score[sel].detach() * op).mean()
                metrics.update({"carve/loss": float(loss.detach()), "carve/n_soft": int(sel.sum())})
        if force_on and self._phi is not None:
            fsel = self._score > self.cfg.force_score_min
            if int(fsel.sum()) > 0:
                xyz_sel = gaussians.get_xyz[fsel]          # grad flows to xyz
                phi_v = self._phi_at(xyz_sel)
                f_loss = self.cfg.lambda_force * (self._score[fsel].detach() * phi_v).mean()
                loss = f_loss if loss is None else loss + f_loss
                metrics.update({"carve/force_loss": float(f_loss.detach()),
                                "carve/n_force": int(fsel.sum()),
                                "carve/phi_mean": float(phi_v.mean().detach())})
        return loss, metrics

    def post_backward(self, gaussians, iteration: int,
                      densify_happened: bool = False) -> Dict:
        """Birth gate + budget prune.  Call inside torch.no_grad() after densify."""
        if not self._ready:
            return {}
        cfg = self.cfg
        metrics: Dict = {}

        if cfg.gate_enabled and densify_happened:
            self._refresh_score(gaussians, iteration, force=True)
            newborn = gaussians.birth_step == iteration
            kill = newborn & (self._score > cfg.gate_score)
            n = int(kill.sum())
            if n > 0:
                gaussians.prune_points(kill)
                self._refresh_score(gaussians, iteration, force=True)
            metrics["carve/gate_pruned"] = n

        if (cfg.prune_enabled and iteration >= cfg.prune_start_iter
                and iteration % cfg.prune_interval == 0
                and self._harm_spent < cfg.budget_total):
            self._refresh_score(gaussians, iteration, force=True)
            opac = gaussians.get_opacity.detach()[:, 0]
            scales = gaussians.get_scaling.detach()
            s_sorted, _ = scales.sort(dim=1)
            area = s_sorted[:, 1] * s_sorted[:, 2]
            visw = gaussians.accum_visibility.float().clamp(min=1)
            contrib = opac * area * visw
            total = float(contrib.sum())
            budget = min(cfg.budget_per_cycle,
                         cfg.budget_total - self._harm_spent) * total

            order = torch.argsort(self._score, descending=True)
            csum = torch.cumsum(contrib[order], dim=0)
            eligible = self._score[order] > cfg.prune_score_min
            k = int((eligible & (csum <= budget)).sum())
            if k > 0:
                kill = torch.zeros_like(opac, dtype=torch.bool)
                kill[order[:k]] = True
                spent = float(contrib[kill].sum()) / total
                gaussians.prune_points(kill)
                self._harm_spent += spent
                self._refresh_score(gaussians, iteration, force=True)
                metrics.update({"carve/budget_pruned": k,
                                "carve/harm_spent": self._harm_spent})
        return metrics


# ---------------------------------------------------------------------------
# Field construction (CPU, once at init)
# ---------------------------------------------------------------------------

def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]])


def _load_cameras(source_path: str):
    Rs, ts = [], []
    for line in open(Path(source_path) / "sparse/0/images.txt"):
        p = line.strip().split()
        if len(p) < 9 or p[0].startswith("#"):
            continue
        q = np.array([float(x) for x in p[1:5]])
        t = np.array([float(x) for x in p[5:8]])
        Rs.append(_qvec2rotmat(q))
        ts.append(t)
    return np.array(Rs, np.float32), np.array(ts, np.float32)


def _load_points3d(source_path: str) -> np.ndarray:
    pts = []
    for line in open(Path(source_path) / "sparse/0/points3D.txt"):
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        if len(tok) >= 7:
            pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return np.array(pts, np.float32)


def _build_fields(Rs, ts, slam, lo, dims, cfg: CarveLossConfig):
    n_vox = int(np.prod(dims))
    transit = np.zeros(n_vox, dtype=np.float32)
    terminal = np.zeros(n_vox, dtype=np.float32)
    max_steps = int(cfg.max_depth / cfg.ray_step)
    step_d = np.arange(max_steps, dtype=np.float32) * cfg.ray_step + cfg.min_t

    def deposit(pts, acc):
        idx = np.floor((pts - lo) / cfg.voxel).astype(np.int64)
        inb = ((idx >= 0) & (idx < dims[None, :])).all(1)
        idx = idx[inb]
        flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
        acc += np.bincount(flat, minlength=n_vox).astype(np.float32)

    for ci in range(len(Rs)):
        R, t = Rs[ci], ts[ci]
        C = (-R.T @ t).astype(np.float32)
        pc = slam @ R.T + t
        z = pc[:, 2]
        ok = z > 0.2
        u = pc[:, 0] / np.clip(z, 1e-6, None) * cfg.fx + cfg.cx
        v = pc[:, 1] / np.clip(z, 1e-6, None) * cfg.fy + cfg.cy
        ok &= (u >= 0) & (u < cfg.img_w) & (v >= 0) & (v < cfg.img_h) & (z < cfg.max_depth)
        P = slam[ok]
        if len(P) == 0:
            continue
        dvec = P - C
        dist = np.linalg.norm(dvec, axis=1)
        dirs = dvec / dist[:, None]
        valid = step_d[None, :] < (dist - cfg.surf_margin)[:, None]
        pts = C[None, None, :] + step_d[None, :, None] * dirs[:, None, :]
        deposit(pts[valid], transit)
        term_off = np.linspace(-cfg.surf_margin, cfg.surf_margin, 5, dtype=np.float32)
        tpts = C[None, None, :] + (dist[:, None] + term_off[None, :])[:, :, None] * dirs[:, None, :]
        deposit(tpts.reshape(-1, 3), terminal)
    return transit.reshape(tuple(dims)), terminal.reshape(tuple(dims))
