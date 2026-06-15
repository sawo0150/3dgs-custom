from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


def _as_matrix(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = arr.reshape(4, 4)
    return arr


def load_pose_records(path: str | Path, fmt: str = "auto") -> Dict[int, np.ndarray]:
    path = Path(path)
    if fmt == "auto":
        fmt = path.suffix.lstrip(".").lower()

    if fmt == "jsonl":
        out = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out[int(row["frame_idx"])] = _as_matrix(row["pose"])
        return out

    if fmt == "csv":
        out = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                mat = json.loads(row["pose"])
                out[int(row["frame_idx"])] = _as_matrix(mat)
        return out

    raise ValueError(f"Unsupported pose format: {fmt}")


def attach_pose_records(records: List[Dict[str, Any]], pose_path: str | Path, fmt: str = "auto") -> List[Dict[str, Any]]:
    pose_map = load_pose_records(pose_path, fmt)
    out = []
    for r in records:
        rr = dict(r)
        idx = int(rr.get("frame_idx", -1))
        if idx in pose_map:
            rr["pose"] = pose_map[idx].tolist()
            rr["pose_valid"] = True
        else:
            rr["pose_valid"] = False
        out.append(rr)
    return out


def rotation_angle_deg(T_prev, T_curr) -> float:
    R_prev = np.asarray(T_prev, dtype=float)[:3, :3]
    R_curr = np.asarray(T_curr, dtype=float)[:3, :3]
    R = R_prev.T @ R_curr
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(trace))
    return float(angle)


def translation_distance_m(T_prev, T_curr) -> float:
    t_prev = np.asarray(T_prev, dtype=float)[:3, 3]
    t_curr = np.asarray(T_curr, dtype=float)[:3, 3]
    return float(np.linalg.norm(t_curr - t_prev))
