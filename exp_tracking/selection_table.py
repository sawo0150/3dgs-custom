from __future__ import annotations

from typing import List, Dict, Any, Tuple


def build_selection_table_rows(debug_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in debug_records:
        rows.append({
            "frame_idx": r.get("frame_idx"),
            "image_name": r.get("image_name"),
            "keyframe_flag": r.get("keyframe_flag", 0),
            "reason": r.get("reason", "unknown"),
            "score": r.get("score", 0.0),
            "delta_trans_m": r.get("delta_trans_m", 0.0),
            "delta_rot_deg": r.get("delta_rot_deg", 0.0),
            "parallax_px": r.get("parallax_px", 0.0),
        })
    return rows


def build_wandb_table_payload(debug_records: List[Dict[str, Any]]) -> Tuple[list[str], list[list[Any]]]:
    columns = [
        "frame_idx", "image_name", "keyframe_flag", "reason", "score",
        "delta_trans_m", "delta_rot_deg", "parallax_px",
    ]
    rows = []
    for r in build_selection_table_rows(debug_records):
        rows.append([r.get(c) for c in columns])
    return columns, rows
