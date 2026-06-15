from __future__ import annotations

from typing import List, Dict, Any


def summarize_selection(debug_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(debug_records)
    if total == 0:
        return {"total_frames": 0, "selected_frames": 0, "ratio": 0.0}

    selected = [r for r in debug_records if int(r.get("keyframe_flag", 0)) == 1]
    sel_indices = [i for i, r in enumerate(debug_records) if int(r.get("keyframe_flag", 0)) == 1]
    gaps = [b - a for a, b in zip(sel_indices[:-1], sel_indices[1:])]

    return {
        "total_frames": total,
        "selected_frames": len(selected),
        "ratio": len(selected) / total,
        "avg_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "max_gap": max(gaps) if gaps else 0,
        "num_rotation_selected": sum(1 for r in selected if r.get("reason") == "rotation"),
        "num_translation_selected": sum(1 for r in selected if r.get("reason") == "translation"),
        "num_timeout_selected": sum(1 for r in selected if r.get("reason") == "timeout"),
    }
