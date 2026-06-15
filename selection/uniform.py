from __future__ import annotations

from typing import List, Dict, Any, Tuple

from selection.base import KeyframeSelectorBase


class UniformSelector(KeyframeSelectorBase):
    def __init__(self, stride: int = 10, offset: int = 0, max_gap: int = 10**9):
        self.stride = max(1, int(stride))
        self.offset = int(offset)
        self.max_gap = int(max_gap)

    def select(self, records: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
        selected = []
        debug = []
        for i, rec in enumerate(records):
            is_kf = ((i - self.offset) % self.stride == 0)
            if is_kf:
                selected.append(i)
            debug.append({
                "frame_idx": rec.get("frame_idx", i),
                "image_name": rec.get("image_name"),
                "keyframe_flag": int(is_kf),
                "reason": "uniform" if is_kf else "skip",
                "score": 1.0 if is_kf else 0.0,
                "delta_trans_m": 0.0,
                "delta_rot_deg": 0.0,
            })
        return selected, debug
