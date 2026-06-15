from __future__ import annotations

from typing import List, Dict, Any, Tuple

from selection.base import KeyframeSelectorBase


class ParallaxSelector(KeyframeSelectorBase):
    """Placeholder selector.

    Right now this falls back to a max-gap policy because robust feature-based
    parallax requires image loading + matching, which should be added in the next stage.
    """

    def __init__(self, parallax_thresh_px: float = 8.0, max_gap: int = 20, min_gap: int = 0):
        self.parallax_thresh_px = float(parallax_thresh_px)
        self.max_gap = int(max_gap)
        self.min_gap = int(min_gap)

    def select(self, records: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
        if not records:
            return [], []
        selected = [0]
        debug = [{
            "frame_idx": records[0].get("frame_idx", 0),
            "image_name": records[0].get("image_name"),
            "keyframe_flag": 1,
            "reason": "first",
            "score": 1.0,
            "parallax_px": 0.0,
        }]
        last = 0
        for i, rec in enumerate(records[1:], start=1):
            gap = i - last
            is_kf = gap >= self.max_gap
            if is_kf:
                last = i
            debug.append({
                "frame_idx": rec.get("frame_idx", i),
                "image_name": rec.get("image_name"),
                "keyframe_flag": int(is_kf),
                "reason": "timeout" if is_kf else "skip",
                "score": float(gap / max(self.max_gap, 1)),
                "parallax_px": 0.0,
            })
            if is_kf:
                selected.append(i)
        return selected, debug
