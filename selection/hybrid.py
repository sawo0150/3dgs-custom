from __future__ import annotations

from typing import List, Dict, Any, Tuple

from selection.pose_threshold import PoseThresholdSelector


class HybridSelector(PoseThresholdSelector):
    def __init__(self, trans_thresh_m: float = 0.15, rot_thresh_deg: float = 10.0,
                 max_gap: int = 20, min_gap: int = 0,
                 blur_max: float = 1e9, brightness_min: float = 0.0,
                 brightness_max: float = 255.0):
        super().__init__(trans_thresh_m, rot_thresh_deg, max_gap, min_gap)
        self.blur_max = float(blur_max)
        self.brightness_min = float(brightness_min)
        self.brightness_max = float(brightness_max)

    def _passes_quality(self, rec: Dict[str, Any]) -> bool:
        blur_score = float(rec.get("blur_score", 0.0))
        brightness = float(rec.get("brightness_mean", 128.0))
        return blur_score <= self.blur_max and self.brightness_min <= brightness <= self.brightness_max

    def select(self, records: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
        selected, debug = super().select(records)
        selected_set = set(selected)
        filtered_selected = []
        filtered_debug = []

        for i, (rec, dbg) in enumerate(zip(records, debug)):
            is_kf = i in selected_set
            if is_kf and dbg["reason"] not in {"first", "timeout"} and not self._passes_quality(rec):
                dbg = dict(dbg)
                dbg["keyframe_flag"] = 0
                dbg["reason"] = "quality_reject"
                is_kf = False
            if is_kf:
                filtered_selected.append(i)
            filtered_debug.append(dbg)

        if 0 not in filtered_selected and records:
            filtered_selected = [0] + filtered_selected
            filtered_debug[0] = dict(filtered_debug[0])
            filtered_debug[0]["keyframe_flag"] = 1
            filtered_debug[0]["reason"] = "first"
        return filtered_selected, filtered_debug
