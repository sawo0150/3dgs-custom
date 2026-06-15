from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

from selection.base import KeyframeSelectorBase
from data.pose_io import translation_distance_m, rotation_angle_deg


class PoseThresholdSelector(KeyframeSelectorBase):
    def __init__(self, trans_thresh_m: float = 0.15, rot_thresh_deg: float = 10.0,
                 max_gap: int = 20, min_gap: int = 0):
        self.trans_thresh_m = float(trans_thresh_m)
        self.rot_thresh_deg = float(rot_thresh_deg)
        self.max_gap = int(max_gap)
        self.min_gap = int(min_gap)

    def _maybe_pose(self, rec: Dict[str, Any]) -> Optional[list]:
        pose = rec.get("pose")
        if pose is None:
            return None
        return pose

    def select(self, records: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
        if not records:
            return [], []

        selected = [0]
        last_kf_idx = 0
        last_kf_pose = self._maybe_pose(records[0])
        debug = []

        debug.append({
            "frame_idx": records[0].get("frame_idx", 0),
            "image_name": records[0].get("image_name"),
            "keyframe_flag": 1,
            "reason": "first",
            "score": 1.0,
            "delta_trans_m": 0.0,
            "delta_rot_deg": 0.0,
        })

        for i in range(1, len(records)):
            rec = records[i]
            current_pose = self._maybe_pose(rec)
            gap = i - last_kf_idx
            dtrans = 0.0
            drot = 0.0
            reason = "skip"
            is_kf = False

            if gap <= self.min_gap:
                reason = "min_gap"
            elif gap >= self.max_gap:
                is_kf = True
                reason = "timeout"
            elif current_pose is not None and last_kf_pose is not None:
                dtrans = translation_distance_m(last_kf_pose, current_pose)
                drot = rotation_angle_deg(last_kf_pose, current_pose)
                if dtrans >= self.trans_thresh_m:
                    is_kf = True
                    reason = "translation"
                elif drot >= self.rot_thresh_deg:
                    is_kf = True
                    reason = "rotation"
            else:
                reason = "no_pose"

            if is_kf:
                selected.append(i)
                last_kf_idx = i
                last_kf_pose = current_pose

            debug.append({
                "frame_idx": rec.get("frame_idx", i),
                "image_name": rec.get("image_name"),
                "keyframe_flag": int(is_kf),
                "reason": reason,
                "score": max(dtrans / max(self.trans_thresh_m, 1e-9), drot / max(self.rot_thresh_deg, 1e-9)),
                "delta_trans_m": dtrans,
                "delta_rot_deg": drot,
            })

        return selected, debug
