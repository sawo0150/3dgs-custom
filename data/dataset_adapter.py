from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from data.manifest_io import load_jsonl
from data.pose_io import load_pose_records


def _selected_names_from_keyframe_list(path: str | None) -> Optional[Set[str]]:
    if not path:
        return None
    rows = load_jsonl(path)
    names = {row.get("image_name") for row in rows if row.get("image_name") is not None}
    return names


def maybe_apply_keyframe_filter_to_scene_info(scene_info, args):
    keyframe_list_path = getattr(args, "keyframe_list_path", None)
    selected_names = _selected_names_from_keyframe_list(keyframe_list_path)
    if not selected_names:
        return scene_info

    out = copy.deepcopy(scene_info)
    out.train_cameras = [c for c in scene_info.train_cameras if getattr(c, "image_name", None) in selected_names]
    keep_test = bool(getattr(args, "selection_keep_test", True))
    if not keep_test:
        out.test_cameras = [c for c in scene_info.test_cameras if getattr(c, "image_name", None) in selected_names]
    return out


def maybe_apply_pose_override_to_scene_info(scene_info, args):
    pose_override_path = getattr(args, "pose_override_path", None)
    pose_override_format = getattr(args, "pose_override_format", "auto")
    if not pose_override_path:
        return scene_info

    pose_map = load_pose_records(pose_override_path, pose_override_format)
    out = copy.deepcopy(scene_info)

    def _apply(camera_list):
        for cam in camera_list:
            idx = getattr(cam, "frame_idx", None)
            if idx is None:
                continue
            if idx in pose_map:
                if hasattr(cam, "world_view_transform"):
                    # NOTE:
                    # This assumes the override matrix is compatible with the downstream convention.
                    # Adjust here if your local scene reader uses c2w instead of w2c.
                    cam.pose_override = pose_map[idx]
        return camera_list

    out.train_cameras = _apply(out.train_cameras)
    out.test_cameras = _apply(out.test_cameras)
    return out
