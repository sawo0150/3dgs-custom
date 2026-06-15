from __future__ import annotations

from typing import List, Dict, Any


def filter_invalid_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if bool(r.get("valid", True))]


def filter_missing_pose(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if bool(r.get("pose_valid", False))]
