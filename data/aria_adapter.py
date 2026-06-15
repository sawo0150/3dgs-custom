"""Helpers for future Aria-specific offline/export parsing.

Keep this lightweight for now; the main goal is to provide a landing place for
Aria image/timestamp/pose conversion into the common manifest format.
"""

from __future__ import annotations

from typing import Dict, Any


def normalize_aria_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame_idx": raw.get("frame_idx"),
        "image_path": raw.get("image_path"),
        "image_name": raw.get("image_name"),
        "timestamp_ns": raw.get("timestamp_ns"),
        "valid": raw.get("valid", True),
    }
