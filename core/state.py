from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RunState:
    run_dir: str
    selection_path: Optional[str] = None
    manifest_path: Optional[str] = None
    keyframe_count: Optional[int] = None


@dataclass
class SessionState:
    session_id: str
    current_map_version: int = 0
    latest_timestamp_ns: Optional[int] = None
    keyframe_ids: List[int] = field(default_factory=list)
