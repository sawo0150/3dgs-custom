from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IncrementalSession:
    session_id: str
    map_version: int = 0
    latest_frame_idx: Optional[int] = None
    accepted_keyframes: List[int] = field(default_factory=list)
