from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FramePacket:
    frame_idx: int
    image_path: str
    timestamp_ns: Optional[int] = None
    pose: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class KeyframePacket(FramePacket):
    reason: Optional[str] = None
