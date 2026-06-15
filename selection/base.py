from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class KeyframeSelectorBase(ABC):
    @abstractmethod
    def select(self, records: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
        raise NotImplementedError
