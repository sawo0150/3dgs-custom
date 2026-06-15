from __future__ import annotations

from typing import Iterable, Iterator, Dict, Any


class ReplayLoader:
    def __init__(self, records: Iterable[Dict[str, Any]]):
        self.records = list(records)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self.records
