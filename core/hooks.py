from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


class HookBase:
    def on_train_start(self, **kwargs) -> None:
        pass

    def on_iteration_end(self, **kwargs) -> None:
        pass

    def on_eval_end(self, **kwargs) -> None:
        pass

    def on_checkpoint_saved(self, **kwargs) -> None:
        pass

    def on_train_end(self, **kwargs) -> None:
        pass


@dataclass
class HookContext:
    state: Dict[str, Any] = field(default_factory=dict)
