from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class WandbLogger:
    def __init__(self, enabled: bool, run=None):
        self.enabled = enabled
        self.run = run
        self._wandb = None

    @classmethod
    def from_config(cls, cfg, run_dir: Path) -> "WandbLogger":
        enabled = bool(cfg.logging.get("enabled", False))
        if not enabled:
            return cls(False, None)
        import wandb
        run = wandb.init(
            project=cfg.logging.get("project", "3dgs-keyframe"),
            entity=cfg.logging.get("entity", None),
            name=cfg.logging.get("name", None),
            tags=list(cfg.logging.get("tags", [])),
            mode=cfg.logging.get("mode", "online"),
            dir=str(run_dir),
        )
        obj = cls(True, run)
        obj._wandb = wandb
        return obj

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log(data, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        if self.enabled and self.run is not None:
            self.run.config.update(config, allow_val_change=True)

    def log_table(self, name: str, columns, rows) -> None:
        if self.enabled and self.run is not None:
            table = self._wandb.Table(columns=columns, data=rows)
            self.run.log({name: table})

    def log_image(self, name: str, image_path: str, caption: Optional[str] = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log({name: self._wandb.Image(image_path, caption=caption)})

    def log_tensor_image(self, name: str, image, caption: Optional[str] = None, step: Optional[int] = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log({name: self._wandb.Image(image, caption=caption)}, step=step)

    def log_artifact_path(self, path: str, name: str, artifact_type: str = "dataset") -> None:
        if self.enabled and self.run is not None:
            art = self._wandb.Artifact(name=name, type=artifact_type)
            art.add_file(path)
            self.run.log_artifact(art)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
