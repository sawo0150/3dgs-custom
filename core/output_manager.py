from __future__ import annotations

from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def prepare_run_directory(cfg: DictConfig) -> Path:
    output_root = Path(cfg.project.output_root)
    run_dir = output_root / cfg.project.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "selection").mkdir(exist_ok=True)
    (run_dir / "renders").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def dump_resolved_config(cfg: DictConfig, path: Path) -> None:
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
