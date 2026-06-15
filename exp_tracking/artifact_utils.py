from __future__ import annotations

from pathlib import Path


def maybe_log_artifacts(cfg, run_dir: Path, wandb_logger) -> None:
    if not getattr(cfg.logging, "upload_artifacts", False):
        return
    candidate_paths = [
        run_dir / "resolved_config.yaml",
        run_dir / "selection" / "keyframes.jsonl",
        run_dir / "selection" / "selection_debug.csv",
    ]
    for p in candidate_paths:
        if p.exists():
            wandb_logger.log_artifact_path(str(p), name=f"{p.stem}")
