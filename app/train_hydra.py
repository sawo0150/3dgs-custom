from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_bridge import build_legacy_groups, build_schedule_lists, inject_runtime_args
from core.output_manager import prepare_run_directory, dump_resolved_config
from core.trainer import run_legacy_training
from exp_tracking.wandb_logger import WandbLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = prepare_run_directory(cfg)
    dump_resolved_config(cfg, run_dir / "resolved_config.yaml")

    model_args, opt_args, pipe_args = build_legacy_groups(cfg)
    testing_iterations, save_iterations, checkpoint_iterations = build_schedule_lists(cfg)
    inject_runtime_args(cfg, model_args, opt_args, pipe_args, run_dir)

    wb = WandbLogger.from_config(cfg, run_dir)
    if wb.enabled:
        wb.log_config(OmegaConf.to_container(cfg, resolve=True))

    try:
        run_legacy_training(
            dataset=model_args,
            opt=opt_args,
            pipe=pipe_args,
            testing_iterations=testing_iterations,
            saving_iterations=save_iterations,
            checkpoint_iterations=checkpoint_iterations,
            checkpoint=getattr(cfg.train, "start_checkpoint", None),
            debug_from=getattr(cfg.train, "debug_from", -1),
            cfg=cfg,
            run_dir=run_dir,
            wandb_logger=wb,
        )
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
