from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_tracking.artifact_utils import maybe_log_artifacts


def run_legacy_training(dataset, opt, pipe, testing_iterations, saving_iterations,
                        checkpoint_iterations, checkpoint, debug_from,
                        cfg, run_dir: Path, wandb_logger=None):
    """Wrapper around the original 3DGS `training()`.

    This intentionally preserves the original optimization logic and only
    centralizes runtime/viewer/logging setup here.
    """
    import torch
    from utils.general_utils import safe_state
    from gaussian_renderer import network_gui
    from train import training as legacy_training

    safe_state(bool(getattr(opt, "quiet", False)))

    if getattr(opt, "viewer_enabled", False):
        network_gui.init(getattr(opt, "viewer_ip", "127.0.0.1"), int(getattr(opt, "viewer_port", 6009)))

    torch.autograd.set_detect_anomaly(bool(getattr(opt, "detect_anomaly", False)))

    if wandb_logger and wandb_logger.enabled:
        wandb_logger.log({
            "meta/source_path": str(getattr(dataset, "source_path", "")),
            "meta/model_path": str(getattr(dataset, "model_path", "")),
        }, step=0)

    legacy_training(
        dataset,
        opt,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        wandb_logger=wandb_logger,
    )

    if wandb_logger and wandb_logger.enabled:
        maybe_log_artifacts(cfg, run_dir, wandb_logger)
