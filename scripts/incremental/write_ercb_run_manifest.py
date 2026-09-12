#!/usr/bin/env python3
"""Write a reproducibility and zero-tail audit manifest for an ERCB run."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_DIR DATASET_DIR")
    run, dataset = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    repo = Path(__file__).resolve().parents[2]
    scheduler = json.loads((run / "view_scheduler_summary.json").read_text())
    metrics = json.loads((run / "screening_metrics.json").read_text())
    schedule = json.loads((dataset / "causal_arrivals.json").read_text())
    metadata = json.loads((dataset / "vigs_replay_metadata.json").read_text())
    code_files = (
        "train.py",
        "runtime/scheduler.py",
        "scripts/incremental/build_vigs_benchmark_causal_dataset.py",
        "scripts/incremental/run_ercb_utmm_tuning.sh",
        "scripts/incremental/compute_heldout_psnr.py",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.strip()
    payload = {
        "protocol": "ercb_utmm_tuning_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "dataset": str(dataset),
        "dataset_schedule_sha256": sha256(dataset / "causal_arrivals.json"),
        "dataset_metadata_sha256": sha256(dataset / "vigs_replay_metadata.json"),
        "vigs_sources": {
            "pose_sha256": metadata["pose_source_sha256"],
            "keyframe_sha256": metadata["keyframe_source_sha256"],
            "points_sha256": metadata["point_source_sha256"],
        },
        "code": {
            "git_head": head,
            "file_sha256": {name: sha256(repo / name) for name in code_files},
        },
        "scheduler": {
            key: scheduler[key] for key in (
                "name", "seed", "beta", "relative_floor_ratio", "block_size"
            )
        },
        "heldout": {
            "rule": schedule["heldout_rule"],
            "views": metrics["heldout_views"],
            "psnr": metrics["psnr"],
            "worst_q1_psnr": metrics["worst_q1_psnr"],
        },
        "contract": {
            "iterations": scheduler["total_iterations"],
            "selection_count_sum": sum(scheduler["selection_count"].values()),
            "last_arrival_iteration": max(scheduler["arrival_iteration"].values()),
            "tail_iters": schedule["tail_iters"],
            "sensor_eos_boundary_appended": metadata["appended_sensor_eos_boundary"],
            "zero_tail_pass": (
                schedule["tail_iters"] == 0
                and scheduler["total_iterations"] == max(scheduler["arrival_iteration"].values())
                and scheduler["total_iterations"] == sum(scheduler["selection_count"].values())
            ),
        },
    }
    (run / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
