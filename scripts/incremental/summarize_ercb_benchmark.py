#!/usr/bin/env python3
"""Collect final held-out metrics and scheduler statistics for one ablation."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np


def final_metric(payload: dict) -> tuple[str, dict]:
    key = max(payload, key=lambda item: int(item.rsplit("_", 1)[-1]))
    return key, payload[key]


def per_view_psnr(run: Path, iteration_key: str) -> tuple[list[str], list[float]]:
    method = iteration_key
    base = run / "test" / method
    renders = sorted((base / "renders").glob("*.png"))
    gt_root = base / "gt"
    values = []
    for render_path in renders:
        rendered = cv2.imread(str(render_path), cv2.IMREAD_COLOR).astype(np.float64) / 255.0
        target = cv2.imread(str(gt_root / render_path.name), cv2.IMREAD_COLOR).astype(np.float64) / 255.0
        mse = float(np.mean((rendered - target) ** 2))
        values.append(float("inf") if mse == 0.0 else -10.0 * math.log10(mse))
    return [path.name for path in renders], values


def quartile_mean(values: list[float]) -> float:
    count = max(1, math.ceil(len(values) / 4))
    return statistics.fmean(sorted(values)[:count])


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_ROOT RUN_LABEL")
    root, label = Path(sys.argv[1]), sys.argv[2]
    rows = []
    for run in sorted(root.glob(f"{label}_zerotail_*_r4_s*")):
        if not run.is_dir() or not (run / "results.json").exists():
            continue
        iteration, metrics = final_metric(json.loads((run / "results.json").read_text()))
        scheduler = json.loads((run / "view_scheduler_summary.json").read_text())
        names, frame_psnr = per_view_psnr(run, iteration)
        counts = list(scheduler["selection_count"].values())
        count_mean = statistics.fmean(counts)
        rows.append({
            "run": run.name,
            "iteration": iteration,
            "scheduler": scheduler["name"],
            "seed": scheduler["seed"],
            "psnr": metrics["PSNR"],
            "ssim": metrics["SSIM"],
            "lpips": metrics["LPIPS"],
            "worst_q1_psnr": quartile_mean(frame_psnr),
            "thirds_psnr": [statistics.fmean(part.tolist()) for part in np.array_split(frame_psnr, 3)],
            "count_cv": statistics.pstdev(counts) / count_mean,
            "test_image_names": names,
            "per_view_psnr": frame_psnr,
        })
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scheduler"], []).append(row["psnr"])
    summary = {
        "label": label,
        "runs": rows,
        "mean_psnr_by_scheduler": {
            scheduler: statistics.fmean(values) for scheduler, values in grouped.items()
        },
    }
    # RR-hard-Q1 fixes the difficult frame IDs on each seed's RR baseline.
    for row in rows:
        baseline = next(candidate for candidate in rows
                        if candidate["seed"] == row["seed"] and candidate["scheduler"] == "causal_rr")
        hard_count = max(1, math.ceil(len(baseline["per_view_psnr"]) / 4))
        hard_indices = sorted(range(len(baseline["per_view_psnr"])),
                              key=baseline["per_view_psnr"].__getitem__)[:hard_count]
        row["rr_hard_q1_psnr"] = statistics.fmean(row["per_view_psnr"][index] for index in hard_indices)
    output = root / f"{label}_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
