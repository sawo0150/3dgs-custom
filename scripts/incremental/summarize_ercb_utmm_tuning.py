#!/usr/bin/env python3
"""Aggregate bundle-wide UTMM ERCB tuning with paired RR deltas."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import cv2


EXPECTED_SCENES = (
    "ego-centric-1",
    "ego-centric-2",
    "ego-drive",
    "fast-straight",
    "slow-straight-2",
    "square-1",
)


def config_key(scheduler: dict) -> str:
    if scheduler["name"] == "causal_rr":
        return "rr"
    odds = math.exp(float(scheduler["beta"]))
    return f"rho={scheduler['relative_floor_ratio']:g},odds={odds:g},K={scheduler['block_size']}"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_ROOT")
    root = Path(sys.argv[1])
    rows = []
    for scene in EXPECTED_SCENES:
        for run in sorted((root / scene).glob("*_r4_s*")):
            if not (run / "screening_metrics.json").exists() or not (run / "view_scheduler_summary.json").exists():
                continue
            metrics = json.loads((run / "screening_metrics.json").read_text())
            scheduler = json.loads((run / "view_scheduler_summary.json").read_text())
            counts = list(scheduler["selection_count"].values())
            count_mean = statistics.fmean(counts)
            rows.append({
                "scene": scene,
                "run": run.name,
                "config": config_key(scheduler),
                "scheduler": scheduler["name"],
                "seed": int(scheduler["seed"]),
                "gamma": float(scheduler["beta"]),
                "relative_floor_ratio": float(scheduler["relative_floor_ratio"]),
                "block_size": int(scheduler["block_size"]),
                "psnr": float(metrics["psnr"]),
                "worst_q1_psnr": float(metrics["worst_q1_psnr"]),
                "count_cv": statistics.pstdev(counts) / count_mean,
                "_test_image_names": metrics["test_image_names"],
                "_per_view_psnr": metrics["per_view_psnr"],
            })

    baselines = {(row["scene"], row["seed"]): row for row in rows if row["config"] == "rr"}
    for baseline in baselines.values():
        baseline["rr_hard_q1_psnr"] = baseline["worst_q1_psnr"]
    for row in rows:
        baseline = baselines.get((row["scene"], row["seed"]))
        if baseline is not None:
            if row["_test_image_names"] != baseline["_test_image_names"]:
                raise ValueError(f"held-out view mismatch for {row['run']}")
            hard_count = max(1, math.ceil(len(baseline["_per_view_psnr"]) / 4))
            hard_indices = sorted(
                range(len(baseline["_per_view_psnr"])),
                key=baseline["_per_view_psnr"].__getitem__,
            )[:hard_count]
            row["rr_hard_q1_psnr"] = statistics.fmean(
                row["_per_view_psnr"][index] for index in hard_indices
            )
            row["delta_psnr_vs_rr"] = row["psnr"] - baseline["psnr"]
            row["delta_worst_q1_vs_rr"] = row["worst_q1_psnr"] - baseline["worst_q1_psnr"]
            row["delta_rr_hard_q1_vs_rr"] = (
                row["rr_hard_q1_psnr"] - baseline["rr_hard_q1_psnr"]
            )

    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row["config"] != "rr" and "delta_psnr_vs_rr" in row:
            grouped.setdefault((row["config"], row["seed"]), []).append(row)
    configs = []
    for (config, seed), members in sorted(grouped.items()):
        scene_names = sorted(row["scene"] for row in members)
        configs.append({
            "config": config,
            "seed": seed,
            "complete": scene_names == sorted(EXPECTED_SCENES),
            "scenes": scene_names,
            "mean_delta_psnr_vs_rr": statistics.fmean(row["delta_psnr_vs_rr"] for row in members),
            "mean_delta_worst_q1_vs_rr": statistics.fmean(row["delta_worst_q1_vs_rr"] for row in members),
            "mean_delta_rr_hard_q1_vs_rr": statistics.fmean(
                row["delta_rr_hard_q1_vs_rr"] for row in members
            ),
            "mean_count_cv": statistics.fmean(row["count_cv"] for row in members),
            "psnr_win_count": sum(row["delta_psnr_vs_rr"] > 0 for row in members),
        })
    eligible = [item for item in configs if item["complete"] and item["seed"] == 0]
    selected = max(
        eligible,
        key=lambda item: (item["mean_delta_psnr_vs_rr"], item["mean_delta_worst_q1_vs_rr"]),
        default=None,
    )
    selected_config = None if selected is None else selected["config"]
    selected_validation = [
        item for item in configs
        if item["complete"] and item["config"] == selected_config
    ]
    selected_rows = [
        row for row in rows
        if row["config"] == selected_config and "delta_psnr_vs_rr" in row
        and any(
            item["complete"] and item["seed"] == row["seed"]
            for item in selected_validation
        )
    ]
    final_validation = None
    if selected_rows:
        final_validation = {
            "config": selected_config,
            "complete_seeds": sorted({row["seed"] for row in selected_rows}),
            "scene_seed_pairs": len(selected_rows),
            "mean_delta_psnr_vs_rr": statistics.fmean(
                row["delta_psnr_vs_rr"] for row in selected_rows
            ),
            "mean_delta_worst_q1_vs_rr": statistics.fmean(
                row["delta_worst_q1_vs_rr"] for row in selected_rows
            ),
            "mean_delta_rr_hard_q1_vs_rr": statistics.fmean(
                row["delta_rr_hard_q1_vs_rr"] for row in selected_rows
            ),
            "mean_count_cv": statistics.fmean(row["count_cv"] for row in selected_rows),
            "psnr_win_count": sum(row["delta_psnr_vs_rr"] > 0 for row in selected_rows),
        }
        complete_keys = {(row["scene"], row["seed"]) for row in selected_rows}
        selected_baselines = [
            row for row in rows
            if row["config"] == "rr" and (row["scene"], row["seed"]) in complete_keys
        ]
        final_validation.update({
            "rr_mean_psnr": statistics.fmean(row["psnr"] for row in selected_baselines),
            "ercb_mean_psnr": statistics.fmean(row["psnr"] for row in selected_rows),
            "rr_mean_count_cv": statistics.fmean(row["count_cv"] for row in selected_baselines),
            "per_scene": [],
        })
        for scene in EXPECTED_SCENES:
            scene_rows = [row for row in selected_rows if row["scene"] == scene]
            scene_baselines = [row for row in selected_baselines if row["scene"] == scene]
            final_validation["per_scene"].append({
                "scene": scene,
                "rr_psnr": statistics.fmean(row["psnr"] for row in scene_baselines),
                "ercb_psnr": statistics.fmean(row["psnr"] for row in scene_rows),
                "delta_psnr_vs_rr": statistics.fmean(row["delta_psnr_vs_rr"] for row in scene_rows),
                "delta_worst_q1_vs_rr": statistics.fmean(
                    row["delta_worst_q1_vs_rr"] for row in scene_rows
                ),
                "delta_rr_hard_q1_vs_rr": statistics.fmean(
                    row["delta_rr_hard_q1_vs_rr"] for row in scene_rows
                ),
                "psnr_win_count": sum(row["delta_psnr_vs_rr"] > 0 for row in scene_rows),
            })
    for row in rows:
        row.pop("_test_image_names")
        row.pop("_per_view_psnr")
    payload = {
        "dataset_bundle": "UTMM",
        "expected_scenes": list(EXPECTED_SCENES),
        "selection_rule": "maximum seed-0 equal-scene mean paired held-out PSNR delta; worst-Q1 breaks exact ties",
        "selected_seed0_config": selected,
        "selected_validation_by_seed": selected_validation,
        "final_selected_validation": final_validation,
        "configs": configs,
        "runs": rows,
    }
    output = root / "utmm_tuning_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    compact = {
        "dataset_bundle": payload["dataset_bundle"],
        "expected_scenes": payload["expected_scenes"],
        "selection_rule": payload["selection_rule"],
        "selected_seed0_config": payload["selected_seed0_config"],
        "selected_validation_by_seed": payload["selected_validation_by_seed"],
        "final_selected_validation": payload["final_selected_validation"],
        "k_sensitivity_seed0": [
            item for item in payload["configs"]
            if item["seed"] == 0 and item["complete"]
            and item["config"] in {
                "rho=0.75,odds=1.5,K=4",
                "rho=0.75,odds=1.5,K=8",
                "rho=0.75,odds=1.5,K=16",
            }
        ],
        "full_summary": str(output),
    }
    compact_output = root.parent / "utmm_tuning_v1_compact.json"
    compact_output.write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
