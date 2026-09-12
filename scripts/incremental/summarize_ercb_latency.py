#!/usr/bin/env python3
"""Summarize fixed-parameter ERCB latency transfer runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("schedule_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", default="ercb_latency_exp02_final_v3")
    args = parser.parse_args()
    audit = json.loads(args.schedule_audit.read_text())["scenes"]
    rows = []
    for scene_root in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for run in sorted(scene_root.glob("*_r4_s0")):
            quality_path = run / "screening_metrics.json"
            latency_path = run / "service_latency_metrics.json"
            if not quality_path.exists() or not latency_path.exists():
                continue
            quality = json.loads(quality_path.read_text())
            latency = json.loads(latency_path.read_text())
            wall = float(Path(str(run) + ".wall_seconds.txt").read_text())
            arm = run.name.rsplit("_r4_s0", 1)[0]
            rows.append({
                "scene": scene_root.name, "arm": arm,
                "psnr": quality["psnr"], "worst_q1_psnr": quality["worst_q1_psnr"],
                "served_fraction": latency["served_fraction"],
                "right_censored_unserved": latency["right_censored_unserved"],
                "p95_latency_seconds": latency["latency_seconds"]["p95"],
                "p99_latency_seconds": latency["latency_seconds"]["p99"],
                "optimizer_updates": latency["optimizer_updates"],
                "backward_draws": latency["backward_draws"],
                "wall_seconds": wall,
                "capacity_ratio_mu_over_lambda": audit[scene_root.name]["capacity_ratio_mu_over_lambda"],
            })
    arms = sorted(set(row["arm"] for row in rows))
    scenes = sorted(set(row["scene"] for row in rows))
    if len(rows) != len(arms) * len(scenes):
        raise ValueError(f"incomplete rectangular results: {len(rows)} rows")
    baseline = {row["scene"]: row for row in rows if row["arm"] == "capture_rr"}
    summaries = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        feasible = [row for row in selected if row["capacity_ratio_mu_over_lambda"] >= 1.0]
        summaries[arm] = {
            "mean_psnr": statistics.fmean(row["psnr"] for row in selected),
            "mean_worst_q1_psnr": statistics.fmean(row["worst_q1_psnr"] for row in selected),
            "mean_psnr_delta_vs_rr": statistics.fmean(
                row["psnr"] - baseline[row["scene"]]["psnr"] for row in selected
            ),
            "scene_wins_vs_rr": sum(
                row["psnr"] > baseline[row["scene"]]["psnr"] for row in selected
            ),
            "mean_served_fraction": statistics.fmean(row["served_fraction"] for row in selected),
            "feasible_scene_mean_p95_latency_seconds": statistics.fmean(
                row["p95_latency_seconds"] for row in feasible
            ),
            "feasible_scene_mean_p95_delta_vs_rr_seconds": statistics.fmean(
                row["p95_latency_seconds"] - baseline[row["scene"]]["p95_latency_seconds"]
                for row in feasible
            ),
            "mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in selected),
        }
    payload = {
        "protocol": args.protocol,
        "clock": "physical RGB capture timestamp",
        "completion": "gradient included in an actually applied optimizer update",
        "tuning_scene": "ego-drive",
        "transfer_scenes": scenes,
        "feasible_scenes": [scene for scene in scenes if audit[scene]["capacity_ratio_mu_over_lambda"] >= 1.0],
        "infeasible_capacity_controls": [scene for scene in scenes if audit[scene]["capacity_ratio_mu_over_lambda"] < 1.0],
        "arms": summaries, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
