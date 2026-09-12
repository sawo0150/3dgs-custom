#!/usr/bin/env python3
"""Compute capture-to-applied-service latency with explicit censoring."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "view_scheduler_summary.json").read_text())
    schedule = json.loads(args.schedule.read_text())
    capture = schedule["capture_monotonic_by_name"]
    completions = schedule["optimizer_slot_completion_monotonic"]
    first = summary.get("first_service_iteration", {})
    latency, invalid = [], []
    for name, step in first.items():
        if step < 1 or step > len(completions):
            invalid.append({"name": name, "iteration": step})
            continue
        value = float(completions[step - 1]) - float(capture[name])
        if value < -1e-9:
            invalid.append({"name": name, "iteration": step, "latency": value})
        else:
            latency.append(value)
    all_names = set(capture)
    served = set(first)
    censored = sorted(all_names - served)
    deadline_steps = int(summary.get("deadline_steps", 0))
    deadline_violations = 0
    if deadline_steps:
        arrivals = schedule["arrival_iteration_by_name"]
        deadline_violations = sum(
            name not in first or first[name] - int(arrivals[name]) + 1 > deadline_steps
            for name in all_names if int(arrivals[name]) <= len(completions)
        )
    payload = {
        "clock": schedule["clock_origin"],
        "completion_contract": schedule["service_completion"],
        "frames": len(all_names), "served": len(served),
        "served_fraction": len(served) / len(all_names),
        "right_censored_unserved": len(censored),
        "right_censored_names": censored,
        "latency_seconds": {
            "mean": statistics.fmean(latency) if latency else None,
            "median": percentile(latency, .5), "p90": percentile(latency, .9),
            "p95": percentile(latency, .95), "p99": percentile(latency, .99),
            "max": max(latency) if latency else None,
        },
        "deadline_steps": deadline_steps,
        "deadline_violations_or_unserved": deadline_violations,
        "optimizer_updates": summary.get("optimizer_update_count"),
        "backward_draws": sum(summary["selection_count"].values()),
        "invalid_causal_completions": invalid,
    }
    output = args.run / "service_latency_metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
