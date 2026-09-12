#!/usr/bin/env python3
"""Map physical RGB capture timestamps onto measured VIGS optimizer slots."""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path


SCENES = (
    "ego-centric-1", "ego-centric-2", "ego-drive", "fast-straight",
    "slow-straight-2", "square-1",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--vigs-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = {"clock": "physical RGB capture timestamp", "scenes": {}}
    for scene in SCENES:
        prepared = args.prepared_root / scene / "rgb_timestamp"
        vigs = args.vigs_root / scene / "seed0"
        dataset = args.dataset_root / scene
        audit = json.loads((vigs / "sensor_eos_audit.json").read_text())
        slots = json.loads((vigs / "optimizer_completion_times.json").read_text())
        slot_starts = [float(pair[0]) for pair in slots]
        slot_ends = [float(pair[1]) for pair in slots]
        images = sorted(prepared.glob("*.png"), key=lambda p: int(p.stem))
        if not images or not slots:
            raise ValueError(f"empty capture/optimizer trace: {scene}")
        sensor_t0 = int(images[0].stem) * 1e-9
        mono_t0 = float(audit["producer_start_monotonic"])
        arrivals, captures, censored = {}, {}, []
        for index, image in enumerate(images):
            if index % 8 == 0:
                continue
            name = image.name
            capture = mono_t0 + (int(image.stem) * 1e-9 - sensor_t0)
            slot = bisect.bisect_left(slot_starts, capture)
            # T+1 means observed before/at EOS but no optimizer service slot
            # remained.  It is right-censored, never silently dropped.
            arrival = slot + 1 if slot < len(slots) else len(slots) + 1
            arrivals[name] = arrival
            captures[name] = capture
            if arrival > len(slots):
                censored.append(name)
        dataset_names = sorted(p.name for p in (dataset / "images").glob("*.png"))
        expected = [name for index, name in enumerate(dataset_names) if index % 8 != 0]
        if sorted(expected) != sorted(arrivals):
            missing = sorted(set(expected) - set(arrivals))
            extra = sorted(set(arrivals) - set(expected))
            raise ValueError(f"dataset mismatch {scene}: missing={missing[:1]} extra={extra[:1]}")
        schedule = {
            "arrival_iteration_by_name": arrivals,
            "capture_monotonic_by_name": captures,
            "optimizer_slot_start_monotonic": slot_starts,
            "optimizer_slot_completion_monotonic": slot_ends,
            "total_iterations": len(slots),
            "tail_iters": 0,
            "clock_origin": "physical RGB capture timestamp aligned to VIGS producer_start_monotonic",
            "service_completion": "backward gradient included in an actually applied optimizer update",
            "right_censor_monotonic": float(audit["producer_eos_monotonic"]),
            "right_censored_without_slot": censored,
            "heldout_rule": "sorted COLMAP image index modulo 8 equals zero",
            "source_dataset": str(dataset.resolve()),
            "source_vigs_run": str(vigs.resolve()),
        }
        out = args.output / f"{scene}.json"
        out.write_text(json.dumps(schedule, indent=2) + "\n")
        duration = float(audit["producer_eos_monotonic"]) - mono_t0
        train_count = len(arrivals)
        rate = len(slots) / duration
        arrival_rate = train_count / duration
        bundle["scenes"][scene] = {
            "schedule": str(out.resolve()), "train_frames": train_count,
            "optimizer_slots": len(slots), "duration_seconds": duration,
            "optimizer_slots_per_second": rate,
            "training_frames_per_second": arrival_rate,
            "capacity_ratio_mu_over_lambda": rate / arrival_rate,
            "right_censored_without_slot": len(censored),
        }
    (args.output / "schedule_audit.json").write_text(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
