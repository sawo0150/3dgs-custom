from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.manifest_io import load_manifest, save_jsonl, save_csv
from data.pose_io import attach_pose_records
from selection.factory import build_selector_from_flat_args
from selection.metrics import summarize_selection
from exp_tracking.selection_table import build_selection_table_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run keyframe selection on a manifest.")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--selector", type=str, default="pose_threshold",
                   choices=["all", "uniform", "pose_threshold", "parallax", "hybrid"])
    p.add_argument("--pose-path", type=str, default=None)
    p.add_argument("--pose-format", type=str, default="auto")
    p.add_argument("--trans-thresh-m", type=float, default=0.15)
    p.add_argument("--rot-thresh-deg", type=float, default=10.0)
    p.add_argument("--max-gap", type=int, default=20)
    p.add_argument("--min-gap", type=int, default=0)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--blur-max", type=float, default=1e9)
    p.add_argument("--brightness-min", type=float, default=0.0)
    p.add_argument("--brightness-max", type=float, default=255.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.manifest)
    if args.pose_path:
        records = attach_pose_records(records, args.pose_path, args.pose_format)

    selector = build_selector_from_flat_args(args)
    selected_indices, debug_records = selector.select(records)

    selected_rows = [records[i] for i in selected_indices]
    keyframes_jsonl = out_dir / "keyframes.jsonl"
    selection_debug_csv = out_dir / "selection_debug.csv"
    summary_json = out_dir / "selection_summary.json"

    save_jsonl(selected_rows, keyframes_jsonl)
    save_csv(debug_records, selection_debug_csv)

    summary = summarize_selection(debug_records)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # convenience table export (csv-like rows, easy to inspect)
    rows = build_selection_table_rows(debug_records)
    save_csv(rows, out_dir / "selection_table_rows.csv")

    print(f"[selection] total_frames={len(records)}")
    print(f"[selection] selected={len(selected_indices)}")
    print(f"[selection] saved={keyframes_jsonl}")
    print(f"[selection] debug={selection_debug_csv}")
    print(f"[selection] summary={summary_json}")


if __name__ == "__main__":
    main()
