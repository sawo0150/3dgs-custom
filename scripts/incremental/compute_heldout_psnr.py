#!/usr/bin/env python3
"""Compute inexpensive held-out PSNR statistics from saved render PNGs."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import cv2


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_DIR")
    run = Path(sys.argv[1])
    methods = sorted(
        (path for path in (run / "test").glob("ours_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not methods:
        raise ValueError(f"no rendered test method under {run}")
    method = methods[-1]
    per_view = []
    names = []
    for render_path in sorted((method / "renders").glob("*.png")):
        target_path = method / "gt" / render_path.name
        rendered = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if rendered is None or target is None or rendered.shape != target.shape:
            raise ValueError(f"invalid render/GT pair: {render_path}, {target_path}")
        difference = rendered.astype(float) / 255.0 - target.astype(float) / 255.0
        mse = float((difference * difference).mean())
        per_view.append(float("inf") if mse == 0.0 else -10.0 * math.log10(mse))
        names.append(render_path.name)
    if not per_view:
        raise ValueError(f"no render/GT pairs under {method}")
    q1_count = max(1, math.ceil(len(per_view) / 4))
    payload = {
        "method": method.name,
        "heldout_views": len(per_view),
        "psnr": statistics.fmean(per_view),
        "worst_q1_psnr": statistics.fmean(sorted(per_view)[:q1_count]),
        "test_image_names": names,
        "per_view_psnr": per_view,
        "source": "8-bit saved render/GT PNG pairs",
    }
    (run / "screening_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
