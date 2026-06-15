from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.manifest_io import build_basic_manifest_from_image_dir, save_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a manifest from an image directory.")
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--glob", type=str, default="*.png")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--timestamp-mode", type=str, default="index", choices=["index", "filename", "none"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = build_basic_manifest_from_image_dir(
        images_dir=Path(args.images_dir),
        pattern=args.glob,
        recursive=args.recursive,
        start_index=args.start_index,
        timestamp_mode=args.timestamp_mode,
    )
    save_manifest(records, args.output)
    print(f"[manifest] wrote {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
