from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_basic_manifest_from_image_dir(images_dir: Path, pattern: str = "*.png", recursive: bool = False,
                                        start_index: int = 0, timestamp_mode: str = "index") -> List[Dict[str, Any]]:
    globber = images_dir.rglob if recursive else images_dir.glob
    paths = sorted([p for p in globber(pattern) if p.suffix.lower() in IMAGE_EXTS])
    records = []
    for i, p in enumerate(paths, start=start_index):
        ts = None
        if timestamp_mode == "index":
            ts = i
        elif timestamp_mode == "filename":
            stem = p.stem
            digits = ''.join(ch for ch in stem if ch.isdigit())
            ts = int(digits) if digits else None
        rec = {
            "frame_idx": i,
            "image_path": str(p.resolve()),
            "image_name": p.name,
            "timestamp_ns": ts,
            "valid": True,
        }
        records.append(rec)
    return records


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_csv(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    records = list(records)
    if not records:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    if path.suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported manifest format: {path}")


def save_manifest(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".jsonl":
        return save_jsonl(records, path)
    if path.suffix == ".csv":
        return save_csv(records, path)
    raise ValueError(f"Unsupported manifest format: {path}")
