#!/usr/bin/env python3
"""Build an exp74-style causal replay dataset from a VIGS benchmark run.

This keeps the exp74/75 scheduler-isolation contract: final online VIGS poses,
actual VIGS keyframe events, and the cumulative geometry-only depth-anchor log
are fixed across scheduler arms.  The resulting dataset is offline/noncausal;
it is not a strict streaming claim.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


RPNG_K = (416.85223429743274, 414.92069080087543,
          421.02459311003213, 237.76180565241077)
RPNG_DIST = (-0.045761895748285604, 0.03423951132164367,
             -0.00040139057556727315, 0.000431371425853453)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("rpng", "utmm"), required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="prepared benchmark sequence")
    parser.add_argument("--vigs-run", type=Path, required=True,
                        help="completed VIGS run with pose/KF/anchor exports")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters-per-event", type=int, default=60)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory(path: Path) -> np.ndarray:
    trajectory = np.loadtxt(path, dtype=np.float64)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.shape[1] < 8 or len(trajectory) < 2:
        raise ValueError(f"invalid TUM trajectory: {path}")
    trajectory = trajectory[np.argsort(trajectory[:, 0])]
    if np.any(np.diff(trajectory[:, 0]) <= 0):
        raise ValueError(f"trajectory timestamps are not unique: {path}")
    return trajectory[:, :8]


def nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(reference, query), 1, len(reference) - 1)
    left = right - 1
    selected = np.where(
        np.abs(reference[left] - query) <= np.abs(reference[right] - query),
        left,
        right,
    )
    return np.asarray(sorted(set(map(int, selected))), dtype=np.int64)


def interpolate_poses(query: np.ndarray, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(query, trajectory[0, 0], trajectory[-1, 0])
    positions = np.stack([
        np.interp(clipped, trajectory[:, 0], trajectory[:, axis])
        for axis in (1, 2, 3)
    ], axis=1)
    rotations = Slerp(
        trajectory[:, 0], Rotation.from_quat(trajectory[:, 4:8])
    )(clipped).as_matrix()
    return positions, rotations


def inventory(kind: str, root: Path) -> tuple[list[Path], np.ndarray, tuple[float, ...], tuple[float, ...]]:
    rgb_root = root / ("rgb" if kind == "rpng" else "rgb_timestamp")
    images = sorted(rgb_root.glob("*.png"), key=lambda path: int(path.stem))
    if len(images) < 2:
        raise ValueError(f"no timestamped PNG sequence under {rgb_root}")
    timestamps = np.asarray([int(path.stem) * 1e-9 for path in images], dtype=np.float64)
    if kind == "rpng":
        return images, timestamps, RPNG_K, RPNG_DIST
    values = [float(value) for value in (root / "intrinsics_ours.txt").read_text().split()]
    if len(values) != 4:
        raise ValueError("UTMM intrinsics_ours.txt must contain fx fy cx cy")
    return images, timestamps, tuple(values), ()


def rotmat_to_qvec(matrix: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    qvec = np.asarray([w, x, y, z], dtype=np.float64)
    if qvec[0] < 0.0:
        qvec *= -1.0
    return qvec


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.iters_per_event < 1:
        raise ValueError("--iters-per-event must be positive")

    pose_path = args.vigs_run / "traj_full_online_eval.txt"
    keyframe_path = args.vigs_run / "traj_kf_beforeBA.txt"
    point_path = args.vigs_run / "points3D.txt"
    images, timestamps, intrinsics, distortion = inventory(args.kind, args.input)
    trajectory = load_trajectory(pose_path)
    keyframes = load_trajectory(keyframe_path)
    positions, rotations_c2w = interpolate_poses(timestamps, trajectory)
    mapped_boundaries = nearest_indices(keyframes[:, 0], timestamps)
    # VIGS includes the bootstrap keyframe at RGB index 0.  llffhold-8 reserves
    # that image for evaluation, so treating index 0 as a closed interval would
    # leave the scheduler with no eligible training view at iteration 1.  The
    # historical exp74 datasets start with the first non-empty training cohort;
    # discard only leading boundaries that precede it.
    first_train_index = next(index for index in range(len(images)) if index % 8 != 0)
    boundaries = mapped_boundaries[mapped_boundaries >= first_train_index]
    if not len(boundaries):
        raise ValueError("no VIGS keyframe boundaries mapped to RGB frames")

    args.output.mkdir(parents=True)
    image_root = args.output / "images"
    sparse_root = args.output / "sparse" / "0"
    image_root.mkdir()
    sparse_root.mkdir(parents=True)

    fx, fy, cx, cy = intrinsics
    camera_matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    width = height = 0
    image_lines: list[str] = []
    arrivals: dict[str, int] = {}
    heldout: list[str] = []
    for index, (source, position, rotation_c2w) in enumerate(
        zip(images, positions, rotations_c2w)
    ):
        destination = image_root / source.name
        if args.kind == "rpng":
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not read {source}")
            height, width = image.shape[:2]
            undistorted = cv2.undistort(image, camera_matrix, np.asarray(distortion))
            if not cv2.imwrite(str(destination), undistorted):
                raise OSError(f"failed to write {destination}")
        else:
            if index == 0:
                sample = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if sample is None:
                    raise ValueError(f"could not read {source}")
                height, width = sample.shape[:2]
            os.symlink(source.resolve(), destination)

        rotation_w2c = rotation_c2w.T
        translation = -rotation_w2c @ position
        qvec = rotmat_to_qvec(rotation_w2c)
        image_lines.append(
            f"{index + 1} {qvec[0]:.12g} {qvec[1]:.12g} {qvec[2]:.12g} "
            f"{qvec[3]:.12g} {translation[0]:.12g} {translation[1]:.12g} "
            f"{translation[2]:.12g} 1 {source.name}\n\n"
        )
        if index % 8 == 0:
            heldout.append(source.name)
        else:
            event = min(int(np.searchsorted(boundaries, index, side="left")), len(boundaries) - 1)
            arrivals[source.name] = 1 + event * args.iters_per_event

    (sparse_root / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n"
    )
    (sparse_root / "images.txt").write_text("".join(image_lines))

    points = np.loadtxt(point_path, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] < 4 or not np.isfinite(points[:, 1:4]).all():
        raise ValueError(f"invalid VIGS depth-anchor point file: {point_path}")
    with (sparse_root / "points3D.txt").open("w", encoding="utf-8") as stream:
        for point_id, xyz in enumerate(points[:, 1:4]):
            stream.write(
                f"{point_id} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} "
                "128 128 128 0\n"
            )

    group_sizes = collections.Counter(arrivals.values())
    group_histogram = collections.Counter(group_sizes.values())
    total_iterations = max(arrivals.values())
    schedule = {
        "arrival_iteration_by_name": arrivals,
        "total_iterations": total_iterations,
        "iters_per_event": args.iters_per_event,
        "tail_iters": 0,
        "events": len(boundaries),
        "all_frames": len(images),
        "train_frames": len(arrivals),
        "heldout_frames": len(heldout),
        "heldout_rule": "sorted COLMAP image index modulo 8 equals zero",
        "interval_definition": "actual VIGS keyframe timestamps mapped to nearest RGB frame",
    }
    (args.output / "causal_arrivals.json").write_text(json.dumps(schedule, indent=2) + "\n")
    metadata = {
        "kind": args.kind,
        "source": str(args.input.resolve()),
        "vigs_run": str(args.vigs_run.resolve()),
        "pose_source": "VIGS final online full-frame trajectory",
        "keyframe_source": "VIGS traj_kf_beforeBA keyframe timestamps",
        "initialization": "VIGS cumulative geometry-only depth-anchor log",
        "depth_anchor_stride": 40,
        "depth_anchor_valid_range_m": [0.01, 20.0],
        "initial_point_color": [128, 128, 128],
        "point_count": len(points),
        "point_source_sha256": sha256(point_path),
        "pose_source_sha256": sha256(pose_path),
        "keyframe_source_sha256": sha256(keyframe_path),
        "raw_vigs_keyframes": len(keyframes),
        "mapped_rgb_keyframe_boundaries": len(mapped_boundaries),
        "deduplicated_rgb_keyframe_boundaries": len(boundaries),
        "dropped_leading_heldout_only_boundaries": len(mapped_boundaries) - len(boundaries),
        "max_full_pose_to_rgb_timestamp_error_seconds": float(
            np.max(np.min(np.abs(trajectory[:, 0, None] - timestamps[None, :]), axis=1))
        ),
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "width": width, "height": height},
        "rpng_undistorted": args.kind == "rpng",
        "arrival_group_size_histogram": {
            str(size): count for size, count in sorted(group_histogram.items())
        },
        "offline_noncausal_fixed_pose_init": True,
        **{key: value for key, value in schedule.items() if key != "arrival_iteration_by_name"},
    }
    (args.output / "vigs_replay_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
