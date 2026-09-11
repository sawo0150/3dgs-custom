#!/usr/bin/env python3
"""Build a fixed-pose causal COLMAP dataset for RPNG-AR or UTMM.

The output is intentionally shared by every scheduler arm.  Images are ordered
by acquisition time, every eighth image is held out by the stock 3DGS loader,
and each non-overlapping group of eight RGB frames arrives together.  The final
training-view arrival is also the training budget (zero-tail).

This is an offline scheduler-isolation harness: ground-truth poses and a shared
deterministic multi-depth ray scaffold are used at initialization.  It is not a
claim of strict RGB+IMU online operation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np


RPNG_K = (416.85223429743274, 414.92069080087543,
          421.02459311003213, 237.76180565241077)
RPNG_DIST = (-0.045761895748285604, 0.03423951132164367,
             -0.00040139057556727315, 0.000431371425853453)
# Camera optical axes in the UTMM robot frame, matching MM3DGS' loader.
UTMM_C2R = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("rpng", "utmm"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 keeps the full sequence")
    parser.add_argument("--rgb-per-interval", type=int, default=8)
    parser.add_argument("--iters-per-interval", type=int, default=60)
    parser.add_argument("--point-frame-stride", type=int, default=16)
    parser.add_argument("--point-pixel-stride", type=int, default=24)
    return parser.parse_args()


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values = [float(value) for value in line.split()]
            if len(values) >= 8:
                rows.append(values[:8])
    data = np.asarray(rows, dtype=np.float64)
    if len(data) < 2:
        raise ValueError(f"trajectory has fewer than two poses: {path}")
    order = np.argsort(data[:, 0])
    data = data[order]
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def interpolate_pose(timestamp: float, times: np.ndarray, positions: np.ndarray,
                     quaternions_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right <= 0:
        left = right = 0
        alpha = 0.0
    elif right >= len(times):
        left = right = len(times) - 1
        alpha = 0.0
    else:
        left = right - 1
        alpha = float((timestamp - times[left]) / (times[right] - times[left]))
    position = (1.0 - alpha) * positions[left] + alpha * positions[right]
    quaternion = quaternions_xyzw[left].copy()
    if left != right:
        other = quaternions_xyzw[right].copy()
        if float(np.dot(quaternion, other)) < 0.0:
            other *= -1.0
        quaternion = (1.0 - alpha) * quaternion + alpha * other
    quaternion /= np.linalg.norm(quaternion)
    return position, quaternion


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotmat_to_qvec(matrix: np.ndarray) -> np.ndarray:
    # COLMAP ordering is qw, qx, qy, qz.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25*s, (matrix[2, 1]-matrix[1, 2])/s,
                      (matrix[0, 2]-matrix[2, 0])/s,
                      (matrix[1, 0]-matrix[0, 1])/s])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = math.sqrt(1.0 + matrix[0, 0]-matrix[1, 1]-matrix[2, 2]) * 2.0
            q = np.array([(matrix[2, 1]-matrix[1, 2])/s, 0.25*s,
                          (matrix[0, 1]+matrix[1, 0])/s,
                          (matrix[0, 2]+matrix[2, 0])/s])
        elif axis == 1:
            s = math.sqrt(1.0 + matrix[1, 1]-matrix[0, 0]-matrix[2, 2]) * 2.0
            q = np.array([(matrix[0, 2]-matrix[2, 0])/s,
                          (matrix[0, 1]+matrix[1, 0])/s, 0.25*s,
                          (matrix[1, 2]+matrix[2, 1])/s])
        else:
            s = math.sqrt(1.0 + matrix[2, 2]-matrix[0, 0]-matrix[1, 1]) * 2.0
            q = np.array([(matrix[1, 0]-matrix[0, 1])/s,
                          (matrix[0, 2]+matrix[2, 0])/s,
                          (matrix[1, 2]+matrix[2, 1])/s, 0.25*s])
    if q[0] < 0.0:
        q *= -1.0
    return q / np.linalg.norm(q)


def utmm_timestamps_and_intrinsics(root: Path, count: int) -> tuple[list[float], tuple[float, ...]]:
    rows = [line for line in (root / "intrinsics.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != count:
        raise ValueError(f"UTMM RGB/intrinsics count mismatch: {count} vs {len(rows)}")
    timestamps = [float(line.split(maxsplit=1)[0]) for line in rows]
    match = re.search(r"\(([^)]+)\)", rows[0])
    if match is None:
        raise ValueError("could not parse UTMM intrinsics")
    values = [float(value.strip()) for value in match.group(1).split(",")]
    return timestamps, (values[0], values[4], values[2], values[5])


def image_inventory(kind: str, root: Path) -> tuple[list[Path], list[float], tuple[float, ...], tuple[float, ...]]:
    images = sorted((root / "rgb").glob("*.png"))
    if not images:
        raise ValueError(f"no PNG images under {root / 'rgb'}")
    if kind == "rpng":
        timestamps = [int(path.stem) / 1e9 for path in images]
        return images, timestamps, RPNG_K, RPNG_DIST
    timestamps, intrinsics = utmm_timestamps_and_intrinsics(root, len(images))
    return images, timestamps, intrinsics, ()


def write_scaffold(path: Path, images: list[Path], names: list[str],
                   poses_c2w: list[np.ndarray], intrinsics: tuple[float, ...],
                   frame_stride: int, pixel_stride: int, kind: str) -> int:
    fx, fy, cx, cy = intrinsics
    depths = (0.75, 1.5, 3.0, 5.0) if kind == "rpng" else (2.0, 5.0, 10.0, 20.0)
    point_id = 1
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Shared deterministic multi-depth ray scaffold\n")
        for frame_index in range(1, len(images), frame_stride):
            if frame_index % 8 == 0:
                continue
            image = cv2.imread(str(images[frame_index]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not read {images[frame_index]}")
            height, width = image.shape[:2]
            c2w = poses_c2w[frame_index]
            for v in range(pixel_stride // 2, height, pixel_stride):
                for u in range(pixel_stride // 2, width, pixel_stride):
                    color = image[v, u, ::-1]
                    ray = np.array([(u-cx)/fx, (v-cy)/fy, 1.0])
                    for depth in depths:
                        point = c2w[:3, :3] @ (ray * depth) + c2w[:3, 3]
                        stream.write(
                            f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                            f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n")
                        point_id += 1
    return point_id - 1


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.rgb_per_interval < 1 or args.iters_per_interval < 1:
        raise ValueError("interval sizes must be positive")
    images, timestamps, intrinsics, distortion = image_inventory(args.kind, args.input)
    if args.max_frames > 0:
        images, timestamps = images[:args.max_frames], timestamps[:args.max_frames]
    trajectory_name = "gt.txt" if args.kind == "rpng" else "groundtruth.txt"
    pose_times, positions, quaternions = read_trajectory(args.input / trajectory_name)

    args.output.mkdir(parents=True)
    image_root = args.output / "images"
    sparse_root = args.output / "sparse" / "0"
    image_root.mkdir()
    sparse_root.mkdir(parents=True)

    names: list[str] = []
    poses_c2w: list[np.ndarray] = []
    fx, fy, cx, cy = intrinsics
    width = height = 0
    for index, (source, timestamp) in enumerate(zip(images, timestamps)):
        name = f"frame_{index:06d}.png"
        destination = image_root / name
        if args.kind == "rpng":
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            undistorted = cv2.undistort(image, matrix, np.asarray(distortion))
            if not cv2.imwrite(str(destination), undistorted):
                raise OSError(f"failed to write {destination}")
        else:
            if index == 0:
                sample = cv2.imread(str(source), cv2.IMREAD_COLOR)
                height, width = sample.shape[:2]
            os.symlink(source.resolve(), destination)
        position, quaternion = interpolate_pose(timestamp, pose_times, positions, quaternions)
        rotation = quat_xyzw_to_rotmat(quaternion)
        if args.kind == "utmm":
            rotation = rotation @ UTMM_C2R
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = position
        poses_c2w.append(c2w)
        names.append(name)

    (sparse_root / "cameras.txt").write_text(
        f"# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")
    with (sparse_root / "images.txt").open("w", encoding="utf-8") as stream:
        stream.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for image_id, (name, c2w) in enumerate(zip(names, poses_c2w), start=1):
            w2c_rotation = c2w[:3, :3].T
            translation = -w2c_rotation @ c2w[:3, 3]
            qvec = rotmat_to_qvec(w2c_rotation)
            stream.write(
                f"{image_id} {qvec[0]:.12g} {qvec[1]:.12g} {qvec[2]:.12g} {qvec[3]:.12g} "
                f"{translation[0]:.12g} {translation[1]:.12g} {translation[2]:.12g} 1 {name}\n\n")

    points = write_scaffold(sparse_root / "points3D.txt", sorted(image_root.glob("*.png")),
                            names, poses_c2w, intrinsics, args.point_frame_stride,
                            args.point_pixel_stride, args.kind)
    arrivals = {}
    heldout = []
    for frame_index, name in enumerate(names):
        if frame_index % 8 == 0:
            heldout.append(name)
        else:
            arrivals[name] = 1 + (frame_index // args.rgb_per_interval) * args.iters_per_interval
    total_iterations = max(arrivals.values())
    schedule = {
        "arrival_iteration_by_name": arrivals,
        "total_iterations": total_iterations,
        "iters_per_event": args.iters_per_interval,
        "tail_iters": 0,
        "events": 1 + (len(names)-1) // args.rgb_per_interval,
        "all_frames": len(names),
        "train_frames": len(arrivals),
        "heldout_frames": len(heldout),
        "heldout_rule": "sorted COLMAP image index modulo 8 equals zero",
        "interval_definition": f"fixed non-overlapping {args.rgb_per_interval}-RGB-frame intervals",
    }
    (args.output / "causal_arrivals.json").write_text(json.dumps(schedule, indent=2) + "\n")
    metadata = {
        "kind": args.kind,
        "source": str(args.input.resolve()),
        "pose_source": "ground truth (offline scheduler-isolation harness)",
        "initialization": "shared deterministic multi-depth ray scaffold from training images",
        "point_count": points,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "width": width, "height": height},
        "rpng_undistorted": args.kind == "rpng",
        **{key: value for key, value in schedule.items() if key != "arrival_iteration_by_name"},
    }
    (args.output / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
