#!/usr/bin/env python3
"""Generate KITTI road-plane files from linefit ground segmentation.

The script fits one plane per scene in rectified camera coordinates.

For converted V2X-SPD -> KITTI roots, it can:
1. write the real plane txt files back into the original raw scene folders:
   `V2X-SPD/{training,validation}/{scene}/planes/<frame>.txt`
2. create a flat `shared/planes/<scene>_<frame>.txt` symlink layer
3. point `<data-root>/training/planes` at that shared plane directory

This matches OpenPCDet's KITTI `road_plane` loading path while keeping the
source-of-truth plane files on the original dataset layout.
"""

from __future__ import annotations

import os
import argparse
from collections import defaultdict
from contextlib import contextmanager
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pcdet.utils import calibration_kitti


@contextmanager
def suppress_native_stdout():
    saved_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="KITTI-style dataset root")
    parser.add_argument(
        "--split",
        choices=["train", "val", "trainval"],
        default="trainval",
        help="ImageSets split used to discover sample ids",
    )
    parser.add_argument("--scene-id", type=str, default=None, help="Only process one scene")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Debug limit. 0 means all matched scenes",
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=0,
        help="Debug limit. 0 means use all frames from the scene",
    )
    parser.add_argument(
        "--max-ground-points-per-frame",
        type=int,
        default=20000,
        help="Randomly subsample ground points from each frame before plane fitting",
    )
    parser.add_argument(
        "--linefit-config",
        type=Path,
        default=None,
        help="Optional TOML config for linefit. Recommended for roadside LiDAR with custom sensor height.",
    )
    parser.add_argument(
        "--refine-dist-thresh",
        type=float,
        default=0.15,
        help="Inlier threshold in meters for iterative plane refinement",
    )
    parser.add_argument(
        "--refine-iters",
        type=int,
        default=2,
        help="Number of robust refit iterations after the initial fit",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point subsampling")
    parser.add_argument(
        "--write-raw-scene-planes",
        action="store_true",
        help="Write actual plane files into resolved raw scene directories instead of only writing flat KITTI files.",
    )
    parser.add_argument(
        "--build-shared-links",
        action="store_true",
        help="Create <data-root parent>/shared/planes/<sample_id>.txt symlinks that point to raw scene plane files.",
    )
    parser.add_argument(
        "--link-training-planes",
        action="store_true",
        help="Make <data-root>/training/planes a symlink to ../../shared/planes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fit and print planes without writing files")
    return parser.parse_args()


def load_ground_seg():
    try:
        from linefit import ground_seg
    except ImportError as exc:
        raise SystemExit(
            "Failed to import linefit. Install the Python package first, or make sure it is available in the environment."
        ) from exc
    return ground_seg


def read_split_ids(data_root: Path, split: str) -> list[str]:
    split_file = data_root / "ImageSets" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def collect_sample_ids(data_root: Path, split: str) -> list[str]:
    if split != "trainval":
        return read_split_ids(data_root, split)

    merged = read_split_ids(data_root, "train") + read_split_ids(data_root, "val")
    return list(dict.fromkeys(merged))


def infer_scene_id(data_root: Path, sample_id: str) -> str:
    calib_file = data_root / "training" / "calib" / f"{sample_id}.txt"
    if calib_file.exists():
        resolved = calib_file.resolve()
        if resolved.parent.name == "calib" and len(resolved.parents) >= 2:
            return resolved.parent.parent.name

    if "_" in sample_id:
        return sample_id.rsplit("_", 1)[0]

    raise ValueError(f"cannot infer scene id from sample id: {sample_id}")


def group_sample_ids_by_scene(data_root: Path, sample_ids: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        grouped[infer_scene_id(data_root, sample_id)].append(sample_id)
    return dict(grouped)


def resolve_raw_calib(data_root: Path, sample_id: str) -> Path:
    calib_file = data_root / "training" / "calib" / f"{sample_id}.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"calibration file not found: {calib_file}")
    return calib_file.resolve()


def resolve_raw_scene_frame(data_root: Path, sample_id: str) -> tuple[Path, str]:
    raw_calib = resolve_raw_calib(data_root, sample_id)
    if raw_calib.parent.name != "calib":
        raise RuntimeError(f"unexpected raw calibration layout: {raw_calib}")
    scene_root = raw_calib.parent.parent
    frame_id = raw_calib.stem
    return scene_root, frame_id


def load_lidar_points(data_root: Path, sample_id: str) -> np.ndarray:
    lidar_file = data_root / "training" / "velodyne" / f"{sample_id}.bin"
    if not lidar_file.exists():
        raise FileNotFoundError(f"lidar file not found: {lidar_file}")
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)


def load_calibration(data_root: Path, sample_id: str) -> calibration_kitti.Calibration:
    calib_file = data_root / "training" / "calib" / f"{sample_id}.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"calibration file not found: {calib_file}")
    return calibration_kitti.Calibration(calib_file)


def fit_plane_svd(points_rect: np.ndarray) -> np.ndarray:
    if points_rect.shape[0] < 3:
        raise ValueError("need at least 3 points to fit a plane")

    centroid = points_rect.mean(axis=0)
    _, _, vh = np.linalg.svd(points_rect - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    # KITTI loader expects the normal to face upward in rect camera coordinates.
    if normal[1] > 0:
        normal = -normal

    d = -float(np.dot(normal, centroid))
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def refine_plane(points_rect: np.ndarray, dist_thresh: float, refine_iters: int) -> tuple[np.ndarray, np.ndarray]:
    plane = fit_plane_svd(points_rect)
    inliers = np.ones(points_rect.shape[0], dtype=bool)

    for _ in range(refine_iters):
        dist = np.abs(points_rect @ plane[:3] + plane[3])
        new_inliers = dist <= dist_thresh
        if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers
        plane = fit_plane_svd(points_rect[inliers])

    if inliers.sum() >= 3:
        plane = fit_plane_svd(points_rect[inliers])

    return plane, inliers


def format_plane_kitti(plane: np.ndarray) -> str:
    return (
        "# Matrix\n"
        "WIDTH 4\n"
        "HEIGHT 1\n"
        f"{plane[0]:.6e} {plane[1]:.6e} {plane[2]:.6e} {plane[3]:.6e}\n"
    )


def sample_ground_points_rect(
    data_root: Path,
    sample_ids: list[str],
    groundseg,
    rng: np.random.Generator,
    max_frames_per_scene: int,
    max_ground_points_per_frame: int,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    points_rect_parts: list[np.ndarray] = []
    frame_stats: list[tuple[str, int, int]] = []

    if max_frames_per_scene > 0 and len(sample_ids) > max_frames_per_scene:
        indices = np.linspace(0, len(sample_ids) - 1, num=max_frames_per_scene, dtype=int)
        chosen_ids = [sample_ids[i] for i in indices]
    else:
        chosen_ids = sample_ids
    for sample_id in chosen_ids:
        points_lidar = load_lidar_points(data_root, sample_id)
        with suppress_native_stdout():
            labels = np.asarray(groundseg.run(points_lidar[:, :3]), dtype=np.uint8)
        ground_mask = labels.astype(bool)
        num_ground = int(ground_mask.sum())
        if num_ground < 3:
            frame_stats.append((sample_id, points_lidar.shape[0], num_ground))
            continue

        ground_points_lidar = points_lidar[ground_mask, :3]
        if max_ground_points_per_frame > 0 and ground_points_lidar.shape[0] > max_ground_points_per_frame:
            choice = rng.choice(ground_points_lidar.shape[0], size=max_ground_points_per_frame, replace=False)
            ground_points_lidar = ground_points_lidar[choice]

        calib = load_calibration(data_root, sample_id)
        ground_points_rect = calib.lidar_to_rect(ground_points_lidar)
        points_rect_parts.append(ground_points_rect.astype(np.float64, copy=False))
        frame_stats.append((sample_id, points_lidar.shape[0], num_ground))

    if not points_rect_parts:
        raise RuntimeError("no usable ground points collected from the selected scene")

    return np.concatenate(points_rect_parts, axis=0), frame_stats


def ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")

    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def write_scene_planes_raw(data_root: Path, sample_ids: list[str], plane: np.ndarray, dry_run: bool) -> int:
    content = format_plane_kitti(plane)
    count = 0
    for sample_id in sample_ids:
        scene_root, frame_id = resolve_raw_scene_frame(data_root, sample_id)
        plane_file = scene_root / "planes" / f"{frame_id}.txt"
        if not dry_run:
            plane_file.parent.mkdir(parents=True, exist_ok=True)
            plane_file.write_text(content)
        count += 1
    return count


def write_scene_planes_flat(data_root: Path, sample_ids: list[str], plane: np.ndarray, dry_run: bool) -> int:
    plane_dir = data_root / "training" / "planes"
    if not dry_run:
        plane_dir.mkdir(parents=True, exist_ok=True)

    content = format_plane_kitti(plane)
    count = 0
    for sample_id in sample_ids:
        if not dry_run:
            (plane_dir / f"{sample_id}.txt").write_text(content)
        count += 1
    return count


def create_shared_plane_links(data_root: Path, sample_ids: list[str], dry_run: bool) -> int:
    shared_plane_dir = data_root.parent / "shared" / "planes"
    count = 0
    for sample_id in sample_ids:
        scene_root, frame_id = resolve_raw_scene_frame(data_root, sample_id)
        raw_plane = scene_root / "planes" / f"{frame_id}.txt"
        if not raw_plane.exists() and not dry_run:
            raise FileNotFoundError(f"raw plane file not found: {raw_plane}")
        flat_link = shared_plane_dir / f"{sample_id}.txt"
        if not dry_run:
            relative_target = Path(os.path.relpath(raw_plane, start=shared_plane_dir))
            ensure_symlink(flat_link, relative_target)
        count += 1
    return count


def link_training_planes_dir(data_root: Path, dry_run: bool) -> Path:
    training_planes = data_root / "training" / "planes"
    relative_target = Path("../../shared/planes")
    if not dry_run:
        ensure_symlink(training_planes, relative_target)
    return training_planes


def main() -> None:
    args = parse_args()

    if not (args.data_root / "training").exists():
        raise FileNotFoundError(f"training directory not found under data root: {args.data_root}")

    if args.linefit_config is not None and not args.linefit_config.exists():
        raise FileNotFoundError(f"linefit config not found: {args.linefit_config}")

    sample_ids = collect_sample_ids(args.data_root, args.split)
    grouped = group_sample_ids_by_scene(args.data_root, sample_ids)

    scene_ids = sorted(grouped)
    if args.scene_id is not None:
        if args.scene_id not in grouped:
            raise KeyError(f"scene id not found in split {args.split}: {args.scene_id}")
        scene_ids = [args.scene_id]

    if args.max_scenes > 0:
        scene_ids = scene_ids[:args.max_scenes]

    if not scene_ids:
        raise RuntimeError("no scenes matched the selection")

    if not any([args.write_raw_scene_planes, args.build_shared_links, args.link_training_planes]):
        args.write_raw_scene_planes = True
        args.build_shared_links = True
        args.link_training_planes = True

    ground_seg = load_ground_seg()
    if args.linefit_config is None:
        print("using default linefit parameters")
    else:
        print(f"using linefit config: {args.linefit_config}")

    groundseg = ground_seg(None if args.linefit_config is None else str(args.linefit_config))
    rng = np.random.default_rng(args.seed)

    total_written = 0
    for scene_id in scene_ids:
        scene_sample_ids = grouped[scene_id]
        print(f"\nscene: {scene_id}")
        print(f"frames in split: {len(scene_sample_ids)}")

        points_rect, frame_stats = sample_ground_points_rect(
            args.data_root,
            scene_sample_ids,
            groundseg,
            rng,
            args.max_frames_per_scene,
            args.max_ground_points_per_frame,
        )
        plane, inliers = refine_plane(points_rect, args.refine_dist_thresh, args.refine_iters)

        used_frames = len(frame_stats)
        total_ground = int(points_rect.shape[0])
        inlier_ratio = float(inliers.mean()) if inliers.size > 0 else 0.0
        print(f"frames used for fit: {used_frames}")
        print(f"ground points used for fit: {total_ground}")
        print(f"plane: {plane[0]:.6e} {plane[1]:.6e} {plane[2]:.6e} {plane[3]:.6e}")
        print(f"inlier ratio after refinement: {inlier_ratio:.4f}")

        written = 0
        if args.write_raw_scene_planes:
            written = write_scene_planes_raw(args.data_root, scene_sample_ids, plane, args.dry_run)
            action = "would write" if args.dry_run else "wrote"
            print(f"{action} {written} raw-scene plane files")
        elif not args.build_shared_links:
            written = write_scene_planes_flat(args.data_root, scene_sample_ids, plane, args.dry_run)
            action = "would write" if args.dry_run else "wrote"
            print(f"{action} {written} flat KITTI plane files")

        if args.build_shared_links:
            linked = create_shared_plane_links(args.data_root, scene_sample_ids, args.dry_run)
            action = "would create" if args.dry_run else "created"
            print(f"{action} {linked} shared flat plane symlinks")

        total_written += len(scene_sample_ids)

    if args.link_training_planes:
        training_planes = link_training_planes_dir(args.data_root, args.dry_run)
        action = "would link" if args.dry_run else "linked"
        print(f"{action} training plane dir: {training_planes} -> ../../shared/planes")

    summary_action = "would write" if args.dry_run else "wrote"
    print(f"\nDone: {summary_action} {total_written} plane files across {len(scene_ids)} scene(s)")


if __name__ == "__main__":
    main()
