#!/usr/bin/env python3
"""Convert the current V2X-SPD layout into KITTI-like roots for LION.

The source dataset is sequence based:
    V2X-SPD/{training,validation}/{scene}/velodyne/*.pcd
    V2X-SPD/{training,validation}/{scene}/image_02/*.jpg
    V2X-SPD/{training,validation}/{scene}/calib/*.txt
    V2X-SPD/{training,validation}/{scene}/label_02_split/*.txt

The output is:
    V2X-SPD-KITTI/velodyne_bin/*.bin
    V2X-SPD-KITTI/shared/{image_2,calib,ImageSets}
    V2X-SPD-KITTI/{merge3,strict3}/training/{velodyne,image_2,calib,label_2}
    V2X-SPD-KITTI/{merge3,strict3}/ImageSets
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


MERGE3 = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Bus": "Car",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    "Motorcyclist": "Cyclist",
}

STRICT3 = {
    "Car": "Car",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
}


def read_split_ids(root: Path, split: str) -> list[str]:
    path = root / "ImageSets" / f"{split}.txt"
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def scene_part(root: Path, scene_id: str) -> str:
    if (root / "training" / scene_id).is_dir():
        return "training"
    if (root / "validation" / scene_id).is_dir():
        return "validation"
    raise FileNotFoundError(f"scene {scene_id} not found in training/validation")


def ensure_symlink(link: Path, target: Path, force: bool = False) -> None:
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        if not force:
            raise FileExistsError(f"symlink exists with different target: {link} -> {link.readlink()}")
        link.unlink()
    elif link.exists():
        if not force:
            raise FileExistsError(f"path exists and is not a symlink: {link}")
        if link.is_dir():
            raise IsADirectoryError(f"refusing to replace existing directory: {link}")
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def convert_pcd_to_bin_open3d(pcd_path: Path, bin_path: Path) -> int:
    import open3d as o3d

    pc = o3d.t.io.read_point_cloud(str(pcd_path))
    xyz = pc.point.positions.numpy().astype(np.float32, copy=False)
    if "intensity" in pc.point:
        intensity = pc.point.intensity.numpy().astype(np.float32, copy=False)
        if intensity.ndim == 1:
            intensity = intensity.reshape(-1, 1)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)

    points = np.concatenate([xyz, intensity[:, :1]], axis=1).astype(np.float32, copy=False)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    points.tofile(bin_path)
    return int(points.shape[0])


def convert_pcd_task(task: tuple[str, str]) -> tuple[str, int]:
    pcd_path, bin_path = task
    num_points = convert_pcd_to_bin_open3d(Path(pcd_path), Path(bin_path))
    return bin_path, num_points


def convert_label(src: Path, dst: Path, mapping: dict[str, str]) -> tuple[int, int]:
    """Convert frame-level SPD tracking-like labels to KITTI detection labels."""
    kept = 0
    dropped = 0
    out_lines: list[str] = []
    for line in src.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        tok = line.split()
        if len(tok) < 17:
            dropped += 1
            continue

        cls = tok[1]
        mapped = mapping.get(cls)
        if mapped is None:
            dropped += 1
            continue

        # SPD split label format:
        # frame_id class track_id truncated occluded alpha bbox4 h w l cam_xyz ry lidar_xyz lidar_yaw ...
        trunc, occ, alpha = tok[3], tok[4], tok[5]
        bbox = tok[6:10]
        hwl = tok[10:13]
        loc = tok[13:16]
        ry = tok[16]
        out_lines.append(" ".join([mapped, trunc, occ, alpha, *bbox, *hwl, *loc, ry]))
        kept += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return kept, dropped


def iter_frames(src_root: Path, scene_ids: Iterable[str]) -> Iterable[tuple[str, str, Path, str]]:
    for scene_id in scene_ids:
        part = scene_part(src_root, scene_id)
        scene_root = src_root / part / scene_id
        for pcd in sorted((scene_root / "velodyne").glob("*.pcd")):
            frame = pcd.stem
            flat = f"{scene_id}_{frame}"
            yield scene_id, frame, scene_root, flat


def write_imagesets(shared_imagesets: Path, split_to_ids: dict[str, list[str]]) -> None:
    shared_imagesets.mkdir(parents=True, exist_ok=True)
    for split, ids in split_to_ids.items():
        (shared_imagesets / f"{split}.txt").write_text("".join(f"{x}\n" for x in ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, default=Path("/root/autodl-tmp/V2X-SPD"))
    parser.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/V2X-SPD-KITTI"))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-links", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0, help="debug limit per split; 0 means all")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    args = parser.parse_args()

    src_root: Path = args.src_root
    out_root: Path = args.out_root
    velodyne_bin = out_root / "velodyne_bin"
    shared = out_root / "shared"
    shared_image = shared / "image_2"
    shared_calib = shared / "calib"
    shared_imagesets = shared / "ImageSets"

    train_scenes = read_split_ids(src_root, "train")
    val_scenes = read_split_ids(src_root, "val")

    split_to_frame_ids: dict[str, list[str]] = {"train": [], "val": []}
    counts = {
        "frames": 0,
        "pcd_converted": 0,
        "pcd_skipped": 0,
        "merge3_kept": 0,
        "merge3_dropped": 0,
        "strict3_kept": 0,
        "strict3_dropped": 0,
    }
    pcd_tasks: list[tuple[str, str]] = []

    for split, scenes in [("train", train_scenes), ("val", val_scenes)]:
        frame_iter = iter_frames(src_root, scenes)
        for idx, (scene_id, frame, scene_root, flat) in enumerate(frame_iter, 1):
            if args.limit and idx > args.limit:
                break
            split_to_frame_ids[split].append(flat)
            counts["frames"] += 1

            src_pcd = scene_root / "velodyne" / f"{frame}.pcd"
            dst_bin = velodyne_bin / f"{flat}.bin"
            if args.skip_existing and dst_bin.exists():
                counts["pcd_skipped"] += 1
            else:
                pcd_tasks.append((str(src_pcd), str(dst_bin)))

            src_img = scene_root / "image_02" / f"{frame}.jpg"
            src_calib = scene_root / "calib" / f"{frame}.txt"
            ensure_symlink(shared_image / f"{flat}.png", src_img, force=args.force_links)
            ensure_symlink(shared_calib / f"{flat}.txt", src_calib, force=args.force_links)

            src_label = scene_root / "label_02_split" / f"{frame}.txt"
            m_kept, m_drop = convert_label(src_label, out_root / "merge3" / "training" / "label_2" / f"{flat}.txt", MERGE3)
            s_kept, s_drop = convert_label(src_label, out_root / "strict3" / "training" / "label_2" / f"{flat}.txt", STRICT3)
            counts["merge3_kept"] += m_kept
            counts["merge3_dropped"] += m_drop
            counts["strict3_kept"] += s_kept
            counts["strict3_dropped"] += s_drop

            if counts["frames"] % 500 == 0:
                print(f"prepared frames: {counts['frames']}")

    if pcd_tasks:
        print(f"converting PCD files with {args.workers} workers: {len(pcd_tasks)}")
        if args.workers <= 1:
            for i, task in enumerate(pcd_tasks, 1):
                convert_pcd_task(task)
                counts["pcd_converted"] += 1
                if i % 200 == 0:
                    print(f"converted pcd: {i}/{len(pcd_tasks)}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(convert_pcd_task, task) for task in pcd_tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    counts["pcd_converted"] += 1
                    if i % 200 == 0:
                        print(f"converted pcd: {i}/{len(pcd_tasks)}")

    split_to_frame_ids["trainval"] = split_to_frame_ids["train"] + split_to_frame_ids["val"]
    write_imagesets(shared_imagesets, split_to_frame_ids)

    for variant in ["merge3", "strict3"]:
        root = out_root / variant
        training = root / "training"
        training.mkdir(parents=True, exist_ok=True)
        ensure_symlink(training / "velodyne", Path("../../velodyne_bin"), force=args.force_links)
        ensure_symlink(training / "image_2", Path("../../shared/image_2"), force=args.force_links)
        ensure_symlink(training / "calib", Path("../../shared/calib"), force=args.force_links)
        ensure_symlink(root / "ImageSets", Path("../shared/ImageSets"), force=args.force_links)

    print("done")
    for key, val in counts.items():
        print(f"{key}: {val}")
    for split in ["train", "val", "trainval"]:
        print(f"{split}_frames: {len(split_to_frame_ids[split])}")


if __name__ == "__main__":
    main()
