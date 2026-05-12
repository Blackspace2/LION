#!/usr/bin/env python3
"""Create KITTI info pkl files for converted V2X-SPD KITTI-like roots."""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
from pathlib import Path

import _init_path  # noqa: F401
import yaml
from easydict import EasyDict

from pcdet.datasets.kitti.kitti_dataset import KittiDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-cfg",
        type=Path,
        default=Path("tools/cfgs/dataset_configs/kitti_dataset.yaml"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--count-inside-pts",
        action="store_true",
        help="Also compute annos['num_points_in_gt'] in infos. This uses KITTI image FOV filtering.",
    )
    parser.add_argument(
        "--fov-points-only",
        action="store_true",
        help="Enable image-FOV point filtering in the dataset config. Keep disabled for V2X-SPD LiDAR-only training.",
    )
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument("--db-only", action="store_true", help="Only create gt_database from existing train info pkl.")
    return parser.parse_args()


def dump_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    print(f"saved {path} ({len(obj)} records)")


def get_infos_quiet(dataset: KittiDataset, *, num_workers: int, has_label: bool, count_inside_pts: bool):
    """Run OpenPCDet info collection while hiding its per-frame stdout spam."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return dataset.get_infos(
            num_workers=num_workers,
            has_label=has_label,
            count_inside_pts=count_inside_pts,
        )


def create_groundtruth_database_quiet(dataset: KittiDataset, info_path: Path, split: str) -> None:
    """Create GT database while hiding OpenPCDet's per-sample stdout spam."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        dataset.create_groundtruth_database(info_path, split=split)


def main() -> None:
    args = parse_args()
    dataset_cfg = EasyDict(yaml.safe_load(args.dataset_cfg.read_text()))
    dataset_cfg.DATA_PATH = str(args.data_root)
    dataset_cfg.FOV_POINTS_ONLY = bool(args.fov_points_only)

    class_names = ["Car", "Pedestrian", "Cyclist"]
    dataset = KittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=args.data_root,
        training=False,
        logger=None,
    )

    train_info_path = args.data_root / "kitti_infos_train.pkl"

    if not args.db_only:
        print("---------------Start to generate V2X-SPD KITTI infos---------------")
        print(f"data_root: {args.data_root}")
        print(f"fov_points_only: {dataset_cfg.FOV_POINTS_ONLY}")
        print(f"count_inside_pts: {args.count_inside_pts}")
        dataset.set_split("train")
        print(f"collecting train infos with {args.workers} workers...")
        train_infos = get_infos_quiet(
            dataset, num_workers=args.workers, has_label=True, count_inside_pts=args.count_inside_pts
        )
        dump_pickle(train_info_path, train_infos)

        dataset.set_split("val")
        print(f"collecting val infos with {args.workers} workers...")
        val_infos = get_infos_quiet(
            dataset, num_workers=args.workers, has_label=True, count_inside_pts=args.count_inside_pts
        )
        dump_pickle(args.data_root / "kitti_infos_val.pkl", val_infos)

        trainval_infos = train_infos + val_infos
        dump_pickle(args.data_root / "kitti_infos_trainval.pkl", trainval_infos)

    if not args.skip_db:
        print("---------------Start create groundtruth database---------------")
        dataset.set_split("train")
        create_groundtruth_database_quiet(dataset, train_info_path, split="train")
        print(f"saved {args.data_root / 'kitti_dbinfos_train.pkl'}")
        print(f"saved {args.data_root / 'gt_database'}")

    print("---------------V2X-SPD KITTI info preparation done---------------")


if __name__ == "__main__":
    main()
