#!/usr/bin/env python3
"""Smoke test the OpenPCDet KITTI `USE_ROAD_PLANE` training path."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import torch
from torch.nn.utils import clip_grad_norm_

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.augmentor.database_sampler import DataBaseSampler
from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional checkpoint for resume-smoke steps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 so road-plane call counting stays in-process")
    parser.add_argument("--num-data-batches", type=int, default=8, help="Pure dataloader stress-test batches")
    parser.add_argument("--num-train-steps", type=int, default=3, help="Resume-smoke optimization steps")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path")
    return parser.parse_args()


def ensure_road_plane_enabled() -> None:
    aug_list = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
    found = False
    for aug in aug_list:
        if aug.NAME == "gt_sampling":
            aug.USE_ROAD_PLANE = True
            found = True
    if not found:
        raise RuntimeError("gt_sampling augmentor not found in config")


def make_logger() -> logging.Logger:
    logger = logging.getLogger("use_road_plane_smoke")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


def count_plane_files(root_path: Path) -> int:
    plane_dir = root_path / "training" / "planes"
    if not plane_dir.exists():
        raise FileNotFoundError(f"training/planes not found: {plane_dir}")
    return sum(1 for _ in plane_dir.glob("*.txt"))


def main() -> None:
    args = parse_args()
    os.chdir(ROOT_DIR / "tools")
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    ensure_road_plane_enabled()

    logger = make_logger()
    data_root = Path(cfg.DATA_CONFIG.DATA_PATH)
    plane_file_count = count_plane_files(data_root)
    logger.info("data_root=%s", data_root)
    logger.info("plane_files=%d", plane_file_count)

    call_state = {"count": 0}
    original = DataBaseSampler.put_boxes_on_road_planes

    def wrapped(gt_boxes, road_planes, calib):
        call_state["count"] += 1
        return original(gt_boxes, road_planes, calib)

    DataBaseSampler.put_boxes_on_road_planes = staticmethod(wrapped)
    try:
        train_set, train_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=True,
        )

        data_batches = 0
        first_batch_shapes = None
        loader_iter = iter(train_loader)
        while data_batches < args.num_data_batches:
            batch = next(loader_iter)
            data_batches += 1
            if first_batch_shapes is None:
                first_batch_shapes = {
                    "voxels": tuple(batch["voxels"].shape),
                    "voxel_coords": tuple(batch["voxel_coords"].shape),
                    "gt_boxes": tuple(batch["gt_boxes"].shape),
                }

        data_plane_calls = call_state["count"]
        if data_plane_calls <= 0:
            raise RuntimeError("road-plane placement path was never called during dataloader smoke")

        train_plane_calls = 0
        train_losses: list[float] = []
        grad_norms: list[float] = []
        if args.ckpt is not None:
            model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set).cuda()
            optimizer = build_optimizer(model, cfg.OPTIMIZATION)
            model.load_params_with_optimizer(str(args.ckpt), to_cpu=False, optimizer=optimizer, logger=logger)
            model_func = model_fn_decorator()
            model.train()

            for _ in range(args.num_train_steps):
                batch = next(loader_iter)
                optimizer.zero_grad()
                loss, tb_dict, disp_dict = model_func(model, batch)
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"non-finite smoke loss: {loss}")
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
                if not torch.isfinite(grad_norm).all():
                    raise RuntimeError(f"non-finite grad norm: {grad_norm}")
                optimizer.step()
                torch.cuda.synchronize()
                train_losses.append(float(loss.detach().cpu()))
                grad_norms.append(float(grad_norm.detach().cpu()))

            train_plane_calls = call_state["count"] - data_plane_calls

        summary = {
            "data_root": str(data_root),
            "plane_files": plane_file_count,
            "data_batches": data_batches,
            "data_plane_calls": data_plane_calls,
            "train_steps": args.num_train_steps if args.ckpt is not None else 0,
            "train_plane_calls": train_plane_calls,
            "first_batch_shapes": first_batch_shapes,
            "train_losses": train_losses,
            "grad_norms": grad_norms,
            "checkpoint": None if args.ckpt is None else str(args.ckpt),
        }

        print(json.dumps(summary, indent=2))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n")

    finally:
        DataBaseSampler.put_boxes_on_road_planes = original


if __name__ == "__main__":
    main()
