#!/usr/bin/env python3
"""Synthetic smoke tests for the KITTI BEV occupancy guidance pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=Path("tools/cfgs/kitti_models/second_with_lion_mamba_64dim_bev_occupancy_guidance.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument(
        "--mode",
        choices=["random", "boundary", "crowded", "empty_gt", "stress"],
        default="random",
    )
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


class DummyBEVOccupancyKittiDataset(KittiDataset):
    def include_kitti_data(self, mode):
        self.kitti_infos = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        num_points = self.num_points
        if self.mode_name == "stress":
            num_points = max(num_points, 16384)

        points = np.zeros((num_points, 4), dtype=np.float32)
        points[:, 0] = np.random.uniform(0.1, 70.0, num_points)
        points[:, 1] = np.random.uniform(-39.0, 39.0, num_points)
        points[:, 2] = np.random.uniform(-2.9, 0.9, num_points)
        points[:, 3] = np.random.uniform(0.0, 1.0, num_points)

        if self.mode_name == "boundary":
            gt_boxes = np.array([
                [0.8, -39.0, -1.0, 4.0, 1.6, 1.5, 0.0],
                [69.4, 39.0, -1.0, 0.8, 0.6, 1.7, 1.57],
                [35.2, 0.0, -1.0, 1.8, 0.6, 1.7, 0.78],
            ], dtype=np.float32)
            gt_names = np.array(["Car", "Pedestrian", "Cyclist"])
        elif self.mode_name == "crowded":
            gt_boxes = np.array([
                [20.0, 5.0, -1.0, 4.0, 1.6, 1.5, 0.0],
                [20.8, 5.5, -1.0, 4.0, 1.6, 1.5, 0.2],
                [10.0, -5.0, -1.0, 0.8, 0.6, 1.7, 1.57],
                [10.6, -5.4, -1.0, 0.8, 0.6, 1.7, 1.1],
                [30.0, 0.0, -1.0, 1.8, 0.6, 1.7, 0.0],
                [30.8, 0.4, -1.0, 1.8, 0.6, 1.7, 0.5],
            ], dtype=np.float32)
            gt_names = np.array(["Car", "Car", "Pedestrian", "Pedestrian", "Cyclist", "Cyclist"])
        elif self.mode_name == "empty_gt":
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names = np.zeros((0,), dtype="<U1")
        else:
            gt_boxes = np.array([
                [20.0, 5.0, -1.0, 4.0, 1.6, 1.5, 0.0],
                [10.0, -5.0, -1.0, 0.8, 0.6, 1.7, 1.57],
                [30.0, 0.0, -1.0, 1.8, 0.6, 1.7, 0.0],
            ], dtype=np.float32)
            gt_names = np.array(["Car", "Pedestrian", "Cyclist"])

        return self.prepare_data(
            {
                "frame_id": str(index),
                "points": points,
                "gt_boxes": gt_boxes,
                "gt_names": gt_names,
            }
        )


def collect_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    prefixes = [
        "map_to_bev_module.context_encoder",
        "map_to_bev_module.occupancy_head",
        "map_to_bev_module.inject_head",
        "map_to_bev_module.gate_head",
        "map_to_bev_module.fusion_head",
    ]
    output = {}
    for prefix in prefixes:
        sq_norm = 0.0
        matches = 0
        for name, param in model.named_parameters():
            if prefix in name and param.grad is not None:
                sq_norm += float(param.grad.detach().float().norm().item() ** 2)
                matches += 1
        output[prefix] = sq_norm ** 0.5 if matches > 0 else 0.0
    return output


def main() -> None:
    args = parse_args()
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    if "DATA_AUGMENTOR" in cfg.DATA_CONFIG:
        cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
            aug.NAME for aug in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
        ]

    dataset = DummyBEVOccupancyKittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path("/root/autodl-tmp/kitti-offical"),
        logger=None,
    )
    dataset.num_samples = max(args.batch_size, 2)
    dataset.num_points = args.num_points
    dataset.mode_name = args.mode

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_batch,
        num_workers=0,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    assert torch.cuda.is_available(), "CUDA is required for this smoke test"
    model = model.cuda()
    model.train()

    batch_dict = next(iter(dataloader))
    load_data_to_gpu(batch_dict)
    ret_dict, tb_dict, _ = model(batch_dict)
    loss = ret_dict["loss"]

    print(f"mode={args.mode}")
    print(f"loss={float(loss.item()):.6f}")
    print(f"points={int(batch_dict['points'].shape[0])}")
    print(f"occupancy_target_sum={float(batch_dict['bev_occupancy_target_map'].sum().item()):.6f}")
    print(f"context_valid_ratio={float(batch_dict['bev_occupancy_context_valid_mask'].float().mean().item()):.6f}")
    print(f"tb_keys={len(tb_dict)}")
    for key in [
        'bev_occupancy/loss_aux',
        'bev_occupancy/fg_prob_mean',
        'bev_occupancy/bg_prob_mean',
        'bev_occupancy/gate_mean',
        'bev_occupancy/delta_rel_l2',
    ]:
        print(f"{key}={tb_dict.get(key, None)}")

    if args.backward:
        loss.backward()
        for key, value in collect_grad_norms(model).items():
            print(f"{key}={value:.9f}")


if __name__ == "__main__":
    main()
