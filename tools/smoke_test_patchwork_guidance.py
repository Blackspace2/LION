#!/usr/bin/env python3
"""Synthetic smoke tests for the KITTI Patchwork guidance pipeline."""

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
        default=Path("tools/cfgs/kitti_models/second_with_lion_mamba_64dim_patchwork_guidance.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument(
        "--mode",
        choices=["random", "near-origin", "stress"],
        default="random",
        help="random: standard synthetic sample; near-origin: force empty patch_infos; stress: larger point load",
    )
    parser.add_argument("--backward", action="store_true", help="Run one backward pass and report grad norms")
    return parser.parse_args()


class DummyPatchworkKittiDataset(KittiDataset):
    def include_kitti_data(self, mode):
        self.kitti_infos = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        num_points = self.num_points
        points = np.zeros((num_points, 4), dtype=np.float32)
        if self.mode_name == "near-origin":
            points[:, 0] = np.random.uniform(0.1, 2.0, num_points)
            points[:, 1] = np.random.uniform(-1.0, 1.0, num_points)
            points[:, 2] = np.random.uniform(-2.0, 0.0, num_points)
        else:
            points[:, 0] = np.random.uniform(0.1, 70.0, num_points)
            points[:, 1] = np.random.uniform(-39.0, 39.0, num_points)
            points[:, 2] = np.random.uniform(-2.9, 0.9, num_points)
        points[:, 3] = np.random.uniform(0.0, 1.0, num_points)

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
        "backbone_3d.patch_token_encoder",
        "backbone_3d.patch_context_encoder",
        "backbone_3d.patch_guidance_pre",
        "backbone_3d.patch_guidance_logits",
        "backbone_3d.patch_guidance_gates",
        "backbone_3d.patch_guidance_residuals",
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

    num_points = args.num_points
    if args.mode == "stress":
        num_points = max(num_points, 16384)

    dataset = DummyPatchworkKittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path("data/kitti"),
        logger=None,
    )
    dataset.num_samples = max(args.batch_size, 2)
    dataset.num_points = num_points
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
    print(f"patch_infos={int(batch_dict['patch_infos'].shape[0])}")
    print(f"patch_aux={tb_dict.get('patch_guidance/loss_aux', None)}")
    print(f"tb_keys={len(tb_dict)}")

    if args.backward:
        loss.backward()
        for key, value in collect_grad_norms(model).items():
            print(f"{key}={value:.9f}")


if __name__ == "__main__":
    main()
