#!/usr/bin/env python3
"""Visualize BEV occupancy guidance internals on KITTI val samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu


DEFAULT_CFG = Path("tools/cfgs/kitti_models/second_with_lion_mamba_64dim_bev_occupancy_guidance.yaml")
DEFAULT_CKPT = Path(
    "output/LION_output/cfgs/kitti_models/second_with_lion_mamba_64dim_bev_occupancy_guidance/"
    "bev_occupancy_guidance_fromscratch_bs4_e40_seed666/ckpt/checkpoint_epoch_40.pth"
)
DEFAULT_OUTDIR = Path(
    ".planning/2026-05-23-bev-occupancy-guidance-redesign/artifacts/feature_vis_epoch40"
)
DEFAULT_SAMPLE_IDS = ["006682", "001640", "003982", "006475"]


class _SilentLogger:
    def info(self, *args, **kwargs):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-path", type=Path, default=Path("/root/autodl-tmp/kitti-offical"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sample-ids", nargs="*", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--max-preds", type=int, default=20)
    return parser.parse_args()


def box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """boxes: [N, 7] in lidar coords x,y,z,dx,dy,dz,heading."""
    corners_all = []
    for box in boxes:
        x, y, _, dx, dy, _, heading = box[:7]
        local = np.array(
            [
                [dx / 2, dy / 2],
                [dx / 2, -dy / 2],
                [-dx / 2, -dy / 2],
                [-dx / 2, dy / 2],
            ],
            dtype=np.float32,
        )
        c = np.cos(heading)
        s = np.sin(heading)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners = local @ rot.T
        corners[:, 0] += x
        corners[:, 1] += y
        corners_all.append(corners)
    return np.stack(corners_all, axis=0) if corners_all else np.zeros((0, 4, 2), dtype=np.float32)


def draw_boxes(ax, boxes: np.ndarray, color: str, linewidth: float = 1.0, alpha: float = 1.0, linestyle: str = "-"):
    corners = box_corners_bev(boxes)
    for poly in corners:
        ax.add_patch(
            patches.Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        )


def make_dataset(cfg_file: Path, data_path: Path) -> KittiDataset:
    cfg_from_yaml_file(str(cfg_file), cfg)
    if "DATA_AUGMENTOR" in cfg.DATA_CONFIG:
        cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
            aug.NAME for aug in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
        ]
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=data_path,
        logger=None,
    )
    return dataset


def make_model(dataset: KittiDataset, ckpt: Path) -> torch.nn.Module:
    assert torch.cuda.is_available(), "CUDA is required"
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt), logger=_SilentLogger(), to_cpu=False)
    model.cuda()
    model.eval()
    return model


def select_index(dataset: KittiDataset, lidar_idx: str) -> int:
    for idx, info in enumerate(dataset.kitti_infos):
        if info["point_cloud"]["lidar_idx"] == lidar_idx:
            return idx
    raise KeyError(f"lidar_idx {lidar_idx} not found")


def collate_one(dataset: KittiDataset, index: int):
    sample = dataset[index]
    annos = dataset.kitti_infos[index].get("annos", {})
    gt_names = np.array([n for n in annos.get("name", []) if n in {"Car", "Pedestrian", "Cyclist"}])
    cpu_batch = dataset.collate_batch([sample])
    return sample, gt_names, cpu_batch


def clone_cpu_batch(cpu_batch: Dict) -> Dict:
    out = {}
    for k, v in cpu_batch.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def run_once(model, cpu_batch: Dict, enable_fusion: bool):
    batch = clone_cpu_batch(cpu_batch)
    load_data_to_gpu(batch)
    old_fusion = model.map_to_bev_module.enable_fusion
    model.map_to_bev_module.enable_fusion = enable_fusion
    try:
        with torch.no_grad():
            batch["global_step"] = int(model.global_step.item())
            for module in model.module_list:
                batch = module(batch)
            pred_dicts, _ = model.post_processing(batch)
            dense_ret = model.dense_head.forward_ret_dict
            map_ret = model.map_to_bev_module.forward_ret_dict
            outputs = {
                "pred_dict": pred_dicts[0],
                "cls_preds": dense_ret["cls_preds"][0].detach().cpu().numpy(),
                "occupancy_prob": batch["bev_occupancy_prob"][0, 0].detach().cpu().numpy(),
                "gate_map": batch["bev_occupancy_gate_map"][0, 0].detach().cpu().numpy(),
                "feature_delta": batch["bev_occupancy_feature_delta"][0].detach().cpu().numpy(),
                "target_map": map_ret["target_map"][0, 0].detach().cpu().numpy(),
                "positive_weight_map": map_ret["positive_weight_map"][0, 0].detach().cpu().numpy(),
                "valid_mask": map_ret["valid_mask"][0, 0].detach().cpu().numpy(),
            }
    finally:
        model.map_to_bev_module.enable_fusion = old_fusion
    return outputs


def objectness_heatmap(cls_preds: np.ndarray) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-cls_preds))
    return prob.max(axis=-1)


def pred_boxes_for_plot(pred_dict: Dict, score_thresh: float, max_preds: int) -> Tuple[np.ndarray, np.ndarray]:
    boxes = pred_dict["pred_boxes"].detach().cpu().numpy()
    scores = pred_dict["pred_scores"].detach().cpu().numpy()
    labels = pred_dict["pred_labels"].detach().cpu().numpy()
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    order = np.argsort(-scores)[:max_preds]
    return boxes[order], labels[order]


def fg_bg_stats(pred_prob: np.ndarray, target_map: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float]:
    fg = (target_map > 1e-4) & (valid_mask > 0.5)
    bg = (target_map <= 1e-4) & (valid_mask > 0.5)
    fg_mean = float(pred_prob[fg].mean()) if fg.any() else 0.0
    bg_mean = float(pred_prob[bg].mean()) if bg.any() else 0.0
    return fg_mean, bg_mean


def delta_stats(delta_norm: np.ndarray, target_map: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float]:
    fg = (target_map > 1e-4) & (valid_mask > 0.5)
    bg = (target_map <= 1e-4) & (valid_mask > 0.5)
    fg_mean = float(delta_norm[fg].mean()) if fg.any() else 0.0
    bg_mean = float(delta_norm[bg].mean()) if bg.any() else 0.0
    return fg_mean, bg_mean


def plot_sample(
    sample_id: str,
    sample: Dict,
    gt_names: np.ndarray,
    outputs_on: Dict,
    outputs_off: Dict,
    output_path: Path,
    score_thresh: float,
    max_preds: int,
):
    points = sample["points"]
    gt_boxes = sample["gt_boxes"]
    x_min, y_min, _, x_max, y_max, _ = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    extent = [x_min, x_max, y_min, y_max]

    target_map = outputs_on["target_map"]
    pred_prob = outputs_on["occupancy_prob"]
    gate_map = outputs_on["gate_map"]
    delta_norm = np.linalg.norm(outputs_on["feature_delta"], axis=0)
    obj_off = objectness_heatmap(outputs_off["cls_preds"])
    obj_on = objectness_heatmap(outputs_on["cls_preds"])
    valid_mask = outputs_on["valid_mask"]

    fg_prob, bg_prob = fg_bg_stats(pred_prob, target_map, valid_mask)
    fg_delta, bg_delta = delta_stats(delta_norm, target_map, valid_mask)

    pred_boxes_on, pred_labels_on = pred_boxes_for_plot(outputs_on["pred_dict"], score_thresh, max_preds)
    pred_boxes_off, pred_labels_off = pred_boxes_for_plot(outputs_off["pred_dict"], score_thresh, max_preds)

    counts = {
        "Car": int((gt_names == "Car").sum()),
        "Ped": int((gt_names == "Pedestrian").sum()),
        "Cyc": int((gt_names == "Cyclist").sum()),
    }

    gate_valid = gate_map[valid_mask > 0.5]
    gate_mean = float(gate_valid.mean()) if gate_valid.size > 0 else float(gate_map.mean())
    gate_max = float(gate_valid.max()) if gate_valid.size > 0 else float(gate_map.max())

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(points[:, 0], points[:, 1], s=0.2, c="black", alpha=0.35)
    draw_boxes(ax, gt_boxes, color="#00aa66", linewidth=1.2)
    ax.set_title(f"{sample_id} points + GT\nCar {counts['Car']} | Ped {counts['Ped']} | Cyc {counts['Cyc']}")

    ax = axes[0, 1]
    ax.imshow(target_map, origin="lower", extent=extent, cmap="magma")
    draw_boxes(ax, gt_boxes, color="white", linewidth=0.8, alpha=0.8)
    ax.set_title("Soft occupancy target")

    ax = axes[0, 2]
    ax.imshow(pred_prob, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
    draw_boxes(ax, gt_boxes, color="white", linewidth=0.8, alpha=0.8)
    ax.set_title(f"Pred occupancy prob\nfg {fg_prob:.3f} vs bg {bg_prob:.3f}")

    ax = axes[0, 3]
    ax.imshow(gate_map, origin="lower", extent=extent, cmap="cividis", vmin=0.0, vmax=1.0)
    draw_boxes(ax, gt_boxes, color="white", linewidth=0.8, alpha=0.8)
    ax.set_title(f"Guidance gate map\nmean {gate_mean:.3f} | max {gate_max:.3f}")

    ax = axes[1, 0]
    ax.imshow(delta_norm, origin="lower", extent=extent, cmap="inferno")
    draw_boxes(ax, gt_boxes, color="cyan", linewidth=0.8, alpha=0.9)
    ax.set_title(f"Feature delta norm\nfg {fg_delta:.4f} vs bg {bg_delta:.4f}")

    ax = axes[1, 1]
    ax.imshow(obj_off, origin="lower", extent=extent, cmap="plasma", vmin=0.0, vmax=1.0)
    draw_boxes(ax, gt_boxes, color="white", linewidth=0.8, alpha=0.8)
    draw_boxes(ax, pred_boxes_off, color="#ff5555", linewidth=1.0, alpha=0.9)
    ax.set_title(f"Dense cls heatmap (no fusion)\npreds >= {score_thresh}: {len(pred_boxes_off)}")

    ax = axes[1, 2]
    ax.imshow(obj_on, origin="lower", extent=extent, cmap="plasma", vmin=0.0, vmax=1.0)
    draw_boxes(ax, gt_boxes, color="white", linewidth=0.8, alpha=0.8)
    draw_boxes(ax, pred_boxes_on, color="#00e0ff", linewidth=1.0, alpha=0.9)
    ax.set_title(f"Dense cls heatmap (with fusion)\npreds >= {score_thresh}: {len(pred_boxes_on)}")

    axes[1, 3].axis("off")

    for ax in axes.flat:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

    fig.suptitle(
        "BEV occupancy guidance evidence: target -> predicted prior -> feature modulation -> detector response",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args.cfg_file, args.data_path)
    model = make_model(dataset, args.ckpt)

    index_lookup = {info["point_cloud"]["lidar_idx"]: idx for idx, info in enumerate(dataset.kitti_infos)}
    generated = []
    for sample_id in args.sample_ids:
        if sample_id not in index_lookup:
            continue
        sample, gt_names, cpu_batch = collate_one(dataset, index_lookup[sample_id])
        outputs_off = run_once(model, cpu_batch, enable_fusion=False)
        outputs_on = run_once(model, cpu_batch, enable_fusion=True)
        out_path = args.output_dir / f"{sample_id}_bev_occupancy_evidence.png"
        plot_sample(sample_id, sample, gt_names, outputs_on, outputs_off, out_path, args.score_thresh, args.max_preds)
        generated.append(out_path)

    print("generated_files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
