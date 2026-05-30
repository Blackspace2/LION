#!/usr/bin/env python3
"""Reproduce the LION paper Fig. 6 foreground/background feature-response plot.

This script extracts sparse tensors from a trained LION model and renders BEV
heat maps with red GT boxes. By default it visualizes response magnitude as
mean(abs(feature)); paper_mean is still available as a diagnostic option.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import _init_path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


DEFAULT_CFG = (
    "tools/cfgs/kitti_models/second_with_lion_mamba_64dim_sbsd/"
    "second_with_lion_mamba_64dim_sbsd_baseline_fromscratch.yaml"
)
DEFAULT_CKPT = (
    "output/LION_output/cfgs/kitti_models/second_with_lion_mamba_64dim_sbsd/"
    "second_with_lion_mamba_64dim_sbsd_baseline_fromscratch/"
    "baseline_from_scratch_20p24e_onecycle_lr1e3_ema997_ped20_limitgt/"
    "ckpt/checkpoint_epoch_24.pth"
)
DEFAULT_OUTPUT_DIR = (
    ".planning/2026-05-27-igvg-diagnostic-review-improvement/"
    "visualizations/fig6_feature_response"
)
DEFAULT_DATA_PATH = "/root/autodl-tmp/kitti-offical"

SELECTION_STAGES = (
    ("Block 1", "dow1", np.array([1, 1, 1], dtype=np.float32)),
    ("Block 2", "dow2", np.array([1, 1, 2], dtype=np.float32)),
    ("Block 3", "dow3", np.array([1, 1, 4], dtype=np.float32)),
    ("Block 4", "dow4", np.array([1, 1, 8], dtype=np.float32)),
)
GENERATED_STAGES = (
    ("Block 1", "x_conv1"),
    ("Block 2", "x_conv2"),
    ("Block 3", "x_conv3"),
    ("Block 4", "x_conv4"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file", default=DEFAULT_CFG, help="LION model config.")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint to load.")
    parser.add_argument("--data-path", default=None, help="KITTI root. Defaults to local data if present.")
    parser.add_argument("--frame-id", default="003219", help="KITTI frame id to visualize.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for PNG/CSV outputs.")
    parser.add_argument(
        "--source",
        choices=("selection", "generated", "both"),
        default="both",
        help=(
            "selection: tensors before voxel generation/merge, matching the paper response definition; "
            "generated: x_conv tensors after generation/merge."
        ),
    )
    parser.add_argument(
        "--response-mode",
        choices=("paper_mean", "mean_abs", "l2"),
        default="mean_abs",
        help="mean_abs is abs(feature) then channel mean. paper_mean is Eq. (1): raw channel mean.",
    )
    parser.add_argument("--top-ratio", type=float, default=0.2, help="Top response ratio used for separation stats.")
    parser.add_argument("--percentile-min", type=float, default=2.0, help="Shared color lower percentile.")
    parser.add_argument("--percentile-max", type=float, default=98.0, help="Shared color upper percentile.")
    parser.add_argument("--point-size", type=float, default=6.0, help="Sparse voxel point radius in the BEV plot.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def normalize_frame_id(frame_id):
    return str(frame_id).zfill(6)


def resolve_data_path(args):
    if args.data_path:
        return Path(args.data_path)
    default_path = Path(DEFAULT_DATA_PATH)
    if default_path.exists():
        return default_path
    return None


def make_logger(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    return common_utils.create_logger(output_dir / "log_lion_fig6_feature_response.txt", rank=0)


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_cfg_with_tools_relative_base(cfg_file):
    cfg_path = Path(cfg_file)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path

    old_cwd = Path.cwd()
    try:
        os.chdir(repo_root() / "tools")
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)


def build_dataset_and_model(args, logger):
    load_cfg_with_tools_relative_base(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    data_path = resolve_data_path(args)
    dataset, _loader, _sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        root_path=data_path,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return dataset, model, device


def find_dataset_index(dataset, frame_id):
    target = normalize_frame_id(frame_id)
    for idx, info in enumerate(dataset.kitti_infos):
        cur_id = normalize_frame_id(info["point_cloud"]["lidar_idx"])
        if cur_id == target:
            return idx
    raise ValueError(f"Frame {target} not found in dataset with {len(dataset)} samples")


def move_batch_to_device(batch_dict, device):
    for key, val in list(batch_dict.items()):
        if not isinstance(val, np.ndarray):
            continue
        if key in ("frame_id", "metadata", "calib"):
            continue
        if key == "image_shape":
            batch_dict[key] = torch.from_numpy(val).int().to(device)
        else:
            batch_dict[key] = torch.from_numpy(val).float().to(device)


def clone_sparse_tensor(x):
    return {
        "features": x.features.detach().cpu(),
        "indices": x.indices.detach().cpu(),
        "spatial_shape": np.asarray(x.spatial_shape, dtype=np.int64),
        "batch_size": int(x.batch_size),
    }


def register_selection_hooks(model, captured):
    handles = []
    backbone = getattr(model, "backbone_3d", None)
    if backbone is None:
        raise RuntimeError("Model has no backbone_3d module")

    for _label, attr, _stride in SELECTION_STAGES:
        module = getattr(backbone, attr, None)
        if module is None:
            raise RuntimeError(f"backbone_3d has no {attr}; cannot capture Fig.6 selection tensors")

        def _make_hook(name):
            def _hook(_module, inputs, _output):
                captured[name] = clone_sparse_tensor(inputs[0])

            return _hook

        handles.append(module.register_forward_hook(_make_hook(attr)))
    return handles


def extract_sample(dataset, model, device, frame_id):
    index = find_dataset_index(dataset, frame_id)
    data_dict = dataset[index]
    gt_boxes = data_dict.get("gt_boxes", np.zeros((0, 8), dtype=np.float32)).copy()
    batch_dict = dataset.collate_batch([data_dict])

    captured = {}
    handles = register_selection_hooks(model, captured)
    try:
        move_batch_to_device(batch_dict, device)
        with torch.no_grad():
            _pred_dicts, _recall_dicts = model(batch_dict)
    finally:
        for handle in handles:
            handle.remove()

    ret = model.forward_ret_dict
    return data_dict, gt_boxes, captured, ret


def sparse_centers_xyz(indices, stride_xyz, voxel_size, point_cloud_range):
    coords = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
    stride = np.asarray(stride_xyz, dtype=np.float32).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(3)
    pc_range = np.asarray(point_cloud_range, dtype=np.float32).reshape(6)

    centers = np.zeros((coords.shape[0], 3), dtype=np.float32)
    centers[:, 0] = pc_range[0] + (coords[:, 3].astype(np.float32) + 0.5) * voxel_size[0] * stride[0]
    centers[:, 1] = pc_range[1] + (coords[:, 2].astype(np.float32) + 0.5) * voxel_size[1] * stride[1]
    centers[:, 2] = pc_range[2] + (coords[:, 1].astype(np.float32) + 0.5) * voxel_size[2] * stride[2]
    return centers


def feature_response(features, mode):
    feat = features.detach().float().cpu() if torch.is_tensor(features) else torch.as_tensor(features).float()
    if mode == "paper_mean":
        response = feat.mean(dim=1)
    elif mode == "mean_abs":
        response = feat.abs().mean(dim=1)
    elif mode == "l2":
        response = feat.norm(dim=1) / math.sqrt(max(int(feat.shape[1]), 1))
    else:
        raise ValueError(f"Unsupported response mode: {mode}")
    return torch.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0).numpy()


def response_description(mode):
    return {
        "paper_mean": "raw channel mean",
        "mean_abs": "abs-then-mean magnitude",
        "l2": "channel L2 magnitude",
    }[mode]


def box_corners_bev(box):
    x, y, _z, dx, dy, _dz, heading = [float(v) for v in box[:7]]
    c, s = math.cos(heading), math.sin(heading)
    local = np.array(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
        dtype=np.float32,
    )
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def points_in_rotated_box_bev(points_xy, box):
    x, y, _z, dx, dy, _dz, heading = [float(v) for v in box[:7]]
    shifted = points_xy - np.array([x, y], dtype=np.float32)
    c, s = math.cos(-heading), math.sin(-heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = shifted @ rot.T
    return (np.abs(local[:, 0]) <= dx / 2) & (np.abs(local[:, 1]) <= dy / 2)


def foreground_mask_bev(centers_xyz, gt_boxes):
    if len(gt_boxes) == 0 or len(centers_xyz) == 0:
        return np.zeros((len(centers_xyz),), dtype=bool)
    points_xy = centers_xyz[:, :2].astype(np.float32)
    mask = np.zeros((len(points_xy),), dtype=bool)
    for box in gt_boxes:
        mask |= points_in_rotated_box_bev(points_xy, box)
    return mask


def clean_gt_boxes(gt_boxes):
    if gt_boxes is None:
        return np.zeros((0, 7), dtype=np.float32)
    boxes = np.asarray(gt_boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)
    boxes7 = boxes[:, :7]
    valid = np.all(np.isfinite(boxes7), axis=1) & np.all(boxes7[:, 3:6] > 0, axis=1)
    if boxes.shape[1] >= 8:
        valid &= boxes[:, 7] > 0
    return boxes7[valid]


def collect_stage_maps(source, captured, ret, response_mode, voxel_size, point_cloud_range):
    stage_maps = []
    if source == "selection":
        for label, key, stride in SELECTION_STAGES:
            if key not in captured:
                raise RuntimeError(f"Missing captured tensor {key}")
            tensor = captured[key]
            centers = sparse_centers_xyz(tensor["indices"], stride, voxel_size, point_cloud_range)
            response = feature_response(tensor["features"], response_mode)
            stage_maps.append({"label": label, "key": key, "centers": centers, "response": response})
    elif source == "generated":
        features = ret["multi_scale_3d_features"]
        strides = ret["multi_scale_3d_strides"]
        for label, key in GENERATED_STAGES:
            tensor = features[key]
            stride = strides[key].detach().cpu().numpy() if torch.is_tensor(strides[key]) else np.asarray(strides[key])
            centers = sparse_centers_xyz(tensor.indices, stride, voxel_size, point_cloud_range)
            response = feature_response(tensor.features, response_mode)
            stage_maps.append({"label": label, "key": key, "centers": centers, "response": response})
    else:
        raise ValueError(source)
    return stage_maps


def stage_stats(stage_map, gt_boxes, top_ratio):
    response = stage_map["response"]
    centers = stage_map["centers"]
    fg_mask = foreground_mask_bev(centers, gt_boxes)
    bg_mask = ~fg_mask

    fg_mean = float(response[fg_mask].mean()) if bool(fg_mask.any()) else float("nan")
    bg_mean = float(response[bg_mask].mean()) if bool(bg_mask.any()) else float("nan")
    delta = fg_mean - bg_mean if np.isfinite(fg_mean) and np.isfinite(bg_mean) else float("nan")
    k = max(1, min(len(response), int(math.ceil(len(response) * float(top_ratio)))))
    top_idx = np.argsort(response)[-k:] if len(response) else np.array([], dtype=np.int64)
    top_fg_rate = float(fg_mask[top_idx].mean()) if len(top_idx) else 0.0
    all_fg_rate = float(fg_mask.mean()) if len(fg_mask) else 0.0
    enrichment = top_fg_rate / max(all_fg_rate, 1e-6)
    return {
        "stage": stage_map["label"],
        "key": stage_map["key"],
        "num_voxels": int(len(response)),
        "num_fg_voxels": int(fg_mask.sum()),
        "num_bg_voxels": int(bg_mask.sum()),
        "fg_mean": fg_mean,
        "bg_mean": bg_mean,
        "fg_minus_bg": delta,
        "top_ratio": float(top_ratio),
        "top_fg_rate": top_fg_rate,
        "all_fg_rate": all_fg_rate,
        "top_fg_enrichment": enrichment,
    }


def draw_gt_boxes(ax, gt_boxes):
    for box in gt_boxes:
        corners = box_corners_bev(box)
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor="red",
                linewidth=1.2,
                linestyle="-",
                alpha=0.95,
            )
        )


def color_limits(stage_maps, pmin, pmax):
    values = np.concatenate([m["response"] for m in stage_maps if len(m["response"]) > 0])
    if len(values) == 0:
        return 0.0, 1.0
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, pmin))
    vmax = float(np.percentile(values, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
        center = float(np.nanmean(values)) if len(values) else 0.0
        return center - 1.0, center + 1.0
    return vmin, vmax


def axis_limits(stage_maps, gt_boxes, point_cloud_range):
    centers = np.concatenate([m["centers"][:, :2] for m in stage_maps if len(m["centers"]) > 0], axis=0)
    if len(centers) == 0:
        return (point_cloud_range[0], point_cloud_range[3]), (point_cloud_range[1], point_cloud_range[4])

    xs = [centers[:, 0]]
    ys = [centers[:, 1]]
    if len(gt_boxes) > 0:
        gt_corners = np.concatenate([box_corners_bev(box) for box in gt_boxes], axis=0)
        xs.append(gt_corners[:, 0])
        ys.append(gt_corners[:, 1])
    x_min, x_max = float(np.min(np.concatenate(xs))), float(np.max(np.concatenate(xs)))
    y_min, y_max = float(np.min(np.concatenate(ys))), float(np.max(np.concatenate(ys)))
    pad_x = max((x_max - x_min) * 0.05, 2.0)
    pad_y = max((y_max - y_min) * 0.08, 2.0)
    return (
        max(float(point_cloud_range[0]), x_min - pad_x),
        min(float(point_cloud_range[3]), x_max + pad_x),
    ), (
        max(float(point_cloud_range[1]), y_min - pad_y),
        min(float(point_cloud_range[4]), y_max + pad_y),
    )


def render_figure(stage_maps, stats, gt_boxes, frame_id, source, response_mode, args, point_cloud_range):
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.90, top=0.80, bottom=0.15, wspace=0.22)
    vmin, vmax = color_limits(stage_maps, args.percentile_min, args.percentile_max)
    xlim, ylim = axis_limits(stage_maps, gt_boxes, point_cloud_range)

    scatter = None
    for ax, stage_map, stat in zip(axes, stage_maps, stats):
        centers = stage_map["centers"]
        response = stage_map["response"]
        scatter = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c=response,
            s=args.point_size,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.95,
            rasterized=True,
        )
        draw_gt_boxes(ax, gt_boxes)
        ax.set_title(
            f"{stage_map['label']} ({stage_map['key']})\n"
            f"FG={stat['fg_mean']:.3f}, BG={stat['bg_mean']:.3f}, "
            f"Delta={stat['fg_minus_bg']:+.3f}\n"
            f"top{int(args.top_ratio * 100)} FG={stat['top_fg_rate']:.2f}, "
            f"enrich={stat['top_fg_enrichment']:.1f}x",
            fontsize=10,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e0e0e0", linewidth=0.5, alpha=0.55)
        ax.set_xlabel("x forward (m)")
        if ax is axes[0]:
            ax.set_ylabel("y left (m)")
        else:
            ax.set_yticklabels([])

    source_text = {
        "selection": "pre-generation selection tensors",
        "generated": "post-generation x_conv tensors",
    }[source]
    fig.suptitle(
        f"LION Fig.6 reproduction | frame {normalize_frame_id(frame_id)} | "
        f"{source_text} | {response_description(response_mode)}",
        fontsize=14,
        fontweight="bold",
    )
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.028, pad=0.012)
        cbar.set_label(f"feature response ({response_mode})")
    legend = [Line2D([0], [0], color="red", lw=1.6, label="GT boxes")]
    fig.legend(handles=legend, loc="center right", bbox_to_anchor=(0.985, 0.52), frameon=False)
    return fig


def save_stats_csv(path, stats):
    fieldnames = [
        "stage",
        "key",
        "num_voxels",
        "num_fg_voxels",
        "num_bg_voxels",
        "fg_mean",
        "bg_mean",
        "fg_minus_bg",
        "top_ratio",
        "top_fg_rate",
        "all_fg_rate",
        "top_fg_enrichment",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)


def run_source(source, captured, ret, gt_boxes, frame_id, args, output_dir, voxel_size, point_cloud_range):
    stage_maps = collect_stage_maps(
        source=source,
        captured=captured,
        ret=ret,
        response_mode=args.response_mode,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    )
    stats = [stage_stats(stage_map, gt_boxes, args.top_ratio) for stage_map in stage_maps]
    fig = render_figure(
        stage_maps=stage_maps,
        stats=stats,
        gt_boxes=gt_boxes,
        frame_id=frame_id,
        source=source,
        response_mode=args.response_mode,
        args=args,
        point_cloud_range=point_cloud_range,
    )
    stem = f"lion_fig6_feature_response_{source}_{normalize_frame_id(frame_id)}_{args.response_mode}"
    png_path = output_dir / f"{stem}.png"
    csv_path = output_dir / f"{stem}.csv"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    save_stats_csv(csv_path, stats)
    return png_path, csv_path, stats


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = make_logger(output_dir)
    dataset, model, device = build_dataset_and_model(args, logger)
    data_dict, raw_gt_boxes, captured, ret = extract_sample(dataset, model, device, args.frame_id)
    gt_boxes = clean_gt_boxes(raw_gt_boxes)

    voxel_size = np.asarray(dataset.voxel_size, dtype=np.float32)
    point_cloud_range = np.asarray(dataset.point_cloud_range, dtype=np.float32)

    sources = ("selection", "generated") if args.source == "both" else (args.source,)
    outputs = []
    for source in sources:
        outputs.append(
            run_source(
                source=source,
                captured=captured,
                ret=ret,
                gt_boxes=gt_boxes,
                frame_id=data_dict.get("frame_id", args.frame_id),
                args=args,
                output_dir=output_dir,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
            )
        )

    print(f"frame_id={normalize_frame_id(data_dict.get('frame_id', args.frame_id))}")
    print(f"gt_boxes={len(gt_boxes)}")
    for png_path, csv_path, stats in outputs:
        print(f"saved_png={png_path}")
        print(f"saved_csv={csv_path}")
        for stat in stats:
            print(
                f"  {stat['stage']} {stat['key']}: "
                f"N={stat['num_voxels']} FG={stat['fg_mean']:.4f} BG={stat['bg_mean']:.4f} "
                f"Delta={stat['fg_minus_bg']:+.4f} topFG={stat['top_fg_rate']:.3f} "
                f"enrich={stat['top_fg_enrichment']:.2f}x"
            )


if __name__ == "__main__":
    main()
