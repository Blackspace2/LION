#!/usr/bin/env python3
import argparse
import pickle
import random
from pathlib import Path

import numpy as np


CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a KITTI train subset split txt and matching info pkl for smoke / small-sample runs."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="KITTI root, e.g. /root/autodl-tmp/kitti-offical")
    parser.add_argument("--source-split", type=str, default="train", help="Source split name under ImageSets. Default: train")
    parser.add_argument(
        "--source-info",
        type=str,
        default="kitti_infos_train.pkl",
        help="Source info pkl relative to data-root. Default: kitti_infos_train.pkl",
    )
    parser.add_argument("--subset-name", type=str, required=True, help="New subset split name, e.g. train_patchwork_smoke128")
    parser.add_argument("--subset-size", type=int, required=True, help="Number of samples to keep")
    parser.add_argument("--seed", type=int, default=666, help="Random seed. Default: 666")
    parser.add_argument(
        "--selection",
        choices=["random", "head", "stratified"],
        default="random",
        help="Subset selection policy. Default: random",
    )
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[0, -40, -3, 70.4, 40, 1],
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Range used for scene voxel-count statistics in stratified mode.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[0.2, 0.2, 0.125],
        metavar=("VX", "VY", "VZ"),
        help="Voxel size used for scene voxel-count statistics in stratified mode.",
    )
    parser.add_argument(
        "--low-point-threshold",
        type=int,
        default=10,
        help="Objects with num_points_in_gt <= this value count as low-point objects in stratified mode.",
    )
    parser.add_argument(
        "--stratified-swap-iters",
        type=int,
        default=50000,
        help="Number of deterministic random-swap refinement iterations for stratified selection.",
    )
    parser.add_argument(
        "--output-info",
        type=str,
        default=None,
        help="Output info pkl filename relative to data-root. Default: kitti_infos_<subset-name>.pkl",
    )
    return parser.parse_args()


def _target_mask(names):
    return np.isin(names, np.array(CLASS_NAMES))


def _scene_voxel_count(data_root, sample_id, point_cloud_range, voxel_size):
    velodyne_path = data_root / "training" / "velodyne" / f"{sample_id}.bin"
    if not velodyne_path.exists():
        return 0

    points = np.fromfile(str(velodyne_path), dtype=np.float32).reshape(-1, 4)[:, :3]
    pc_range = np.asarray(point_cloud_range, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    in_range = np.all((points >= pc_range[:3]) & (points < pc_range[3:]), axis=1)
    points = points[in_range]
    if points.shape[0] == 0:
        return 0

    coords = np.floor((points - pc_range[:3]) / voxel_size).astype(np.int32)
    grid_size = np.ceil((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int64)
    linear_coords = (
        coords[:, 0].astype(np.int64) * grid_size[1] * grid_size[2]
        + coords[:, 1].astype(np.int64) * grid_size[2]
        + coords[:, 2].astype(np.int64)
    )
    return int(np.unique(linear_coords).shape[0])


def _frame_stats(info, data_root, point_cloud_range, voxel_size, low_point_threshold):
    annos = info.get("annos", {})
    names = np.asarray(annos.get("name", []))
    target = _target_mask(names)
    target_names = names[target]
    valid_object_mask = np.asarray(annos.get("index", np.arange(names.shape[0]))) >= 0
    valid_target_names = names[valid_object_mask]
    valid_target = _target_mask(valid_target_names)

    stats = {}
    for class_name in CLASS_NAMES:
        count = int(np.sum(target_names == class_name))
        stats[f"num_{class_name.lower()[:3]}"] = count
        stats[f"has_{class_name.lower()[:3]}"] = int(count > 0)

    stats["num_objects_total"] = int(target.sum())

    difficulty = np.asarray(annos.get("difficulty", []))
    if difficulty.size:
        stats["num_moderate_hard"] = int(np.sum(target & (difficulty >= 1)))
    else:
        stats["num_moderate_hard"] = 0

    gt_boxes = np.asarray(annos.get("gt_boxes_lidar", []))
    if gt_boxes.size:
        centers = gt_boxes[:, :3]
        distance = np.linalg.norm(centers[:, :2], axis=1)
        if distance.shape[0] != valid_target.shape[0]:
            raise RuntimeError(
                f"Mismatched valid object count for {info['point_cloud']['lidar_idx']}: "
                f"gt_boxes={distance.shape[0]} valid_names={valid_target.shape[0]}"
            )
        stats["num_far_objects_30m"] = int(np.sum(valid_target & (distance > 30.0)))
        stats["num_far_objects_40m"] = int(np.sum(valid_target & (distance > 40.0)))
    else:
        stats["num_far_objects_30m"] = 0
        stats["num_far_objects_40m"] = 0

    num_points = np.asarray(annos.get("num_points_in_gt", []))
    if num_points.size:
        if num_points.shape[0] == names.shape[0]:
            valid_num_points = num_points[valid_object_mask]
        elif num_points.shape[0] == valid_target.shape[0]:
            valid_num_points = num_points
        else:
            raise RuntimeError(
                f"Mismatched valid point-count object count for {info['point_cloud']['lidar_idx']}: "
                f"num_points={num_points.shape[0]} valid_names={valid_target.shape[0]}"
            )
        stats["num_low_point_objects"] = int(np.sum(valid_target & (valid_num_points <= low_point_threshold)))
    else:
        stats["num_low_point_objects"] = 0

    sample_id = info["point_cloud"]["lidar_idx"]
    stats["scene_voxel_count"] = _scene_voxel_count(data_root, sample_id, point_cloud_range, voxel_size)
    return stats


def _select_stratified(
    infos,
    sample_ids,
    subset_size,
    seed,
    data_root,
    point_cloud_range,
    voxel_size,
    low_point_threshold,
    swap_iters,
):
    rng = random.Random(seed)
    id_to_info = {info["point_cloud"]["lidar_idx"]: info for info in infos}
    ordered_infos = [id_to_info[sample_id] for sample_id in sample_ids if sample_id in id_to_info]
    if len(ordered_infos) != len(sample_ids):
        missing = sorted(set(sample_ids) - set(id_to_info))
        raise RuntimeError(f"Source split ids missing from info pkl: {missing[:10]}")

    stats = [
        _frame_stats(info, data_root, point_cloud_range, voxel_size, low_point_threshold)
        for info in ordered_infos
    ]
    stat_keys = list(stats[0].keys())
    stat_matrix = np.asarray([[frame_stats[key] for key in stat_keys] for frame_stats in stats], dtype=np.float64)
    target = stat_matrix.sum(axis=0) * (subset_size / len(sample_ids))

    # Normalize each feature so high-count statistics do not dominate low-frequency classes.
    scale = np.maximum(target, 1.0)
    weights = np.ones_like(target)
    for idx, key in enumerate(stat_keys):
        if key.startswith("num_") or key.startswith("has_"):
            weights[idx] = 1.5
        if key == "scene_voxel_count":
            weights[idx] = 0.75

    def score(accumulated):
        return float(np.sum(weights * np.abs((accumulated - target) / scale)))

    selected_indices = rng.sample(range(len(sample_ids)), subset_size)
    selected = set(selected_indices)
    unselected_indices = [idx for idx in range(len(sample_ids)) if idx not in selected]
    accumulated = stat_matrix[selected_indices].sum(axis=0)
    best_score = score(accumulated)

    # Local swap refinement keeps the stochastic nature of random sampling while matching
    # aggregate scene/object statistics more closely than plain random sampling.
    for _ in range(swap_iters):
        selected_pos = rng.randrange(len(selected_indices))
        unselected_pos = rng.randrange(len(unselected_indices))
        out_idx = selected_indices[selected_pos]
        in_idx = unselected_indices[unselected_pos]
        next_accumulated = accumulated - stat_matrix[out_idx] + stat_matrix[in_idx]
        next_score = score(next_accumulated)
        if next_score < best_score:
            selected_indices[selected_pos] = in_idx
            unselected_indices[unselected_pos] = out_idx
            accumulated = next_accumulated
            best_score = next_score

    chosen_ids = sorted(sample_ids[idx] for idx in selected_indices)
    chosen_id_set = set(chosen_ids)
    chosen_matrix = stat_matrix[[idx for idx in selected_indices]]
    return chosen_ids, stat_keys, stat_matrix.sum(axis=0), chosen_matrix.sum(axis=0)


def _print_stratified_report(stat_keys, full_sum, subset_sum, subset_size, full_size):
    expected = full_sum * (subset_size / full_size)
    print("stratified_report:")
    for key, full_value, subset_value, expected_value in zip(stat_keys, full_sum, subset_sum, expected):
        diff = subset_value - expected_value
        print(
            f"  {key}: full={full_value:.1f} expected={expected_value:.2f} "
            f"subset={subset_value:.1f} diff={diff:+.2f}"
        )


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    image_sets_dir = data_root / "ImageSets"
    source_split_path = image_sets_dir / f"{args.source_split}.txt"
    source_info_path = data_root / args.source_info

    if not source_split_path.exists():
        raise FileNotFoundError(f"Missing source split file: {source_split_path}")
    if not source_info_path.exists():
        raise FileNotFoundError(f"Missing source info pkl: {source_info_path}")

    sample_ids = [line.strip() for line in source_split_path.read_text().splitlines() if line.strip()]
    if args.subset_size <= 0 or args.subset_size > len(sample_ids):
        raise ValueError(f"subset-size must be in [1, {len(sample_ids)}], got {args.subset_size}")

    with open(source_info_path, "rb") as f:
        infos = pickle.load(f)

    stratified_report = None
    if args.selection == "head":
        chosen_ids = sample_ids[:args.subset_size]
    elif args.selection == "random":
        rng = random.Random(args.seed)
        chosen_ids = sorted(rng.sample(sample_ids, args.subset_size))
    else:
        chosen_ids, *stratified_report = _select_stratified(
            infos=infos,
            sample_ids=sample_ids,
            subset_size=args.subset_size,
            seed=args.seed,
            data_root=data_root,
            point_cloud_range=args.point_cloud_range,
            voxel_size=args.voxel_size,
            low_point_threshold=args.low_point_threshold,
            swap_iters=args.stratified_swap_iters,
        )

    chosen_id_set = set(chosen_ids)
    subset_infos = [info for info in infos if info["point_cloud"]["lidar_idx"] in chosen_id_set]
    if len(subset_infos) != len(chosen_ids):
        found_ids = {info["point_cloud"]["lidar_idx"] for info in subset_infos}
        missing_ids = sorted(chosen_id_set - found_ids)
        raise RuntimeError(f"Subset ids missing from info pkl: {missing_ids[:10]}")

    output_info_name = args.output_info or f"kitti_infos_{args.subset_name}.pkl"
    output_info_path = data_root / output_info_name
    output_split_path = image_sets_dir / f"{args.subset_name}.txt"

    output_split_path.write_text("".join(f"{sample_id}\n" for sample_id in chosen_ids))
    with open(output_info_path, "wb") as f:
        pickle.dump(subset_infos, f)

    print(f"data_root={data_root}")
    print(f"source_split={source_split_path}")
    print(f"source_info={source_info_path}")
    print(f"subset_name={args.subset_name}")
    print(f"subset_size={len(chosen_ids)}")
    print(f"selection={args.selection}")
    print(f"seed={args.seed}")
    print(f"output_split={output_split_path}")
    print(f"output_info={output_info_path}")
    print(f"first_ids={chosen_ids[:5]}")
    if stratified_report is not None:
        _print_stratified_report(*stratified_report, subset_size=len(chosen_ids), full_size=len(sample_ids))


if __name__ == "__main__":
    main()
