#!/usr/bin/env python3
import argparse
import pickle
import random
from pathlib import Path


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
        choices=["random", "head"],
        default="random",
        help="Subset selection policy. Default: random",
    )
    parser.add_argument(
        "--output-info",
        type=str,
        default=None,
        help="Output info pkl filename relative to data-root. Default: kitti_infos_<subset-name>.pkl",
    )
    return parser.parse_args()


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

    if args.selection == "head":
        chosen_ids = sample_ids[:args.subset_size]
    else:
        rng = random.Random(args.seed)
        chosen_ids = sorted(rng.sample(sample_ids, args.subset_size))

    with open(source_info_path, "rb") as f:
        infos = pickle.load(f)

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


if __name__ == "__main__":
    main()
