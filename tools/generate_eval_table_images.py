#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

THRESHOLD_GROUPS = ("0.5/0.25/0.25", "0.7/0.5/0.5")
CLASS_ORDER = ("Car", "Pedestrain", "Cyclist")
CLASS_NAME_MAP = {
    "Car": "Car",
    "Pedestrian": "Pedestrain",
    "Cyclist": "Cyclist",
}
METRIC_KEYS = {
    "bbox": "bbox AP",
    "bev": "bev  AP",
    "3d": "3d   AP",
}
METRIC_INDEX = {
    "bbox": 0,
    "bev": 1,
    "3d": 2,
}
GROUP_DEFS = {
    "0.5/0.25/0.25": {"Car": "0.50", "Pedestrain": "0.25", "Cyclist": "0.25"},
    "0.7/0.5/0.5": {"Car": "0.70", "Pedestrain": "0.50", "Cyclist": "0.50"},
}
RECALL_HEADER_PATTERNS = {
    "R11": r"^(Car|Pedestrian|Cyclist) AP@(.+):$",
    "R40": r"^(Car|Pedestrian|Cyclist) AP_R40@(.+):$",
}


@dataclass
class EpochMetrics:
    epoch: int
    source_file: Path
    values: Dict[str, Dict[str, List[str]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render R11/R40 evaluation table images from an eval directory. Usually only eval_dir is needed."
    )
    parser.add_argument("eval_dir", type=Path, help="Path to the eval directory that contains epoch_* subdirs")
    parser.add_argument(
        "--metric",
        choices=["bbox", "bev", "3d"],
        default="3d",
        help="Which metric block to render. Default: 3d",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <eval_dir>",
    )
    parser.add_argument(
        "--eval-tag",
        type=str,
        default=None,
        help="Optional leaf eval directory name to prefer when multiple official_eval_py310.txt files exist for one epoch.",
    )
    parser.add_argument(
        "--recalls",
        nargs="+",
        choices=["R11", "R40"],
        default=["R11", "R40"],
        help="Which recall settings to render. Default: R11 R40",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI. Default: 220")
    return parser.parse_args()


def find_eval_files(eval_dir: Path, eval_tag: Optional[str]) -> Dict[int, Path]:
    files_by_epoch: Dict[int, List[Path]] = {}
    for path in eval_dir.rglob("official_eval_py310.txt"):
        match = re.search(r"epoch_(\d+)", str(path))
        if not match:
            continue
        epoch = int(match.group(1))
        files_by_epoch.setdefault(epoch, []).append(path)

    preferred_tag = eval_tag
    if preferred_tag is None:
        tag_stats: Dict[str, Tuple[int, float]] = {}
        for candidates in files_by_epoch.values():
            for path in candidates:
                tag = path.parent.name
                count, latest_mtime = tag_stats.get(tag, (0, float("-inf")))
                tag_stats[tag] = (count + 1, max(latest_mtime, path.stat().st_mtime))
        if tag_stats:
            preferred_tag = max(tag_stats.items(), key=lambda item: (item[1][0], item[1][1], item[0]))[0]

    chosen: Dict[int, Path] = {}
    for epoch, candidates in files_by_epoch.items():
        if preferred_tag is not None:
            tagged = [p for p in candidates if p.parent.name == preferred_tag]
            if tagged:
                candidates = tagged
        candidates = sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
        chosen[epoch] = candidates[0]
    return chosen


def parse_eval_file(path: Path, metric_name: str, recall_mode: str) -> EpochMetrics:
    text = path.read_text()
    metric_prefix = METRIC_KEYS[metric_name]
    metric_idx = METRIC_INDEX[metric_name]
    header_pattern = RECALL_HEADER_PATTERNS[recall_mode]

    values: Dict[str, Dict[str, List[str]]] = {group: {} for group in THRESHOLD_GROUPS}
    raw_values: Dict[str, Dict[str, List[str]]] = {class_name: {} for class_name in CLASS_ORDER}
    current_class: Optional[str] = None
    current_threshold: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        header_match = re.match(header_pattern, line)
        if header_match:
            class_name = CLASS_NAME_MAP[header_match.group(1)]
            threshold_parts = [x.strip() for x in header_match.group(2).split(",")]
            if len(threshold_parts) != 3:
                current_class = None
                current_threshold = None
                continue
            current_class = class_name
            current_threshold = f"{float(threshold_parts[metric_idx]):.2f}"
            continue

        if current_class and current_threshold and line.startswith(metric_prefix):
            _, metrics_text = line.split(":", 1)
            metrics = [x.strip() for x in metrics_text.split(",")]
            if len(metrics) != 3:
                raise ValueError(f"Unexpected metric line in {path}: {line}")
            raw_values[current_class][current_threshold] = metrics
            current_class = None
            current_threshold = None

    for group_name, group_def in GROUP_DEFS.items():
        for class_name, threshold in group_def.items():
            metrics = raw_values.get(class_name, {}).get(threshold)
            if metrics is not None:
                values[group_name][class_name] = metrics

    match = re.search(r"epoch_(\d+)", str(path))
    if not match:
        raise ValueError(f"Could not extract epoch from path: {path}")
    return EpochMetrics(epoch=int(match.group(1)), source_file=path, values=values)


def collect_rows(eval_dir: Path, metric: str, eval_tag: Optional[str], recall_mode: str) -> List[EpochMetrics]:
    files_by_epoch = find_eval_files(eval_dir, eval_tag)
    if not files_by_epoch:
        raise SystemExit(f"No official_eval_py310.txt found under: {eval_dir}")
    return [parse_eval_file(path, metric, recall_mode) for _, path in sorted(files_by_epoch.items())]


def compute_best_values(rows: List[EpochMetrics], group_name: str) -> Dict[Tuple[str, int], float]:
    best_values: Dict[Tuple[str, int], float] = {}
    for class_name in CLASS_ORDER:
        for idx in range(3):
            values = []
            for row in rows:
                raw = row.values.get(group_name, {}).get(class_name, ["", "", ""])[idx]
                if raw != "":
                    values.append(float(raw))
            if values:
                best_values[(class_name, idx)] = max(values)
    return best_values


def draw_group_table(ax, group_name: str, rows: List[EpochMetrics], recall_mode: str) -> None:
    ax.set_axis_off()

    col_widths = [1.25] + [1.0] * 9
    x_edges = [0.0]
    for w in col_widths:
        x_edges.append(x_edges[-1] + w)
    total_width = x_edges[-1]

    title_h = 0.75
    header1_h = 0.8
    header2_h = 0.75
    row_h = 0.72
    total_height = title_h + header1_h + header2_h + row_h * len(rows) + 0.35

    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)

    title_y = 0.35
    top_rule_y = title_h
    header1_center_y = title_h + header1_h * 0.55
    cmidrule_y = title_h + header1_h
    header2_center_y = title_h + header1_h + header2_h * 0.55
    mid_rule_y = title_h + header1_h + header2_h
    bottom_rule_y = mid_rule_y + row_h * len(rows)

    ax.text(total_width / 2, title_y, f"{recall_mode} {group_name}", ha="center", va="center", fontsize=16, fontweight="bold")

    ax.hlines(top_rule_y, x_edges[0], x_edges[-1], linewidth=1.6, color="black")
    ax.hlines(mid_rule_y, x_edges[0], x_edges[-1], linewidth=1.0, color="black")
    ax.hlines(bottom_rule_y, x_edges[0], x_edges[-1], linewidth=1.6, color="black")

    group_spans = {
        "epochs": (0, 1),
        "Car": (1, 4),
        "Pedestrain": (4, 7),
        "Cyclist": (7, 10),
    }
    for label, (start, end) in group_spans.items():
        x0, x1 = x_edges[start], x_edges[end]
        ax.text((x0 + x1) / 2, header1_center_y, label, ha="center", va="center", fontsize=13, fontweight="bold")

    for start, end in [(1, 4), (4, 7), (7, 10)]:
        ax.hlines(cmidrule_y, x_edges[start] + 0.08, x_edges[end] - 0.08, linewidth=0.9, color="black")

    subheaders = [""] + ["Easy", "Moderate", "Hard"] * 3
    for idx, label in enumerate(subheaders):
        x0, x1 = x_edges[idx], x_edges[idx + 1]
        ax.text((x0 + x1) / 2, header2_center_y, label, ha="center", va="center", fontsize=11)

    best_values = compute_best_values(rows, group_name)
    start_y = mid_rule_y

    for row_idx, row in enumerate(rows):
        y = start_y + row_h * row_idx + row_h * 0.55
        ax.text((x_edges[0] + x_edges[1]) / 2, y, str(row.epoch), ha="center", va="center", fontsize=11)

        col_idx = 1
        for class_name in CLASS_ORDER:
            class_metrics = row.values.get(group_name, {}).get(class_name, ["", "", ""])
            for metric_idx, raw in enumerate(class_metrics):
                x0, x1 = x_edges[col_idx], x_edges[col_idx + 1]
                is_best = raw != "" and abs(float(raw) - best_values.get((class_name, metric_idx), float("-inf"))) < 1e-9
                ax.text(
                    (x0 + x1) / 2,
                    y,
                    raw,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold" if is_best else "normal",
                )
                col_idx += 1


def render_table_image(eval_dir: Path, rows: List[EpochMetrics], metric: str, recall_mode: str, output_dir: Path, dpi: int) -> Path:
    fig_height = 2.2 + max(len(rows), 1) * 0.36
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14.5, fig_height * 2),
        constrained_layout=True,
    )

    for ax, group_name in zip(axes, THRESHOLD_GROUPS):
        draw_group_table(ax, group_name, rows, recall_mode)

    output_path = output_dir / f"eval_summary_{metric}_{recall_mode}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    if not eval_dir.is_dir():
        raise SystemExit(f"eval_dir is not a directory: {eval_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for recall_mode in args.recalls:
        rows = collect_rows(eval_dir, args.metric, args.eval_tag, recall_mode)
        output_path = render_table_image(eval_dir, rows, args.metric, recall_mode, output_dir, args.dpi)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
