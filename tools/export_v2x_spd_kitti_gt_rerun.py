#!/usr/bin/env python3
"""Export V2X-SPD-KITTI GT samples to Rerun recordings."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import _init_path  # noqa: F401
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from skimage import io

from pcdet.utils import box_utils, calibration_kitti, object3d_kitti


CLASS_COLORS = {
    "Car": [0, 170, 255, 255],
    "Pedestrian": [255, 180, 0, 255],
    "Cyclist": [180, 80, 255, 255],
}

BAD_SCENES = {f"{idx:04d}" for idx in range(60, 74)}

INTENSITY_LUT = np.array(
    [
        [20, 20, 140],
        [0, 120, 255],
        [0, 210, 120],
        [255, 220, 0],
        [255, 80, 0],
    ],
    dtype=np.float32,
)

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="trainval", choices=["train", "val", "trainval"])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-image-points", type=int, default=20000)
    parser.add_argument("--max-3d-points", type=int, default=70000)
    parser.add_argument("--image-point-radius", type=float, default=2.2)
    return parser.parse_args()


def choose_ids(data_root: Path, split: str, frames: int, seed: int) -> list[str]:
    ids = [
        x.strip()
        for x in (data_root / "ImageSets" / f"{split}.txt").read_text().splitlines()
        if x.strip() and x.strip().split("_", 1)[0] not in BAD_SCENES
    ]
    rng = random.Random(seed)
    if len(ids) <= frames:
        return ids
    return sorted(rng.sample(ids, frames))


def box_rotations(yaws: np.ndarray):
    return [rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=float(yaw)) for yaw in yaws]


def objects_to_lidar_boxes(objs: list[object3d_kitti.Object3d], calib: calibration_kitti.Calibration) -> np.ndarray:
    if not objs:
        return np.zeros((0, 7), dtype=np.float32)
    loc = np.stack([o.loc for o in objs], axis=0).astype(np.float32)
    dims = np.array([[o.l, o.h, o.w] for o in objs], dtype=np.float32)
    rots = np.array([o.ry for o in objs], dtype=np.float32)
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[:, None])], axis=1).astype(np.float32)


def log_3d_boxes(path: str, boxes: np.ndarray, labels: list[str], colors: list[list[int]]) -> None:
    if len(boxes) == 0:
        rr.log(path, rr.Boxes3D(centers=[], sizes=[]))
        return
    rr.log(
        path,
        rr.Boxes3D(
            centers=boxes[:, :3],
            sizes=boxes[:, 3:6],
            rotations=box_rotations(boxes[:, 6]),
            labels=labels,
            colors=colors,
            radii=0.035,
            show_labels=True,
        ),
    )


def projected_box_strips(obj: object3d_kitti.Object3d, calib: calibration_kitti.Calibration, image_shape) -> list[np.ndarray]:
    corners = obj.generate_corners3d().astype(np.float32)
    pts_img, depth = calib.rect_to_img(corners)
    h, w = image_shape[:2]
    strips = []
    for i, j in BOX_EDGES:
        if depth[i] <= 0 or depth[j] <= 0:
            continue
        p0, p1 = pts_img[i], pts_img[j]
        if ((p0[0] < -w or p0[0] > 2 * w or p0[1] < -h or p0[1] > 2 * h) and
                (p1[0] < -w or p1[0] > 2 * w or p1[1] < -h or p1[1] > 2 * h)):
            continue
        strips.append(np.stack([p0, p1], axis=0).astype(np.float32))
    return strips


def sample_rows(arr: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(arr) <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=max_rows, replace=False)
    return arr[idx]


def colors_from_intensity(intensity: np.ndarray) -> np.ndarray:
    values = np.clip(intensity.reshape(-1), 0.0, 1.0)
    pos = values * (len(INTENSITY_LUT) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(INTENSITY_LUT) - 1)
    frac = (pos - lo)[:, None]
    rgb = INTENSITY_LUT[lo] * (1.0 - frac) + INTENSITY_LUT[hi] * frac
    return rgb.astype(np.uint8)


def main() -> None:
    args = parse_args()
    frame_ids = choose_ids(args.data_root, args.split, args.frames, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Image projection", origin="/frame/image"),
            rrb.Spatial3DView(
                name="Point cloud",
                origin="/frame/world",
                background=rrb.Background(color=[255, 255, 255, 255], kind=rrb.BackgroundKind.SolidColor),
            ),
        )
    )
    rr.init(f"v2x_spd_{args.data_root.name}_gt", spawn=False, default_blueprint=blueprint)
    rr.save(args.output)

    rr.log("/frame/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for frame_idx, fid in enumerate(frame_ids):
        points = np.fromfile(args.data_root / "training" / "velodyne" / f"{fid}.bin", dtype=np.float32).reshape(-1, 4)
        image = io.imread(args.data_root / "training" / "image_2" / f"{fid}.png")
        calib = calibration_kitti.Calibration(args.data_root / "training" / "calib" / f"{fid}.txt")
        objs = object3d_kitti.get_objects_from_label(args.data_root / "training" / "label_2" / f"{fid}.txt")
        boxes_lidar = objects_to_lidar_boxes(objs, calib)

        rr.set_time_sequence("sample", frame_idx)
        rr.log("/frame/image/rgb", rr.Image(image))
        rr.log("/frame/meta", rr.TextDocument(f"{args.data_root.name} {args.split} frame {fid}\\nGT boxes: {len(objs)}"))

        pts_3d = sample_rows(points, args.max_3d_points, args.seed + frame_idx)
        colors_3d = np.repeat((np.clip(pts_3d[:, 3:4], 0, 1) * 255).astype(np.uint8), 3, axis=1)
        rr.log("/frame/world/points", rr.Points3D(pts_3d[:, :3], colors=colors_3d, radii=0.02))

        labels = [o.cls_type for o in objs]
        box_colors = [CLASS_COLORS.get(o.cls_type, [255, 255, 255, 255]) for o in objs]
        log_3d_boxes("/frame/world/gt_boxes", boxes_lidar, labels, box_colors)

        pts_img, depth = calib.lidar_to_img(points[:, :3])
        h, w = image.shape[:2]
        mask = (depth > 0) & (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
        proj = np.concatenate([pts_img[mask], points[mask, 3:4]], axis=1)
        proj = sample_rows(proj, args.max_image_points, args.seed + 1000 + frame_idx)
        proj_colors = colors_from_intensity(proj[:, 2])
        rr.log(
            "/frame/image/projected_points",
            rr.Points2D(proj[:, :2], colors=proj_colors, radii=args.image_point_radius),
        )

        strips = []
        strip_colors = []
        for obj in objs:
            obj_strips = projected_box_strips(obj, calib, image.shape)
            strips.extend(obj_strips)
            strip_colors.extend([CLASS_COLORS.get(obj.cls_type, [255, 255, 255, 255])] * len(obj_strips))
        rr.log("/frame/image/projected_gt_boxes", rr.LineStrips2D(strips, colors=strip_colors, radii=2.0))

        print(f"{frame_idx + 1}/{len(frame_ids)} {fid}: points={len(points)} gt={len(objs)} image_points={len(proj)}")

    rr.disconnect()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
