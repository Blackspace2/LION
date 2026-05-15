import argparse
from pathlib import Path

import _init_path  # noqa: F401
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import box_utils
from pcdet.utils import common_utils


CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
CLASS_COLORS = {
    1: [0, 180, 255, 255],
    2: [255, 180, 0, 255],
    3: [180, 80, 255, 255],
}
GT_COLOR = [30, 210, 90, 255]
ROI_COLOR = [255, 255, 255, 255]
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
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


def parse_args():
    parser = argparse.ArgumentParser(description="Export KITTI val predictions to a Rerun .rrd file")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score_thresh", type=float, default=0.1)
    parser.add_argument(
        "--roi",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="ROI box in lidar coordinates. Defaults to DATA_CONFIG.POINT_CLOUD_RANGE",
    )
    parser.add_argument("--hide_roi", action="store_true", help="Do not log the ROI helper box")
    parser.add_argument("--crop_to_roi", action="store_true", help="Only export points and boxes inside ROI")
    parser.add_argument("--max_image_points", type=int, default=20000)
    parser.add_argument("--max_3d_points", type=int, default=70000)
    parser.add_argument("--image_point_radius", type=float, default=2.2)
    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def box_rotations(yaws):
    return [rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=float(yaw)) for yaw in yaws]


def make_roi_box(roi):
    roi = np.asarray(roi, dtype=np.float32)
    roi_min, roi_max = roi[:3], roi[3:]
    center = (roi_min + roi_max) * 0.5
    size = roi_max - roi_min
    return np.array([[center[0], center[1], center[2], size[0], size[1], size[2], 0.0]], dtype=np.float32)


def mask_centers_in_roi(centers, roi):
    roi = np.asarray(roi, dtype=np.float32)
    return np.logical_and(centers >= roi[:3], centers <= roi[3:]).all(axis=1)


def log_boxes(path, boxes, labels, colors):
    if len(boxes) == 0:
        rr.log(path, rr.Boxes3D(centers=[], sizes=[]))
        return

    rr.log(
        path,
        rr.Boxes3D(
            centers=boxes[:, 0:3],
            sizes=boxes[:, 3:6],
            rotations=box_rotations(boxes[:, 6]),
            colors=colors,
            labels=labels,
            show_labels=True,
            radii=0.025,
        ),
    )


def sample_rows(arr, max_rows, seed):
    if len(arr) <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=max_rows, replace=False)
    return arr[idx]


def colors_from_intensity(intensity):
    values = np.clip(intensity.reshape(-1), 0.0, 1.0)
    pos = values * (len(INTENSITY_LUT) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(INTENSITY_LUT) - 1)
    frac = (pos - lo)[:, None]
    rgb = INTENSITY_LUT[lo] * (1.0 - frac) + INTENSITY_LUT[hi] * frac
    return rgb.astype(np.uint8)


def boxes_to_line_strips(boxes, calib, image_shape):
    if len(boxes) == 0:
        return []

    corners_lidar = box_utils.boxes_to_corners_3d(boxes)
    strips = []
    h, w = image_shape[:2]
    for corners in corners_lidar:
        pts_rect = calib.lidar_to_rect(corners)
        pts_img, depth = calib.rect_to_img(pts_rect)
        for i, j in BOX_EDGES:
            if depth[i] <= 0 or depth[j] <= 0:
                continue
            p0, p1 = pts_img[i], pts_img[j]
            if ((p0[0] < -w or p0[0] > 2 * w or p0[1] < -h or p0[1] > 2 * h) and
                    (p1[0] < -w or p1[0] > 2 * w or p1[1] < -h or p1[1] > 2 * h)):
                continue
            strips.append(np.stack([p0, p1], axis=0).astype(np.float32))
    return strips


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    roi = args.roi if args.roi is not None else cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    roi_box = make_roi_box(roi)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger()
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

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
    rr.init("lion_kitti_val_pred", spawn=False, default_blueprint=blueprint)
    rr.save(output_path)
    rr.log("/frame/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    with torch.no_grad():
        for frame_idx, batch_dict in enumerate(test_loader):
            if frame_idx >= args.frames:
                break

            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            frame_id = str(batch_dict["frame_id"][0])
            image = test_set.get_image(frame_id)
            calib = batch_dict["calib"][0]
            raw_points = test_set.get_lidar(frame_id)
            points = raw_points
            if args.crop_to_roi:
                points = points[mask_centers_in_roi(points[:, 0:3], roi)]

            intensities = np.clip(points[:, 3:4], 0.0, 1.0)
            point_colors = np.repeat((intensities * 255).astype(np.uint8), 3, axis=1)

            rr.set_time_sequence("frame", frame_idx)
            rr.log("/frame/image/rgb", rr.Image(image))
            rr.log(
                "/frame/meta",
                rr.TextDocument(f"KITTI val frame {frame_id}, score_thresh={args.score_thresh:.2f}")
            )

            pts_3d = sample_rows(points, args.max_3d_points, frame_idx)
            colors_3d = np.repeat((np.clip(pts_3d[:, 3:4], 0, 1) * 255).astype(np.uint8), 3, axis=1)
            rr.log("/frame/world/points", rr.Points3D(pts_3d[:, 0:3], colors=colors_3d, radii=0.015))
            if not args.hide_roi:
                log_boxes("/frame/world/roi", roi_box, ["ROI"], [ROI_COLOR])

            pts_img, depth = calib.lidar_to_img(points[:, :3])
            h, w = image.shape[:2]
            mask = (depth > 0) & (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
            proj = np.concatenate([pts_img[mask], points[mask, 3:4]], axis=1)
            proj = sample_rows(proj, args.max_image_points, 1000 + frame_idx)
            proj_colors = colors_from_intensity(proj[:, 2])
            rr.log(
                "/frame/image/projected_points",
                rr.Points2D(proj[:, :2], colors=proj_colors, radii=args.image_point_radius),
            )

            gt = batch_dict["gt_boxes"][0].detach().cpu().numpy()
            gt = gt[gt[:, 7] > 0]
            if args.crop_to_roi:
                gt = gt[mask_centers_in_roi(gt[:, 0:3], roi)]
            gt_labels = [f"GT {CLASS_NAMES[int(cls_id) - 1]}" for cls_id in gt[:, 7].astype(np.int64)]
            gt_colors = [GT_COLOR for _ in range(len(gt))]
            log_boxes("/frame/world/gt_boxes", gt[:, :7], gt_labels, gt_colors)
            gt_strips = boxes_to_line_strips(gt[:, :7], calib, image.shape)
            rr.log("/frame/image/projected_gt_boxes", rr.LineStrips2D(gt_strips, colors=gt_colors * 12, radii=2.0))

            pred = pred_dicts[0]
            pred_boxes = pred["pred_boxes"].detach().cpu().numpy()
            pred_scores = pred["pred_scores"].detach().cpu().numpy()
            pred_labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64)

            if args.crop_to_roi and len(pred_boxes) > 0:
                pred_roi_mask = mask_centers_in_roi(pred_boxes[:, 0:3], roi)
                pred_boxes = pred_boxes[pred_roi_mask]
                pred_scores = pred_scores[pred_roi_mask]
                pred_labels = pred_labels[pred_roi_mask]

            score_mask = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[score_mask]
            pred_scores = pred_scores[score_mask]
            pred_labels = pred_labels[score_mask]
            pred_text = [
                f"{CLASS_NAMES[label - 1]} {score:.2f}"
                for label, score in zip(pred_labels, pred_scores)
            ]
            pred_colors = [CLASS_COLORS.get(int(label), [255, 255, 255, 255]) for label in pred_labels]
            log_boxes("/frame/world/pred_boxes", pred_boxes, pred_text, pred_colors)
            pred_strips = boxes_to_line_strips(pred_boxes, calib, image.shape)
            rr.log("/frame/image/projected_pred_boxes", rr.LineStrips2D(pred_strips, colors=pred_colors * 12, radii=2.0))
            logger.info(
                f"Exported frame {frame_idx + 1}/{args.frames}: {frame_id}, "
                f"points={len(points)}, gt={len(gt)}, pred={len(pred_boxes)}, score_thresh={args.score_thresh:.2f}"
            )

    rr.disconnect()
    logger.info(f"Saved Rerun recording to {output_path}")


if __name__ == "__main__":
    main()
