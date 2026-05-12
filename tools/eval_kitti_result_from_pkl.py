import argparse
import importlib
import pickle
import shutil
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KITTI result.pkl without importing full pcdet")
    parser.add_argument("--info_pkl", required=True, help="KITTI info pkl with ground-truth annos")
    parser.add_argument("--result_pkl", required=True, help="OpenPCDet detection result.pkl")
    parser.add_argument("--output", required=True, help="Path to save the official KITTI AP text")
    parser.add_argument("--class_names", nargs="+", default=["Car", "Pedestrian", "Cyclist"])
    return parser.parse_args()


def import_standalone_kitti_eval():
    src_dir = Path(__file__).resolve().parents[1] / "pcdet/datasets/kitti/kitti_object_eval_python"
    tmp_dir = Path(tempfile.mkdtemp(prefix="kitti_eval_"))
    pkg_dir = tmp_dir / "kitti_eval_standalone"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    shutil.copy2(src_dir / "eval.py", pkg_dir / "eval.py")
    shutil.copy2(src_dir / "rotate_iou.py", pkg_dir / "rotate_iou.py")
    sys.path.insert(0, str(tmp_dir))
    return importlib.import_module("kitti_eval_standalone.eval")


def main():
    args = parse_args()

    with open(args.info_pkl, "rb") as f:
        infos = pickle.load(f)
    gt_annos = [info["annos"] for info in infos]

    with open(args.result_pkl, "rb") as f:
        det_annos = pickle.load(f)

    if len(gt_annos) != len(det_annos):
        raise RuntimeError(f"GT/detection length mismatch: {len(gt_annos)} vs {len(det_annos)}")

    kitti_eval = import_standalone_kitti_eval()
    result_str, _ = kitti_eval.get_official_eval_result(gt_annos, det_annos, args.class_names)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result_str)
    print(result_str)
    print(f"Saved official KITTI AP to {output}")


if __name__ == "__main__":
    main()
