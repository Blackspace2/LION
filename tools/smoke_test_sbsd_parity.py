#!/usr/bin/env python3
import argparse
import copy
import os
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.models.backbones_3d.sbsd import SBSD
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Check SBSD parity and activation diagnostics against the baseline on one train batch.")
    parser.add_argument("--baseline-cfg", type=Path, required=True)
    parser.add_argument("--sbsd-cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train_sbsd_smoke128")
    parser.add_argument("--train-info", type=str, default="kitti_infos_train_sbsd_smoke128.pkl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--mode",
        type=str,
        default="phase3_placeholder",
        choices=["phase3_placeholder", "phase4_active"],
    )
    parser.add_argument("--loss-tol", type=float, default=1e-5)
    parser.add_argument("--tb-tol", type=float, default=1e-5)
    parser.add_argument("--active-loss-tol", type=float, default=2e-5)
    parser.add_argument("--active-tb-tol", type=float, default=2e-5)
    parser.add_argument("--grad-tol", type=float, default=1e-6)
    parser.add_argument("--min-proj-grad", type=float, default=1e-7)
    parser.add_argument("--enforce-shared-grad-parity", action="store_true")
    parser.add_argument("--use-spectral-norm", action="store_true")
    return parser.parse_args()


def load_cfg(cfg_path, overrides):
    local_cfg = copy.deepcopy(cfg)
    cfg_from_yaml_file(str(cfg_path), local_cfg)
    cfg_from_list(overrides, local_cfg)
    return local_cfg


def run_one_pass(model, batch, seed):
    model_func = model_fn_decorator()
    model.zero_grad(set_to_none=True)
    common_utils.set_random_seed(seed)
    result = model_func(model, copy.deepcopy(batch))
    result.loss.backward()
    return result


def main():
    args = parse_args()
    launch_cwd = Path.cwd().resolve()
    if not args.baseline_cfg.is_absolute():
        args.baseline_cfg = (launch_cwd / args.baseline_cfg).resolve()
    else:
        args.baseline_cfg = args.baseline_cfg.resolve()
    if not args.sbsd_cfg.is_absolute():
        args.sbsd_cfg = (launch_cwd / args.sbsd_cfg).resolve()
    else:
        args.sbsd_cfg = args.sbsd_cfg.resolve()
    if not args.ckpt.is_absolute():
        args.ckpt = (launch_cwd / args.ckpt).resolve()
    else:
        args.ckpt = args.ckpt.resolve()
    if not args.data_root.is_absolute():
        args.data_root = (launch_cwd / args.data_root).resolve()
    else:
        args.data_root = args.data_root.resolve()
    os.chdir(Path(__file__).resolve().parent)
    logger = common_utils.create_logger()
    common_utils.set_random_seed(args.seed)

    data_overrides = [
        "DATA_CONFIG.DATA_PATH", str(args.data_root),
        "DATA_CONFIG.DATA_SPLIT.train", args.train_split,
        "DATA_CONFIG.INFO_PATH.train", f"['{args.train_info}']",
    ]
    baseline_cfg = load_cfg(args.baseline_cfg, data_overrides)
    sbsd_overrides = data_overrides + ["MODEL.BACKBONE_3D.SBSD.ENABLED", "True"]
    if args.use_spectral_norm:
        sbsd_overrides += ["MODEL.BACKBONE_3D.SBSD.USE_SPECTRAL_NORM", "True"]
    if args.mode == "phase3_placeholder":
        sbsd_overrides += [
            "MODEL.BACKBONE_3D.SBSD.USE_BANDWIDTH", "False",
            "MODEL.BACKBONE_3D.SBSD.USE_SPECTRAL", "False",
            "MODEL.BACKBONE_3D.SBSD.USE_DENSITY", "False",
        ]
    sbsd_cfg = load_cfg(args.sbsd_cfg, sbsd_overrides)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=baseline_cfg.DATA_CONFIG,
        class_names=baseline_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        root_path=args.data_root,
        workers=args.workers,
        seed=args.seed,
        logger=logger,
        training=True,
    )
    batch = next(iter(dataloader))

    baseline_model = build_network(
        model_cfg=baseline_cfg.MODEL,
        num_class=len(baseline_cfg.CLASS_NAMES),
        dataset=dataset,
    ).cuda()
    baseline_model.load_params_from_file(str(args.ckpt), logger=logger, to_cpu=False)
    baseline_model.train()

    sbsd_model = build_network(
        model_cfg=sbsd_cfg.MODEL,
        num_class=len(sbsd_cfg.CLASS_NAMES),
        dataset=dataset,
    ).cuda()
    sbsd_model.load_params_from_file(str(args.ckpt), logger=logger, to_cpu=False)
    sbsd_model.train()

    enabled_sbsd = sum(1 for m in sbsd_model.modules() if isinstance(m, SBSD) and m.enabled)
    print(f"mode={args.mode}")
    print(f"enabled_sbsd_modules={enabled_sbsd}")

    baseline_result = run_one_pass(baseline_model, batch, args.seed)
    sbsd_result = run_one_pass(sbsd_model, batch, args.seed)

    loss_diff = abs(float(baseline_result.loss.item()) - float(sbsd_result.loss.item()))
    print(f"baseline_loss={float(baseline_result.loss.item()):.8f}")
    print(f"sbsd_loss={float(sbsd_result.loss.item()):.8f}")
    print(f"loss_diff={loss_diff:.8e}")

    shared_tb_keys = sorted(set(baseline_result.tb_dict) & set(sbsd_result.tb_dict))
    max_tb_diff = 0.0
    for key in shared_tb_keys:
        diff = abs(float(baseline_result.tb_dict[key]) - float(sbsd_result.tb_dict[key]))
        max_tb_diff = max(max_tb_diff, diff)
    print(f"shared_tb_keys={len(shared_tb_keys)}")
    print(f"max_tb_diff={max_tb_diff:.8e}")

    sbsd_named_params = dict(sbsd_model.named_parameters())
    max_grad_diff = 0.0
    for name, param in baseline_model.named_parameters():
        if param.grad is None:
            continue
        other = sbsd_named_params.get(name)
        if other is None or other.grad is None:
            continue
        diff = float((param.grad - other.grad).abs().max().item())
        max_grad_diff = max(max_grad_diff, diff)
    print(f"max_shared_grad_diff={max_grad_diff:.8e}")

    sbsd_grad_norm = 0.0
    sbsd_scale_grad_norm = 0.0
    for name, param in sbsd_model.named_parameters():
        if param.grad is None:
            continue
        if ".sbsd.proj.weight" in name:
            sbsd_grad_norm = float(param.grad.norm().item())
        if ".sbsd.proj_scale" in name:
            sbsd_scale_grad_norm = float(param.grad.norm().item())
    print(f"sbsd_proj_grad_norm={sbsd_grad_norm:.8e}")
    print(f"sbsd_proj_scale_grad_norm={sbsd_scale_grad_norm:.8e}")

    if enabled_sbsd != 8:
        raise SystemExit(f"Expected 8 enabled SBSD modules, found {enabled_sbsd}")
    if args.mode == "phase3_placeholder":
        loss_tol = args.loss_tol
        tb_tol = args.tb_tol
    else:
        loss_tol = args.active_loss_tol
        tb_tol = args.active_tb_tol

    if loss_diff >= loss_tol:
        raise SystemExit(f"loss_diff {loss_diff:.8e} >= loss_tol {loss_tol:.8e}")
    if max_tb_diff >= tb_tol:
        raise SystemExit(f"max_tb_diff {max_tb_diff:.8e} >= tb_tol {tb_tol:.8e}")
    if args.enforce_shared_grad_parity and max_grad_diff >= args.grad_tol:
        raise SystemExit(f"max_shared_grad_diff {max_grad_diff:.8e} >= grad_tol {args.grad_tol:.8e}")
    if args.mode == "phase3_placeholder":
        if sbsd_grad_norm >= args.grad_tol:
            raise SystemExit(f"sbsd_proj_grad_norm {sbsd_grad_norm:.8e} >= grad_tol {args.grad_tol:.8e}")
    else:
        if sbsd_grad_norm <= args.min_proj_grad:
            raise SystemExit(f"sbsd_proj_grad_norm {sbsd_grad_norm:.8e} <= min_proj_grad {args.min_proj_grad:.8e}")

    print("parity_check=pass")


if __name__ == "__main__":
    main()
