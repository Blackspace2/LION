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
    parser = argparse.ArgumentParser(description="Phase 6D one-iter sanity check for Gaussian-init SBSD.")
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, required=True)
    parser.add_argument("--train-info", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--set", dest="set_cfgs", nargs="*", default=[])
    return parser.parse_args()


def load_cfg(cfg_path, overrides):
    local_cfg = copy.deepcopy(cfg)
    cfg_from_yaml_file(str(cfg_path), local_cfg)
    cfg_from_list(overrides, local_cfg)
    return local_cfg


def main():
    args = parse_args()
    launch_cwd = Path.cwd().resolve()
    args.cfg = (launch_cwd / args.cfg).resolve() if not args.cfg.is_absolute() else args.cfg.resolve()
    args.data_root = (launch_cwd / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    os.chdir(Path(__file__).resolve().parent)

    logger = common_utils.create_logger()
    common_utils.set_random_seed(args.seed)

    overrides = [
        "DATA_CONFIG.DATA_PATH", str(args.data_root),
        "DATA_CONFIG.DATA_SPLIT.train", args.train_split,
        "DATA_CONFIG.INFO_PATH.train", f"['{args.train_info}']",
        "MODEL.BACKBONE_3D.SBSD.ENABLED", "True",
    ] + args.set_cfgs
    local_cfg = load_cfg(args.cfg, overrides)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=local_cfg.DATA_CONFIG,
        class_names=local_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        root_path=args.data_root,
        workers=args.workers,
        seed=args.seed,
        logger=logger,
        training=True,
    )
    batch = next(iter(dataloader))

    model = build_network(
        model_cfg=local_cfg.MODEL,
        num_class=len(local_cfg.CLASS_NAMES),
        dataset=dataset,
    ).cuda()
    model.train()

    sbsd_modules = [m for m in model.modules() if isinstance(m, SBSD) and m.enabled]
    if len(sbsd_modules) != 8:
        raise SystemExit(f"Expected 8 enabled SBSD modules, found {len(sbsd_modules)}")

    proj_stds = []
    proj_scale_values = []
    proj_scale_trainable = []
    for module in sbsd_modules:
        proj_stds.append(float(module.proj.weight.std().item()))
        proj_scale_values.append(float(module.proj_scale.item()))
        proj_scale_trainable.append(bool(getattr(module.proj_scale, "requires_grad", False)))

    model_func = model_fn_decorator()
    model.zero_grad(set_to_none=True)
    result = model_func(model, batch)
    loss = result.loss
    if not torch.isfinite(loss):
        raise SystemExit(f"Non-finite loss before backward: {float(loss.item())}")
    loss.backward()

    grad_finite = True
    max_grad = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_finite = grad_finite and bool(torch.isfinite(param.grad).all().item())
        max_grad = max(max_grad, float(param.grad.abs().max().item()))

    if not grad_finite:
        raise SystemExit("Detected non-finite gradient values during Phase 6D sanity check")

    mean_proj_std = sum(proj_stds) / len(proj_stds)
    min_proj_std = min(proj_stds)
    max_proj_std = max(proj_stds)
    if not (0.0 < mean_proj_std < 1e-2):
        raise SystemExit(f"Unexpected mean proj std: {mean_proj_std:.8e}")
    if any(trainable for trainable in proj_scale_trainable):
        raise SystemExit("Phase 6D expects proj_scale to be fixed, but a trainable proj_scale was found")
    if any(abs(v - 1.0) > 1e-6 for v in proj_scale_values):
        raise SystemExit(f"Phase 6D expects proj_scale=1.0, got {proj_scale_values}")

    print(f"loss={float(loss.item()):.8f}")
    print(f"proj_weight_std_mean={mean_proj_std:.8e}")
    print(f"proj_weight_std_min={min_proj_std:.8e}")
    print(f"proj_weight_std_max={max_proj_std:.8e}")
    print(f"proj_scale_values={proj_scale_values}")
    print(f"proj_scale_trainable={proj_scale_trainable}")
    print(f"max_grad={max_grad:.8e}")
    print("phase6d_sanity=pass")


if __name__ == "__main__":
    main()
