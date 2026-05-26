#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export DATA_PATH="${DATA_PATH:-/root/autodl-tmp/kitti-offical}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/project/LION/output/LION_output}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
export FIX_RANDOM_SEED="${FIX_RANDOM_SEED:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-40}"
export STOP_EPOCHS="${STOP_EPOCHS:-8}"
export CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-2}"
export MAX_CKPT_SAVE_NUM="${MAX_CKPT_SAVE_NUM:-10}"
export LOGGER_ITER_INTERVAL="${LOGGER_ITER_INTERVAL:-20}"
export EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export EVAL_WORKERS="${EVAL_WORKERS:-4}"
export EVAL_TAG="${EVAL_TAG:-py310_ap}"
export GENERATE_TABLES="${GENERATE_TABLES:-1}"
export TABLE_METRICS="${TABLE_METRICS:-3d}"

export TRAIN_SPLIT_NAME="${TRAIN_SPLIT_NAME:-train_patchwork_classaware_smoke512}"
export TRAIN_INFO_PKL="${TRAIN_INFO_PKL:-kitti_infos_train_patchwork_classaware_smoke512.pkl}"
export EXTRA_TAG="${EXTRA_TAG:-patchwork_guidance_classaware_smoke512_fromscratch_e8_seed666}"
export EXTRA_SET_CFGS="${EXTRA_SET_CFGS:-}"

exec ./run_kitti_experiment.sh train ./cfgs/kitti_models/second_with_lion_mamba_64dim_patchwork_guidance.yaml
