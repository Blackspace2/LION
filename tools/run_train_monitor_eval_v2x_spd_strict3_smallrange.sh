#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
export LION_MAX_CONSECUTIVE_NONFINITE_SKIP="${LION_MAX_CONSECUTIVE_NONFINITE_SKIP:-10}"

CFG_FILE="${CFG_FILE:-./cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox25000.yaml}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/V2X-SPD-KITTI/strict3}"
INFO_PKL="${INFO_PKL:-${DATA_ROOT}/kitti_infos_val.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/project/LION/run}"
TRAIN_TAG="${TRAIN_TAG:-v2x_spd_strict3_smallrange_lion_mamba_fp32_bs6_from_merge3_ep64}"
EVAL_TAG="${EVAL_TAG:-official_eval_py310}"
PY310="${PY310:-/root/miniconda3/envs/lion_eval_py310/bin/python}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-None}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${CKPT_SAVE_INTERVAL}}"
MAX_CKPT_SAVE_NUM="${MAX_CKPT_SAVE_NUM:-20}"
LOGGER_ITER_INTERVAL="${LOGGER_ITER_INTERVAL:-20}"
CKPT_SAVE_TIME_INTERVAL="${CKPT_SAVE_TIME_INTERVAL:-1800}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

EXP_GROUP_PATH="cfgs/kitti_models"
CFG_TAG="$(basename "${CFG_FILE}" .yaml)"
EXP_DIR="${OUTPUT_ROOT}/${EXP_GROUP_PATH}/${CFG_TAG}/${TRAIN_TAG}"
CKPT_DIR="${EXP_DIR}/ckpt"

USE_PRETRAINED_MODEL=1
case "${PRETRAINED_MODEL}" in
  ""|"None"|"none"|"NULL"|"null")
    USE_PRETRAINED_MODEL=0
    PRETRAINED_MODEL=""
    ;;
esac

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || die "missing required path: ${path}"
}

latest_train_log() {
  find "${EXP_DIR}" -maxdepth 1 -type f -name 'log_train_*.txt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

scan_train_log() {
  local latest_log
  latest_log="$(latest_train_log || true)"
  if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
    echo "========== scan training log for anomalies =========="
    if grep -Eiq 'nan|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|non-finite|Skip CUDA OOM|out of memory|RuntimeError|Traceback|error|skipped_nonfinite[^0-9]*[1-9]|nonfinite_skip[^0-9]*[1-9]' "${latest_log}"; then
      grep -Ein 'nan|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|non-finite|Skip CUDA OOM|out of memory|RuntimeError|Traceback|error|skipped_nonfinite[^0-9]*[1-9]|nonfinite_skip[^0-9]*[1-9]' "${latest_log}" \
        | tee "${EXP_DIR}/train_anomaly_scan.txt"
      die "training log contains anomaly lines; inspect ${EXP_DIR}/train_anomaly_scan.txt"
    fi
    echo "No obvious NaN/Inf/OOM/skip/traceback patterns found in ${latest_log}."
  fi
}

evaluate_one_ckpt() {
  local ckpt="$1"
  local epoch="$2"

  echo "----- evaluating epoch ${epoch}: ${ckpt} -----"
  set +e
  python test.py \
    --cfg_file "${CFG_FILE}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --extra_tag "${TRAIN_TAG}" \
    --output_dir "${OUTPUT_ROOT}" \
    --ckpt "${ckpt}" \
    --eval_tag "${EVAL_TAG}" \
    --save_to_file
  local test_status=$?
  set -e

  local result_dir="${EXP_DIR}/eval/epoch_${epoch}/val/${EVAL_TAG}"
  local result_pkl="${result_dir}/result.pkl"
  if [[ ! -f "${result_pkl}" ]]; then
    die "test.py status=${test_status}, but result.pkl was not produced: ${result_pkl}"
  fi

  if (( test_status != 0 )); then
    echo "test.py exited with status=${test_status}; result.pkl exists, continue with standalone py310 AP."
  fi

  "${PY310}" eval_kitti_result_from_pkl.py \
    --info_pkl "${INFO_PKL}" \
    --result_pkl "${result_pkl}" \
    --output "${result_dir}/official_eval_py310.txt"
}

echo "========== V2X-SPD strict3 smallrange LION training =========="
echo "CFG_FILE=${CFG_FILE}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "INFO_PKL=${INFO_PKL}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_TAG=${TRAIN_TAG}"
if (( USE_PRETRAINED_MODEL )); then
  echo "PRETRAINED_MODEL=${PRETRAINED_MODEL}"
else
  echo "PRETRAINED_MODEL=<scratch>"
fi
echo "BATCH_SIZE=${BATCH_SIZE}, EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}, WORKERS=${WORKERS}, EPOCHS=${EPOCHS}"
echo "CKPT_SAVE_INTERVAL=${CKPT_SAVE_INTERVAL}, EVAL_INTERVAL=${EVAL_INTERVAL}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

require_path "${CFG_FILE}"
if (( USE_PRETRAINED_MODEL )); then
  require_path "${PRETRAINED_MODEL}"
fi
require_path "${DATA_ROOT}/kitti_infos_train.pkl"
require_path "${DATA_ROOT}/kitti_infos_val.pkl"
require_path "${DATA_ROOT}/kitti_infos_trainval.pkl"
require_path "${DATA_ROOT}/kitti_dbinfos_train.pkl"
require_path "${DATA_ROOT}/gt_database"
require_path "${PY310}"

mkdir -p "${EXP_DIR}"

echo "========== preflight dataloader smoke test =========="
CFG_FILE_ABS="$(realpath "${CFG_FILE}")"
export CFG_FILE_ABS
export PRETRAINED_MODEL
export USE_PRETRAINED_MODEL
python - <<'PY'
import os
import _init_path  # noqa
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

cfg_from_yaml_file(os.environ['CFG_FILE_ABS'], cfg)
ds, loader, _ = build_dataloader(
    cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=None, training=True
)
batch = next(iter(loader))
print('dataset_len', len(ds))
print('batch_size', batch['batch_size'])
print('voxels', tuple(batch['voxels'].shape))
print('gt_boxes', tuple(batch['gt_boxes'].shape))
PY

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  echo "========== smoke-only forward/backward step =========="
  python - <<'PY'
import os
import logging
import torch
from torch.nn.utils import clip_grad_norm_

import _init_path  # noqa
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer

cfg_from_yaml_file(os.environ['CFG_FILE_ABS'], cfg)
train_set, train_loader, _ = build_dataloader(
    cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=None, training=True
)
model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set).cuda()
logger = logging.getLogger("strict3_smallrange_smoke")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
if os.environ['USE_PRETRAINED_MODEL'] == '1':
    model.load_params_from_file(filename=os.environ['PRETRAINED_MODEL'], to_cpu=False, logger=logger)
optimizer = build_optimizer(model, cfg.OPTIMIZATION)
model_func = model_fn_decorator()

model.train()
batch = next(iter(train_loader))
optimizer.zero_grad()
loss, tb_dict, disp_dict = model_func(model, batch)
if not torch.isfinite(loss).all():
    raise RuntimeError(f'non-finite smoke train loss: {loss}')
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
if not torch.isfinite(grad_norm).all():
    raise RuntimeError(f'non-finite smoke grad norm: {grad_norm}')
optimizer.step()
torch.cuda.synchronize()
print('smoke_train_loss', float(loss.detach().cpu()))
print('smoke_grad_norm', float(grad_norm.detach().cpu()))
PY
  echo "========== smoke-only finished =========="
  exit 0
fi

echo "========== start interleaved train/eval =========="
current_epoch=0
while (( current_epoch < EPOCHS )); do
  next_epoch=$(( current_epoch + EVAL_INTERVAL ))
  if (( next_epoch > EPOCHS )); then
    next_epoch="${EPOCHS}"
  fi

  echo "========== train to epoch ${next_epoch} (scheduler horizon ${EPOCHS}) =========="
  train_cmd=(
    python train_no_builtin_eval.py
    --cfg_file "${CFG_FILE}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${next_epoch}"
    --total_epochs "${EPOCHS}"
    --workers "${WORKERS}"
    --extra_tag "${TRAIN_TAG}"
    --output_dir "${OUTPUT_ROOT}"
    --ckpt_save_interval "${CKPT_SAVE_INTERVAL}"
    --max_ckpt_save_num "${MAX_CKPT_SAVE_NUM}"
    --num_epochs_to_eval 0
    --logger_iter_interval "${LOGGER_ITER_INTERVAL}"
    --ckpt_save_time_interval "${CKPT_SAVE_TIME_INTERVAL}"
    --wo_gpu_stat
  )

  if (( USE_PRETRAINED_MODEL )) && ! find "${CKPT_DIR}" -maxdepth 1 -type f -name '*.pth' | grep -q .; then
    train_cmd+=(--pretrained_model "${PRETRAINED_MODEL}")
  fi

  "${train_cmd[@]}" 2>&1 | tee -a "${EXP_DIR}/train_console.log"
  scan_train_log

  ckpt_path="${CKPT_DIR}/checkpoint_epoch_${next_epoch}.pth"
  if [[ ! -f "${ckpt_path}" ]]; then
    if [[ -f "${CKPT_DIR}/latest_model.pth" ]]; then
      ckpt_path="${CKPT_DIR}/latest_model.pth"
      echo "checkpoint_epoch_${next_epoch}.pth not found, fallback to ${ckpt_path}"
    else
      die "checkpoint for epoch ${next_epoch} not found in ${CKPT_DIR}"
    fi
  fi

  evaluate_one_ckpt "${ckpt_path}" "${next_epoch}"
  current_epoch="${next_epoch}"
done

echo "========== interleaved training/evaluation finished =========="
echo "Experiment directory: ${EXP_DIR}"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_ROOT} --host 0.0.0.0 --port 6006"
