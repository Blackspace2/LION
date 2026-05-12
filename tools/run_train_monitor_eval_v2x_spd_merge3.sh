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

CFG_FILE="${CFG_FILE:-./cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml}"
DATA_ROOT="/root/autodl-tmp/V2X-SPD-KITTI/merge3"
INFO_PKL="${DATA_ROOT}/kitti_infos_val.pkl"
OUTPUT_ROOT="/root/project/LION/run"
TRAIN_TAG="${TRAIN_TAG:-v2x_spd_merge3_lion_mamba_fp32_bs2}"
EVAL_TAG="${EVAL_TAG:-final_eval_py310_ap}"
PY310="${PY310:-/root/miniconda3/envs/lion_eval_py310/bin/python}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-5}"
MAX_CKPT_SAVE_NUM="${MAX_CKPT_SAVE_NUM:-20}"
LOGGER_ITER_INTERVAL="${LOGGER_ITER_INTERVAL:-20}"
CKPT_SAVE_TIME_INTERVAL="${CKPT_SAVE_TIME_INTERVAL:-1800}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
FP16="${FP16:-0}"

EXP_GROUP_PATH="cfgs/kitti_models"
CFG_TAG="$(basename "${CFG_FILE}" .yaml)"
EXP_DIR="${OUTPUT_ROOT}/${EXP_GROUP_PATH}/${CFG_TAG}/${TRAIN_TAG}"
CKPT_DIR="${EXP_DIR}/ckpt"
MONITOR_LOG="${EXP_DIR}/monitor_train.log"

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

echo "========== V2X-SPD merge3 LION training =========="
echo "CFG_FILE=${CFG_FILE}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EXP_DIR=${EXP_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "BATCH_SIZE=${BATCH_SIZE}, WORKERS=${WORKERS}, EPOCHS=${EPOCHS}"
echo "FP16=${FP16}"
echo "SMOKE_ONLY=${SMOKE_ONLY}"

require_path "${CFG_FILE}"
require_path "${DATA_ROOT}/kitti_infos_train.pkl"
require_path "${DATA_ROOT}/kitti_infos_val.pkl"
require_path "${DATA_ROOT}/kitti_infos_trainval.pkl"
require_path "${DATA_ROOT}/kitti_dbinfos_train.pkl"
require_path "${DATA_ROOT}/gt_database"

mkdir -p "${EXP_DIR}"

echo "========== preflight dataloader smoke test =========="
python - <<'PY'
from pathlib import Path
import _init_path  # noqa
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

cfg_from_yaml_file('./cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml', cfg)
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
  echo "========== smoke-only gpu train/eval step =========="
  python - <<'PY'
import math
import torch
from torch.nn.utils import clip_grad_norm_

import _init_path  # noqa
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer

cfg_from_yaml_file('./cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml', cfg)

train_set, train_loader, _ = build_dataloader(
    cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=None, training=True
)
model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set).cuda()
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

del train_loader, train_set, batch
torch.cuda.empty_cache()

test_set, test_loader, _ = build_dataloader(
    cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=None, training=False
)
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))
    from pcdet.models import load_data_to_gpu
    load_data_to_gpu(batch)
    pred_dicts, ret_dict = model(batch)
    annos = test_set.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)
torch.cuda.synchronize()
print('smoke_eval_preds', len(annos), len(annos[0]['name']))
print('smoke_eval_ret_keys', sorted(ret_dict.keys()))
PY
  echo "========== smoke-only finished =========="
  exit 0
fi

echo "========== start training =========="
train_args=(
  train_no_builtin_eval.py
  --cfg_file "${CFG_FILE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --workers "${WORKERS}" \
  --extra_tag "${TRAIN_TAG}" \
  --output_dir "${OUTPUT_ROOT}" \
  --ckpt_save_interval "${CKPT_SAVE_INTERVAL}" \
  --max_ckpt_save_num "${MAX_CKPT_SAVE_NUM}" \
  --num_epochs_to_eval 0 \
  --logger_iter_interval "${LOGGER_ITER_INTERVAL}" \
  --ckpt_save_time_interval "${CKPT_SAVE_TIME_INTERVAL}" \
  --wo_gpu_stat
)

if [[ "${FP16}" == "1" ]]; then
  train_args+=(--fp16)
fi

python "${train_args[@]}" 2>&1 | tee "${EXP_DIR}/train_console.log"

echo "========== training finished =========="

latest_log="$(latest_train_log || true)"
if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
  echo "========== scan training log for anomalies =========="
  if grep -Eiq 'nan|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|non-finite|Skip CUDA OOM|out of memory|RuntimeError|Traceback|error|skipped_nonfinite[^0-9]*1|nonfinite_skip[^0-9]*1' "${latest_log}"; then
    grep -Ein 'nan|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|non-finite|Skip CUDA OOM|out of memory|RuntimeError|Traceback|error|skipped_nonfinite[^0-9]*1|nonfinite_skip[^0-9]*1' "${latest_log}" \
      | tee "${EXP_DIR}/train_anomaly_scan.txt"
    die "training log contains anomaly lines; inspect ${EXP_DIR}/train_anomaly_scan.txt"
  fi
  echo "No obvious NaN/Inf/OOM/skip/traceback patterns found in ${latest_log}."
fi

mapfile -t ckpts < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pth' | sort -V)
if (( ${#ckpts[@]} == 0 )); then
  die "no checkpoints found in ${CKPT_DIR}"
fi

if [[ ! -x "${PY310}" ]]; then
  die "missing eval-only Python: ${PY310}"
fi

echo "========== evaluate checkpoints =========="
for ckpt in "${ckpts[@]}"; do
  epoch="$(basename "${ckpt}" .pth)"
  epoch="${epoch#checkpoint_epoch_}"

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
  test_status=$?
  set -e

  result_dir="${EXP_DIR}/eval/epoch_${epoch}/val/${EVAL_TAG}"
  result_pkl="${result_dir}/result.pkl"
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
done

echo "========== done =========="
echo "Experiment directory: ${EXP_DIR}"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_ROOT} --host 0.0.0.0 --port 6006"
