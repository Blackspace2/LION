#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

CFG_FILE="./cfgs/kitti_models/second_with_lion_mamba_64dim.yaml"
DATA_PATH="/root/autodl-tmp/kitti-offical"
INFO_PKL="${DATA_PATH}/kitti_infos_val.pkl"
PY310="/root/miniconda3/envs/lion_eval_py310/bin/python"

TRAIN_TAG="lion_mamba_kitti_single_gpu_bs10_fp16_resume_lr8e-4_clip3"
EVAL_TAG="${TRAIN_TAG}_eval_py310_ap"
CKPT_DIR="/root/autodl-tmp/LION_output/cfgs/kitti_models/second_with_lion_mamba_64dim/${TRAIN_TAG}/ckpt"
RESUME_CKPT="${RESUME_CKPT:-/root/autodl-tmp/LION_output/cfgs/kitti_models/second_with_lion_mamba_64dim/lion_mamba_kitti_single_gpu_bs12_fp16_resume_lr1e-3_clip3/ckpt/checkpoint_epoch_26.pth}"

python train_no_builtin_eval.py \
  --cfg_file "${CFG_FILE}" \
  --extra_tag "${TRAIN_TAG}" \
  --batch_size 10 \
  --epochs 80 \
  --workers 10 \
  --ckpt "${RESUME_CKPT}" \
  --ckpt_save_interval 2 \
  --max_ckpt_save_num 50 \
  --num_epochs_to_eval 0 \
  --logger_iter_interval 20 \
  --ckpt_save_time_interval 1800 \
  --fp16 \
  --set DATA_CONFIG.DATA_PATH "${DATA_PATH}" \
        OPTIMIZATION.LR 0.0008 \
        OPTIMIZATION.GRAD_NORM_CLIP 3

if [[ ! -x "${PY310}" ]]; then
  echo "Missing eval-only Python: ${PY310}" >&2
  exit 1
fi

shopt -s nullglob
ckpts=("${CKPT_DIR}"/checkpoint_epoch_*.pth)
if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No checkpoints found in ${CKPT_DIR}" >&2
  exit 1
fi

for ckpt in "${ckpts[@]}"; do
  epoch="$(basename "${ckpt}" .pth)"
  epoch="${epoch#checkpoint_epoch_}"
  if (( epoch < 26 )); then
    continue
  fi

  echo "Evaluating checkpoint epoch ${epoch}: ${ckpt}"
  set +e
  python test.py \
    --cfg_file cfgs/kitti_models/second_with_lion_mamba_64dim.yaml \
    --extra_tag "${EVAL_TAG}" \
    --batch_size 10 \
    --workers 10 \
    --ckpt "${ckpt}" \
    --eval_tag "py310_ap" \
    --set DATA_CONFIG.DATA_PATH "${DATA_PATH}"
  test_status=$?
  set -e

  result_dir="/root/autodl-tmp/LION_output/kitti_models/second_with_lion_mamba_64dim/${EVAL_TAG}/eval/epoch_${epoch}/val/py310_ap"
  result_pkl="${result_dir}/result.pkl"
  if [[ ! -f "${result_pkl}" ]]; then
    echo "test.py status=${test_status}, but result.pkl was not produced: ${result_pkl}" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    "${PY310}" eval_kitti_result_from_pkl.py \
      --info_pkl "${INFO_PKL}" \
      --result_pkl "${result_pkl}" \
      --output "${result_dir}/official_eval_py310.txt"
done

echo "Training and py310 AP evaluation finished."
