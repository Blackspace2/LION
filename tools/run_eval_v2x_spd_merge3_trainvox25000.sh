#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CFG_FILE="./cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3_trainvox25000.yaml"
INFO_PKL="/root/autodl-tmp/V2X-SPD-KITTI/merge3/kitti_infos_val.pkl"
OUTPUT_ROOT="/root/project/LION/run"
TRAIN_TAG="v2x_spd_merge3_lion_mamba_fp32_bs2_trainvox25000"
EVAL_TAG="final_eval_py310_ap"
PY310="/root/miniconda3/envs/lion_eval_py310/bin/python"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"

EXP_DIR="${OUTPUT_ROOT}/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3_trainvox25000/${TRAIN_TAG}"
CKPT_DIR="${EXP_DIR}/ckpt"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -d "${CKPT_DIR}" ]] || die "checkpoint dir not found: ${CKPT_DIR}"
[[ -f "${INFO_PKL}" ]] || die "info pkl not found: ${INFO_PKL}"
[[ -x "${PY310}" ]] || die "eval python not found: ${PY310}"

EPOCHS_TO_EVAL="${EPOCHS_TO_EVAL:-5 10 15 20 25 30}"

echo "========== V2X-SPD merge3 LION eval =========="
echo "EXP_DIR=${EXP_DIR}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "EPOCHS_TO_EVAL=${EPOCHS_TO_EVAL}"

for epoch in ${EPOCHS_TO_EVAL}; do
  ckpt="${CKPT_DIR}/checkpoint_epoch_${epoch}.pth"
  [[ -f "${ckpt}" ]] || { echo "skip epoch ${epoch}: checkpoint not found"; continue; }

  result_dir="${EXP_DIR}/eval/epoch_${epoch}/val/${EVAL_TAG}"
  result_pkl="${result_dir}/result.pkl"

  if [[ -f "${result_pkl}" ]]; then
    echo "----- epoch ${epoch}: result.pkl exists, skipping test.py, re-running AP only -----"
  else
    echo "----- epoch ${epoch}: running test.py -----"
    mkdir -p "${result_dir}"
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
    if [[ ! -f "${result_pkl}" ]]; then
      die "test.py status=${test_status}, but result.pkl was not produced: ${result_pkl}"
    fi
    echo "test.py done, status=${test_status}"
  fi

  echo "----- epoch ${epoch}: computing AP -----"
  "${PY310}" eval_kitti_result_from_pkl.py \
    --info_pkl "${INFO_PKL}" \
    --result_pkl "${result_pkl}" \
    --output "${result_dir}/official_eval_py310.txt"
  echo "----- epoch ${epoch} done -----"
  echo ""
done

echo "========== all evaluations complete =========="
echo "Results directory: ${EXP_DIR}/eval/"
for epoch in ${EPOCHS_TO_EVAL}; do
  result_file="${EXP_DIR}/eval/epoch_${epoch}/val/${EVAL_TAG}/official_eval_py310.txt"
  if [[ -f "${result_file}" ]]; then
    echo "=== epoch ${epoch} ==="
    cat "${result_file}"
    echo ""
  fi
done
