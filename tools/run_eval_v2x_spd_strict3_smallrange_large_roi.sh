#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CFG_FILE="${CFG_FILE:-./cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000.yaml}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/V2X-SPD-KITTI/strict3}"
INFO_PKL="${INFO_PKL:-${DATA_ROOT}/kitti_infos_val.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/project/LION/run}"
TRAIN_TAG="${TRAIN_TAG:-v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch}"
EVAL_TAG="${EVAL_TAG:-official_eval_py310_roi_0_-60_-3_140_60_2}"
PY310="${PY310:-/root/miniconda3/envs/lion_eval_py310/bin/python}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"
EPOCHS_TO_EVAL="${EPOCHS_TO_EVAL:-all}"
RESULT_ROOT_NAME="${RESULT_ROOT_NAME:-eval_large_range}"

ROI_X_MIN="${ROI_X_MIN:-0}"
ROI_Y_MIN="${ROI_Y_MIN:--60}"
ROI_Z_MIN="${ROI_Z_MIN:--3}"
ROI_X_MAX="${ROI_X_MAX:-140}"
ROI_Y_MAX="${ROI_Y_MAX:-60}"
ROI_Z_MAX="${ROI_Z_MAX:-2}"
TEST_MAX_VOXELS="${TEST_MAX_VOXELS:-120000}"
TRAIN_Z_MIN="${TRAIN_Z_MIN:--3}"
TRAIN_Z_MAX="${TRAIN_Z_MAX:-1}"
TRAIN_VOXEL_Z="${TRAIN_VOXEL_Z:-0.125}"

ORIG_Z_BINS="$(python -c "train_z_min=float('${TRAIN_Z_MIN}'); train_z_max=float('${TRAIN_Z_MAX}'); train_voxel_z=float('${TRAIN_VOXEL_Z}'); print(int(round((train_z_max-train_z_min)/train_voxel_z)))")"
EVAL_VOXEL_Z="${EVAL_VOXEL_Z:-$(python -c "roi_z_min=float('${ROI_Z_MIN}'); roi_z_max=float('${ROI_Z_MAX}'); orig_z_bins=int('${ORIG_Z_BINS}'); print(f'{(roi_z_max-roi_z_min)/orig_z_bins:.8f}')")}"

EXP_GROUP_PATH="cfgs/kitti_models"
CFG_TAG="$(basename "${CFG_FILE}" .yaml)"
EXP_DIR="${OUTPUT_ROOT}/${EXP_GROUP_PATH}/${CFG_TAG}/${TRAIN_TAG}"
CKPT_DIR="${EXP_DIR}/ckpt"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -d "${CKPT_DIR}" ]] || die "checkpoint dir not found: ${CKPT_DIR}"
[[ -f "${INFO_PKL}" ]] || die "info pkl not found: ${INFO_PKL}"
[[ -x "${PY310}" ]] || die "eval python not found: ${PY310}"

TMP_CFG="/tmp/lion_large_roi_eval_${CFG_TAG}.yaml"
cleanup() {
  :
}
trap cleanup EXIT

CFG_FILE_ABS="$(realpath "${CFG_FILE}")"
TMP_TAG="$(basename "${TMP_CFG}" .yaml)"
TMP_EXP_GROUP_PATH="tmp"
TMP_EXP_DIR="${OUTPUT_ROOT}/${TMP_EXP_GROUP_PATH}/${TMP_TAG}/${TRAIN_TAG}"

cat > "${TMP_CFG}" <<EOF
_BASE_CONFIG_: ${CFG_FILE_ABS}

DATA_CONFIG:
    DATASET: 'KittiDataset'
    DATA_PATH: ${DATA_ROOT}
    DATA_SPLIT: {
        'train': train,
        'test': val
    }
    INFO_PATH: {
        'train': [kitti_infos_train.pkl],
        'test': [kitti_infos_val.pkl],
    }
    GET_ITEM_LIST: ["points"]
    FOV_POINTS_ONLY: False
    POINT_CLOUD_RANGE: [${ROI_X_MIN}, ${ROI_Y_MIN}, ${ROI_Z_MIN}, ${ROI_X_MAX}, ${ROI_Y_MAX}, ${ROI_Z_MAX}]
    POINT_FEATURE_ENCODING: {
        encoding_type: absolute_coordinates_encoding,
        used_feature_list: ['x', 'y', 'z', 'intensity'],
        src_feature_list: ['x', 'y', 'z', 'intensity'],
    }
    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
          SHUFFLE_ENABLED: {
              'train': True,
              'test': False
          }

        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.2, 0.2, ${EVAL_VOXEL_Z}]
          MAX_POINTS_PER_VOXEL: 5
          MAX_NUMBER_OF_VOXELS: {
              'train': 16000,
              'test': ${TEST_MAX_VOXELS}
          }
EOF

echo "========== V2X-SPD strict3 large-ROI eval =========="
echo "CFG_FILE=${CFG_FILE}"
echo "TMP_CFG=${TMP_CFG}"
echo "EXP_DIR=${EXP_DIR}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "EPOCHS_TO_EVAL=${EPOCHS_TO_EVAL}"
echo "RESULT_ROOT_NAME=${RESULT_ROOT_NAME}"
echo "POINT_CLOUD_RANGE=[${ROI_X_MIN}, ${ROI_Y_MIN}, ${ROI_Z_MIN}, ${ROI_X_MAX}, ${ROI_Y_MAX}, ${ROI_Z_MAX}]"
echo "EVAL_VOXEL_Z=${EVAL_VOXEL_Z} (preserve ${ORIG_Z_BINS} z bins from training config)"
echo "TEST_MAX_VOXELS=${TEST_MAX_VOXELS}"

if [[ "${EPOCHS_TO_EVAL}" == "all" ]]; then
  EPOCHS_TO_EVAL="$(
    find "${CKPT_DIR}" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pth' \
      | sed -E 's#.*checkpoint_epoch_([0-9]+)\.pth#\1#' \
      | sort -n \
      | tr '\n' ' '
  )"
fi

[[ -n "${EPOCHS_TO_EVAL// }" ]] || die "no checkpoint epochs selected for evaluation"
echo "Resolved epochs: ${EPOCHS_TO_EVAL}"

for epoch in ${EPOCHS_TO_EVAL}; do
  ckpt="${CKPT_DIR}/checkpoint_epoch_${epoch}.pth"
  [[ -f "${ckpt}" ]] || { echo "skip epoch ${epoch}: checkpoint not found"; continue; }

  src_result_dir="${TMP_EXP_DIR}/eval/epoch_${epoch}/val/${EVAL_TAG}"
  result_dir="${EXP_DIR}/${RESULT_ROOT_NAME}/epoch_${epoch}/val/${EVAL_TAG}"
  result_pkl="${result_dir}/result.pkl"
  src_result_pkl="${src_result_dir}/result.pkl"

  if [[ -f "${result_pkl}" ]]; then
    echo "----- epoch ${epoch}: ${RESULT_ROOT_NAME} result.pkl exists, skipping test.py -----"
  elif [[ -f "${src_result_pkl}" ]]; then
    echo "----- epoch ${epoch}: reusing existing raw result from ${src_result_pkl} -----"
  else
    echo "----- epoch ${epoch}: running test.py with enlarged ROI -----"
    mkdir -p "${src_result_dir}"
    set +e
    python test.py \
      --cfg_file "${TMP_CFG}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --workers "${WORKERS}" \
      --extra_tag "${TRAIN_TAG}" \
      --output_dir "${OUTPUT_ROOT}" \
      --ckpt "${ckpt}" \
      --eval_tag "${EVAL_TAG}" \
      --save_to_file
    test_status=$?
    set -e
    if [[ ! -f "${src_result_pkl}" ]]; then
      die "test.py status=${test_status}, but result.pkl was not produced: ${src_result_pkl}"
    fi
    if (( test_status != 0 )); then
      echo "test.py exited with status=${test_status}; result.pkl exists, continue with standalone py310 AP."
    else
      echo "test.py done, status=${test_status}"
    fi
  fi

  mkdir -p "$(dirname "${result_dir}")"
  rm -rf "${result_dir}"
  cp -a "${src_result_dir}" "${result_dir}"

  echo "----- epoch ${epoch}: computing AP -----"
  "${PY310}" eval_kitti_result_from_pkl.py \
    --info_pkl "${INFO_PKL}" \
    --result_pkl "${result_pkl}" \
    --output "${result_dir}/official_eval_py310.txt"
  echo "----- epoch ${epoch} done -----"
  echo ""
done

echo "========== all evaluations complete =========="
echo "Results directory: ${EXP_DIR}/${RESULT_ROOT_NAME}/"
for epoch in ${EPOCHS_TO_EVAL}; do
  result_file="${EXP_DIR}/${RESULT_ROOT_NAME}/epoch_${epoch}/val/${EVAL_TAG}/official_eval_py310.txt"
  if [[ -f "${result_file}" ]]; then
    echo "=== epoch ${epoch} ==="
    cat "${result_file}"
    echo ""
  fi
done
