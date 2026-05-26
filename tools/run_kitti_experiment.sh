#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

sanitize_positive_int_env() {
  local name="$1"
  local fallback="$2"
  local current="${!name:-}"
  if [[ ! "${current}" =~ ^[1-9][0-9]*$ ]]; then
    export "${name}=${fallback}"
  fi
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

usage() {
  cat <<'EOF'
Usage:
  run_kitti_experiment.sh <action> --cfg-file <path> [options] [positional-overrides]
  run_kitti_experiment.sh <action> <cfg-file> [options] [positional-overrides]
  run_kitti_experiment.sh <action> <preset> [options] [positional-overrides]

Actions:
  train           Train or resume a single-stage KITTI experiment
  staged-train    Run the staged training flow for experiments that support it
  eval            Evaluate one checkpoint and compute official KITTI AP
  eval-all        Evaluate every saved checkpoint in a run
  ap              Compute official KITTI AP from an existing result.pkl
  analyze         Compare an eval directory against a baseline eval txt
  posttrain-eval  Run eval-all, then optional analyze / guided-reference compare
  render-tables   Render eval summary tables from an eval directory

Known presets:
  baseline
  ground-guided-diffusion
  ground-defect-guidance
  ground-context-film
  ground-adapter

Core options:
  --cfg-file <path>   Preferred primary input. Use a config file under tools/, e.g. ./cfgs/kitti_models/xxx.yaml
  --preset <name>     Optional preset for known experiment families. Mostly needed for staged-train or shorthand defaults.

Positional overrides after action/options:
  train [extra_tag]
  staged-train [base_tag]
  eval <ckpt_path> [extra_tag]
  eval-all [extra_tag]
  ap <result_pkl> [output_txt]
  analyze [run_eval_dir] [baseline_eval_file] [output_json]
  posttrain-eval [extra_tag]
  render-tables [eval_dir] [metric]

Important env vars:
  DATA_PATH, OUTPUT_ROOT, EXTRA_TAG, VARIANT
  TRAIN_BATCH_SIZE, TRAIN_WORKERS, TOTAL_EPOCHS
  TRAIN_LR, GRAD_NORM_CLIP, PRETRAINED_CKPT, RESUME_CKPT
  FIX_RANDOM_SEED=1 by default
  EVAL_TAG, EVAL_BATCH_SIZE, EVAL_WORKERS, CKPT_PATH, CKPT_DIR
  START_EPOCH, END_EPOCH, SKIP_EXISTING, GENERATE_TABLES, TABLE_METRICS
  BASELINE_EVAL_FILE, GUIDED_REFERENCE_JSON, COMPARE_GUIDED_REFERENCE
  DRY_RUN=1 prints commands without executing them
EOF
}

ACTION="${1:-}"

if [[ -z "${ACTION}" || "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 0
fi

canonicalize_experiment() {
  case "$1" in
    baseline)
      echo "baseline"
      ;;
    ground-guided-diffusion|ground_guided_diffusion|guided)
      echo "ground-guided-diffusion"
      ;;
    ground-defect-guidance|ground_defect_guidance|defect)
      echo "ground-defect-guidance"
      ;;
    ground-context-film|ground_context_film|film)
      echo "ground-context-film"
      ;;
    ground-adapter|ground_adapter|adapter)
      echo "ground-adapter"
      ;;
    *)
      die "Unknown KITTI experiment preset: $1"
      ;;
  esac
}

looks_like_cfg_path() {
  local value="$1"
  [[ "${value}" == *.yaml || "${value}" == *.yml || "${value}" == */* ]]
}

normalize_cfg_file() {
  local raw="$1"
  local abs rel
  abs="$(realpath "${raw}")"
  case "${abs}" in
    "$(pwd)"/*)
      rel="$(realpath --relative-to="$(pwd)" "${abs}")"
      printf './%s' "${rel}"
      ;;
    *)
      die "CFG_FILE must live under $(pwd) so train/test output paths stay stable: ${raw}"
      ;;
  esac
}

infer_experiment_from_cfg() {
  local cfg_basename
  cfg_basename="$(basename "$1")"
  case "${cfg_basename}" in
    second_with_lion_mamba_64dim.yaml)
      echo "baseline"
      ;;
    second_with_lion_mamba_64dim_ground_guided_diffusion.yaml)
      echo "ground-guided-diffusion"
      ;;
    second_with_lion_mamba_64dim_ground_defect_guidance.yaml)
      echo "ground-defect-guidance"
      ;;
    second_with_lion_mamba_64dim_ground_context_film.yaml)
      echo "ground-context-film"
      ;;
    second_with_lion_mamba_64dim_ground_adapter.yaml)
      echo "ground-adapter"
      ;;
    *)
      echo "custom"
      ;;
  esac
}

shift

CFG_FILE_INPUT=""
PRESET_INPUT=""
POSITIONAL_ARGS=()

if [[ $# -gt 0 && "${1}" != --* ]]; then
  if looks_like_cfg_path "${1}"; then
    CFG_FILE_INPUT="${1}"
  else
    PRESET_INPUT="${1}"
  fi
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg-file)
      [[ $# -ge 2 ]] || die "--cfg-file requires a value"
      CFG_FILE_INPUT="$2"
      shift 2
      ;;
    --preset|--experiment)
      [[ $# -ge 2 ]] || die "--preset requires a value"
      PRESET_INPUT="$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL_ARGS+=("$1")
        shift
      done
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${PRESET_INPUT}" ]]; then
  EXPERIMENT="$(canonicalize_experiment "${PRESET_INPUT}")"
else
  EXPERIMENT=""
fi

if [[ -n "${CFG_FILE_INPUT}" ]]; then
  CFG_FILE_INPUT="$(normalize_cfg_file "${CFG_FILE_INPUT}")"
  if [[ -z "${EXPERIMENT}" ]]; then
    EXPERIMENT="$(infer_experiment_from_cfg "${CFG_FILE_INPUT}")"
  fi
fi

if [[ -z "${EXPERIMENT}" ]]; then
  die "Please provide --cfg-file <path> or a known preset"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
sanitize_positive_int_env OMP_NUM_THREADS 8
sanitize_positive_int_env MKL_NUM_THREADS 8

DRY_RUN="${DRY_RUN:-0}"
SAVE_TO_FILE="${SAVE_TO_FILE:-0}"
GENERATE_TABLES="${GENERATE_TABLES:-1}"
TABLE_METRICS="${TABLE_METRICS:-bbox bev 3d}"
METRIC="${METRIC:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
START_EPOCH="${START_EPOCH:-1}"
END_EPOCH="${END_EPOCH:-999999}"
FIX_RANDOM_SEED="${FIX_RANDOM_SEED:-1}"
FP16="${FP16:-0}"
DISABLE_GT_SAMPLING_FOR_SUBSET="${DISABLE_GT_SAMPLING_FOR_SUBSET:-1}"
ENABLE_EMA="${ENABLE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
SAVE_EMA_AS_MODEL="${SAVE_EMA_AS_MODEL:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/project/LION/output/LION_output}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/kitti-offical}"
INFO_PKL="${INFO_PKL:-${DATA_PATH}/kitti_infos_val.pkl}"
TRAIN_SPLIT_NAME="${TRAIN_SPLIT_NAME:-}"
TRAIN_INFO_PKL="${TRAIN_INFO_PKL:-}"
TEST_SPLIT_NAME="${TEST_SPLIT_NAME:-}"
TEST_INFO_PKL="${TEST_INFO_PKL:-}"
EXTRA_SET_CFGS="${EXTRA_SET_CFGS:-}"
PY310="${PY310:-/root/miniconda3/envs/lion_eval_py310/bin/python}"

CFG_FILE=""
CFG_TAG=""
CFG_GROUP_PATH=""
DEFAULT_VARIANT=""
DEFAULT_EXTRA_TAG=""
DEFAULT_TRAIN_BATCH_SIZE=""
DEFAULT_TRAIN_WORKERS=""
DEFAULT_TOTAL_EPOCHS=""
DEFAULT_CKPT_SAVE_INTERVAL=""
DEFAULT_MAX_CKPT_SAVE_NUM=""
DEFAULT_LOGGER_ITER_INTERVAL=""
DEFAULT_CKPT_SAVE_TIME_INTERVAL="1800"
DEFAULT_TRAIN_LR=""
DEFAULT_GRAD_NORM_CLIP=""
DEFAULT_EVAL_AFTER_TRAIN="0"
DEFAULT_POSTTRAIN_ANALYZE="0"
DEFAULT_COMPARE_GUIDED_REFERENCE="0"
DEFAULT_EVAL_BATCH_SIZE="10"
DEFAULT_EVAL_WORKERS="8"

configure_experiment() {
  case "${EXPERIMENT}" in
    baseline)
      CFG_FILE="${CFG_FILE_INPUT:-./cfgs/kitti_models/second_with_lion_mamba_64dim.yaml}"
      DEFAULT_EXTRA_TAG="lion_mamba_kitti_baseline_from_scratch_bs4_e40_ckpt4"
      DEFAULT_TRAIN_BATCH_SIZE="4"
      DEFAULT_TRAIN_WORKERS="8"
      DEFAULT_TOTAL_EPOCHS="40"
      DEFAULT_CKPT_SAVE_INTERVAL="4"
      DEFAULT_MAX_CKPT_SAVE_NUM="50"
      DEFAULT_LOGGER_ITER_INTERVAL="20"
      DEFAULT_TRAIN_LR="0.0002"
      DEFAULT_GRAD_NORM_CLIP="2"
      DEFAULT_EVAL_AFTER_TRAIN="1"
      ;;
    ground-guided-diffusion)
      CFG_FILE="${CFG_FILE_INPUT:-./cfgs/kitti_models/second_with_lion_mamba_64dim_ground_guided_diffusion.yaml}"
      DEFAULT_EXTRA_TAG="lion_mamba_kitti_ground_guided_from_scratch_bs4_e80"
      DEFAULT_TRAIN_BATCH_SIZE="4"
      DEFAULT_TRAIN_WORKERS="8"
      DEFAULT_TOTAL_EPOCHS="80"
      DEFAULT_CKPT_SAVE_INTERVAL="2"
      DEFAULT_MAX_CKPT_SAVE_NUM="50"
      DEFAULT_LOGGER_ITER_INTERVAL="20"
      DEFAULT_TRAIN_LR="0.0002"
      DEFAULT_GRAD_NORM_CLIP="2"
      DEFAULT_VARIANT="response_plus_learned_trust"
      ;;
    ground-defect-guidance)
      CFG_FILE="${CFG_FILE_INPUT:-./cfgs/kitti_models/second_with_lion_mamba_64dim_ground_defect_guidance.yaml}"
      DEFAULT_VARIANT="defect_branch_with_fusion"
      DEFAULT_EXTRA_TAG="${DEFAULT_VARIANT}"
      DEFAULT_TRAIN_BATCH_SIZE="4"
      DEFAULT_TRAIN_WORKERS="4"
      DEFAULT_TOTAL_EPOCHS="40"
      DEFAULT_CKPT_SAVE_INTERVAL="4"
      DEFAULT_MAX_CKPT_SAVE_NUM="10"
      DEFAULT_LOGGER_ITER_INTERVAL="50"
      DEFAULT_POSTTRAIN_ANALYZE="1"
      DEFAULT_COMPARE_GUIDED_REFERENCE="1"
      ;;
    ground-context-film)
      CFG_FILE="${CFG_FILE_INPUT:-./cfgs/kitti_models/second_with_lion_mamba_64dim_ground_context_film.yaml}"
      DEFAULT_VARIANT="full_film"
      DEFAULT_EXTRA_TAG="${DEFAULT_VARIANT}"
      DEFAULT_TRAIN_BATCH_SIZE="4"
      DEFAULT_TRAIN_WORKERS="4"
      DEFAULT_TOTAL_EPOCHS="40"
      DEFAULT_CKPT_SAVE_INTERVAL="4"
      DEFAULT_MAX_CKPT_SAVE_NUM="10"
      DEFAULT_LOGGER_ITER_INTERVAL="50"
      DEFAULT_EVAL_AFTER_TRAIN="1"
      DEFAULT_POSTTRAIN_ANALYZE="1"
      ;;
    ground-adapter)
      CFG_FILE="${CFG_FILE_INPUT:-./cfgs/kitti_models/second_with_lion_mamba_64dim_ground_adapter.yaml}"
      DEFAULT_EXTRA_TAG="lion_mamba_kitti_ground_adapter"
      DEFAULT_TRAIN_BATCH_SIZE="4"
      DEFAULT_TRAIN_WORKERS="8"
      DEFAULT_TOTAL_EPOCHS="8"
      DEFAULT_CKPT_SAVE_INTERVAL="1"
      DEFAULT_MAX_CKPT_SAVE_NUM="50"
      DEFAULT_LOGGER_ITER_INTERVAL="20"
      DEFAULT_TRAIN_LR="0.0005"
      DEFAULT_GRAD_NORM_CLIP="2"
      ;;
    custom)
      [[ -n "${CFG_FILE_INPUT}" ]] || die "Custom cfg mode requires --cfg-file"
      CFG_FILE="${CFG_FILE_INPUT}"
      DEFAULT_EXTRA_TAG="$(basename "${CFG_FILE_INPUT}" .yaml)"
      DEFAULT_TRAIN_BATCH_SIZE="4"
      DEFAULT_TRAIN_WORKERS="4"
      DEFAULT_TOTAL_EPOCHS="40"
      DEFAULT_CKPT_SAVE_INTERVAL="1"
      DEFAULT_MAX_CKPT_SAVE_NUM="30"
      DEFAULT_LOGGER_ITER_INTERVAL="20"
      ;;
  esac

  CFG_TAG="$(basename "${CFG_FILE}" .yaml)"
  CFG_GROUP_PATH="$(dirname "${CFG_FILE#./}")"
}

configure_experiment

VARIANT="${VARIANT:-${DEFAULT_VARIANT}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
TRAIN_WORKERS="${TRAIN_WORKERS:-${DEFAULT_TRAIN_WORKERS}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${DEFAULT_TOTAL_EPOCHS}}"
STOP_EPOCHS="${STOP_EPOCHS:-${TOTAL_EPOCHS}}"
CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-${DEFAULT_CKPT_SAVE_INTERVAL}}"
MAX_CKPT_SAVE_NUM="${MAX_CKPT_SAVE_NUM:-${DEFAULT_MAX_CKPT_SAVE_NUM}}"
LOGGER_ITER_INTERVAL="${LOGGER_ITER_INTERVAL:-${DEFAULT_LOGGER_ITER_INTERVAL}}"
CKPT_SAVE_TIME_INTERVAL="${CKPT_SAVE_TIME_INTERVAL:-${DEFAULT_CKPT_SAVE_TIME_INTERVAL}}"
TRAIN_LR="${TRAIN_LR:-${DEFAULT_TRAIN_LR}}"
GRAD_NORM_CLIP="${GRAD_NORM_CLIP:-${DEFAULT_GRAD_NORM_CLIP}}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-${DEFAULT_EVAL_AFTER_TRAIN}}"
POSTTRAIN_ANALYZE="${POSTTRAIN_ANALYZE:-${DEFAULT_POSTTRAIN_ANALYZE}}"
COMPARE_GUIDED_REFERENCE="${COMPARE_GUIDED_REFERENCE:-${DEFAULT_COMPARE_GUIDED_REFERENCE}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${DEFAULT_EVAL_BATCH_SIZE}}"
EVAL_WORKERS="${EVAL_WORKERS:-${DEFAULT_EVAL_WORKERS}}"
EVAL_TAG="${EVAL_TAG:-py310_ap}"

DEFAULT_BASELINE_EVAL_FILE="${OUTPUT_ROOT}/${CFG_GROUP_PATH}/second_with_lion_mamba_64dim/lion_mamba_kitti_baseline_from_scratch_bs4_e40_ckpt4/eval/epoch_40/val/${EVAL_TAG}/official_eval_ap_r40.txt"
BASELINE_EVAL_FILE="${BASELINE_EVAL_FILE:-${DEFAULT_BASELINE_EVAL_FILE}}"
GUIDED_REFERENCE_JSON="${GUIDED_REFERENCE_JSON:-/root/project/LION/.planning/2026-05-19-kitti-lightweight-ground-defect-guidance/guided_vs_baseline_epoch40.json}"

run_dir_for_tag() {
  local tag="$1"
  printf '%s/%s/%s/%s' "${OUTPUT_ROOT}" "${CFG_GROUP_PATH}" "${CFG_TAG}" "${tag}"
}

eval_dir_for_tag() {
  local tag="$1"
  printf '%s/eval' "$(run_dir_for_tag "${tag}")"
}

ckpt_dir_for_tag() {
  local tag="$1"
  printf '%s/ckpt' "$(run_dir_for_tag "${tag}")"
}

require_file() {
  local path="$1"
  if [[ "${DRY_RUN}" != "1" && ! -f "${path}" ]]; then
    die "Missing file: ${path}"
  fi
}

require_dir() {
  local path="$1"
  if [[ "${DRY_RUN}" != "1" && ! -d "${path}" ]]; then
    die "Missing directory: ${path}"
  fi
}

uses_train_subset() {
  [[ -n "${TRAIN_SPLIT_NAME}" && "${TRAIN_SPLIT_NAME}" != "train" ]] && return 0
  [[ -n "${TRAIN_INFO_PKL}" && "${TRAIN_INFO_PKL}" != "kitti_infos_train.pkl" ]] && return 0
  return 1
}

append_context_film_variant_set_args() {
  case "${VARIANT}" in
    ""|full_film)
      ;;
    ground_context_only)
      SET_ARGS+=(MODEL.BACKBONE_3D.GROUND_CONTEXT_FILM.MODE beta_only)
      ;;
    point_features_only)
      SET_ARGS+=(
        MODEL.VFE.GROUND_CONTEXT_ENABLED False
        MODEL.BACKBONE_3D.GROUND_CONTEXT_FILM.ENABLED False
      )
      ;;
    *)
      die "Unknown ground-context-film VARIANT: ${VARIANT}"
      ;;
  esac
}

append_defect_variant_set_args() {
  case "${VARIANT}" in
    ""|defect_branch_with_fusion)
      ;;
    point_features_only)
      SET_ARGS+=(MODEL.MAP_TO_BEV.NAME HeightCompression)
      ;;
    defect_branch_no_fusion)
      SET_ARGS+=(MODEL.MAP_TO_BEV.ENABLE_FUSION False)
      ;;
    *)
      die "Unknown ground-defect-guidance VARIANT: ${VARIANT}"
      ;;
  esac
}

append_guided_train_set_args() {
  local mode="${GROUND_GUIDED_MODE:-${VARIANT:-response_plus_learned_trust}}"
  local learned_alpha_init="${LEARNED_ALPHA_INIT:-0.05}"
  local trust_logit_init="${TRUST_LOGIT_INIT:--4.0}"
  local diffusion_feature_scale_init="${DIFFUSION_FEATURE_SCALE_INIT:-0.001}"
  local force_disable_train_freeze="${FORCE_DISABLE_TRAIN_FREEZE:-1}"
  SET_ARGS+=(
    MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.ABLATION_MODE "${mode}"
    MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.LEARNED_ALPHA_INIT "${learned_alpha_init}"
    MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.TRUST_LOGIT_INIT "${trust_logit_init}"
    MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.DIFFUSION_FEATURE_SCALE_INIT "${diffusion_feature_scale_init}"
  )
  if [[ "${force_disable_train_freeze}" == "1" ]]; then
    SET_ARGS+=(TRAIN_FREEZE.ENABLED False)
  fi
}

append_guided_eval_set_args() {
  local mode="${GROUND_GUIDED_MODE:-${VARIANT:-response_plus_learned_trust}}"
  if [[ -n "${mode}" ]]; then
    SET_ARGS+=(MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.ABLATION_MODE "${mode}")
  fi
}

append_train_set_args() {
  SET_ARGS=(DATA_CONFIG.DATA_PATH "${DATA_PATH}")
  if [[ -n "${TRAIN_SPLIT_NAME}" ]]; then
    SET_ARGS+=(DATA_CONFIG.DATA_SPLIT.train "${TRAIN_SPLIT_NAME}")
  fi
  if [[ -n "${TRAIN_INFO_PKL}" ]]; then
    SET_ARGS+=(DATA_CONFIG.INFO_PATH.train "['${TRAIN_INFO_PKL}']")
  fi
  if [[ "${DISABLE_GT_SAMPLING_FOR_SUBSET}" == "1" ]] && uses_train_subset; then
    echo "Subset train data detected; disabling gt_sampling. Set DISABLE_GT_SAMPLING_FOR_SUBSET=0 to override."
    SET_ARGS+=(DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST "['gt_sampling']")
  fi
  if [[ -n "${TEST_SPLIT_NAME}" ]]; then
    SET_ARGS+=(DATA_CONFIG.DATA_SPLIT.test "${TEST_SPLIT_NAME}")
  fi
  if [[ -n "${TEST_INFO_PKL}" ]]; then
    SET_ARGS+=(DATA_CONFIG.INFO_PATH.test "['${TEST_INFO_PKL}']")
  fi
  if [[ -n "${TRAIN_LR}" ]]; then
    SET_ARGS+=(OPTIMIZATION.LR "${TRAIN_LR}")
  fi
  if [[ -n "${GRAD_NORM_CLIP}" ]]; then
    SET_ARGS+=(OPTIMIZATION.GRAD_NORM_CLIP "${GRAD_NORM_CLIP}")
  fi

  case "${EXPERIMENT}" in
    baseline)
      if [[ "${FORCE_DISABLE_TRAIN_FREEZE:-1}" == "1" ]] && grep -q '^TRAIN_FREEZE:' "${CFG_FILE}"; then
        SET_ARGS+=(TRAIN_FREEZE.ENABLED False)
      fi
      ;;
    ground-guided-diffusion)
      append_guided_train_set_args
      ;;
    ground-defect-guidance)
      append_defect_variant_set_args
      ;;
    ground-context-film)
      append_context_film_variant_set_args
      ;;
    ground-adapter)
      ;;
  esac
}

append_eval_set_args() {
  SET_ARGS=(DATA_CONFIG.DATA_PATH "${DATA_PATH}")
  if [[ -n "${TRAIN_SPLIT_NAME}" ]]; then
    SET_ARGS+=(DATA_CONFIG.DATA_SPLIT.train "${TRAIN_SPLIT_NAME}")
  fi
  if [[ -n "${TRAIN_INFO_PKL}" ]]; then
    SET_ARGS+=(DATA_CONFIG.INFO_PATH.train "['${TRAIN_INFO_PKL}']")
  fi
  if [[ -n "${TEST_SPLIT_NAME}" ]]; then
    SET_ARGS+=(DATA_CONFIG.DATA_SPLIT.test "${TEST_SPLIT_NAME}")
  fi
  if [[ -n "${TEST_INFO_PKL}" ]]; then
    SET_ARGS+=(DATA_CONFIG.INFO_PATH.test "['${TEST_INFO_PKL}']")
  fi
  case "${EXPERIMENT}" in
    ground-guided-diffusion)
      append_guided_eval_set_args
      ;;
    ground-defect-guidance)
      append_defect_variant_set_args
      ;;
    ground-context-film)
      append_context_film_variant_set_args
      ;;
  esac
}

append_extra_set_args() {
  if [[ -z "${EXTRA_SET_CFGS}" ]]; then
    return
  fi
  local -a extra_args=()
  read -r -a extra_args <<< "${EXTRA_SET_CFGS}"
  SET_ARGS+=("${extra_args[@]}")
}

derive_extra_tag_from_ckpt() {
  local ckpt_path="$1"
  basename "$(dirname "$(dirname "${ckpt_path}")")"
}

derive_epoch_from_ckpt() {
  local ckpt_path="$1"
  local filename
  filename="$(basename "${ckpt_path}")"
  local epoch="${filename#checkpoint_epoch_}"
  epoch="${epoch%.pth}"
  if [[ ! "${epoch}" =~ ^[0-9]+$ ]]; then
    die "Checkpoint name must look like checkpoint_epoch_<N>.pth: ${ckpt_path}"
  fi
  printf '%s' "${epoch}"
}

render_eval_tables() {
  local eval_dir="$1"
  local metrics=()

  if [[ "${METRIC}" != "all" ]]; then
    metrics=("${METRIC}")
  else
    read -r -a metrics <<< "${TABLE_METRICS}"
  fi

  for metric_name in "${metrics[@]}"; do
    run_cmd python generate_eval_table_images.py \
      "${eval_dir}" \
      --metric "${metric_name}" \
      --eval-tag "${EVAL_TAG}" \
      --output-dir "${eval_dir}"
  done
}

run_official_ap() {
  local result_pkl="$1"
  local output_txt="$2"

  require_file "${INFO_PKL}"
  require_file "${PY310}"
  require_file "${result_pkl}"

  run_cmd "${PY310}" eval_kitti_result_from_pkl.py \
    --info_pkl "${INFO_PKL}" \
    --result_pkl "${result_pkl}" \
    --output "${output_txt}"
}

run_eval_single() {
  local ckpt_path="$1"
  local extra_tag="$2"
  local epoch result_dir result_pkl official_txt

  require_file "${CFG_FILE}"
  require_dir "${DATA_PATH}"
  require_file "${ckpt_path}"

  epoch="$(derive_epoch_from_ckpt "${ckpt_path}")"
  result_dir="$(eval_dir_for_tag "${extra_tag}")/epoch_${epoch}/val/${EVAL_TAG}"
  result_pkl="${result_dir}/result.pkl"
  official_txt="${result_dir}/official_eval_ap_r40.txt"

  append_eval_set_args
  append_extra_set_args
  CMD=(
    python test.py
    --cfg_file "${CFG_FILE}"
    --ckpt "${ckpt_path}"
    --extra_tag "${extra_tag}"
    --eval_tag "${EVAL_TAG}"
    --output_dir "${OUTPUT_ROOT}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --workers "${EVAL_WORKERS}"
    --skip_dataset_eval
  )
  if [[ "${SAVE_TO_FILE}" == "1" ]]; then
    CMD+=(--save_to_file)
  fi
  CMD+=(--set "${SET_ARGS[@]}")

  run_cmd "${CMD[@]}"

  if [[ "${DRY_RUN}" != "1" && ! -f "${result_pkl}" ]]; then
    die "Validation did not produce result.pkl: ${result_pkl}"
  fi

  run_official_ap "${result_pkl}" "${official_txt}"
}

run_eval_all() {
  local extra_tag="$1"
  local ckpt_dir="${CKPT_DIR:-$(ckpt_dir_for_tag "${extra_tag}")}"
  local eval_dir epoch ckpt official_txt
  local -a ckpts

  require_file "${CFG_FILE}"
  require_dir "${DATA_PATH}"
  require_dir "${ckpt_dir}"

  shopt -s nullglob
  ckpts=("${ckpt_dir}"/checkpoint_epoch_*.pth)
  shopt -u nullglob
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN: no concrete checkpoints discovered under ${ckpt_dir}; skip checkpoint iteration"
      return 0
    fi
    die "No checkpoints found under: ${ckpt_dir}"
  fi

  IFS=$'\n' read -r -d '' -a ckpts < <(printf '%s\n' "${ckpts[@]}" | sort -V && printf '\0')

  for ckpt in "${ckpts[@]}"; do
    epoch="$(derive_epoch_from_ckpt "${ckpt}")"
    if (( epoch < START_EPOCH || epoch > END_EPOCH )); then
      continue
    fi

    eval_dir="$(eval_dir_for_tag "${extra_tag}")"
    official_txt="${eval_dir}/epoch_${epoch}/val/${EVAL_TAG}/official_eval_ap_r40.txt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${official_txt}" ]]; then
      echo "Skip epoch ${epoch}: ${official_txt} already exists"
      continue
    fi

    echo "Evaluating checkpoint epoch ${epoch}: ${ckpt}"
    run_eval_single "${ckpt}" "${extra_tag}"
  done

  if [[ "${GENERATE_TABLES}" == "1" ]]; then
    render_eval_tables "$(eval_dir_for_tag "${extra_tag}")"
  fi
}

run_analyze() {
  local extra_tag="$1"
  local run_eval_dir="${RUN_EVAL_DIR:-$(eval_dir_for_tag "${extra_tag}")}"
  local output_json="${OUTPUT_JSON:-${run_eval_dir}/analysis_vs_baseline_${EVAL_TAG}.json}"

  require_file "/root/project/LION/tools/analyze_kitti_run_vs_baseline.py"
  require_dir "${run_eval_dir}"
  require_file "${BASELINE_EVAL_FILE}"

  run_cmd python analyze_kitti_run_vs_baseline.py \
    --run-eval-dir "${run_eval_dir}" \
    --baseline-eval-file "${BASELINE_EVAL_FILE}" \
    --eval-tag "${EVAL_TAG}" \
    --output "${output_json}"

  if [[ "${COMPARE_GUIDED_REFERENCE}" == "1" && -f "${GUIDED_REFERENCE_JSON}" && "${EXPERIMENT}" == "ground-defect-guidance" ]]; then
    run_cmd python compare_run_with_guided_reference.py \
      --current-analysis "${output_json}" \
      --guided-analysis "${GUIDED_REFERENCE_JSON}" \
      --output "${run_eval_dir}/analysis_vs_old_guided_${EVAL_TAG}.json"
  fi
}

run_posttrain_eval() {
  local extra_tag="$1"
  run_eval_all "${extra_tag}"
  if [[ "${POSTTRAIN_ANALYZE}" == "1" && -f "${BASELINE_EVAL_FILE}" ]]; then
    run_analyze "${extra_tag}"
  fi
}

run_train_once() {
  local extra_tag="$1"
  local total_epochs="$2"
  local stop_epochs="$3"
  local pretrained_ckpt="$4"
  local resume_ckpt="$5"
  shift 5
  local -a extra_set_args=("$@")
  local -a cmd=(
    python train_no_builtin_eval.py
    --cfg_file "${CFG_FILE}"
    --batch_size "${TRAIN_BATCH_SIZE}"
    --workers "${TRAIN_WORKERS}"
    --epochs "${stop_epochs}"
    --total_epochs "${total_epochs}"
    --extra_tag "${extra_tag}"
    --output_dir "${OUTPUT_ROOT}"
    --ckpt_save_interval "${CKPT_SAVE_INTERVAL}"
    --max_ckpt_save_num "${MAX_CKPT_SAVE_NUM}"
    --logger_iter_interval "${LOGGER_ITER_INTERVAL}"
    --ckpt_save_time_interval "${CKPT_SAVE_TIME_INTERVAL}"
  )

  if [[ "${FIX_RANDOM_SEED}" == "1" ]]; then
    cmd+=(--fix_random_seed)
  fi

  if [[ "${FP16}" == "1" ]]; then
    cmd+=(--fp16)
  fi

  if [[ "${ENABLE_EMA}" == "1" ]]; then
    cmd+=(--ema --ema_decay "${EMA_DECAY}")
    if [[ "${SAVE_EMA_AS_MODEL}" == "1" ]]; then
      cmd+=(--save_ema_as_model)
    fi
  fi

  if [[ -n "${pretrained_ckpt}" ]]; then
    require_file "${pretrained_ckpt}"
    cmd+=(--pretrained_model "${pretrained_ckpt}")
  fi

  if [[ -n "${resume_ckpt}" ]]; then
    require_file "${resume_ckpt}"
    cmd+=(--ckpt "${resume_ckpt}")
  fi

  cmd+=(--set "${extra_set_args[@]}")
  run_cmd "${cmd[@]}"
}

run_train_action() {
  local positional_extra_tag="${1:-}"
  local extra_tag="${EXTRA_TAG:-${positional_extra_tag:-${DEFAULT_EXTRA_TAG}}}"

  if [[ -z "${extra_tag}" ]]; then
    die "EXTRA_TAG is required for train"
  fi

  require_file "${CFG_FILE}"
  require_dir "${DATA_PATH}"
  append_train_set_args
  append_extra_set_args

  run_train_once "${extra_tag}" "${TOTAL_EPOCHS}" "${STOP_EPOCHS}" "${PRETRAINED_CKPT:-}" "${RESUME_CKPT:-}" "${SET_ARGS[@]}"

  if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
    run_posttrain_eval "${extra_tag}"
  fi
}

run_staged_ground_adapter() {
  local base_tag="${1:-${BASE_TAG:-lion_mamba_kitti_ground_adapter}}"
  local stage1_tag="${STAGE1_TAG:-${base_tag}_stage1_adapter_only}"
  local stage2_tag="${STAGE2_TAG:-${base_tag}_stage2_bev_head}"
  local stage3_tag="${STAGE3_TAG:-${base_tag}_stage3_full_ft}"
  local stage1_epochs="${STAGE1_EPOCHS:-8}"
  local stage2_epochs="${STAGE2_EPOCHS:-18}"
  local stage3_epochs="${STAGE3_EPOCHS:-4}"
  local stage1_lr="${STAGE1_LR:-0.0005}"
  local stage2_lr="${STAGE2_LR:-0.0002}"
  local stage3_lr="${STAGE3_LR:-0.00005}"
  local baseline_ckpt="${BASELINE_CKPT:-${OUTPUT_ROOT}/cfgs/kitti_models/second_with_lion_mamba_64dim/lion_mamba_kitti_baseline_from_scratch_bs4_e40_ckpt4/ckpt/checkpoint_epoch_40.pth}"
  local stage1_ckpt stage2_ckpt

  echo "Stage 1: adapter only"
  run_train_once "${stage1_tag}" "${stage1_epochs}" "${stage1_epochs}" "${baseline_ckpt}" "" \
    DATA_CONFIG.DATA_PATH "${DATA_PATH}" \
    OPTIMIZATION.LR "${stage1_lr}" \
    OPTIMIZATION.GRAD_NORM_CLIP "2" \
    TRAIN_FREEZE.ENABLED True \
    TRAIN_FREEZE.TRAINABLE_PREFIXES "['map_to_bev_module']"

  stage1_ckpt="$(ckpt_dir_for_tag "${stage1_tag}")/checkpoint_epoch_${stage1_epochs}.pth"
  require_file "${stage1_ckpt}"

  echo "Stage 2: adapter + bev backbone + dense head"
  run_train_once "${stage2_tag}" "${stage2_epochs}" "${stage2_epochs}" "${stage1_ckpt}" "" \
    DATA_CONFIG.DATA_PATH "${DATA_PATH}" \
    OPTIMIZATION.LR "${stage2_lr}" \
    OPTIMIZATION.GRAD_NORM_CLIP "2" \
    TRAIN_FREEZE.ENABLED True \
    TRAIN_FREEZE.TRAINABLE_PREFIXES "['map_to_bev_module','backbone_2d','dense_head']"

  stage2_ckpt="$(ckpt_dir_for_tag "${stage2_tag}")/checkpoint_epoch_${stage2_epochs}.pth"
  require_file "${stage2_ckpt}"

  if (( stage3_epochs > 0 )); then
    echo "Stage 3: full low-lr finetune"
    run_train_once "${stage3_tag}" "${stage3_epochs}" "${stage3_epochs}" "${stage2_ckpt}" "" \
      DATA_CONFIG.DATA_PATH "${DATA_PATH}" \
      OPTIMIZATION.LR "${stage3_lr}" \
      OPTIMIZATION.GRAD_NORM_CLIP "2" \
      TRAIN_FREEZE.ENABLED False
  fi
}

run_staged_ground_guided_diffusion() {
  local base_tag="${1:-${BASE_TAG:-lion_mamba_kitti_ground_guided_diffusion}}"
  local stage1_tag="${STAGE1_TAG:-${base_tag}_stage1_score_only}"
  local stage2_tag="${STAGE2_TAG:-${base_tag}_stage2_full_ft}"
  local stage1_epochs="${STAGE1_EPOCHS:-20}"
  local stage2_epochs="${STAGE2_EPOCHS:-5}"
  local stage1_lr="${STAGE1_LR:-0.0005}"
  local stage2_lr="${STAGE2_LR:-0.0001}"
  local baseline_ckpt="${BASELINE_CKPT:-${OUTPUT_ROOT}/cfgs/kitti_models/second_with_lion_mamba_64dim/lion_mamba_kitti_baseline_from_scratch_bs4_e40_ckpt4/ckpt/checkpoint_epoch_40.pth}"
  local guided_mode="${GROUND_GUIDED_MODE:-${VARIANT:-response_plus_learned_trust}}"
  local eval_stage1_saved_ckpts="${EVAL_STAGE1_SAVED_CKPTS:-1}"
  local stage1_ckpt stage2_ckpt

  echo "Stage 1: train only the new ground-guided score parameters"
  run_train_once "${stage1_tag}" "${stage1_epochs}" "${stage1_epochs}" "${baseline_ckpt}" "" \
    DATA_CONFIG.DATA_PATH "${DATA_PATH}" \
    MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.ABLATION_MODE "${guided_mode}" \
    OPTIMIZATION.LR "${stage1_lr}" \
    OPTIMIZATION.GRAD_NORM_CLIP "2" \
    TRAIN_FREEZE.ENABLED True \
    TRAIN_FREEZE.TRAINABLE_KEYWORDS "['response_proj','prior_alpha_logit','prior_trust_logit','diffusion_feature_scale_logit']"

  stage1_ckpt="$(ckpt_dir_for_tag "${stage1_tag}")/checkpoint_epoch_${stage1_epochs}.pth"
  require_file "${stage1_ckpt}"

  echo "Stage 2: full finetune"
  run_train_once "${stage2_tag}" "${stage2_epochs}" "${stage2_epochs}" "${stage1_ckpt}" "" \
    DATA_CONFIG.DATA_PATH "${DATA_PATH}" \
    MODEL.BACKBONE_3D.GROUND_GUIDED_DIFFUSION.ABLATION_MODE "${guided_mode}" \
    OPTIMIZATION.LR "${stage2_lr}" \
    OPTIMIZATION.GRAD_NORM_CLIP "2" \
    TRAIN_FREEZE.ENABLED False

  stage2_ckpt="$(ckpt_dir_for_tag "${stage2_tag}")/checkpoint_epoch_${stage2_epochs}.pth"
  require_file "${stage2_ckpt}"

  if [[ "${eval_stage1_saved_ckpts}" == "1" ]]; then
    EXTRA_TAG="${stage1_tag}" run_eval_all "${stage1_tag}"
  fi
  run_eval_all "${stage2_tag}"
}

run_staged_train_action() {
  local base_tag="${1:-}"
  case "${EXPERIMENT}" in
    ground-adapter)
      run_staged_ground_adapter "${base_tag}"
      ;;
    ground-guided-diffusion)
      run_staged_ground_guided_diffusion "${base_tag}"
      ;;
    *)
      die "staged-train is not supported for ${EXPERIMENT}"
      ;;
  esac
}

case "${ACTION}" in
  train)
    run_train_action "${POSITIONAL_ARGS[0]:-}"
    ;;
  staged-train)
    run_staged_train_action "${POSITIONAL_ARGS[0]:-}"
    ;;
  eval)
    CKPT_PATH="${CKPT_PATH:-${POSITIONAL_ARGS[0]:-}}"
    if [[ -z "${CKPT_PATH}" ]]; then
      die "eval requires <ckpt_path> or CKPT_PATH"
    fi
    EXTRA_TAG="${EXTRA_TAG:-${POSITIONAL_ARGS[1]:-$(derive_extra_tag_from_ckpt "${CKPT_PATH}")}}"
    run_eval_single "${CKPT_PATH}" "${EXTRA_TAG}"
    ;;
  eval-all)
    EXTRA_TAG="${EXTRA_TAG:-${POSITIONAL_ARGS[0]:-${DEFAULT_EXTRA_TAG}}}"
    if [[ -z "${EXTRA_TAG}" ]]; then
      die "eval-all requires EXTRA_TAG"
    fi
    run_eval_all "${EXTRA_TAG}"
    ;;
  ap)
    RESULT_PKL="${RESULT_PKL:-${POSITIONAL_ARGS[0]:-}}"
    OUTPUT_TXT="${OUTPUT_TXT:-${POSITIONAL_ARGS[1]:-}}"
    if [[ -z "${RESULT_PKL}" ]]; then
      die "ap requires <result_pkl> or RESULT_PKL"
    fi
    if [[ -z "${OUTPUT_TXT}" ]]; then
      OUTPUT_TXT="$(dirname "${RESULT_PKL}")/official_eval_ap_r40.txt"
    fi
    run_official_ap "${RESULT_PKL}" "${OUTPUT_TXT}"
    ;;
  analyze)
    EXTRA_TAG="${EXTRA_TAG:-${DEFAULT_EXTRA_TAG}}"
    RUN_EVAL_DIR="${RUN_EVAL_DIR:-${POSITIONAL_ARGS[0]:-$(eval_dir_for_tag "${EXTRA_TAG}")}}"
    BASELINE_EVAL_FILE="${BASELINE_EVAL_FILE:-${POSITIONAL_ARGS[1]:-${DEFAULT_BASELINE_EVAL_FILE}}}"
    OUTPUT_JSON="${OUTPUT_JSON:-${POSITIONAL_ARGS[2]:-}}"
    run_analyze "${EXTRA_TAG}"
    ;;
  posttrain-eval)
    EXTRA_TAG="${EXTRA_TAG:-${POSITIONAL_ARGS[0]:-${DEFAULT_EXTRA_TAG}}}"
    if [[ -z "${EXTRA_TAG}" ]]; then
      die "posttrain-eval requires EXTRA_TAG"
    fi
    run_posttrain_eval "${EXTRA_TAG}"
    ;;
  render-tables)
    EXTRA_TAG="${EXTRA_TAG:-${DEFAULT_EXTRA_TAG}}"
    RUN_EVAL_DIR="${RUN_EVAL_DIR:-${POSITIONAL_ARGS[0]:-$(eval_dir_for_tag "${EXTRA_TAG}")}}"
    METRIC="${POSITIONAL_ARGS[1]:-${METRIC}}"
    render_eval_tables "${RUN_EVAL_DIR}"
    ;;
  *)
    usage >&2
    die "Unknown action: ${ACTION}"
    ;;
esac
