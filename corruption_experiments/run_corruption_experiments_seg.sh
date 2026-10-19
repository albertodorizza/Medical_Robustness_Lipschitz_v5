#!/bin/bash
# =============================================================================
# run_corruption_experiments_seg.sh
# =============================================================================
# Sweep launcher for corruption_experiments_seg.py
#
# Only the soft-masking variant is launched for every checkpoint and every
# eval dataset, matching run_corruption_experiments_cls.sh for classification.
#
# Usage:
#   chmod +x run_corruption_experiments_seg.sh
#   ./run_corruption_experiments_seg.sh
# =============================================================================

set -euo pipefail

# -- Project root -------------------------------------------------------------
# `python3 -m <package>.<module>` only resolves from the directory that holds
# the packages.  Walk up from this script until that directory is found and cd
# there, so the launcher works from anywhere, including from inside its own
# subfolder as the README suggests.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [ ! -d "${PROJECT_ROOT}/finetuning" ] || [ ! -d "${PROJECT_ROOT}/shared" ]; do
    parent="$(dirname "$PROJECT_ROOT")"
    if [ "$parent" = "$PROJECT_ROOT" ]; then
        echo "Cannot find the project root (a directory holding finetuning/ and shared/)." >&2
        exit 1
    fi
    PROJECT_ROOT="$parent"
done
cd "$PROJECT_ROOT"

# ── Paths ─────────────────────────────────────────────────────────────────────
# -- Paths --------------------------------------------------------------------
# Every path is an environment variable with a default, so one `export` reaches
# all eight launchers and nothing has to be edited to move machine.  The three
# below MUST be identical across every launcher of the pipeline.
RESULTS_ROOT="${RESULTS_ROOT:-/home/a.dorizza/OUTPUTS_MEDICAL}"
SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_ROOT}/splits}"
export MEDICAL_ROOT="${MEDICAL_ROOT:-/home/datasets/medical}"
# Read by the interpreter at startup, so it has to be exported here rather than
# set from inside reset_seed().
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# -- GPU scheduling -------------------------------------------------------------
# No jobs-per-GPU constant here.  shared/gpu_pool.sh admits a job only when the
# card has room for it NOW and would still have room LATER, if every job this
# launcher has running there grew to its peak.  The concurrency is therefore
# derived from the real size of the card and from what these jobs actually use,
# instead of a hand-tuned constant that is wrong in both directions: too low
# and most of an H200 sits idle, too high and the run dies hours in.
#
# The per-job peak calibrates itself off the first job, which runs alone, and
# is revised upward whenever a job is seen using more than the estimate -- that
# is what covers jobs whose memory grows as they run.  Tune with
# GPU_POOL_HEADROOM_MB (memory never handed out, per card), GPU_POOL_GROWTH
# (safety factor on the measured peak) or GPU_POOL_PEAK_MB (pin it outright).
source "${PROJECT_ROOT}/shared/gpu_pool.sh"


# ── Dataset grid: every trainable segmentation dataset ────────────────────────
DATASETS=("kvasir_seg" "cvc_clinicdb" "cvc_colondb" "etis_larib" "aircare_all")
POLYP_FAMILY=("kvasir_seg" "cvc_clinicdb" "cvc_colondb" "etis_larib")

declare -A ARCH_MAP=([resnet]=resnet34 [vit]=vit_b_16 [dinov2]=dinov2)

# ── Corruption sweep parameters ───────────────────────────────────────────────
CORRUPTION_TYPES="circular superpixel squared"
PERTURBATION_RATIOS="0.3 0.6 0.9"
SEVERITY_LEVELS="1 3 5"
SQUARED_SIZES="32"        # must divide 224 exactly
PERTURBATION_TYPES='{"NOISE": ["Gaussian Noise"], "BLUR": ["Defocus Blur"], "COLOR": ["Brightness", "Contrast"]}'
METRIC_NAMES="mse cos"

BATCH_SIZE=32
NUM_WORKERS=8
SEED=29                             # corruption masks / noise, NOT split_seed

# ── Speed knobs ───────────────────────────────────────────────────────────────
# COMBO_CHUNK combinations share one pass over the data: the image is decoded
# and the clean forward pass is run once per chunk instead of once per
# combination.  Higher is faster but keeps COMBO_CHUNK corrupted copies of each
# batch in memory, so lower it (or lower BATCH_SIZE) if the host runs short.
# AMP runs the forward passes in bfloat16/float16; metrics stay float32.
COMBO_CHUNK=4
USE_AMP=true

# Per-worker LRU of SLIC segmentations. The segmentation of an image does not
# depend on ratio, severity or perturbation type, so every superpixel
# combination of a chunk reuses it. 0 disables the cache.
export SLIC_CACHE_SIZE=256

# ── Depth profile ─────────────────────────────────────────────────────────────
# HOOK_DEPTH=1 also measures the feature drift at every intermediate block of
# the backbone, not only at the pre-head feature map, so the depth study that
# exists for classification can be produced for segmentation too.  Off by
# default: it multiplies the metric work by the number of hooked layers and
# makes each metrics.json about an order of magnitude larger.
HOOK_DEPTH="${HOOK_DEPTH:-0}"
HOOK_FLAG=""
[[ "${HOOK_DEPTH}" == "1" ]] && HOOK_FLAG="--hook_depth"

# ── Masking strategy ──────────────────────────────────────────────────────────
# Only soft masking is launched, matching run_corruption_experiments_cls.py.
# This is also the python default; pass --no-soft_masking for the binary one.
SOFT_MASKING_FLAG="--soft_masking"
VARIANT_DIR="rawfeat_softmasking"

# -- Resume --------------------------------------------------------------------
# A rerun launches only what has not already produced a complete result, so the
# whole chain can be restarted after a crash, a preemption or an added seed
# without redoing days of work.  FORCE=1 redoes everything.
#
# "Complete" means the result file exists AND is non-empty: a job killed while
# writing leaves a zero-byte file, and treating that as done would put a silent
# hole in the analysis.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

USE_WANDB=true
WANDB_MODE="online"

GPU_POOL_DEVICES="${GPU_POOL_DEVICES:-0 1 2 3}"

# -- Zero-shot evaluation ------------------------------------------------------
# OFF by default: each checkpoint is evaluated only on the held-out test split
# of the dataset it was trained on, so a training set produces exactly one
# corruption run, named "<run>_on_<that same dataset>_rawfeat_softmasking".
#
# Turning this on multiplies the job count by the size of the polyp family
# (4 training sets x 3 externals = 12 extra runs per checkpoint, 108 corruption
# combinations each) and answers a different question: how a model degrades
# under corruption on data from another centre. Worth running, but as its own
# experiment rather than as a side effect of the in-distribution sweep.
CORRUPTION_INCLUDE_EXTERNAL="${CORRUPTION_INCLUDE_EXTERNAL:-0}"

# external_datasets_for TRAIN_DATASET -> prints the other members of its family
external_datasets_for() {
    local train=$1
    [ "$CORRUPTION_INCLUDE_EXTERNAL" = "1" ] || return 0
    local family=()
    case "$train" in
        aircare_all) family=() ;;                # pooled: no external set here
        *)           family=("${POLYP_FAMILY[@]}") ;;
    esac
    for ds in "${family[@]}"; do
        [ "$ds" != "$train" ] && echo "$ds"
    done
}

# launch_one TRAIN_DS EVAL_DS SHORT ARCH CKPT SPLIT_SEED EVAL_SPLIT OUTDIR GPU [--soft_masking]
launch_one() {
    local TRAIN_DS=$1 EVAL_DS=$2 SHORT=$3 ARCH=$4 CKPT=$5 SPLIT_SEED=$6 EVAL_SPLIT=$7 OUTDIR=$8 GPU=$9
    local SOFT_FLAG="${10:-}"

    local VARIANT="rawfeat"
    [[ "$SOFT_FLAG" == "--soft_masking" ]] && VARIANT="rawfeat_softmasking"

    local WANDB_FLAGS=""
    [[ "$USE_WANDB" == "true" ]] && WANDB_FLAGS="--use_wandb --wandb_mode ${WANDB_MODE}"

    local AMP_FLAG="--no-amp"
    [[ "$USE_AMP" == "true" ]] && AMP_FLAG="--amp"

    echo "Launching: ${TRAIN_DS}/${SHORT} -> eval on ${EVAL_DS} (${EVAL_SPLIT}) [${VARIANT}]  ->  GPU ${GPU}"

    CUDA_VISIBLE_DEVICES="${GPU}" nohup python3 -m corruption_experiments.corruption_experiments_seg \
        --arch                "$ARCH" \
        --checkpoint          "$CKPT" \
        --dataset             "$EVAL_DS" \
        --eval_split          "$EVAL_SPLIT" \
        --output_dir          "$OUTDIR" \
        --corruption_types    ${CORRUPTION_TYPES} \
        --perturbation_ratios ${PERTURBATION_RATIOS} \
        --severity_levels     ${SEVERITY_LEVELS} \
        --squared_sizes       ${SQUARED_SIZES} \
        --perturbation_types  "${PERTURBATION_TYPES}" \
        --metric_names        ${METRIC_NAMES} \
        --batch_size          "${BATCH_SIZE}" \
        --num_workers         "${NUM_WORKERS}" \
        --combo_chunk         "${COMBO_CHUNK}" \
        ${AMP_FLAG} \
        --gpu                 0 \
        --seed                "${SEED}" \
        --split_dir           "${SPLIT_ROOT}" \
        --split_seed          "${SPLIT_SEED}" \
        ${WANDB_FLAGS} \
        ${HOOK_FLAG} \
        ${SOFT_FLAG} \
        > "${OUTDIR}/log_${SHORT}_on_${EVAL_DS}_${VARIANT}.out" 2>&1 &

    gpu_pool_register $!
}

N_SKIPPED=0
N_LAUNCHED=0
gpu_pool_init

echo "Datasets     : ${DATASETS[*]}"
echo "Zero-shot    : $([ "$CORRUPTION_INCLUDE_EXTERNAL" = "1" ] \
    && echo "on (in-distribution + external)" \
    || echo "off (in-distribution only)  -- set CORRUPTION_INCLUDE_EXTERNAL=1 to enable")"
echo ""

for dataset in "${DATASETS[@]}"; do
    CKPT_BASE="${RESULTS_ROOT}/finetuning/segmentation/${dataset}"
    OUTPUT_DIR="${RESULTS_ROOT}/corruption_experiments/segmentation/${dataset}"
    mkdir -p "$OUTPUT_DIR"

    [ -d "$CKPT_BASE" ] || { echo "  [skip] no checkpoints under ${CKPT_BASE}"; continue; }

    # `|| true`: external_datasets_for prints nothing when zero-shot is off,
    # and an empty EXTERNALS array is exactly what the loop below should see.
    EXTERNALS=()
    while IFS= read -r _ds; do
        [ -n "$_ds" ] && EXTERNALS+=("$_ds")
    done < <(external_datasets_for "$dataset")

    while IFS= read -r CKPT; do
        RUN_DIR=$(basename "$(dirname "$CKPT")")
        SHORT=$(basename "$CKPT" .pth)
        ARCH_SHORT="${SHORT%%_*}"
        ARCH="${ARCH_MAP[$ARCH_SHORT]:-}"
        [ -z "$ARCH" ] && { echo "  [skip] unrecognised arch in ${CKPT}"; continue; }

        # Recover split_seed from a trailing "_seedN"; no suffix = seed 0.
        # The python side then writes the suffix out explicitly, so even a
        # checkpoint folder without it produces a "..._seed0_..." output name.
        SPLIT_SEED=0
        if [[ "$RUN_DIR" =~ _seed([0-9]+)$ ]]; then
            SPLIT_SEED="${BASH_REMATCH[1]}"
        fi

        # In-distribution + every external dataset in the family.
        for EVAL_DS in "$dataset" "${EXTERNALS[@]}"; do
            if [ "$EVAL_DS" = "$dataset" ]; then EVAL_SPLIT="test"; else EVAL_SPLIT="all"; fi

            # corruption_experiments_seg.py writes
            # <output_dir>/<ckpt folder>_on_<eval ds>_<variant>/metrics.json
            DONE_FILE="${OUTPUT_DIR}/${RUN_DIR}_on_${EVAL_DS}_${VARIANT_DIR}/metrics.json"
            if [ "$SKIP_EXISTING" = "1" ] && [ "${FORCE:-0}" != "1" ] && [ -s "$DONE_FILE" ]; then
                echo "Skipping (already done): ${dataset}/${RUN_DIR} -> ${EVAL_DS}"
                N_SKIPPED=$(( N_SKIPPED + 1 ))
                continue
            fi

            N_LAUNCHED=$(( N_LAUNCHED + 1 ))
            gpu_pool_acquire
            GPU="$GPU_POOL_SELECTED"
            launch_one "$dataset" "$EVAL_DS" "${RUN_DIR}_${SHORT}" "$ARCH" "$CKPT" \
                       "$SPLIT_SEED" "$EVAL_SPLIT" "$OUTPUT_DIR" "$GPU" "$SOFT_MASKING_FLAG"
            sleep 1
        done
    done < <(find "$CKPT_BASE" -type f -name "*.pth" | sort)
done

echo ""
echo "Launched ${N_LAUNCHED}, skipped ${N_SKIPPED} already-complete run(s)."
echo "All jobs launched (queued or running). Waiting for completion..."
gpu_pool_wait || true
echo "All segmentation corruption experiments are done."
