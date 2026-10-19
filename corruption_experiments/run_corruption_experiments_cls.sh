#!/bin/bash
# =============================================================================
# run_corruption_experiments_cls.sh
# =============================================================================
# Sweep launcher for corruption_experiments_cls.py.
#
# Scans:
#   RESULTS_ROOT/finetuning/classification/<dataset>/<run>/<short>_<freeze>.pth
# for every dataset in DATASETS, every run produced by run_finetuning_cls.sh
# (any architecture, freeze mode, lambda, seed).  --num_classes is left unset
# and read from the checkpoint itself, which is required for Hyper-Kvasir and
# Kvasir-Capsule since their class count is discovered at training time.
#
#
# Only the soft-masking variant is launched (continuous [0,1] weights on the
# projected corruption mask), not the binary-threshold one.
#
# Concurrency is decided by shared/gpu_pool.sh from the free memory of each
# card, rather than by a fixed jobs-per-GPU constant.

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
# instead of a hand-tuned constant that is wrong in both directions: too low and
# most of an H200 sits idle, too high and the run dies hours in.
#
# The per-job peak calibrates itself off the first job, which runs alone, and is
# revised upward whenever a job is seen using more than the estimate -- that is
# what covers jobs whose memory grows as they run.  Tune with
# GPU_POOL_HEADROOM_MB (memory never handed out, per card), GPU_POOL_GROWTH
# (safety factor on the measured peak) or GPU_POOL_PEAK_MB (pin it outright).
source "${PROJECT_ROOT}/shared/gpu_pool.sh"

EXPERIMENT_SCRIPT="corruption_experiments.corruption_experiments_cls"

# ── Dataset grid ──────────────────────────────────────────────────────────────
DATASETS=("kvasir_v2" "hyperkvasir" "kvasir_capsule")

# Short run-folder name -> the --arch value the python script expects.
declare -A ARCH_MAP=([resnet]=resnet34 [vit]=vit_b_16 [dinov2]=dinov2)

# ── Corruption sweep parameters ───────────────────────────────────────────────
INPUT_SIZE="224 224"
CORRUPTION_TYPES="circular superpixel squared"
PERTURBATION_RATIOS="0.3 0.6 0.9"
SEVERITY_LEVELS="1 3 5"
PERTURBATION_TYPES='{"NOISE": ["Gaussian Noise"], "BLUR": ["Defocus Blur"], "COLOR": ["Brightness", "Contrast"]}'
SQUARED_SIZES="32"                  # must divide 224 exactly
METRIC_NAMES="mse cos"

# ── DataLoader config ─────────────────────────────────────────────────────────
BATCH_SIZE=128
NUM_WORKERS=16
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

# ── Masking strategy ──────────────────────────────────────────────────────────
# Only soft masking is launched: continuous [0,1] weights on the projected
# corruption mask, rather than a hard 0.5 threshold.  See
# corruption_experiments/corruption_metrics.py for the definition.
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

# ── WandB ─────────────────────────────────────────────────────────────────────
# Entity and project come from shared/wandb_logging.py; only the mode is a
# per-launcher choice.  Use "offline" on a node without outbound network and
# sync afterwards with `wandb sync <dir>`.
USE_WANDB=true
WANDB_MODE="online"

GPU_POOL_DEVICES="${GPU_POOL_DEVICES:-0 1 2 3}"

# launch_one DATASET SHORT_NAME ARCH CKPT_PATH SPLIT_SEED OUTPUT_DIR GPU [--soft_masking]
#   OUTPUT_DIR is the per-dataset root; corruption_experiments_cls.py appends its
#   own <run>_<rawfeat|normfeat>[_softmasking] folder underneath, matching the
#   layout documented in shared/paths.py -- do NOT pre-create a run subfolder
#   here, or the path gains an extra, unwanted nesting level.
launch_one() {
    local DATASET=$1 SHORT=$2 ARCH=$3 CKPT=$4 SPLIT_SEED=$5 OUTDIR=$6 GPU=$7
    local SOFT_FLAG="${8:-}"

    local VARIANT="rawfeat"
    [[ "$SOFT_FLAG" == "--soft_masking" ]] && VARIANT="rawfeat_softmasking"

    local WANDB_FLAGS=""
    [[ "$USE_WANDB" == "true" ]] && WANDB_FLAGS="--use_wandb --wandb_mode ${WANDB_MODE}"

    local AMP_FLAG="--no-amp"
    [[ "$USE_AMP" == "true" ]] && AMP_FLAG="--amp"

    echo "Launching: ${DATASET}/${SHORT} (${ARCH}, split_seed=${SPLIT_SEED}) [${VARIANT}]  ->  GPU ${GPU}"

    # CUDA_VISIBLE_DEVICES exposes only the target GPU, so torch sees it as
    # cuda:0 and --gpu is always 0 here.
    CUDA_VISIBLE_DEVICES="${GPU}" nohup python3 -m "${EXPERIMENT_SCRIPT}" \
        --dataset             "${DATASET}" \
        --output_dir          "${OUTDIR}" \
        --checkpoint          "${CKPT}" \
        --arch                "${ARCH}" \
        --input_size          ${INPUT_SIZE} \
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
        ${SOFT_FLAG} \
        > "${OUTDIR}/log_${SHORT}_${VARIANT}.out" 2>&1 &

    gpu_pool_register $!
}

N_SKIPPED=0
N_LAUNCHED=0
gpu_pool_init

echo "Datasets     : ${DATASETS[*]}"
echo ""

# =============================================================================
# Discover every checkpoint and dispatch
# =============================================================================
for dataset in "${DATASETS[@]}"; do
    CKPT_BASE="${RESULTS_ROOT}/finetuning/classification/${dataset}"
    OUTPUT_DIR="${RESULTS_ROOT}/corruption_experiments/classification/${dataset}"
    mkdir -p "$OUTPUT_DIR"

    [ -d "$CKPT_BASE" ] || { echo "  [skip] no checkpoints under ${CKPT_BASE}"; continue; }

    while IFS= read -r CKPT; do
        RUN_DIR=$(basename "$(dirname "$CKPT")")      # e.g. dinov2_head+last_lambda100_seed2
        SHORT=$(basename "$CKPT" .pth)                 # e.g. dinov2_head+last
        ARCH_SHORT="${SHORT%%_*}"                       # e.g. dinov2
        ARCH="${ARCH_MAP[$ARCH_SHORT]:-}"
        if [ -z "$ARCH" ]; then
            echo "  [skip] unrecognised architecture in ${CKPT}"
            continue
        fi

        # Recover split_seed from a trailing "_seedN"; no suffix means seed 0.
        # The python side then writes the suffix out explicitly, so even a
        # checkpoint folder without it produces a "..._seed0_..." output name.
        SPLIT_SEED=0
        if [[ "$RUN_DIR" =~ _seed([0-9]+)$ ]]; then
            SPLIT_SEED="${BASH_REMATCH[1]}"
        fi

        RUN_OUT="${OUTPUT_DIR}"

        # corruption_experiments_cls.py writes
        # <output_dir>/<ckpt folder>_<variant>/metrics.json
        DONE_FILE="${OUTPUT_DIR}/${RUN_DIR}_${VARIANT_DIR}/metrics.json"
        if [ "$SKIP_EXISTING" = "1" ] && [ "${FORCE:-0}" != "1" ] && [ -s "$DONE_FILE" ]; then
            echo "Skipping (already done): ${dataset}/${RUN_DIR}"
            N_SKIPPED=$(( N_SKIPPED + 1 ))
            continue
        fi

        N_LAUNCHED=$(( N_LAUNCHED + 1 ))
        gpu_pool_acquire
            GPU="$GPU_POOL_SELECTED"
        launch_one "$dataset" "${RUN_DIR}_${SHORT}" "$ARCH" "$CKPT" "$SPLIT_SEED" "$RUN_OUT" "$GPU" "$SOFT_MASKING_FLAG"
        sleep 1
    done < <(find "$CKPT_BASE" -type f -name "*.pth" | sort)
done

echo ""
echo "Launched ${N_LAUNCHED}, skipped ${N_SKIPPED} already-complete run(s)."
echo "All jobs launched (queued or running). Waiting for completion..."
gpu_pool_wait || true
echo "All classification corruption experiments are done."
