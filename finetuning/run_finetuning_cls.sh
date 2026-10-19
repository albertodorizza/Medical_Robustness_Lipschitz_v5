#!/bin/bash
# =============================================================================
# run_finetuning_cls.sh
# =============================================================================
# Sweep launcher for finetuning_cls.py — ALL classification datasets.
#
# Loops over:
#   DATASETS  x  MODEL_TYPES  x  FREEZE_MODES  x  LAMBDA_VALUES  x  SPLIT_SEEDS
#
# Concurrency is decided by shared/gpu_pool.sh from the free memory of each
# card, rather than by a fixed jobs-per-GPU constant.

set -euo pipefail

# ── Project root ──────────────────────────────────────────────────────────────
# `python3 -m finetuning.<module>` only resolves from the directory that holds
# the finetuning/ and shared/ packages.  Walk up from this script until that
# directory is found and cd there, so the launcher works from anywhere and
# whether it sits in the project root or inside finetuning/.
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

# Cache of the frozen backbone's features, used by the freeze_mode=head half of
# the grid (see shared/featcache.py).  It lives next to the results rather than
# in the repo: it is derived data, every launcher shares it, and the entries for
# the larger datasets are hundreds of MB each.
export FEATCACHE_DIR="${FEATCACHE_DIR:-${RESULTS_ROOT}/featcache}"

# -- GPU scheduling -------------------------------------------------------------
# No hand-tuned jobs-per-GPU constant here, only a safety ceiling (below).
# shared/gpu_pool.sh admits a job only when the card has room for it NOW and would still have room LATER, if every job this
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
#
# Hard ceiling on our own jobs per card, on top of the memory rule.  The memory
# rule is about not running out of VRAM; once the freeze_mode=head runs replay
# cached features they are small enough that it would admit far more of them
# than the card has compute for, and past ~8 they stop getting faster and just
# queue on each other (and on the node's CPUs).  GPU_POOL_MAX_PER_GPU=0 goes
# back to the memory rule alone.
GPU_POOL_MAX_PER_GPU="${GPU_POOL_MAX_PER_GPU:-8}"

source "${PROJECT_ROOT}/shared/gpu_pool.sh"

FINETUNING_MODULE="finetuning.finetuning_cls"   # invoked as: python -m <MODULE>
# Resume: a rerun launches only the configurations that have not already
# produced a complete result.  FORCE=1 overrides and redoes everything.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# ── Dataset grid ──────────────────────────────────────────────────────────────
# Every classification dataset registered in shared/DatasetRegistry.py.
DATASETS=("kvasir_v2" "hyperkvasir" "kvasir_capsule")

# ── Training length ───────────────────────────────────────────────────────────
EPOCHS=50
BATCH_SIZE=128
NUM_WORKERS=4

# ── Experiment grid ───────────────────────────────────────────────────────────
MODEL_TYPES=("resnet" "vit" "dinov2")
FREEZE_MODES=("head" "head+last")
LAMBDA_VALUES=(0 0.3)

# Set by run_all.sh from the ablation result; unset when run by hand.
if [ -n "${LAMBDA_VALUES_OVERRIDE:-}" ]; then
    read -r -a LAMBDA_VALUES <<< "$LAMBDA_VALUES_OVERRIDE"
fi

# 5 independent repetitions: each seed varies BOTH the stratified split and the
# weight initialisation (see shared/seed.py -- reset_seed(args.split_seed)).
SPLIT_SEEDS=(0 1 2 3 4)

SPLIT_DIR="${SPLIT_ROOT}"

# -- Optimisation ---------------------------------------------------------------
# These are the argparse defaults of finetuning_cls.py, spelled out and passed
# explicitly.  They ARE the values every reported run used; leaving them
# implicit meant the paper's hyperparameters lived in a docstring and would
# change silently if a default ever moved.
LR=1e-4
OPTIMIZER="adam"
WEIGHT_DECAY=0.0

# -- Class filtering ------------------------------------------------------------
# MIN_PER_CLASS drops classes too small to split 70/10/20.  MAX_PER_CLASS is
# deliberately EMPTY: the reported runs used no cap, so Hyper-Kvasir and
# Kvasir-Capsule were trained on their full, heavily imbalanced label
# distribution.  Set it to a number to cap; the flag is then passed through.
MIN_PER_CLASS=20
MAX_PER_CLASS=""

GPU_POOL_DEVICES="${GPU_POOL_DEVICES:-0 1 2 3}"

N_SKIPPED=0
N_LAUNCHED=0
gpu_pool_init

echo "Datasets     : ${DATASETS[*]}"
echo ""

# =============================================================================
# Main dispatch loop
# =============================================================================
for dataset in "${DATASETS[@]}"; do
    BASE_DIR="${RESULTS_ROOT}/finetuning/classification/${dataset}"

    for split_seed in "${SPLIT_SEEDS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
    for freeze_mode in "${FREEZE_MODES[@]}"; do
    for lambda_sn in "${LAMBDA_VALUES[@]}"; do

        

        RUN_NAME="${model_type}_${freeze_mode}_lambda${lambda_sn}_seed${split_seed}"
        SAVE_ROOT="${BASE_DIR}/${RUN_NAME}"
        mkdir -p "$SAVE_ROOT"

        # A finished run leaves BOTH a checkpoint and its log.  The
        # checkpoint is written at the best epoch, so a job killed after
        # that point leaves a .pth with no _log.json -- which the analysis
        # would then silently drop.  Requiring both means a rerun picks up
        # exactly the runs that did not finish.  FORCE=1 reruns everything.
        CKPT_FILE="${SAVE_ROOT}/${model_type}_${freeze_mode}.pth"
        LOG_FILE="${SAVE_ROOT}/${model_type}_${freeze_mode}_log.json"
        if [ "$SKIP_EXISTING" = "1" ] && [ "${FORCE:-0}" != "1" ] \
           && [ -f "$CKPT_FILE" ] && [ -s "$LOG_FILE" ]; then
            echo "Skipping (already done): ${dataset}/${RUN_NAME}"
            N_SKIPPED=$(( N_SKIPPED + 1 ))
            continue
        fi
        if [ -f "$CKPT_FILE" ] && [ ! -s "$LOG_FILE" ]; then
            echo "Redoing (checkpoint without log, run did not finish): ${dataset}/${RUN_NAME}"
        fi

        gpu_pool_acquire
        DEVICE_ID="$GPU_POOL_SELECTED"

        echo "Launching: ${dataset}/${RUN_NAME}  ->  GPU ${DEVICE_ID}"
        N_LAUNCHED=$(( N_LAUNCHED + 1 ))

        nohup python3 -m "$FINETUNING_MODULE" \
            --dataset      "$dataset"      \
            --save_root    "$SAVE_ROOT"    \
            --device_id    "$DEVICE_ID"    \
            --model_type   "$model_type"   \
            --freeze_mode  "$freeze_mode"  \
            --lambda_sn    "$lambda_sn"    \
            --split_dir    "$SPLIT_DIR"    \
            --split_seed   "$split_seed"   \
            --epochs       "$EPOCHS"       \
            --batch_size   "$BATCH_SIZE"   \
            --lr           "$LR"           \
            --optimizer    "$OPTIMIZER"    \
            --weight_decay "$WEIGHT_DECAY" \
            --min_per_class "$MIN_PER_CLASS" \
            ${MAX_PER_CLASS:+--max_per_class "$MAX_PER_CLASS"} \
            --num_workers  "$NUM_WORKERS"  \
            --use_wandb                    \
            --wandb_mode   offline          \
            > "${SAVE_ROOT}/log.txt" 2>&1 &

        gpu_pool_register $! "$DEVICE_ID"
        # A short spacer only.  Waiting for a new job to actually allocate is
        # gpu_pool_acquire's business (GPU_POOL_SETTLE_SEC), and it handles it by
        # discounting not-yet-settled jobs from the free memory rather than by
        # blocking.  The fixed pause that used to sit here cost a quarter of a
        # minute per launch even when the cards were empty -- two hours across a
        # 480-run sweep, spent doing nothing.
        sleep 1

    done
    done
    done
    done
done

echo ""
echo "Launched ${N_LAUNCHED}, skipped ${N_SKIPPED} already-complete run(s)."
echo "All jobs launched (queued or running). Waiting for completion..."
# gpu_pool_wait waits on every PID individually and reports how many failed;
# a bare `wait` returns the status of the last job only, which is how a run
# can finish "successfully" with half its jobs dead.
gpu_pool_wait || true
echo "All classification fine-tuning jobs are done."
