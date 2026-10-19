#!/bin/bash
# =============================================================================
# run_drift_analysis_seg.sh
# =============================================================================
# Sweep launcher for drift_analysis_seg.py — every trained segmentation
# checkpoint, discovered on disk, evaluated in-distribution and on every other
# member of its family (see run_corruption_experiments_seg.sh for the family
# definition and the reasoning behind excluding the training set from its own
# external list).
#
# Usage:
#   chmod +x run_drift_analysis_seg.sh
#   ./run_drift_analysis_seg.sh
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


DATASETS=("kvasir_seg" "cvc_clinicdb" "cvc_colondb" "etis_larib" "aircare_all")
POLYP_FAMILY=("kvasir_seg" "cvc_clinicdb" "cvc_colondb" "etis_larib")

declare -A ARCH_MAP=([resnet]=resnet34 [vit]=vit_b_16 [dinov2]=dinov2)

PERTURBATIONS=("gaussian" "Contrast")
SEVERITY=3
EPS=0.01

BATCH_SIZE=32
NUM_WORKERS=8
PIN_CORES=0                          # 0 = no CPU pinning
SEED=29                              # perturbation noise, NOT split_seed

USE_WANDB=true
WANDB_MODE="online"

GPU_POOL_DEVICES="${GPU_POOL_DEVICES:-0 1 2 3}"

# -- Zero-shot evaluation ------------------------------------------------------
# ON by default here, unlike the corruption launcher: the bound is a property
# of the head and holds under distribution shift too, and checking that its
# slack does not blow up off-distribution is cheap (one pass per dataset, not
# 108 corruption combinations). Set DRIFT_INCLUDE_EXTERNAL=0 to restrict this
# stage to the in-distribution split as well.
DRIFT_INCLUDE_EXTERNAL="${DRIFT_INCLUDE_EXTERNAL:-1}"

external_datasets_for() {
    local train=$1
    [ "$DRIFT_INCLUDE_EXTERNAL" = "1" ] || return 0
    local family=()
    case "$train" in
        aircare_all) family=() ;;                # pooled: no external set here
        *)           family=("${POLYP_FAMILY[@]}") ;;
    esac
    for ds in "${family[@]}"; do
        [ "$ds" != "$train" ] && echo "$ds"
    done
}

# -- Resume --------------------------------------------------------------------
# A rerun launches only what has not already produced a complete result, so the
# chain can be restarted after a crash or a preemption without redoing days of
# work.  FORCE=1 redoes everything.
#
# The result file is matched by GLOB rather than by name: drift_analysis_*.py
# builds the stem from the architecture string it resolves internally, and
# reconstructing that in bash would be a second, divergent copy of the rule.
# The run folder plus the perturbation tag already identify the job uniquely.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# done_glob RUN_DIR EVAL_DS TAG -> non-empty if the result file already exists
drift_result_exists() {
    local out_root=$1 run_dir=$2 eval_ds=$3 tag=$4 f
    for f in "${out_root}/${run_dir}"/*_"${eval_ds}"_"${tag}"_results.json; do
        [ -s "$f" ] && return 0
    done
    return 1
}

N_SKIPPED=0
N_LAUNCHED=0
gpu_pool_init

echo "Datasets      : ${DATASETS[*]}"
echo "Perturbations : ${PERTURBATIONS[*]}"
echo ""

for PERTURBATION in "${PERTURBATIONS[@]}"; do
    if [ "$PERTURBATION" = "gaussian" ]; then
        PERTURB_ARGS="--perturbation ${PERTURBATION} --eps ${EPS}"
    else
        PERTURB_ARGS="--perturbation ${PERTURBATION} --severity ${SEVERITY}"
    fi
    TAG="drift_${PERTURBATION// /}"

    for dataset in "${DATASETS[@]}"; do
        CKPT_BASE="${RESULTS_ROOT}/finetuning/segmentation/${dataset}"
        OUT_ROOT="${RESULTS_ROOT}/drift_analysis/segmentation/${dataset}/${TAG}"
        mkdir -p "$OUT_ROOT"

        [ -d "$CKPT_BASE" ] || { echo "  [skip] no checkpoints under ${CKPT_BASE}"; continue; }

        # external_datasets_for prints nothing when zero-shot is off; build the
        # array by hand so an empty one is an empty array rather than a single
        # empty element, which "${EXTERNALS[@]}" under set -u would trip over.
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

            SPLIT_SEED=0
            if [[ "$RUN_DIR" =~ _seed([0-9]+)$ ]]; then
                SPLIT_SEED="${BASH_REMATCH[1]}"
            fi

            WANDB_FLAGS=""
            [[ "$USE_WANDB" == "true" ]] && WANDB_FLAGS="--use_wandb --wandb_mode ${WANDB_MODE}"

            # in-distribution + every external dataset in the same family
            for EVAL_DS in "$dataset" "${EXTERNALS[@]}"; do
                if [ "$EVAL_DS" = "$dataset" ]; then EVAL_SPLIT="test"; else EVAL_SPLIT="all"; fi

                if [ "$PERTURBATION" = "gaussian" ]; then
                    PTAG="gaussian_eps${EPS}"
                else
                    PTAG="${PERTURBATION// /}_s${SEVERITY}"
                fi
                if [ "$SKIP_EXISTING" = "1" ] && [ "${FORCE:-0}" != "1" ] \
                   && drift_result_exists "$OUT_ROOT" "$RUN_DIR" "$EVAL_DS" "$PTAG"; then
                    echo "Skipping (already done): ${dataset}/${RUN_DIR} -> ${EVAL_DS} [${PERTURBATION}]"
                    N_SKIPPED=$(( N_SKIPPED + 1 ))
                    continue
                fi

                N_LAUNCHED=$(( N_LAUNCHED + 1 ))
                gpu_pool_acquire
            GPU="$GPU_POOL_SELECTED"
                echo "Launching: ${dataset}/${RUN_DIR} -> eval on ${EVAL_DS} (${EVAL_SPLIT}) [${PERTURBATION}]  ->  GPU ${GPU}"

                # CUDA_VISIBLE_DEVICES exposes only the target GPU, so torch
                # sees it as cuda:0 and --gpu is always 0 here.
                # --output_dir is the PARENT: drift_analysis_seg.py appends its
                # own "<run>_seed<N>" folder underneath, matching the layout in
                # shared/paths.py -- do NOT pre-create a run subfolder here, or
                # the path gains an extra, unwanted nesting level.
                CUDA_VISIBLE_DEVICES="${GPU}" nohup python3 -m drift_analysis.drift_analysis_seg \
                    --model_path  "$CKPT" \
                    --arch        "$ARCH" \
                    --dataset     "$EVAL_DS" \
                    --eval_split  "$EVAL_SPLIT" \
                    --output_dir  "$OUT_ROOT" \
                    --batch_size  "$BATCH_SIZE" \
                    --num_workers "$NUM_WORKERS" \
                    --pin_cores   "$PIN_CORES" \
                    --gpu         0 \
                    --seed        "$SEED" \
                    --split_dir   "$SPLIT_ROOT" \
                    --split_seed  "$SPLIT_SEED" \
                    $PERTURB_ARGS \
                    ${WANDB_FLAGS} \
                    > "${OUT_ROOT}/log_${RUN_DIR}_on_${EVAL_DS}.out" 2>&1 &

                gpu_pool_register $!
                sleep 1
            done
        done < <(find "$CKPT_BASE" -type f -name "*.pth" | sort)
    done
done

echo ""
echo "Launched ${N_LAUNCHED}, skipped ${N_SKIPPED} already-complete run(s)."
echo "All jobs launched (queued or running). Waiting for completion..."
gpu_pool_wait || true
echo "All segmentation drift-analysis jobs are done."
