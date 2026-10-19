#!/bin/bash
# =============================================================================
# check_runs.sh
# =============================================================================
# Verifies that every expected fine-tuning run completed, AND flags duplicates:
#
# 1. Missing / incomplete runs: checkpoint file missing, or present without a
#    matching "selection" block in the log -- torch.save() is the LAST thing
#    train_model() does, so a missing .pth means the run crashed or never
#    finished.
#
# 2. Duplicate run folders: two directories on disk that normalize to the
#    SAME logical config (arch, freeze_mode, lambda, split_seed) but have
#    different folder names, e.g. "..._lambda100" and "..._lambda100.0" --
#    the same lambda written differently by two separate launches. Both are
#    real fine-tuning runs, but only one should count as the result for that
#    config; the other silently doubles the seed count in any script that
#    averages over "all folders found on disk" (analysis.io).
#
# 3. Duplicate checkpoints inside one run folder: more than one *.pth file in
#    the same directory, which should never happen (finetuning_cls.py writes
#    exactly one) and points at a run directory reused for two different
#    invocations.
#
# Edit the grids below to match what run_finetuning_cls.sh / run_finetuning_seg.sh
# were actually configured with.
#
# Usage:
#   chmod +x check_runs.sh
#   ./check_runs.sh
# =============================================================================

set -uo pipefail   # no -e: we want to keep checking after a miss

RESULTS_ROOT="${RESULTS_ROOT:-/home/a.dorizza/OUTPUTS_MEDICAL}"

# ── Expected grids (must match the launcher scripts) ─────────────────────────
CLS_DATASETS=("kvasir_v2" "hyperkvasir" "kvasir_capsule")
SEG_DATASETS=("kvasir_seg" "cvc_clinicdb" "cvc_colondb" "etis_larib" "aircare_all")

MODEL_TYPES=("resnet" "vit" "dinov2")
FREEZE_MODES=("head" "head+last")
LAMBDA_VALUES=(0 100)
SPLIT_SEEDS=(0 1 2 3 4)

TOTAL=0
MISSING=0
declare -a MISSING_LIST=()

check_one() {
    local task=$1 dataset=$2 model_type=$3 freeze_mode=$4 lambda_sn=$5 split_seed=$6

    local run_name="${model_type}_${freeze_mode}_lambda${lambda_sn}_seed${split_seed}"
    local save_root="${RESULTS_ROOT}/finetuning/${task}/${dataset}/${run_name}"
    local ckpt="${save_root}/${model_type}_${freeze_mode}.pth"
    local log="${save_root}/${model_type}_${freeze_mode}_log.json"

    TOTAL=$((TOTAL + 1))

    if [ ! -f "$ckpt" ]; then
        MISSING=$((MISSING + 1))
        MISSING_LIST+=("${task}/${dataset}/${run_name}  [no checkpoint]")
        return
    fi

    if [ ! -f "$log" ] || ! grep -q '"selection"' "$log"; then
        MISSING=$((MISSING + 1))
        MISSING_LIST+=("${task}/${dataset}/${run_name}  [checkpoint exists, but log is missing/incomplete]")
        return
    fi
}

echo "Checking classification runs ..."
for dataset in "${CLS_DATASETS[@]}"; do
for split_seed in "${SPLIT_SEEDS[@]}"; do
for model_type in "${MODEL_TYPES[@]}"; do
for freeze_mode in "${FREEZE_MODES[@]}"; do
for lambda_sn in "${LAMBDA_VALUES[@]}"; do
    check_one "classification" "$dataset" "$model_type" "$freeze_mode" "$lambda_sn" "$split_seed"
done; done; done; done; done

echo "Checking segmentation runs ..."
for dataset in "${SEG_DATASETS[@]}"; do
for split_seed in "${SPLIT_SEEDS[@]}"; do
for model_type in "${MODEL_TYPES[@]}"; do
for freeze_mode in "${FREEZE_MODES[@]}"; do
for lambda_sn in "${LAMBDA_VALUES[@]}"; do
    check_one "segmentation" "$dataset" "$model_type" "$freeze_mode" "$lambda_sn" "$split_seed"
done; done; done; done; done

echo ""
echo "================================================================"
echo "Total expected : ${TOTAL}"
echo "Complete       : $((TOTAL - MISSING))"
echo "Missing/failed : ${MISSING}"
echo "================================================================"

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "Missing or incomplete runs:"
    for entry in "${MISSING_LIST[@]}"; do
        echo "  - $entry"
    done
fi

# =============================================================================
# Duplicate check
# =============================================================================
# Normalizes a lambda string ("100", "100.0", "100.00") to a canonical form so
# two folders that spell the same lambda differently are recognised as the
# same logical config.  %g also matches how shared/paths.run_name() formats
# it, so this stays consistent with the naming convention used elsewhere.
normalize_lambda() {
    awk -v v="$1" 'BEGIN { printf "%g", v+0 }'
}

echo ""
echo "Checking for duplicate run folders ..."
DUPLICATES=0
declare -a DUP_LIST=()

check_duplicates_in() {
    local task=$1 dataset=$2
    local base="${RESULTS_ROOT}/finetuning/${task}/${dataset}"
    [ -d "$base" ] || return

    declare -A seen_key_to_folder=()

    for dir in "$base"/*/; do
        [ -d "$dir" ] || continue
        local name
        name=$(basename "$dir")

        # Parse "<arch>_<freeze>_lambda<L>[_seed<N>]"
        if [[ "$name" =~ ^([a-z0-9]+)_(head\+last|head)_lambda([0-9.]+)(_seed([0-9]+))?$ ]]; then
            local arch="${BASH_REMATCH[1]}"
            local freeze="${BASH_REMATCH[2]}"
            local lambda_raw="${BASH_REMATCH[3]}"
            local seed="${BASH_REMATCH[5]:-0}"
            local lambda_norm
            lambda_norm=$(normalize_lambda "$lambda_raw")
            local key="${arch}_${freeze}_lambda${lambda_norm}_seed${seed}"

            if [ -n "${seen_key_to_folder[$key]:-}" ]; then
                DUPLICATES=$((DUPLICATES + 1))
                DUP_LIST+=("${task}/${dataset}: '${seen_key_to_folder[$key]}' and '${name}' are the same config (${key})")
            else
                seen_key_to_folder[$key]="$name"
            fi
        fi

        # Duplicate checkpoints inside a single run folder
        local n_ckpt
        n_ckpt=$(find "$dir" -maxdepth 1 -name "*.pth" | wc -l)
        if [ "$n_ckpt" -gt 1 ]; then
            DUPLICATES=$((DUPLICATES + 1))
            DUP_LIST+=("${task}/${dataset}/${name}: ${n_ckpt} .pth files in one folder (expected 1)")
        fi
    done
}

for dataset in "${CLS_DATASETS[@]}"; do check_duplicates_in "classification" "$dataset"; done
for dataset in "${SEG_DATASETS[@]}"; do check_duplicates_in "segmentation" "$dataset"; done

echo ""
echo "================================================================"
echo "Duplicates found: ${DUPLICATES}"
echo "================================================================"
if [ "$DUPLICATES" -gt 0 ]; then
    for entry in "${DUP_LIST[@]}"; do
        echo "  - $entry"
    done
fi

if [ "$MISSING" -gt 0 ] || [ "$DUPLICATES" -gt 0 ]; then
    exit 1
fi

echo ""
echo "All expected fine-tuning runs are present, complete, and unique."