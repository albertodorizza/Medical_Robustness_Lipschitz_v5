#!/bin/bash
# =============================================================================
# fullres_certificate/run_fullres_seg.sh
# =============================================================================
# Full-resolution certificate for every segmentation checkpoint, evaluated on
# its own test split (the runs behind the certificate figure of the paper).
# Same discovery, split seeds and perturbations as
# drift_analysis/run_drift_analysis_seg.sh; writes to its own tree:
#
#     ${RESULTS_ROOT}/fullres_certificate/segmentation/<dataset>/<tag>/<run>/
#
# Usage:
#     RESULTS_ROOT=/home/a.dorizza/OUTPUTS_MEDICAL bash fullres_certificate/run_fullres_seg.sh
#
# Environment: PERTURBATIONS="gaussian Contrast", DATASETS="kvasir_seg ...",
# GPU_POOL_DEVICES="0 1 2 3", FORCE=1 to redo finished runs.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-/home/a.dorizza/OUTPUTS_MEDICAL}"
SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_ROOT}/splits}"
export MEDICAL_ROOT="${MEDICAL_ROOT:-/home/datasets/medical}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
GPU_POOL_DEVICES="${GPU_POOL_DEVICES:-0 1 2 3}"
source "${PROJECT_ROOT}/shared/gpu_pool.sh"

read -r -a DATASETS <<< "${DATASETS:-kvasir_seg cvc_clinicdb cvc_colondb etis_larib aircare_all}"
read -r -a PERTURBATIONS <<< "${PERTURBATIONS:-gaussian Contrast}"
declare -A ARCH_MAP=([resnet]=resnet34 [vit]=vit_b_16 [dinov2]=dinov2)
SEVERITY=3
EPS=0.01
SEED=29          # perturbation seed, same as the existing drift analysis

N_LAUNCHED=0
N_SKIPPED=0
gpu_pool_init

for PERT in "${PERTURBATIONS[@]}"; do
    if [ "$PERT" = "gaussian" ]; then
        PARGS="--perturbation gaussian --eps ${EPS}"; PTAG="gaussian_eps${EPS}"
    else
        PARGS="--perturbation ${PERT} --severity ${SEVERITY}"; PTAG="${PERT// /}_s${SEVERITY}"
    fi
    for DS in "${DATASETS[@]}"; do
        CKPT_BASE="${RESULTS_ROOT}/finetuning/segmentation/${DS}"
        OUT_ROOT="${RESULTS_ROOT}/fullres_certificate/segmentation/${DS}/${PTAG}"
        [ -d "$CKPT_BASE" ] || { echo "  [skip] no checkpoints under ${CKPT_BASE}"; continue; }
        mkdir -p "$OUT_ROOT"

        while IFS= read -r CKPT; do
            RUN_DIR=$(basename "$(dirname "$CKPT")")
            ARCH="${ARCH_MAP[$(basename "$CKPT" .pth | cut -d_ -f1)]:-}"
            [ -z "$ARCH" ] && { echo "  [skip] unrecognised arch in ${CKPT}"; continue; }
            SPLIT_SEED=0
            [[ "$RUN_DIR" =~ _seed([0-9]+)$ ]] && SPLIT_SEED="${BASH_REMATCH[1]}"

            if [ "${FORCE:-0}" != "1" ] && \
               ls "${OUT_ROOT}/${RUN_DIR}"/*_"${DS}"_"${PTAG}"_fullres.json >/dev/null 2>&1; then
                N_SKIPPED=$(( N_SKIPPED + 1 )); continue
            fi

            gpu_pool_acquire
            GPU="$GPU_POOL_SELECTED"
            echo "Launching: ${DS}/${RUN_DIR} [${PERT}] -> GPU ${GPU}"
            CUDA_VISIBLE_DEVICES="${GPU}" nohup python3 -m fullres_certificate.certificate_seg_fullres \
                --model_path "$CKPT" --arch "$ARCH" --dataset "$DS" --eval_split test \
                --output_dir "$OUT_ROOT" --split_dir "$SPLIT_ROOT" --split_seed "$SPLIT_SEED" \
                --seed "$SEED" --gpu 0 $PARGS \
                > "${OUT_ROOT}/log_${RUN_DIR}.out" 2>&1 &
            gpu_pool_register $!
            N_LAUNCHED=$(( N_LAUNCHED + 1 ))
            sleep 1
        done < <(find "$CKPT_BASE" -type f -name "*.pth" | sort)
    done
done

echo "Launched ${N_LAUNCHED}, skipped ${N_SKIPPED}. Waiting..."
gpu_pool_wait || true
echo "Done. Summarise with:"
echo "  python -m fullres_certificate.summarize --root ${RESULTS_ROOT}/fullres_certificate/segmentation"
