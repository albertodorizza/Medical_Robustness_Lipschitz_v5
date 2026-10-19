#!/bin/bash
# =============================================================================
# run_all.sh
# =============================================================================
# Unattended overnight chain:
#
#   1. lambda ablation, classification  (kvasir_v2)
#   2. lambda ablation, segmentation    (kvasir_seg)
#   3. lambda* selection on validation  (select_lambda.py)
#   4. full sweeps, with the selected lambda injected per task
#   5. corruption sweep, classification
#   6. corruption sweep, segmentation
#   7. drift / bound analysis, classification
#   8. drift / bound analysis, segmentation
#
# Stages 5-8 run STRICTLY ONE AT A TIME.  Each of those launchers throttles
# only the jobs it started itself, so two of them sharing the GPU pool add up
# and oversubscribe it -- that is what produced the OutOfMemoryError crashes
# the launcher comments warn about.  Set SKIP_EVAL=1 to stop after stage 4.
#
# Step 4 is skipped when the ablation is inconclusive.  Two ways that happens:
#
#   * lambda* lands on either edge of the grid  -> the plateau was not
#     bracketed and the value cannot be trusted;
#   * the selection is FRAGILE -> the tightest (architecture, freeze mode)
#     clears the tolerance by a small fraction of it, so a different draw of
#     seeds would return a different coefficient.  Since every downstream
#     experiment is run at this coefficient, shipping a value that is one
#     resample away from changing is worse than losing a night.
#
# Set ALLOW_FRAGILE=1 to proceed anyway (and say so in the paper), or rerun
# ablation/select_lambda.py with --margin 1 to take the next grid point down.
#
# Every stage resumes.  A run that already produced a complete result -- a
# checkpoint with its log, a metrics.json, a drift results file -- is skipped,
# so this script can be restarted after a crash, a preemption or an added seed
# and will launch only what is missing.  FORCE=1 redoes everything.
#
# This script lives in the PROJECT ROOT and addresses every other script
# through $PROJECT_ROOT.  It used to sit in a run_all/ folder holding private
# copies of the four launchers and of select_lambda.py; those copies drifted
# from the real ones (they carried a stale LAMBDA_VALUES) and the whole thing
# broke as soon as the tree was tidied up.  There is now exactly one copy of
# each script, and this launcher calls it where it actually lives.
#
# Usage:
#   chmod +x run_all.sh
#   nohup setsid ./run_all.sh > run_all.log 2>&1 < /dev/null &
#   disown
#
# Environment (all optional, defaults shown in the launchers):
#   RESULTS_ROOT   where OUTPUTS_MEDICAL lives
#   SPLIT_ROOT     where splits/<dataset>/split_seed<N>.json lives
#   MEDICAL_ROOT   dataset root read by shared/DatasetRegistry.py
#   SELECT_MARGIN  grid steps to back off in the lambda selection (default 0)
#   SELECT_MIN_TOL equivalence margin of the selection rule (default 0.02)
#   ALLOW_FRAGILE  1 to sweep even when the selection is decided by noise
#   SKIP_EVAL      1 to stop after the fine-tuning sweeps
#   FORCE          1 to redo runs that already produced a complete result
#   CORRUPTION_INCLUDE_EXTERNAL  0 (default) or 1; zero-shot corruption runs
#   DRIFT_INCLUDE_EXTERNAL       1 (default) or 0; zero-shot drift runs
#   GPU_POOL_HEADROOM_MB   memory never handed out, per card (default 8000)
#   GPU_POOL_PEAK_MB       pin the per-job peak instead of calibrating it
#   GPU_POOL_DEVICES       "0 1 2 3"; default is every GPU on the machine
#   GPU_POOL_MAX_PER_GPU   jobs per card ceiling (default 8; 0 = memory rule only)
#   FEATCACHE_DIR          backbone feature cache (default RESULTS_ROOT/featcache)
# =============================================================================

set -uo pipefail        # NB: no -e, the guard below inspects exit codes itself

# -- Project root -------------------------------------------------------------
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
# Exported, so the eight launchers below inherit exactly these values instead
# of each falling back to its own default.
export RESULTS_ROOT="${RESULTS_ROOT:-/home/a.dorizza/OUTPUTS_MEDICAL}"
export SPLIT_ROOT="${SPLIT_ROOT:-${PROJECT_ROOT}/splits}"
export MEDICAL_ROOT="${MEDICAL_ROOT:-/home/datasets/medical}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Backbone-feature cache for the freeze_mode=head half of both fine-tuning
# grids (see shared/featcache.py).  Derived data, shared by the ablation and
# the sweep, so it goes next to the results and not in the repo.
export FEATCACHE_DIR="${FEATCACHE_DIR:-${RESULTS_ROOT}/featcache}"

# Safety ceiling on concurrent jobs per card, on top of gpu_pool.sh's memory
# rule; 0 disables it and leaves the memory rule alone.
export GPU_POOL_MAX_PER_GPU="${GPU_POOL_MAX_PER_GPU:-8}"

# Zero-shot evaluation: off for the corruption sweep (each checkpoint is
# evaluated only on its own dataset), on for the cheap drift stage.
export CORRUPTION_INCLUDE_EXTERNAL="${CORRUPTION_INCLUDE_EXTERNAL:-0}"
export DRIFT_INCLUDE_EXTERNAL="${DRIFT_INCLUDE_EXTERNAL:-1}"

ASSETS_DIR="${ASSETS_DIR:-${PROJECT_ROOT}/paper_assets}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ASSETS_DIR" "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED: $*"; exit 1; }

log "project root : ${PROJECT_ROOT}"
log "results root : ${RESULTS_ROOT}"
log "split root   : ${SPLIT_ROOT}"
log "dataset root : ${MEDICAL_ROOT}"
log "zero-shot    : corruption=${CORRUPTION_INCLUDE_EXTERNAL} drift=${DRIFT_INCLUDE_EXTERNAL}"

log "=== 1/8  classification ablation ==="
"${PROJECT_ROOT}/ablation/run_ablation_cls.sh" \
    > "${LOG_DIR}/ablation_cls_${STAMP}.log" 2>&1 || fail "run_ablation_cls.sh"

log "=== 2/8  segmentation ablation ==="
"${PROJECT_ROOT}/ablation/run_ablation_seg.sh" \
    > "${LOG_DIR}/ablation_seg_${STAMP}.log" 2>&1 || fail "run_ablation_seg.sh"

log "=== 3/8  lambda selection (validation) ==="
# --margin / --min-tol are the knobs of the selection rule; they are passed
# explicitly (rather than left to the defaults) so this log records exactly
# which rule produced the coefficients the sweeps then use.
SELECT_MARGIN="${SELECT_MARGIN:-0}"
SELECT_MIN_TOL="${SELECT_MIN_TOL:-0.02}"
python3 "${PROJECT_ROOT}/ablation/select_lambda.py" "${RESULTS_ROOT}/ablation" \
    --outdir "$ASSETS_DIR" \
    --margin "$SELECT_MARGIN" --min-tol "$SELECT_MIN_TOL" \
    > "${LOG_DIR}/select_lambda_${STAMP}.log" 2>&1 || fail "select_lambda.py"
cat "${LOG_DIR}/select_lambda_${STAMP}.log"

SEL_JSON="${ASSETS_DIR}/selected_lambda.json"
[ -f "$SEL_JSON" ] || fail "no ${SEL_JSON}"

read_sel() {   # $1 = task, $2 = field  (floats printed %g, so 1.0 -> 1)
    python3 -c "
import json
v = json.load(open('${SEL_JSON}'))['$1']['$2']
print(('%g' % v) if isinstance(v, float) else v)"
}

LAMBDA_CLS="$(read_sel classification lambda_star)"
LAMBDA_SEG="$(read_sel segmentation  lambda_star)"
EDGE_CLS="$(read_sel classification at_grid_edge)"
EDGE_SEG="$(read_sel segmentation  at_grid_edge)"
FRAGILE_CLS="$(read_sel classification fragile)"
FRAGILE_SEG="$(read_sel segmentation  fragile)"
ALT_CLS="$(read_sel classification lambda_star_margin1)"
ALT_SEG="$(read_sel segmentation  lambda_star_margin1)"

log "lambda* classification = ${LAMBDA_CLS}  (grid edge: ${EDGE_CLS}, fragile: ${FRAGILE_CLS}, margin-1 value: ${ALT_CLS})"
log "lambda* segmentation   = ${LAMBDA_SEG}  (grid edge: ${EDGE_SEG}, fragile: ${FRAGILE_SEG}, margin-1 value: ${ALT_SEG})"

if [ "$EDGE_CLS" = "True" ] || [ "$EDGE_SEG" = "True" ]; then
    log "STOPPING: lambda* is on a grid edge, so the plateau was not bracketed."
    log "Widen LAMBDA_VALUES in the ablation scripts and rerun before sweeping."
    exit 2
fi

if [ "${ALLOW_FRAGILE:-0}" != "1" ] \
   && { [ "$FRAGILE_CLS" = "True" ] || [ "$FRAGILE_SEG" = "True" ]; }; then
    log "STOPPING: the selection is decided by seed noise (see 'FRAGILE' in the"
    log "report above).  Every downstream experiment would inherit a coefficient"
    log "that a resample could change.  Options, in order of preference:"
    log "  1. add seeds to SPLIT_SEEDS in the ablation scripts and rerun;"
    log "  2. rerun with SELECT_MARGIN=1 to take the next grid point down;"
    log "  3. accept it deliberately with ALLOW_FRAGILE=1 and report it."
    exit 3
fi

log "=== 4/8  full fine-tuning sweeps ==="
# The launchers already carry the selected coefficients as their default
# (0 0.3 for classification, 0 0.1 for segmentation).  The override below is
# what makes the chain self-consistent: if a rerun of the ablation moves
# lambda*, the sweep follows it instead of silently using the old default.
LAMBDA_VALUES_OVERRIDE="0 ${LAMBDA_CLS}" "${PROJECT_ROOT}/finetuning/run_finetuning_cls.sh" \
    > "${LOG_DIR}/sweep_cls_${STAMP}.log" 2>&1 || fail "run_finetuning_cls.sh"

LAMBDA_VALUES_OVERRIDE="0 ${LAMBDA_SEG}" "${PROJECT_ROOT}/finetuning/run_finetuning_seg.sh" \
    > "${LOG_DIR}/sweep_seg_${STAMP}.log" 2>&1 || fail "run_finetuning_seg.sh"

N_CKPT=$(find "${RESULTS_ROOT}/finetuning" -name '*.pth' 2>/dev/null | wc -l)
log "checkpoints in finetuning/: ${N_CKPT}"

if [ "${SKIP_EVAL:-0}" = "1" ]; then
    log "=== stopping after stage 4 (SKIP_EVAL=1) ==="
    exit 0
fi

# The evaluation stages read every checkpoint the sweeps produced.  They are
# run sequentially and never in parallel: see the note at the top.
log "=== 5/8  corruption sweep, classification ==="
"${PROJECT_ROOT}/corruption_experiments/run_corruption_experiments_cls.sh" \
    > "${LOG_DIR}/corruption_cls_${STAMP}.log" 2>&1 \
    || fail "run_corruption_experiments_cls.sh"

log "=== 6/8  corruption sweep, segmentation ==="
"${PROJECT_ROOT}/corruption_experiments/run_corruption_experiments_seg.sh" \
    > "${LOG_DIR}/corruption_seg_${STAMP}.log" 2>&1 \
    || fail "run_corruption_experiments_seg.sh"

log "=== 7/8  drift analysis, classification ==="
"${PROJECT_ROOT}/drift_analysis/run_drift_analysis_cls.sh" \
    > "${LOG_DIR}/drift_cls_${STAMP}.log" 2>&1 || fail "run_drift_analysis_cls.sh"

log "=== 8/8  drift analysis, segmentation ==="
"${PROJECT_ROOT}/drift_analysis/run_drift_analysis_seg.sh" \
    > "${LOG_DIR}/drift_seg_${STAMP}.log" 2>&1 || fail "run_drift_analysis_seg.sh"

N_CORR=$(find "${RESULTS_ROOT}/corruption_experiments" -name 'metrics.json' 2>/dev/null | wc -l)
N_DRIFT=$(find "${RESULTS_ROOT}/drift_analysis" -name '*_results.json' 2>/dev/null | wc -l)
log "=== done ===  ${N_CKPT} checkpoints, ${N_CORR} corruption runs, ${N_DRIFT} drift runs"
log "paper assets in ${ASSETS_DIR}/ : per-task table + figure, tolerance"
log "sensitivity table, selected_lambda.json (records the rule, the binding"
log "constraint and the slack, so the choice can be quoted as a procedure)."
log "logs in ${LOG_DIR}/"

# The fine-tuning launchers log to W&B offline: with dozens of concurrent jobs,
# syncing while they run is network contention that buys nothing.  Everything
# is pushed here instead, once, and a failure is reported but never fails the
# pipeline -- the results are already on disk.
if command -v wandb >/dev/null 2>&1 && [ -d "${PROJECT_ROOT}/wandb" ]; then
    log "syncing offline W&B runs ..."
    wandb sync --sync-all "${PROJECT_ROOT}/wandb"         > "${LOG_DIR}/wandb_sync_${STAMP}.log" 2>&1         || log "WARNING: wandb sync failed, see ${LOG_DIR}/wandb_sync_${STAMP}.log"
fi
log ""
log "Next:  bash analysis/make_paper_figures.sh"
