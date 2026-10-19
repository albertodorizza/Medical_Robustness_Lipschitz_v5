# =============================================================================
# shared/gpu_pool.sh
# =============================================================================
# Memory-aware GPU admission control, shared by every launcher.
#
# The problem with a fixed jobs-per-GPU limit
# -------------------------------------------
# `MAX_PER_GPU=6` is a guess about memory dressed up as a scheduling policy.
# It is wrong in both directions: too low and three quarters of an H200 sits
# idle, too high and the run dies hours in with an OutOfMemoryError.  Worse, it
# counts only the jobs THIS script launched, so two launchers sharing the same
# GPUs each stay under their own limit while together they blow past the card.
#
# What this does instead
# ----------------------
# A job is admitted onto a GPU only when both of these hold:
#
#   1. there is room RIGHT NOW -- free memory on the card, as reported by
#      nvidia-smi, covers one more job at its estimated peak plus the headroom,
#      which accounts for other users and other launchers automatically;
#
#   2. there will still be room LATER -- if every job this pool has running on
#      that GPU grew to its peak at the same time, they would still fit under
#      (total - headroom).
#
# Condition 2 is the one that matters here, because these jobs do not allocate
# their full footprint at startup: a corruption sweep grows as it moves to
# larger combination chunks, and a training job grows when it first hits the
# validation pass.  Admitting on current free memory alone fills the card with
# jobs that are all still small, and they then OOM together.  Condition 2
# derives the concurrency cap from the actual card size rather than a constant
# in the script, which is why no launcher declares one any more.
#
# The peak estimate calibrates itself
# -----------------------------------
# The first job runs alone.  Once it has settled, its real memory is read back
# and multiplied by GPU_POOL_GROWTH to leave room for later growth; from then
# on the estimate is revised upward whenever any job is observed using more
# than the current estimate.  Nothing has to be guessed in advance, and a
# launcher whose jobs are heavier than expected slows down instead of crashing.
#
# Usage
# -----
#     source "${PROJECT_ROOT}/shared/gpu_pool.sh"
#     gpu_pool_init
#     ...
#     gpu_pool_acquire            # blocks until a GPU has room
#     GPU="$GPU_POOL_SELECTED"
#     CUDA_VISIBLE_DEVICES="$GPU" nohup python3 ... &
#     gpu_pool_register $!
#     ...
#     gpu_pool_wait
#
# Environment
# -----------
#   GPU_POOL_DEVICES     "0 1 2 3"; default: every GPU nvidia-smi reports
#   GPU_POOL_HEADROOM_MB memory never handed out, per GPU        (default 8000)
#   GPU_POOL_PEAK_MB     fixed per-job peak; 0 = calibrate          (default 0)
#   GPU_POOL_PEAK_INIT_MB  first guess before calibration       (default 10000)
#   GPU_POOL_GROWTH      safety factor on the measured peak       (default 1.5)
#   GPU_POOL_SETTLE_SEC  how long a new job is assumed not to have allocated
#                        its memory yet                             (default 60)
#   GPU_POOL_POLL_SEC    how often admission is retried              (default 10)
#   GPU_POOL_MAX_PER_GPU hard cap on top of the memory rule; 0 = none (default 0)
#   GPU_POOL_QUIET       1 to silence the admission log               (default 0)
# =============================================================================

GPU_POOL_HEADROOM_MB="${GPU_POOL_HEADROOM_MB:-8000}"
GPU_POOL_PEAK_MB="${GPU_POOL_PEAK_MB:-0}"
GPU_POOL_PEAK_INIT_MB="${GPU_POOL_PEAK_INIT_MB:-10000}"
GPU_POOL_GROWTH="${GPU_POOL_GROWTH:-1.5}"
GPU_POOL_SETTLE_SEC="${GPU_POOL_SETTLE_SEC:-60}"
GPU_POOL_POLL_SEC="${GPU_POOL_POLL_SEC:-10}"
GPU_POOL_MAX_PER_GPU="${GPU_POOL_MAX_PER_GPU:-0}"
GPU_POOL_QUIET="${GPU_POOL_QUIET:-0}"

GPU_POOL_SELECTED=""

declare -a _GPU_POOL_DEVICES=()
declare -A _GPU_POOL_PIDS=()        # gpu -> "pid pid pid"
declare -A _GPU_POOL_START=()       # pid -> epoch seconds at launch
declare -a _GPU_POOL_ALL_PIDS=()    # every pid ever launched, for the final wait
declare -A _GPU_POOL_TOTAL=()       # gpu -> total memory, MB
_GPU_POOL_PEAK=0                    # current per-job peak estimate, MB
_GPU_POOL_CALIBRATED=0
_GPU_POOL_HAVE_SMI=1

_gpu_pool_log() {
    [ "$GPU_POOL_QUIET" = "1" ] || echo "[gpu-pool] $*"
}

_gpu_pool_smi() {
    nvidia-smi "$@" 2>/dev/null
}

_gpu_pool_free_mb() {
    local v
    v="$(_gpu_pool_smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | head -1 | tr -d ' ')"
    # An unreadable card reports 0, so it is simply never selected.
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

_gpu_pool_total_mb() {
    local v
    v="$(_gpu_pool_smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" | head -1 | tr -d ' ')"
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v" || echo 0
}

# Largest amount of GPU memory currently used by any single one of OUR jobs.
# Per-process rather than per-card on purpose: the card also holds other
# people's work, and what the admission rule needs is the size of one job.
_gpu_pool_observed_peak_mb() {
    local pids=" "
    local g pid
    for g in "${_GPU_POOL_DEVICES[@]}"; do
        pids+="${_GPU_POOL_PIDS[$g]} "
    done

    local max=0 line p mem
    while IFS=, read -r p mem; do
        p="$(tr -d ' ' <<< "$p")"
        mem="$(tr -d ' ' <<< "$mem")"
        [[ "$mem" =~ ^[0-9]+$ ]] || continue
        [[ "$pids" == *" $p "* ]] || continue
        [ "$mem" -gt "$max" ] && max="$mem"
    done < <(_gpu_pool_smi --query-compute-apps=pid,used_gpu_memory \
                           --format=csv,noheader,nounits)
    echo "$max"
}

_gpu_pool_prune() {
    local g="$1" alive=() pid
    for pid in ${_GPU_POOL_PIDS[$g]}; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        else
            unset "_GPU_POOL_START[$pid]"
        fi
    done
    _GPU_POOL_PIDS[$g]="${alive[*]:-}"
}

_gpu_pool_live() {   # number of live jobs on GPU $1
    local n
    n=$(wc -w <<< "${_GPU_POOL_PIDS[$1]}")
    echo "$n"
}

# Jobs launched less than SETTLE_SEC ago: assumed to be holding nothing yet, so
# their eventual footprint has to be subtracted from the free memory by hand.
_gpu_pool_pending() {
    local g="$1" now pid started n=0
    now=$(date +%s)
    for pid in ${_GPU_POOL_PIDS[$g]}; do
        started="${_GPU_POOL_START[$pid]:-0}"
        [ $(( now - started )) -lt "$GPU_POOL_SETTLE_SEC" ] && n=$(( n + 1 ))
    done
    echo "$n"
}

_gpu_pool_total_live() {
    local g total=0
    for g in "${_GPU_POOL_DEVICES[@]}"; do
        total=$(( total + $(_gpu_pool_live "$g") ))
    done
    echo "$total"
}

# Revise the per-job peak estimate upward from what the jobs are actually using.
_gpu_pool_update_peak() {
    [ "$GPU_POOL_PEAK_MB" -gt 0 ] && return      # pinned by the caller
    local observed scaled
    observed="$(_gpu_pool_observed_peak_mb)"
    [ "$observed" -le 0 ] && return

    scaled=$(awk -v m="$observed" -v g="$GPU_POOL_GROWTH" 'BEGIN{printf "%d", m*g}')
    if [ "$scaled" -gt "$_GPU_POOL_PEAK" ]; then
        _gpu_pool_log "per-job peak estimate ${_GPU_POOL_PEAK} -> ${scaled} MB" \
                      "(largest live job uses ${observed} MB)"
        _GPU_POOL_PEAK="$scaled"
    fi
    _GPU_POOL_CALIBRATED=1
}

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
gpu_pool_init() {
    local g
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        _GPU_POOL_HAVE_SMI=0
        _gpu_pool_log "nvidia-smi not found; falling back to 1 job per GPU."
    fi

    if [ -n "${GPU_POOL_DEVICES:-}" ]; then
        read -r -a _GPU_POOL_DEVICES <<< "$GPU_POOL_DEVICES"
    elif [ "$_GPU_POOL_HAVE_SMI" = "1" ]; then
        mapfile -t _GPU_POOL_DEVICES < <(
            _gpu_pool_smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
    fi
    [ "${#_GPU_POOL_DEVICES[@]}" -gt 0 ] || _GPU_POOL_DEVICES=(0)

    for g in "${_GPU_POOL_DEVICES[@]}"; do
        _GPU_POOL_PIDS[$g]=""
        _GPU_POOL_TOTAL[$g]="$(_gpu_pool_total_mb "$g")"
    done

    if [ "$GPU_POOL_PEAK_MB" -gt 0 ]; then
        _GPU_POOL_PEAK="$GPU_POOL_PEAK_MB"
        _GPU_POOL_CALIBRATED=1
        _gpu_pool_log "per-job peak pinned to ${_GPU_POOL_PEAK} MB"
    else
        _GPU_POOL_PEAK="$GPU_POOL_PEAK_INIT_MB"
        _gpu_pool_log "per-job peak will be calibrated from the first job" \
                      "(initial guess ${_GPU_POOL_PEAK} MB, growth factor ${GPU_POOL_GROWTH})"
    fi

    _gpu_pool_log "devices: ${_GPU_POOL_DEVICES[*]}  headroom: ${GPU_POOL_HEADROOM_MB} MB/GPU"
    for g in "${_GPU_POOL_DEVICES[@]}"; do
        _gpu_pool_log "  GPU ${g}: ${_GPU_POOL_TOTAL[$g]} MB total, $(_gpu_pool_free_mb "$g") MB free now"
    done
}

# Block until some GPU can take another job; the choice lands in
# GPU_POOL_SELECTED.  Never call this inside $( ), or the bookkeeping it does
# happens in a subshell and is lost.
gpu_pool_acquire() {
    local g free total live pending effective future best best_score best_live

    while true; do
        _gpu_pool_update_peak

        # Until the first job has been observed, run exactly one job so the
        # calibration measures a job on its own rather than a guess.
        if [ "$_GPU_POOL_CALIBRATED" = "0" ] && [ "$(_gpu_pool_total_live)" -gt 0 ]; then
            sleep "$GPU_POOL_POLL_SEC"
            continue
        fi

        best=""
        best_score=-1
        best_live=0
        for g in "${_GPU_POOL_DEVICES[@]}"; do
            _gpu_pool_prune "$g"
            live="$(_gpu_pool_live "$g")"

            if [ "$GPU_POOL_MAX_PER_GPU" -gt 0 ] \
               && [ "$live" -ge "$GPU_POOL_MAX_PER_GPU" ]; then
                continue
            fi

            if [ "$_GPU_POOL_HAVE_SMI" = "0" ]; then
                [ "$live" -lt 1 ] && { best="$g"; break; }
                continue
            fi

            free="$(_gpu_pool_free_mb "$g")"
            total="${_GPU_POOL_TOTAL[$g]}"
            pending="$(_gpu_pool_pending "$g")"

            # (1) room now, discounting jobs that have not allocated yet
            effective=$(( free - pending * _GPU_POOL_PEAK ))
            [ "$effective" -lt $(( _GPU_POOL_PEAK + GPU_POOL_HEADROOM_MB )) ] && continue

            # (2) room later, if every job of ours on this GPU hits its peak
            future=$(( total - GPU_POOL_HEADROOM_MB - (live + 1) * _GPU_POOL_PEAK ))
            [ "$future" -lt 0 ] && continue

            # Spread the load: most room to spare wins, and on a tie the GPU
            # running fewer of our jobs does.  The tie matters at the start of a
            # sweep, when every card reads identical and idle: without it the
            # first jobs all queue behind GPU 0 until their memory shows up.
            if [ "$effective" -gt "$best_score" ] \
               || { [ "$effective" -eq "$best_score" ] && [ "$live" -lt "$best_live" ]; }; then
                best_score="$effective"
                best_live="$live"
                best="$g"
            fi
        done

        if [ -n "$best" ]; then
            GPU_POOL_SELECTED="$best"
            return 0
        fi
        sleep "$GPU_POOL_POLL_SEC"
    done
}

# Record a launched PID against the GPU that gpu_pool_acquire just handed out.
gpu_pool_register() {
    local pid="$1" g="${2:-$GPU_POOL_SELECTED}"
    _GPU_POOL_PIDS[$g]="${_GPU_POOL_PIDS[$g]} ${pid}"
    _GPU_POOL_START[$pid]="$(date +%s)"
    # Separate list: _GPU_POOL_PIDS is pruned as jobs finish, so it cannot be
    # used to account for everything at the end.
    _GPU_POOL_ALL_PIDS+=("$pid")
}

# Wait for every job, reporting how many failed.  `wait` alone would return the
# status of the last one only, which is how a run can finish "successfully"
# with half its jobs dead.
gpu_pool_wait() {
    local pid failed=0 total=0
    for pid in "${_GPU_POOL_ALL_PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        total=$(( total + 1 ))
        wait "$pid" || failed=$(( failed + 1 ))
    done
    if [ "$failed" -gt 0 ]; then
        _gpu_pool_log "WARNING: ${failed}/${total} job(s) exited non-zero." \
                      "Check the per-run .out logs before trusting the results."
        return 1
    fi
    _gpu_pool_log "all ${total} job(s) finished cleanly."
    return 0
}
