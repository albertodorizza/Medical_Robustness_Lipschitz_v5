#!/bin/bash
# =============================================================================
# check_performance.sh
# =============================================================================
# Scans finetuning logs, flags models w/ test perf below threshold.
# cls: test_acc < THRESH_ACC (default chance-ish, 8-class Kvasir ~0.125)
# seg: test_dice < THRESH_DICE (default 0.3, binary seg)
#
# Usage: ./check_performance.sh
# =============================================================================

set -uo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-/home/a.dorizza/OUTPUTS_MEDICAL}"
THRESH_ACC=0.5      # below this = probably broken (chance ~0.125 for 8 classes)
THRESH_DICE=0.5     # below this = probably broken

check_task() {
    local task=$1 metric_key=$2 thresh=$3
    find "${RESULTS_ROOT}/finetuning/${task}" -name "*_log.json" 2>/dev/null | sort |
    while read -r log; do
        val=$(python3 -c "
import json, sys
try:
    d = json.load(open('$log'))['selection']
    print(d.get('$metric_key', 'NA'))
except Exception:
    print('ERR')
")
        run=$(basename "$(dirname "$log")")
        ds=$(basename "$(dirname "$(dirname "$log")")")

        if [ "$val" = "ERR" ]; then
            echo "  [BAD LOG] ${task}/${ds}/${run}"
        elif [ "$val" = "NA" ]; then
            echo "  [NO ${metric_key}] ${task}/${ds}/${run}"
        else
            below=$(python3 -c "print(1 if $val < $thresh else 0)")
            if [ "$below" = "1" ]; then
                printf "  [LOW %s=%.4f] %s/%s/%s\n" "$metric_key" "$val" "$task" "$ds" "$run"
            fi
        fi
    done
}

echo "=== classification (test_acc < ${THRESH_ACC}) ==="
check_task "classification" "test_acc" "$THRESH_ACC"

echo ""
echo "=== segmentation (test_dice < ${THRESH_DICE}) ==="
check_task "segmentation" "test_dice" "$THRESH_DICE"