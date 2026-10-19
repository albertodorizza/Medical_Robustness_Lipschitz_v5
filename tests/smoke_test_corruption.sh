#!/bin/bash
# =============================================================================
# smoke_test_corruption.sh
# =============================================================================
# Fast sanity check for corruption_experiments/, one combination only.
# Run from the project root (the folder that contains shared/, corruptions/,
# corruption_experiments/, ...).
#
# What it checks:
#   1. the sweep produces at least one experiment (catches the old B1 bug)
#   2. metadata.dataset is set (catches the old B3 bug)
#   3. effective_ratio is close to the nominal perturbation_ratio
#   4. num_workers=0 vs num_workers=4 give the SAME effective_ratio and
#      accuracy_corrupted, i.e. the per-sample seed does not depend on workers
#
# Requires a real checkpoint. Point CKPT / ARCH / DATASET at something small
# you already have on disk (a classification checkpoint + kvasir_v2 is the
# fastest combination to test).
#
# Usage:
#   chmod +x smoke_test_corruption.sh
#   ./smoke_test_corruption.sh /path/to/checkpoint.pth dinov2 kvasir_v2
# =============================================================================
set -euo pipefail

CKPT="${1:?usage: $0 CKPT ARCH DATASET}"
ARCH="${2:?usage: $0 CKPT ARCH DATASET}"
DATASET="${3:?usage: $0 CKPT ARCH DATASET}"

OUT=$(mktemp -d)
echo "Output dir: $OUT"

run_once() {
    local WORKERS=$1 TAG=$2
    python3 -m corruption_experiments.corruption_experiments_cls \
        --dataset             "$DATASET" \
        --checkpoint          "$CKPT" \
        --arch                "$ARCH" \
        --output_dir          "$OUT/$TAG" \
        --corruption_types    circular \
        --perturbation_ratios 0.3 \
        --severity_levels     3 \
        --perturbation_types  '{"NOISE": ["Gaussian Noise"]}' \
        --batch_size          8 \
        --num_workers         "$WORKERS" \
        --split_dir           "$OUT/splits" \
        > "$OUT/log_${TAG}.out" 2>&1
    find "$OUT/$TAG" -name metrics.json
}

echo "Run 1: num_workers=0"
JSON0=$(run_once 0 w0)
echo "Run 2: num_workers=4"
JSON4=$(run_once 4 w4)

python3 - "$JSON0" "$JSON4" <<'PY'
import json, sys

j0, j4 = (json.load(open(p)) for p in sys.argv[1:3])

for name, blob in (("num_workers=0", j0), ("num_workers=4", j4)):
    exps = blob["all_experiments"]
    assert exps, f"[{name}] all_experiments is EMPTY -- sweep produced nothing"
    assert blob["metadata"].get("dataset"), f"[{name}] metadata.dataset is missing"

    exp = exps[0]
    nominal = exp["perturbation_ratio"]
    eff = exp["effective_ratio"]
    assert abs(eff - nominal) < 0.1, (
        f"[{name}] effective_ratio={eff:.3f} far from nominal={nominal}"
    )
    print(f"[{name}] OK  n_experiments={len(exps)}  "
          f"effective_ratio={eff:.3f}  acc_clean={exp['accuracy_clean']:.4f}  "
          f"acc_corrupted={exp['accuracy_corrupted']:.4f}")

e0, e4 = j0["all_experiments"][0], j4["all_experiments"][0]
assert abs(e0["effective_ratio"] - e4["effective_ratio"]) < 1e-6, (
    "effective_ratio changed with num_workers -- seeding is NOT worker-independent"
)
assert abs(e0["accuracy_corrupted"] - e4["accuracy_corrupted"]) < 1e-6, (
    "accuracy_corrupted changed with num_workers -- seeding is NOT worker-independent"
)
print("Worker-independence: OK (num_workers=0 and 4 give identical results)")
print("\nALL CHECKS PASSED")
PY
