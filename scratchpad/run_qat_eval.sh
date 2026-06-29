#!/usr/bin/env bash
# Run the QAT model on the 17 gate users: fp32 (its own baseline) + deployed state quant, saving distinct
# pred sets, then evaluate. Reuses the existing iter36 fp32 baseline (rust_pred_fp32).
#   args: $1 = QAT weights (.safetensors), $2 = deploy state-quant scope, $3 = NPROC (default 10)
#
# PARALLEL across users (2026-06-29): the gate is embarrassingly parallel -- each run_user writes its own
# reference/rust_pred_{u}.json, so we run the per-user rust passes concurrently (<= NPROC at a time), each
# process pinned single-threaded (RAYON/OMP=1) so NPROC procs use NPROC cores cleanly. Bit-identical to the
# old sequential gate (same per-user compute) but ~8x faster (verified iter45: 841s -> 102s at NPROC=10).
# Arch-agnostic (loops whatever users/shapes appear). For a sequential run, pass NPROC=1.
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
BIN=./rust/rwkv-infer/target/release/rwkv-infer.exe
W=${1:-reference/rwkv_iter45.safetensors}   # pass the QAT safetensors as arg 1
SCOPE=${2:-card:int2,note:int2}             # deploy state-quant scope (arg 2)
NPROC=${3:-10}                              # parallel rust processes (arg 3)
LR=${4:-}                                   # deploy LOW-RANK scope (arg 4, e.g. "card:2" or "card:2:int4")
PY=.venv/Scripts/python.exe
echo "weights: $W  scope: $SCOPE  nproc: $NPROC  lowrank: ${LR:-none}"
USERS="107 110 116 120 121 128 136 146 150 151 156 159 162 165 175 176 187"

# run $BIN on every user concurrently (<= NPROC at a time), then tag each user's pred file.
# Empty scope / lowrank env vars are treated as "off" by the engine, so always passing them is safe.
par_pass() {
  local scope="$1"; local tag="$2"; local lr="$3"; local i=0
  for u in $USERS; do
    (
      RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W \
        RWKV_STATE_QUANT_SCOPE="$scope" RWKV_STATE_LOWRANK_SCOPE="$lr" $BIN $u >/dev/null 2>&1
      cp reference/rust_pred_${u}.json reference/rust_pred_${tag}_${u}.json
    ) &
    i=$((i+1))
    if [ $((i % NPROC)) -eq 0 ]; then wait; fi
  done
  wait
}

echo "=== QAT model fp32 (no quant) ==="
par_pass "" qatfp32 ""
echo "=== QAT model + deploy quant ($SCOPE) lowrank (${LR:-none}) ==="
par_pass "$SCOPE" qatq "$LR"

echo "=== EVAL ==="
$PY scratchpad/qat_eval.py 2>&1 | grep -vE "FutureWarning|warnings.warn"
echo "=== QAT EVAL DONE ==="
