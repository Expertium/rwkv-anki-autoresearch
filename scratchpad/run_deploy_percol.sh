#!/usr/bin/env bash
# Per-column int2 low-rank measure (Andrew 2026-06-30): does per-COLUMN factor scaling rescue rank-2 int2?
# Compares fp32 / int4 / int2-shared-scale / int2-per-column on the 17-user gate (decay15 champion). The
# Rust sort-bug fix means int2 no longer panics, so all 17 should now score.
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
BIN=./rust/rwkv-infer/target/release/rwkv-infer.exe
W=reference/champ_decay15.safetensors
PY=.venv/Scripts/python.exe
NPROC=${1:-10}
USERS="107 110 116 120 121 128 136 146 150 151 156 159 162 165 175 176 187"

par_pass() {  # $1 lowrank-scope  $2 quant-shifts  $3 percol(0/1)  $4 tag
  local i=0
  for u in $USERS; do
    ( RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$2" RWKV_LOWRANK_PERCOL="$3" $BIN $u >/dev/null 2>&1
      cp reference/rust_pred_${u}.json reference/rust_pred_${4}_${u}.json ) &
    i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
}

echo "=== fp32 ==="
par_pass "" "0" "0" fp32
echo "=== int4 (shared scale) ==="
par_pass "card:2:int4,note:2:int4" "1" "0" int4
echo "=== int2 SHARED scale ==="
par_pass "card:2:int2,note:2:int2" "1" "0" int2sh
echo "=== int2 PER-COLUMN scale ==="
par_pass "card:2:int2,note:2:int2" "1" "1" int2pc

echo "=== SCORE (17-user gate) ==="
$PY scratchpad/deploy_eval_range.py 101 201 fp32 int4 int2sh int2pc
echo "PERCOL_DONE"
