#!/usr/bin/env bash
# BOTH-low-rank deploy measure for the decay15 champion (Andrew 2026-06-30): card AND note state each
# rank-2 low-rank with quantized factors + quantized shifts. Compares int4 vs int2 factors. Reuses the
# weight-independent trace_user_{u} inputs (17-user gate subset). fp32 -> the model's own Rust baseline.
#   bothlr4 = card:2:int4 + note:2:int4 + int4 shifts  (card 96 B + note 288 B)
#   bothlr2 = card:2:int2 + note:2:int2 + int2 shifts  (card 48 B + note 144 B)  <- Andrew's request
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
BIN=./rust/rwkv-infer/target/release/rwkv-infer.exe
W=reference/champ_decay15.safetensors
PY=.venv/Scripts/python.exe
NPROC=${1:-10}
USERS="107 110 116 120 121 128 136 146 150 151 156 159 162 165 175 176 187"

echo "=== export decay15 weights ==="
RWKV_CHAMP_CKPT=scratchpad/tuner/decay15/decay15_640.pth RWKV_CHAMP_SFT=champ_decay15.safetensors \
  $PY scratchpad/export_weights_only.py

par_pass() {  # $1 lowrank-scope  $2 quant-shifts(0/1)  $3 tag
  local i=0
  for u in $USERS; do
    ( RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$2" $BIN $u >/dev/null 2>&1
      cp reference/rust_pred_${u}.json reference/rust_pred_${3}_${u}.json ) &
    i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
}

echo "=== rust fp32 ==="
par_pass "" "0" fp32
echo "=== both-low-rank int4 factors + int4 shifts (card 96B + note 288B) ==="
par_pass "card:2:int4,note:2:int4" "1" bothlr4
echo "=== both-low-rank int2 factors + int2 shifts (card 48B + note 144B) ==="
par_pass "card:2:int2,note:2:int2" "1" bothlr2

echo "=== SCORE (17-user gate subset) ==="
$PY scratchpad/deploy_eval_range.py 101 201 fp32 bothlr4 bothlr2
echo "BOTHLR_DONE"
