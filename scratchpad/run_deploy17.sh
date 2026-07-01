#!/usr/bin/env bash
# FAST deployed-eval penalty read for the decay15 champion on the 17-user gate subset (reuses existing
# weight-independent trace_user_{u} INPUTS; re-exports only the decay15 weights). Runs Rust fp32 + two
# deploy configs, scores by-user-mean imm/ahead. The penalty (deploy - fp32) added to the 100-user fp32
# champion (imm 0.280200 / ahead 0.314807) estimates the 100-user DEPLOYED number.
#   deploy_n4 = card rank-2 int4 low-rank + note int4 + quantized shifts  (conservative PTQ)
#   deploy_n2 = card rank-2 int4 low-rank + note int2 + quantized shifts  (the >=2x-note target)
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

par_pass() {  # $1 quant-scope  $2 lowrank-scope  $3 quant-shifts(0/1)  $4 tag
  local i=0
  for u in $USERS; do
    ( RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W \
        RWKV_STATE_QUANT_SCOPE="$1" RWKV_STATE_LOWRANK_SCOPE="$2" RWKV_QUANT_SHIFTS="$3" $BIN $u >/dev/null 2>&1
      cp reference/rust_pred_${u}.json reference/rust_pred_${4}_${u}.json ) &
    i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
}

echo "=== rust fp32 pass ==="
par_pass "" "" "0" fp32
echo "=== rust deploy_n4 pass (card rank2-int4 lowrank + note int4 + shifts) ==="
par_pass "note:int4" "card:2:int4" "1" deploy_n4
echo "=== rust deploy_n2 pass (card rank2-int4 lowrank + note int2 + shifts) ==="
par_pass "note:int2" "card:2:int4" "1" deploy_n2

echo "=== SCORE (17-user gate subset) ==="
$PY scratchpad/deploy_eval_range.py 101 201 fp32 deploy_n4 deploy_n2
echo "DEPLOY17_DONE"
