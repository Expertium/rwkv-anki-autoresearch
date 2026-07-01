#!/usr/bin/env bash
# INT2-RESCUE deploy eval (techniques #1 percol [always on], #3 Hadamard, #4 4-level) on the 17-user
# gate, champion = champ_decay15. Measures by-user-mean imm/ahead LOGLOSS (the real metric; the Frobenius
# screen was only a proxy). Compares int2 both-low-rank variants vs the current int4 deploy + fp32.
# Goal: can int2 (card 48 B + note 144 B) match int4 (card 96 B + note 288 B, ~free)? Run on a CLEAN CPU
# (after build_1500). Monitor scratchpad/int2rescue.log (INT2RESCUE_DONE). Usage: bash run_int2rescue.sh [NPROC]
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

# $1 lowrank-scope  $2 quant-shifts  $3 percol(0/1)  $4 hadamard(0/1)  $5 4level(0/1)  $6 tag
par_pass() {
  local i=0
  for u in $USERS; do
    ( RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 RWKV_WEIGHTS=$W \
        RWKV_STATE_LOWRANK_SCOPE="$1" RWKV_QUANT_SHIFTS="$2" \
        RWKV_LOWRANK_PERCOL="$3" RWKV_LOWRANK_HADAMARD="$4" RWKV_LOWRANK_4LEVEL="$5" \
        $BIN $u >/dev/null 2>&1
      cp reference/rust_pred_${u}.json reference/rust_pred_${6}_${u}.json ) &
    i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait
  done; wait
}

echo "=== fp32 (no quant) ==="                         ; par_pass ""                  0 0 0 0 fp32
echo "=== int4 both-low-rank (current deploy target) ==="; par_pass "card:2:int4,note:2:int4" 1 1 0 0 i4
echo "=== int2 both-low-rank percol (the 'dies' base) ==="; par_pass "card:2:int2,note:2:int2" 1 1 0 0 i2
echo "=== int2 + 4LEVEL (#4) ==="                       ; par_pass "card:2:int2,note:2:int2" 1 1 0 1 i2_4l
echo "=== int2 + 4LEVEL + HADAMARD (#4+#3 kitchen sink) ==="; par_pass "card:2:int2,note:2:int2" 1 1 1 1 i2_both

echo "=== SCORE (17-user gate; deltas vs fp32) ==="
$PY scratchpad/deploy_eval_range.py 101 201 fp32 i4 i2 i2_4l i2_both
echo "INT2RESCUE_DONE"
