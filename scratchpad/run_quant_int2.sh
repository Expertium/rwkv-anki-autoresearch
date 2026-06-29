#!/usr/bin/env bash
# Probe card int2 (ternary, 0.27 KiB/card) — alone and with note int8. Reuses rust_pred_fp32_{u}.
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
W=reference/rwkv_iter36_124.safetensors
PY=.venv/Scripts/python.exe
export OMP_NUM_THREADS=3

USERS=$(.venv/Scripts/python.exe - <<'PY'
from pathlib import Path
cand=[107,110,116,120,121,128,136,146,150,151,156,159,162,165,175,176,187,198]
print(" ".join(str(u) for u in cand
      if (Path("reference")/f"trace_user_{u}.safetensors").exists()
      and (Path("reference")/f"rust_pred_fp32_{u}.json").exists()))
PY
)
echo "USERS: $USERS"

run_gate () {
  echo "=== $1 ==="
  RWKV_WEIGHTS=$W RWKV_STATE_QUANT_SCOPE=$2 \
    ./rust/rwkv-infer/target/release/rwkv-infer.exe $USERS >/dev/null 2>&1
  $PY scratchpad/quant_gate_users.py $USERS 2>&1 | grep -vE "FutureWarning|warnings.warn"
  echo
}

run_gate "card int2 (card 0.27 KiB)" "card:int2"
run_gate "card int2 + note int8 (card 0.27 + note 3.19 KiB)" "card:int2,note:int8"
echo "=== INT2 DONE ==="
