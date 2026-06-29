#!/usr/bin/env bash
# Extra config: card int4 + note int4 (worst-user 2.13 GB, 7.8x). Run AFTER run_quant_gate.sh
# (reuses the rust_pred_fp32_{u}.json baselines it created). Gates by-user-mean LogLoss vs iter0.
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
echo "=== CONFIG: card int4 + note int4 (card 0.53 + note 1.59 KiB) ==="
RWKV_WEIGHTS=$W RWKV_STATE_QUANT_SCOPE=card:int4,note:int4 \
  ./rust/rwkv-infer/target/release/rwkv-infer.exe $USERS >/dev/null 2>&1
$PY scratchpad/quant_gate_users.py $USERS 2>&1 | grep -vE "FutureWarning|warnings.warn"
echo "=== INT4INT4 DONE ==="
