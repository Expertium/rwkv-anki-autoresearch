#!/usr/bin/env bash
# Run the state-quant gate over whichever of the 18 target users have traces present.
# For each config: run rust over the users, then gate by-user-mean LogLoss vs iter0.
# fp32 baseline is captured first (rust_pred_fp32_{u}.json).
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
W=reference/rwkv_iter36_124.safetensors
PY=.venv/Scripts/python.exe
export OMP_NUM_THREADS=3

# Users with traces present (subset of the 18 smallest).
USERS=$(.venv/Scripts/python.exe - <<'PY'
from pathlib import Path
cand=[107,110,116,120,121,128,136,146,150,151,156,159,162,165,175,176,187,198]
print(" ".join(str(u) for u in cand if (Path("reference")/f"trace_user_{u}.safetensors").exists()))
PY
)
echo "USERS WITH TRACES: $USERS"

run_rust () { RWKV_WEIGHTS=$W "$@" ./rust/rwkv-infer/target/release/rwkv-infer.exe $USERS >/dev/null 2>&1; }

echo "=== FP32 baseline ==="
run_rust
for u in $USERS; do cp reference/rust_pred_$u.json reference/rust_pred_fp32_$u.json; done

echo; echo "=== CONFIG: card int8 (card 1.06 KiB) ==="
run_rust env RWKV_STATE_QUANT_SCOPE=card:int8
$PY scratchpad/quant_gate_users.py $USERS 2>&1 | grep -vE "FutureWarning|warnings.warn"

echo; echo "=== CONFIG: card int4 (card 0.53 KiB, <1 KB) ==="
run_rust env RWKV_STATE_QUANT_SCOPE=card:int4
$PY scratchpad/quant_gate_users.py $USERS 2>&1 | grep -vE "FutureWarning|warnings.warn"

echo; echo "=== CONFIG: card+note int8 (card 1.06 + note 3.19 KiB) ==="
run_rust env RWKV_STATE_QUANT_SCOPE=card:int8,note:int8
$PY scratchpad/quant_gate_users.py $USERS 2>&1 | grep -vE "FutureWarning|warnings.warn"

echo; echo "=== CONFIG: card int4 + note int8 (card 0.53 + note 3.19 KiB) ==="
run_rust env RWKV_STATE_QUANT_SCOPE=card:int4,note:int8
$PY scratchpad/quant_gate_users.py $USERS 2>&1 | grep -vE "FutureWarning|warnings.warn"

echo; echo "=== GATE RUN DONE ==="
