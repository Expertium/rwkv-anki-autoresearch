#!/usr/bin/env bash
# PROVE the state quant actually affects the logloss predictions (not a no-op).
# Run the SAME QAT model (iter39) three ways on 3 ref users and compare imm predictions + logloss:
#   (a) fp32 (no quant)         -- baseline
#   (b) card:int2,note:int4     -- the DEPLOY config (QAT was trained for this -> should be ~free)
#   (c) all streams int2        -- HARSH, untrained (user/global int2, long recurrence -> must DEGRADE)
# If (c) predictions == (a), quant is a no-op (BUG). If (c) degrades a lot, quant is LIVE.
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
BIN=./rust/rwkv-infer/target/release/rwkv-infer.exe
W=reference/rwkv_iter39_124.safetensors
PY=.venv/Scripts/python.exe
U="107 136 156"

echo "(a) fp32"
RWKV_WEIGHTS=$W $BIN $U >/dev/null 2>&1
for u in $U; do cp reference/rust_pred_${u}.json reference/rpv_fp32_${u}.json; done
echo "(b) card:int2,note:int4 (deploy)"
RWKV_WEIGHTS=$W RWKV_STATE_QUANT_SCOPE="card:int2,note:int4" $BIN $U >/dev/null 2>&1
for u in $U; do cp reference/rust_pred_${u}.json reference/rpv_deploy_${u}.json; done
echo "(c) ALL streams int2 (harsh, untrained)"
RWKV_WEIGHTS=$W RWKV_STATE_QUANT=int2 RWKV_STATE_QUANT_SCOPE="all" $BIN $U >/dev/null 2>&1
for u in $U; do cp reference/rust_pred_${u}.json reference/rpv_allint2_${u}.json; done

$PY - <<'PY'
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss
REF = Path("reference"); U = [107,136,156]
def load(tag,u):
    d=json.load(open(REF/f"rpv_{tag}_{u}.json")); rth=d["review_th"]
    return {rt:p for rt,p in zip(rth,d["pred_imm"])}
def ll(u, pm):
    m=json.load(open(REF/f"trace_user_{u}.json")); eq=m["equalize_review_ths"]
    lr={int(k):int(v) for k,v in m["label_rating"].items()}
    lb={rt:int(np.clip(lr[rt],0,1)) for rt in eq}
    return log_loss([lb[rt] for rt in eq],[pm[rt] for rt in eq],labels=[0,1])
print(f"{'tag':<10} {'imm_logloss(mean)':>18} {'maxΔpred vs fp32':>18}")
base={u:load('fp32',u) for u in U}
for tag in ['fp32','deploy','allint2']:
    preds={u:load(tag,u) for u in U}
    lls=np.mean([ll(u,preds[u]) for u in U])
    md=max(max(abs(preds[u][rt]-base[u][rt]) for rt in base[u]) for u in U)
    print(f"{tag:<10} {lls:>18.6f} {md:>18.3e}")
print()
print("VERDICT: if 'deploy' maxΔ>0 the quant IS applied (just tolerated); if 'allint2' logloss")
print("blows up vs fp32, the quant path definitively changes predictions -> NOT a no-op.")
PY
echo "=== VERIFY DONE ==="
