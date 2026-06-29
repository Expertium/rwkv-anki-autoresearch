"""Gate the card-only state-quant Rust run over a range of users.

Baseline = the Python fp32 preds stored in reference/trace_user_{u}.json (parity proved
rust-fp32 == python-fp32 bit-exactly, so we skip a separate fp32 rust run). Candidate =
reference/rust_pred_{u}.json (produced by running rwkv-infer with RWKV_STATE_QUANT=...,
RWKV_STATE_QUANT_SCOPE=card over the same users). Reports by-user-mean imm/ahead LogLoss
for fp32 and quant, the delta, and the gate vs iter0 (+0.0015).

Usage: python scratchpad/state_quant_gate.py START END   (inclusive START, exclusive END)
"""
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

REF = Path("reference")
ITER0_IMM = 0.319475
ITER0_AHEAD = 0.374046
BUDGET = 0.0015


def ll(label_bin, predmap, keys):
    y = [label_bin[rt] for rt in keys]
    p = [predmap[rt] for rt in keys]
    return log_loss(y_true=y, y_pred=p, labels=[0, 1])


def user_pair(u):
    meta = json.load(open(REF / f"trace_user_{u}.json"))
    rust = json.load(open(REF / f"rust_pred_{u}.json"))
    eq = meta["equalize_review_ths"]
    label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
    label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}

    py_imm = {int(k): float(v) for k, v in meta["py_pred_imm"].items()}
    py_ahead = {int(k): float(v) for k, v in meta["py_pred_ahead"].items()}
    rth = rust["review_th"]
    r_imm = {rt: p for rt, p in zip(rth, rust["pred_imm"])}
    r_ahead = {rt: p for rt, p in zip(rth, rust["pred_ahead"]) if p is not None}

    imm_keys = [rt for rt in eq if rt in py_imm and rt in r_imm]
    ahead_keys = [rt for rt in eq if rt in py_ahead and rt in r_ahead]
    return {
        "imm_fp32": ll(label_bin, py_imm, imm_keys),
        "imm_quant": ll(label_bin, r_imm, imm_keys),
        "ahead_fp32": ll(label_bin, py_ahead, ahead_keys),
        "ahead_quant": ll(label_bin, r_ahead, ahead_keys),
    }


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 201
    rows = {}
    for u in range(start, end):
        if (REF / f"trace_user_{u}.json").exists() and (REF / f"rust_pred_{u}.json").exists():
            rows[u] = user_pair(u)
    n = len(rows)
    if n == 0:
        print("no users with both trace + rust_pred present")
        return
    imm_fp32 = np.mean([rows[u]["imm_fp32"] for u in rows])
    imm_quant = np.mean([rows[u]["imm_quant"] for u in rows])
    ah_fp32 = np.mean([rows[u]["ahead_fp32"] for u in rows])
    ah_quant = np.mean([rows[u]["ahead_quant"] for u in rows])

    print(f"users: {n}  (range {start}..{end-1})")
    print(f"  imm   fp32 {imm_fp32:.6f} -> quant {imm_quant:.6f}  delta {imm_quant-imm_fp32:+.6f}")
    print(f"  ahead fp32 {ah_fp32:.6f} -> quant {ah_quant:.6f}  delta {ah_quant-ah_fp32:+.6f}")
    print(f"\nGATE vs iter0 (+{BUDGET}):")
    imm_ok = imm_quant <= ITER0_IMM + BUDGET
    ah_ok = ah_quant <= ITER0_AHEAD + BUDGET
    print(f"  imm   {imm_quant:.6f} <= {ITER0_IMM+BUDGET:.6f}  {'PASS' if imm_ok else 'FAIL'}")
    print(f"  ahead {ah_quant:.6f} <= {ITER0_AHEAD+BUDGET:.6f}  {'PASS' if ah_ok else 'FAIL'}")
    print(f"  => {'PASS' if (imm_ok and ah_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
