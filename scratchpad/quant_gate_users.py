"""Gate a state-quant Rust run over an explicit user list, using a rust FP32 run as the
baseline (parity proved rust-fp32 == python bit-exactly, and the fast feature-only export
has no py preds). Reads labels from reference/trace_user_{u}.json (equalize set + label_rating),
fp32 preds from reference/rust_pred_fp32_{u}.json, quant preds from reference/rust_pred_{u}.json.

Reports by-user-mean imm/ahead LogLoss for fp32 and quant, the delta, and the gate vs iter0
(imm 0.319475, ahead 0.374046; budget +0.0015).

Usage: python scratchpad/quant_gate_users.py [u1 u2 ...]   (default = 18 smallest of 101-200)
"""
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

REF = Path("reference")
ITER0_IMM, ITER0_AHEAD, BUDGET = 0.319475, 0.374046, 0.0015
DEFAULT_USERS = [107, 110, 116, 120, 121, 128, 136, 146, 150, 151,
                 156, 159, 162, 165, 175, 176, 187, 198]


def ll(label_bin, predmap, keys):
    y = [label_bin[rt] for rt in keys]
    p = [predmap[rt] for rt in keys]
    return log_loss(y_true=y, y_pred=p, labels=[0, 1])


def load_preds(path):
    d = json.load(open(path))
    rth = d["review_th"]
    imm = {rt: p for rt, p in zip(rth, d["pred_imm"])}
    ahead = {rt: p for rt, p in zip(rth, d["pred_ahead"]) if p is not None}
    return imm, ahead


def user_row(u):
    meta = json.load(open(REF / f"trace_user_{u}.json"))
    eq = meta["equalize_review_ths"]
    label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
    label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}
    b_imm, b_ahead = load_preds(REF / f"rust_pred_fp32_{u}.json")
    c_imm, c_ahead = load_preds(REF / f"rust_pred_{u}.json")
    imm_keys = [rt for rt in eq if rt in b_imm and rt in c_imm]
    ah_keys = [rt for rt in eq if rt in b_ahead and rt in c_ahead]
    return {
        "imm_fp32": ll(label_bin, b_imm, imm_keys),
        "imm_quant": ll(label_bin, c_imm, imm_keys),
        "ahead_fp32": ll(label_bin, b_ahead, ah_keys),
        "ahead_quant": ll(label_bin, c_ahead, ah_keys),
    }


def main():
    users = [int(x) for x in sys.argv[1:]] or DEFAULT_USERS
    rows = {}
    for u in users:
        if (REF / f"trace_user_{u}.json").exists() and \
           (REF / f"rust_pred_fp32_{u}.json").exists() and \
           (REF / f"rust_pred_{u}.json").exists():
            rows[u] = user_row(u)
    n = len(rows)
    if n == 0:
        print("no users with trace + fp32 + quant preds present")
        return
    imm_fp32 = np.mean([rows[u]["imm_fp32"] for u in rows])
    imm_quant = np.mean([rows[u]["imm_quant"] for u in rows])
    ah_fp32 = np.mean([rows[u]["ahead_fp32"] for u in rows])
    ah_quant = np.mean([rows[u]["ahead_quant"] for u in rows])
    print(f"users gated: {n}  {sorted(rows)}")
    print(f"  imm   fp32 {imm_fp32:.6f} -> quant {imm_quant:.6f}  delta {imm_quant-imm_fp32:+.6f}")
    print(f"  ahead fp32 {ah_fp32:.6f} -> quant {ah_quant:.6f}  delta {ah_quant-ah_fp32:+.6f}")
    imm_ok = imm_quant <= ITER0_IMM + BUDGET
    ah_ok = ah_quant <= ITER0_AHEAD + BUDGET
    print(f"\nGATE vs iter0 (+{BUDGET}):")
    print(f"  imm   {imm_quant:.6f} <= {ITER0_IMM+BUDGET:.6f}  {'PASS' if imm_ok else 'FAIL'}")
    print(f"  ahead {ah_quant:.6f} <= {ITER0_AHEAD+BUDGET:.6f}  {'PASS' if ah_ok else 'FAIL'}")
    print(f"  => {'PASS' if imm_ok and ah_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
