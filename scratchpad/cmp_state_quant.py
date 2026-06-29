"""Compare two Rust prediction sets (fp32 baseline vs a state-quant candidate) on the
parity users, reusing the trace_user_*.json labels/equalize set that verify_rust.py uses.

Usage:
  python scratchpad/cmp_state_quant.py BASE_SUFFIX CAND_SUFFIX
where the Rust preds live at reference/rust_pred{BASE_SUFFIX}_{u}.json etc.
(suffix "" = reference/rust_pred_{u}.json). Prints per-user + by-user-mean imm/ahead
LogLoss for both, the delta (cand - base), and max abs per-review prediction drift.

This is a DIRECTIONAL signal on the 3 parity users (107/136/156), not the 100-user gate.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

REF_USERS = [107, 136, 156]
REF = Path("reference")


def load_rust(suffix, u):
    p = REF / f"rust_pred{suffix}_{u}.json" if suffix else REF / f"rust_pred_{u}.json"
    d = json.load(open(p))
    rth = d["review_th"]
    imm = {rt: p for rt, p in zip(rth, d["pred_imm"])}
    ahead = {rt: p for rt, p in zip(rth, d["pred_ahead"]) if p is not None}
    return imm, ahead


def user_logloss(u, base_suffix, cand_suffix):
    meta = json.load(open(REF / f"trace_user_{u}.json"))
    eq = meta["equalize_review_ths"]
    label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
    label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}

    b_imm, b_ahead = load_rust(base_suffix, u)
    c_imm, c_ahead = load_rust(cand_suffix, u)

    def ll(predmap, keys):
        y = [label_bin[rt] for rt in keys]
        p = [predmap[rt] for rt in keys]
        return log_loss(y_true=y, y_pred=p, labels=[0, 1])

    imm_keys = [rt for rt in eq if rt in b_imm and rt in c_imm]
    ahead_keys = [rt for rt in eq if rt in b_ahead and rt in c_ahead]

    res = {
        "imm_base": ll(b_imm, imm_keys),
        "imm_cand": ll(c_imm, imm_keys),
        "ahead_base": ll(b_ahead, ahead_keys),
        "ahead_cand": ll(c_ahead, ahead_keys),
    }
    # max abs drift over the equalized imm set
    res["imm_maxdiff"] = max(abs(c_imm[rt] - b_imm[rt]) for rt in imm_keys)
    res["ahead_maxdiff"] = max(abs(c_ahead[rt] - b_ahead[rt]) for rt in ahead_keys) if ahead_keys else 0.0
    return res


def main():
    base_suffix = sys.argv[1] if len(sys.argv) > 1 else "_fp32"
    cand_suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"base = rust_pred{base_suffix}_*  cand = rust_pred{cand_suffix or '_'}*\n")
    rows = {u: user_logloss(u, base_suffix, cand_suffix) for u in REF_USERS}
    for u in REF_USERS:
        r = rows[u]
        print(
            f"user {u}: imm {r['imm_base']:.6f}->{r['imm_cand']:.6f} "
            f"(d {r['imm_cand']-r['imm_base']:+.6f}, maxdiff {r['imm_maxdiff']:.2e}) | "
            f"ahead {r['ahead_base']:.6f}->{r['ahead_cand']:.6f} "
            f"(d {r['ahead_cand']-r['ahead_base']:+.6f})"
        )
    imm_b = np.mean([rows[u]["imm_base"] for u in REF_USERS])
    imm_c = np.mean([rows[u]["imm_cand"] for u in REF_USERS])
    ah_b = np.mean([rows[u]["ahead_base"] for u in REF_USERS])
    ah_c = np.mean([rows[u]["ahead_cand"] for u in REF_USERS])
    print(
        f"\nBY-USER MEAN (3 users):"
        f"\n  imm   {imm_b:.6f} -> {imm_c:.6f}  delta {imm_c-imm_b:+.6f}"
        f"\n  ahead {ah_b:.6f} -> {ah_c:.6f}  delta {ah_c-ah_b:+.6f}"
    )
    print("\nNOTE: 3-user directional signal only; the gate is by-user mean over 101-200.")


if __name__ == "__main__":
    main()
