"""Verify the Rust RWKV port matches the Python RNN reference.

For each reference user we compare, on the equalized benchmark review set:
  - per-review predictions (Rust vs Python) -> max abs diff (should be ~f32 precision),
  - mean LogLoss (sklearn, same as get_stats) for imm + ahead.
Gate (CLAUDE.md §5): each mode's by-user-mean LogLoss within +/-0.0005 of Python.

Run: python verify_rust.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

REF_USERS = [107, 136, 156]
REF = Path("reference")
TOL = 0.0005


def logloss(review_ths, pred, label_bin):
    y = [label_bin[rt] for rt in review_ths]
    p = [pred[rt] for rt in review_ths]
    return log_loss(y_true=y, y_pred=p, labels=[0, 1])


def verify_user(u):
    meta = json.load(open(REF / f"trace_user_{u}.json"))
    rust = json.load(open(REF / f"rust_pred_{u}.json"))

    eq = meta["equalize_review_ths"]
    label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
    label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}
    py_imm = {int(k): float(v) for k, v in meta["py_pred_imm"].items()}
    py_ahead = {int(k): float(v) for k, v in meta["py_pred_ahead"].items()}

    rth = rust["review_th"]
    r_imm = {rt: p for rt, p in zip(rth, rust["pred_imm"])}
    r_ahead = {
        rt: p for rt, p in zip(rth, rust["pred_ahead"]) if p is not None
    }

    # per-review max abs diff on the equalize set
    d_imm = max(abs(r_imm[rt] - py_imm[rt]) for rt in eq)
    d_ahead = max(abs(r_ahead[rt] - py_ahead[rt]) for rt in eq)

    ll_imm_r = logloss(eq, r_imm, label_bin)
    ll_imm_py = logloss(eq, py_imm, label_bin)
    ll_ahead_r = logloss(eq, r_ahead, label_bin)
    ll_ahead_py = logloss(eq, py_ahead, label_bin)

    print(
        f"user {u}: size {len(eq)}  "
        f"imm[py {ll_imm_py:.6f} rust {ll_imm_r:.6f} dpred {d_imm:.2e}]  "
        f"ahead[py {ll_ahead_py:.6f} rust {ll_ahead_r:.6f} dpred {d_ahead:.2e}]"
    )
    return {
        "imm_r": ll_imm_r, "imm_py": ll_imm_py,
        "ahead_r": ll_ahead_r, "ahead_py": ll_ahead_py,
        "d_imm": d_imm, "d_ahead": d_ahead,
    }


def main():
    rows = [verify_user(u) for u in REF_USERS]
    mean_imm_r = np.mean([r["imm_r"] for r in rows])
    mean_imm_py = np.mean([r["imm_py"] for r in rows])
    mean_ahead_r = np.mean([r["ahead_r"] for r in rows])
    mean_ahead_py = np.mean([r["ahead_py"] for r in rows])
    max_dpred = max(max(r["d_imm"], r["d_ahead"]) for r in rows)

    ref = json.load(open(REF / "ref_metrics.json"))
    print("\n=== by-user mean LogLoss ===")
    print(f"imm   : rust {mean_imm_r:.6f}  python {mean_imm_py:.6f}  "
          f"ref {ref['mean_imm_logloss']:.6f}")
    print(f"ahead : rust {mean_ahead_r:.6f}  python {mean_ahead_py:.6f}  "
          f"ref {ref['mean_ahead_logloss']:.6f}")
    di = abs(mean_imm_r - mean_imm_py)
    da = abs(mean_ahead_r - mean_ahead_py)
    print(f"\nmean-logloss |rust-python|: imm {di:.6f}, ahead {da:.6f}  (tol {TOL})")
    print(f"max per-review |rust-python| pred diff: {max_dpred:.2e}")

    ok = di <= TOL and da <= TOL
    print("\nPARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
