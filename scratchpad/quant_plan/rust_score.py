"""Score Rust engine predictions on the parity traces: deploy LogLoss per head, one row per config.

Mirrors the metric in `verify_rust.py` (sklearn log_loss over the trace's own equalize set, labels
binned from the row rating) so the numbers are on the same footing as every other Rust parity
number in this project. Use it to measure the DEPLOY quantization tax: run the engine twice on the
same traces, once unquantized and once with the candidate env, and compare the two rows.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/rust_score.py --trace-dir reference \
      --preds plain=preds_plain --preds int2=preds_int2 [--users 107,136,156]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[2]


def score_user(trace_dir, preds_dir, u):
    meta_path = trace_dir / f"trace_user_{u}.json"
    pred_path = preds_dir / f"rust_pred_{u}.json"
    if not meta_path.exists() or not pred_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rust = json.loads(pred_path.read_text(encoding="utf-8"))
    eq = meta["equalize_review_ths"]
    rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
    label_bin = {rt: int(np.clip(rating[rt], 0, 1)) for rt in eq if rt in rating}
    eq = [rt for rt in eq if rt in label_bin]
    rth = rust["review_th"]
    r_imm = {rt: p for rt, p in zip(rth, rust["pred_imm"])}
    r_ahead = {rt: p for rt, p in zip(rth, rust["pred_ahead"]) if p is not None}
    eq_imm = [rt for rt in eq if rt in r_imm]
    eq_ahead = [rt for rt in eq if rt in r_ahead]
    out = {"n_eq": len(eq), "n_imm": len(eq_imm), "n_ahead": len(eq_ahead)}
    out["imm"] = float(log_loss([label_bin[rt] for rt in eq_imm], [r_imm[rt] for rt in eq_imm],
                                labels=[0, 1])) if eq_imm else None
    out["ahead"] = float(log_loss([label_bin[rt] for rt in eq_ahead], [r_ahead[rt] for rt in eq_ahead],
                                  labels=[0, 1])) if eq_ahead else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--preds", action="append", required=True, help="name=dir (repeatable)")
    ap.add_argument("--users", default="107,136,156")
    args = ap.parse_args()

    trace_dir = ROOT / args.trace_dir
    users = [int(u) for u in args.users.split(",")]
    rows = {}
    for spec in args.preds:
        name, _, d = spec.partition("=")
        preds_dir = ROOT / d
        per_user = {u: score_user(trace_dir, preds_dir, u) for u in users}
        got = {u: v for u, v in per_user.items() if v}
        if not got:
            print(f"{name:8s} NO PREDS in {preds_dir}")
            continue
        mean_imm = float(np.mean([v["imm"] for v in got.values() if v["imm"] is not None]))
        mean_ahead = float(np.mean([v["ahead"] for v in got.values() if v["ahead"] is not None]))
        rows[name] = {"imm": mean_imm, "ahead": mean_ahead, "users": len(got),
                      "per_user": {str(u): v for u, v in got.items()}}
        print(f"{name:8s} n_users={len(got)}  ahead={mean_ahead:.6f}  imm={mean_imm:.6f}")

    if len(rows) > 1:
        base = next(iter(rows))
        for name, r in rows.items():
            if name == base:
                continue
            print(f"tax({name} - {base}): ahead={r['ahead'] - rows[base]['ahead']:+.6f}  "
                  f"imm={r['imm'] - rows[base]['imm']:+.6f}")
    out = ROOT / "scratchpad" / "quant_plan" / "rust_score.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
