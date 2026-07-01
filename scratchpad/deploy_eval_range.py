"""Deployed-eval scorer over a USER RANGE (generalizes qat_eval.py from the 17-user gate to 101-200).

Reads, per user u in [START, END):
  reference/trace_user_{u}.json          -- labels (label_rating, equalize_review_ths) + Python fp32 logloss
  reference/rust_pred_{tag}_{u}.json      -- Rust predictions for each requested tag (pred_imm/pred_ahead)
Computes by-user-mean imm/ahead LogLoss for each tag (sklearn log_loss on the equalized review set, the
same metric as get_result), plus the Python-fp32 by-user mean (parity sanity) and tag-vs-fp32 deltas.

Usage:
  python scratchpad/deploy_eval_range.py START END tag1 [tag2 ...]
  e.g. python scratchpad/deploy_eval_range.py 101 201 fp32 deploy
A user is included only if its trace + ALL requested tag preds exist (skips with a count of missing)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

REF = Path("reference")


def load_preds(path):
    d = json.load(open(path))
    rth = d["review_th"]
    imm = {rt: p for rt, p in zip(rth, d["pred_imm"])}
    ahead = {rt: p for rt, p in zip(rth, d["pred_ahead"]) if p is not None}
    return imm, ahead


def ll(label_bin, predmap, keys):
    return log_loss([label_bin[rt] for rt in keys], [predmap[rt] for rt in keys], labels=[0, 1])


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 201
    tags = sys.argv[3:] if len(sys.argv) > 3 else ["fp32", "deploy"]

    agg = {t: {"imm": [], "ahead": []} for t in tags}
    py = {"imm": [], "ahead": []}
    used, missing = [], 0
    for u in range(start, end):
        meta_p = REF / f"trace_user_{u}.json"
        if not meta_p.exists() or not all((REF / f"rust_pred_{t}_{u}.json").exists() for t in tags):
            missing += 1
            continue
        meta = json.load(open(meta_p))
        eq = meta["equalize_review_ths"]
        label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
        label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}
        preds = {t: load_preds(REF / f"rust_pred_{t}_{u}.json") for t in tags}
        imm_keys = [rt for rt in eq if all(rt in preds[t][0] for t in tags)]
        ah_keys = [rt for rt in eq if all(rt in preds[t][1] for t in tags)]
        if not imm_keys or not ah_keys:
            missing += 1
            continue
        for t in tags:
            agg[t]["imm"].append(ll(label_bin, preds[t][0], imm_keys))
            agg[t]["ahead"].append(ll(label_bin, preds[t][1], ah_keys))
        # py_* is the trace's Python fp32 logloss; present only if the trace was exported for THIS
        # champion. When reusing traces from a prior champion (inputs are weight-independent), it's the
        # OLD model's number -> optional + not a parity reference here.
        if "py_imm_logloss" in meta and "py_ahead_logloss" in meta:
            py["imm"].append(meta["py_imm_logloss"])
            py["ahead"].append(meta["py_ahead_logloss"])
        used.append(u)

    if not used:
        print(f"NO USERS scored (missing {missing}). Export traces + run Rust passes first.")
        return
    m = {t: {k: float(np.mean(v)) for k, v in agg[t].items()} for t in tags}
    print(f"users scored: {len(used)}  (skipped {missing} for missing trace/preds)")
    print(f"\n{'set':<16} {'imm':>10} {'ahead':>10}")
    for t in tags:
        print(f"{('rust_'+t):<16} {m[t]['imm']:>10.6f} {m[t]['ahead']:>10.6f}")
    if "fp32" in tags:  # the REAL signal: deploy-quant penalty vs the model's own Rust fp32
        print("\nQUANT PENALTY (rust_<deploy> - rust_fp32):")
        for t in tags:
            if t == "fp32":
                continue
            print(f"  {t:<10}: imm {m[t]['imm']-m['fp32']['imm']:+.6f}  ahead {m[t]['ahead']-m['fp32']['ahead']:+.6f}")
    if py["imm"]:  # only meaningful if traces were exported for THIS champion
        pym = {k: float(np.mean(v)) for k, v in py.items()}
        print(f"\n(trace python_fp32 [stale if traces reused]: imm {pym['imm']:.6f} ahead {pym['ahead']:.6f})")


if __name__ == "__main__":
    main()
