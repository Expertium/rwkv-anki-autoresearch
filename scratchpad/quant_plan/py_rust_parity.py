"""Python<->Rust parity on ONE trace dir, for ONE recipe.

Generalises the repo-root `verify_rust.py` in the two ways a recipe check needs:

* it does not read `ref_metrics.json` -- that file is the June-2026 d=128 baseline, so quoting it
  says nothing about the current lineage. The parity target here is the trace's OWN
  `py_pred_imm` / `py_pred_ahead`, which the exporter wrote from the same weights and the same
  quantization env the engine is about to run (see `corruption/export_parity_trace.py`);
* it takes the quantization env from `rust_grid.CONFIGS`, so the trace and the engine are described
  by the same config name.

Gate (AGENTS.md sec.5): |mean LogLoss(rust) - mean LogLoss(python)| <= 0.0005 per head, on the
trace's equalize set with the row rating binned to 0/1.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/py_rust_parity.py \
      --weights scratchpad/row17_l2i8s4_bundle/weights/deploy_iter17_l2i8s4.safetensors \
      --trace-dir scratchpad/quant_plan/trace_iter17_parity --config l2i8s4 --users 107
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[2]  # quant_plan -> scratchpad -> repo root
if not (ROOT / "rwkv").is_dir():
    raise SystemExit(f"repo root not found (looked for {ROOT / 'rwkv'})")
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402

TOL = 0.0005


def logloss(review_ths, pred, label_bin):
    return float(log_loss(y_true=[label_bin[rt] for rt in review_ths],
                          y_pred=[pred[rt] for rt in review_ths], labels=[0, 1]))


def run_engine(weights, trace_dir, users, name, preds_dir):
    env = dict(os.environ)
    env["RWKV_WEIGHTS"] = str(weights)
    env["RWKV_TRACE_DIR"] = str(trace_dir)
    env["RWKV_PRED_DIR"] = str(preds_dir)
    env.pop("RWKV_BENCH", None)
    for k in ("RWKV_STATE_QUANT_SCOPE", "RWKV_STATE_LOWRANK_SCOPE", "RWKV_QUANT_SHIFTS",
              "RWKV_STATE_SHIFT_LEVEL", "RWKV_LOWRANK_HADAMARD", "RWKV_LOWRANK_PERCOL",
              "RWKV_LOWRANK_ALS", "RWKV_LOWRANK_4LEVEL", "RWKV_LOWRANK_MIXED53",
              "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in rg.CONFIGS[name].items()})
    r = subprocess.run([str(rg.ENGINE)] + [str(u) for u in users], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-500:]


def check_user(trace_dir, preds_dir, u):
    meta = json.loads((trace_dir / f"trace_user_{u}.json").read_text(encoding="utf-8"))
    rust = json.loads((preds_dir / f"rust_pred_{u}.json").read_text(encoding="utf-8"))
    eq = [int(t) for t in meta["equalize_review_ths"]]
    rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
    label_bin = {rt: int(np.clip(rating[rt], 0, 1)) for rt in eq if rt in rating}
    py_imm = {int(k): float(v) for k, v in meta["py_pred_imm"].items()}
    py_ahead = {int(k): float(v) for k, v in meta["py_pred_ahead"].items()}
    rth = [int(t) for t in rust["review_th"]]
    r_imm = {rt: p for rt, p in zip(rth, rust["pred_imm"])}
    r_ahead = {rt: p for rt, p in zip(rth, rust["pred_ahead"]) if p is not None}
    common = [rt for rt in eq if rt in label_bin and rt in r_imm and rt in py_imm]
    out = {"n_eq": len(eq), "n_common": len(common)}
    for head, r_pred, p_pred in (("imm", r_imm, py_imm), ("ahead", r_ahead, py_ahead)):
        got = [rt for rt in common if rt in r_pred and rt in p_pred]
        if not got:
            out[head] = None
            continue
        ll_r = logloss(got, r_pred, label_bin)
        ll_p = logloss(got, p_pred, label_bin)
        out[head] = {"n": len(got), "ll_rust": ll_r, "ll_py": ll_p, "d_ll": ll_r - ll_p,
                     "max_dpred": max(abs(r_pred[rt] - p_pred[rt]) for rt in got)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--config", default="l2i8s4")
    ap.add_argument("--users", default="107")
    ap.add_argument("--tag", default="parity")
    args = ap.parse_args()

    trace_dir = ROOT / args.trace_dir
    users = [int(u) for u in args.users.split(",") if u.strip()]
    preds_dir = ROOT / "scratchpad" / "quant_plan" / "parity_preds" / args.tag
    preds_dir.mkdir(parents=True, exist_ok=True)

    rc, tail = run_engine(Path(args.weights), trace_dir, users, args.config, preds_dir)
    print(f"engine rc={rc} config={args.config} preds={preds_dir}")
    if rc != 0:
        print(tail)
        return 1

    doc = {"weights": str(args.weights), "trace_dir": str(trace_dir), "config": args.config,
           "env": {k: str(v) for k, v in rg.CONFIGS[args.config].items()}, "users": {}}
    for u in users:
        rec = check_user(trace_dir, preds_dir, u)
        doc["users"][str(u)] = rec
        if rec.get("imm") is None or rec.get("ahead") is None:
            print(f"user {u}: INCOMPLETE (no common reviews)")
            continue
        print(f"user {u}: n={rec['n_common']}  "
              f"imm[rust {rec['imm']['ll_rust']:.6f} py {rec['imm']['ll_py']:.6f} "
              f"d {rec['imm']['d_ll']:+.6f} maxdpred {rec['imm']['max_dpred']:.2e}]  "
              f"ahead[rust {rec['ahead']['ll_rust']:.6f} py {rec['ahead']['ll_py']:.6f} "
              f"d {rec['ahead']['d_ll']:+.6f} maxdpred {rec['ahead']['max_dpred']:.2e}]")

    ok = all(abs(v[h]["d_ll"]) <= TOL for v in doc["users"].values() if v.get("imm")
             for h in ("imm", "ahead"))
    doc["tolerance"] = TOL
    doc["verdict"] = "PASS" if ok else "FAIL"
    out = ROOT / "scratchpad" / "quant_plan" / f"py_rust_parity_{args.tag}.json"
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nPARITY: {doc['verdict']} (tol {TOL})   written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
