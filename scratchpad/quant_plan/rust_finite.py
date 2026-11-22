"""Deploy-path finiteness check: run the RUST engine with RWKV_FINITE_CHECK=1 and score the counters.

This is the deploy twin of `scratchpad/quant_plan/corruption/` (Andrew's finiteness gate, 2026-11-21).
The Python probe watches the TRAINING path; this one watches the engine that actually ships, where a
non-finite value is otherwise invisible because `san()` maps a non-finite p(recall) to 0.5 and
`compress_wkv_state` skips degenerate components.

A config FAILS when any user reports a non-zero count in `state_raw_bad` (the recurrence blew up),
`state_quant_bad` (quantization produced garbage), `vec_raw_bad` / `vec_quant_bad` (token shifts), or
`p_bad` (a p(recall) that `san()` had to hide). `maxabs` is reported as the state-growth signal.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/rust_finite.py --weights <sft> \
      --configs plain,l2i8s4,l2i8n2i8 --users 107,136,156 [--trace-dir reference] [--strict] [--tag t]

Writes scratchpad/quant_plan/rust_finite_<tag>.json. `--strict` makes the engine itself exit 3 on any
non-finite value (a second, independent witness of the same fact).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402

RE_FIN = re.compile(
    r"\[finite\] user=(-?\d+) compressions=(\d+) state_vals=(\d+) state_raw_bad=(\d+) "
    r"state_quant_bad=(\d+) vec_vals=(\d+) vec_raw_bad=(\d+) vec_quant_bad=(\d+) "
    r"p_vals=(\d+) p_bad=(\d+) state_maxabs=([\d.eE+-]+) vec_maxabs=([\d.eE+-]+)")
RE_TOTAL = re.compile(r"\[finite\] TOTAL (.*)$", re.M)
RE_KV = re.compile(r"([a-z_]+)=(-?[\d.eE+]+)")
KEYED = ("compressions", "state_vals", "state_raw_bad", "state_quant_bad", "vec_vals",
         "vec_raw_bad", "vec_quant_bad", "p_vals", "p_bad")


def parse_total(text):
    """Parse the TOTAL line as generic key=value pairs: adding a counter must not break the driver.

    (The regex that hard-coded `p_bad=.. planted=..` silently stopped matching when the maxima were
    split into two fields, which turned a planted STRICT failure into `INCOMPLETE`.)
    """
    m = RE_TOTAL.search(text)
    if not m:
        return {}
    kv = {k: float(v) for k, v in RE_KV.findall(m.group(1))}
    return {"total_p_bad": int(kv.get("p_bad", 0)), "planted": int(kv.get("planted", 0)),
            "non_finite_total": int(kv.get("non_finite_total", 0))}


def run_one(weights, name, users, trace_dir, preds_root, strict):
    env = dict(os.environ)
    env["RWKV_WEIGHTS"] = str(weights)
    env["RWKV_TRACE_DIR"] = str(trace_dir)
    env["RWKV_PRED_DIR"] = str(preds_root / name)
    env["RWKV_FINITE_CHECK"] = "strict" if strict else "1"
    env.pop("RWKV_BENCH", None)
    for k in ("RWKV_STATE_QUANT_SCOPE", "RWKV_STATE_LOWRANK_SCOPE", "RWKV_QUANT_SHIFTS",
              "RWKV_STATE_SHIFT_LEVEL", "RWKV_LOWRANK_HADAMARD", "RWKV_LOWRANK_PERCOL",
              "RWKV_LOWRANK_ALS", "RWKV_LOWRANK_4LEVEL", "RWKV_LOWRANK_MIXED53",
              "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in rg.CONFIGS[name].items()})
    r = subprocess.run([str(rg.ENGINE)] + [str(u) for u in users], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    per_user, totals = {}, {}
    for m in RE_FIN.finditer(r.stdout + r.stderr):
        rec = dict(zip(KEYED, (int(x) for x in m.groups()[1:10])))
        rec["state_maxabs"] = float(m.group(11))
        rec["vec_maxabs"] = float(m.group(12))
        per_user[int(m.group(1))] = rec
    return per_user, parse_total(r.stdout + r.stderr), r.returncode, (r.stdout + r.stderr)[-600:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--configs", required=True, help="comma list of rust_grid.CONFIGS names, `plain` first")
    ap.add_argument("--users", default="107,136,156")
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--tag", default="check")
    ap.add_argument("--strict", action="store_true",
                    help="let the engine exit 3 on any non-finite value (an independent witness)")
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    users = [int(u) for u in args.users.split(",") if u.strip()]
    preds_root = ROOT / "scratchpad" / "quant_plan" / "rust_finite_preds" / args.tag

    out = {"tag": args.tag, "weights": str(args.weights), "users": users, "trace_dir": args.trace_dir,
           "strict": bool(args.strict), "configs": {}}
    print(f"engine {rg.ENGINE}")
    print(f"weights {args.weights}\nusers {users}  configs {names}  strict={args.strict}\n")
    for name in names:
        if name not in rg.CONFIGS:
            raise SystemExit(f"unknown config {name!r}")
        per_user, totals, rc, tail = run_one(args.weights, name, users, args.trace_dir, preds_root,
                                             args.strict)
        bad_keys = [k for rec in per_user.values() for k in
                    ("state_raw_bad", "state_quant_bad", "vec_raw_bad", "vec_quant_bad", "p_bad")
                    if rec[k] > 0]
        if totals.get("planted", 0) > 0:
            bad_keys.append("planted")  # the RWKV_FINITE_SELFTEST witness
        # INCOMPLETE, never a pass: no [finite] lines at all means the counter never ran.
        if not per_user:
            status = "INCOMPLETE"
        elif rc != 0 and not args.strict:
            status = "INCOMPLETE"
        else:
            status = "FAIL" if bad_keys else ("PASS" if rc == 0 else "INCOMPLETE")
        rec = {"status": status, "rc": rc, "per_user": per_user, "totals": totals,
               "offending_keys": sorted(set(bad_keys)), "tail": tail}
        out["configs"][name] = rec
        worst_state = max((v["state_maxabs"] for v in per_user.values()), default=float("nan"))
        worst_vec = max((v["vec_maxabs"] for v in per_user.values()), default=float("nan"))
        print(f"{name:<12} {status:<10} rc={rc} users={len(per_user)} "
              f"state_vals={sum(v['state_vals'] for v in per_user.values())} "
              f"state_raw_bad={sum(v['state_raw_bad'] for v in per_user.values())} "
              f"state_quant_bad={sum(v['state_quant_bad'] for v in per_user.values())} "
              f"vec_raw_bad={sum(v['vec_raw_bad'] for v in per_user.values())} "
              f"vec_quant_bad={sum(v['vec_quant_bad'] for v in per_user.values())} "
              f"p_vals={sum(v['p_vals'] for v in per_user.values())} "
              f"p_bad={sum(v['p_bad'] for v in per_user.values())} "
              f"state_maxabs={worst_state:.4e} vec_maxabs={worst_vec:.4e}")
        if status != "PASS":
            print(f"    offending keys: {rec['offending_keys']}  tail: {tail.strip()[-200:]}")

    dst = ROOT / "scratchpad" / "quant_plan" / f"rust_finite_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten to {dst}")
    return 0 if all(c["status"] == "PASS" for c in out["configs"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
