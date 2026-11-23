"""Measure the DEPLOY engine's inference speed per quantization config.

Constraint 6 of the state-quantization task is about the Rust engine (Andrew: "I meant Rust"):
quantized inference must be no more than 25% slower than unquantized, i.e. ratio >= 0.75x.
The engine prints two speed numbers per user:

    user <u>: <n> reviews in <t>s (<rate> rev/s) -> <path>            (stdout)
    [speed] user <u>: R/s (query pass, retrievability) <q>   states/s (state pass, review_features) <s>  (stderr)

`rev/s` is the per-review total (what "how long eval takes" means); `states/s` is the official
DEPLOY_SPEED_LOG deploy metric. Both are reported absolutely and relative to the unquantized run,
which is always measured first, plus the median of the per-user ratios (DEPLOY_SPEED_LOG pairs on a
user).

Configs reuse `rust_grid.CONFIGS` so the speed table and the loss table describe the same thing.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/rust_speed.py --configs plain,l1i4,cn2 [--reps 2]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402

RE_USER = re.compile(r"^user (\d+): (\d+) reviews in ([\d.]+)s \(([\d.]+) rev/s\)", re.M)
RE_SPEED = re.compile(
    r"\[speed\] user (\d+): R/s \(query pass, retrievability\) ([\d.]+)\s+"
    r"states/s \(state pass, review_features\) ([\d.]+)")


def run_once(weights, trace_dir, users, name, preds_root):
    """Run the engine for one config; return {user: (reviews, secs, rev_s, r_s, states_s)} or None."""
    env = dict(os.environ)
    env["RWKV_WEIGHTS"] = str(weights)
    env["RWKV_TRACE_DIR"] = str(trace_dir)
    env["RWKV_PRED_DIR"] = str(preds_root / name)
    env.pop("RWKV_BENCH", None)
    for k in ("RWKV_STATE_QUANT_SCOPE", "RWKV_STATE_LOWRANK_SCOPE", "RWKV_QUANT_SHIFTS",
              "RWKV_STATE_SHIFT_LEVEL", "RWKV_LOWRANK_HADAMARD", "RWKV_LOWRANK_PERCOL",
              "RWKV_LOWRANK_ALS", "RWKV_LOWRANK_4LEVEL", "RWKV_LOWRANK_MIXED53",
              "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT",
              # The fixed-basis keys MUST be cleared too. Without them a run that puts a codec config
              # before a non-codec one leaks the basis into the later config (rust_grid.py already
              # clears both); with them the order cannot matter. Added 2026-11-23, when the first
              # fixed-basis speed cells were about to be measured.
              "RWKV_STATE_FIXEDBASIS_SCOPE", "RWKV_FIXEDBASIS_PATH"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in rg.CONFIGS[name].items()})
    r = subprocess.run([str(rg.ENGINE)] + [str(u) for u in users], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-400:]
    out = {}
    for u, n, secs, rate in RE_USER.findall(r.stdout):
        out[int(u)] = {"n": int(n), "secs": float(secs), "rev_s": float(rate)}
    for u, q, s in RE_SPEED.findall(r.stdout + r.stderr):
        if int(u) in out:
            out[int(u)]["r_s"] = float(q)
            out[int(u)]["states_s"] = float(s)
    return out, None


def summarise(per_user):
    tot_n = sum(v["n"] for v in per_user.values())
    tot_t = sum(v["secs"] for v in per_user.values())
    agg = {"rev_s": tot_n / tot_t if tot_t else float("nan"),
           "states_s": statistics.mean([v["states_s"] for v in per_user.values()
                                        if "states_s" in v] or [float("nan")]),
           "r_s": statistics.mean([v["r_s"] for v in per_user.values() if "r_s" in v]
                                  or [float("nan")])}
    agg["per_user_rev_s"] = {u: v["rev_s"] for u, v in per_user.items()}
    agg["per_user_states_s"] = {u: v.get("states_s") for u, v in per_user.items()}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(rg.ART / "adaptive173416.safetensors"))
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--users", default="107,136,156")
    ap.add_argument("--configs", default="plain,cn2,l1i4,l1i4c,l1i4n,l1i8")
    ap.add_argument("--preds-root", default=str(rg.ART.parent / "rust_speed_preds"))
    ap.add_argument("--tag", default="speed")
    ap.add_argument("--reps", type=int, default=1, help="runs per config; the best is kept")
    args = ap.parse_args()

    users = [int(u) for u in args.users.split(",")]
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    if "plain" not in names:
        names = ["plain"] + names
    preds_root = Path(args.preds_root)
    preds_root.mkdir(parents=True, exist_ok=True)
    trace_dir = ROOT / args.trace_dir

    rows = {}
    for name in names:
        best, err = None, None
        for _ in range(max(1, args.reps)):
            pu, err = run_once(Path(args.weights), trace_dir, users, name, preds_root)
            if pu is None:
                break
            s = summarise(pu)
            if best is None or s["rev_s"] > best["rev_s"]:
                best = s
        if best is None:
            print(f"{name:8s} ENGINE FAILED: {err}")
            continue
        rows[name] = best
        print(f"{name:8s} rev/s={best['rev_s']:9.1f}  states/s={best['states_s']:8.1f}  "
              f"R/s={best['r_s']:8.1f}")

    if "plain" in rows:
        base = rows["plain"]
        print(f"\n{'config':9s} {'rev/s':>9s} {'x rev':>7s} {'states/s':>9s} {'x st':>7s} "
              f"{'median per-user x':>17s}  constraint 6 (>= 0.75x)")
        for name, r in rows.items():
            xr = r["rev_s"] / base["rev_s"]
            xs = r["states_s"] / base["states_s"]
            common = sorted(set(r["per_user_rev_s"]) & set(base["per_user_rev_s"]))
            ratios = [r["per_user_rev_s"][u] / base["per_user_rev_s"][u] for u in common]
            med = statistics.median(ratios) if ratios else float("nan")
            gate = "PASS" if med >= 0.75 else "FAIL"
            print(f"{name:9s} {r['rev_s']:9.1f} {xr:7.3f} {r['states_s']:9.1f} {xs:7.3f} "
                  f"{med:17.3f}  {gate}")
    out = rg.ART.parent / f"rust_speed_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
