"""Correctness gate for the FAST rank-r deploy path (`RWKV_LOWRANK_FAST`, the E2 work in model.rs).

The rank>=2 branch of `compress_wkv_state` used to be a KxK Gram matrix + `SymmetricEigen` per layer
per step, which cost 0.600x rev/s on a rank-2 NOTE (a FAIL on the 0.75x deploy floor). The fast path
replaces it with a warm-started block subspace iteration + Rayleigh-Ritz. It is APPROXIMATE, so it
needs three measured facts before it can be trusted:

  1. CHECK is EXACT. `RWKV_LOWRANK_FAST_CHECK=1` runs both solvers and writes the REFERENCE factors, so
     a checked run must reproduce the reference run BIT-FOR-BIT -- and its
     `max_relFrobenius_vs_reference` tells us how far the fast reconstruction is from the reference one.
  2. FAST drifts how much? `RWKV_LOWRANK_FAST=1` (no check) writes the fast factors. The engine's own
     `rust_pred_<user>.json` dumps are compared against the reference run: text identity plus the
     largest per-review change in p(recall).
  3. The counters must add up: every rank>=2 call goes through the fast path (`reference fallbacks=0`
     on a covered shape), and a rank-1 config (`l2i8s4`) must report `calls=0`.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/rust_fastrank_check.py \
      --weights scratchpad/quant_plan/rust_artifacts/it36l2i8n2i8_124.safetensors \
      --configs l2i8s4,l2i8n2i8,l3i8n2i8 --users 107,136,156 --iters 2

Writes `scratchpad/quant_plan/rust_fastrank_<tag>.json` plus the prediction dumps under
`scratchpad/quant_plan/rust_fastrank_preds_<tag>/<mode>/<config>_it<iters>/`.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402

RE_FAST = re.compile(
    r"\[fastrank\] mode=(\S+) iters=(\d+) rank>=2 calls=(\d+) reference fallbacks=(\d+) "
    r"max_relFrobenius_vs_reference=([\d.eE+-]+) rel_buckets=(\d+)/(\d+)/(\d+)/(\d+)/(\d+)")
CLEAR = ("RWKV_STATE_QUANT_SCOPE", "RWKV_STATE_LOWRANK_SCOPE", "RWKV_QUANT_SHIFTS",
         "RWKV_STATE_SHIFT_LEVEL", "RWKV_LOWRANK_HADAMARD", "RWKV_LOWRANK_PERCOL",
         "RWKV_LOWRANK_ALS", "RWKV_LOWRANK_4LEVEL", "RWKV_LOWRANK_MIXED53",
         "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT",
         "RWKV_LOWRANK_FAST", "RWKV_LOWRANK_FAST_CHECK", "RWKV_LOWRANK_FAST_ITERS")


def run(weights, trace_dir, users, cfg, iters, mode, preds_dir):
    env = dict(os.environ)
    for k in CLEAR:
        env.pop(k, None)
    env["RWKV_WEIGHTS"] = str(weights)
    env["RWKV_TRACE_DIR"] = str(trace_dir)
    env["RWKV_PRED_DIR"] = str(preds_dir)
    env.pop("RWKV_BENCH", None)
    env.update({k: str(v) for k, v in rg.CONFIGS[cfg].items()})
    env["RWKV_LOWRANK_FAST_ITERS"] = str(iters)
    if mode == "check":
        env["RWKV_LOWRANK_FAST_CHECK"] = "1"
    elif mode == "fast":
        env["RWKV_LOWRANK_FAST"] = "1"
    elif mode == "ref":
        # The deploy config CARRIES RWKV_LOWRANK_FAST=1 (rust_grid.CONFIGS), so the reference mode must
        # REMOVE it again -- otherwise "ref" silently measures the fast path and every comparison below
        # becomes a comparison of the fast path with itself (this happened once; the numbers were
        # internally consistent and therefore invisible without the counters).
        env.pop("RWKV_LOWRANK_FAST", None)
        env.pop("RWKV_LOWRANK_FAST_CHECK", None)
    r = subprocess.run([str(rg.ENGINE)] + [str(u) for u in users], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    m = RE_FAST.search(text)
    fast = None
    if m:
        fast = {"mode": m.group(1), "iters": int(m.group(2)), "calls": int(m.group(3)),
                "fallbacks": int(m.group(4)), "max_rel_frobenius": float(m.group(5)),
                "rel_buckets": [int(m.group(6 + i)) for i in range(5)]}
    return {"rc": r.returncode, "fast": fast, "tail": text[-500:]}


def snapshot(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for p in sorted(src.glob("rust_pred_*.json")):
        shutil.copy2(p, dst / p.name)
        names.append(p.name)
    return names


def compare(a_dir, b_dir, names):
    """Text identity plus the largest per-review p(recall) change over the saved dumps."""
    out = {"files": 0, "text_identical": True, "max_abs_diff": 0.0, "worst": None}
    for name in names:
        pa, pb = a_dir / name, b_dir / name
        ta, tb = pa.read_text(encoding="utf-8"), pb.read_text(encoding="utf-8")
        out["files"] += 1
        if ta != tb:
            out["text_identical"] = False
        a, b = json.loads(ta), json.loads(tb)
        for key in ("pred_imm", "pred_ahead"):
            va, vb = a.get(key) or [], b.get(key) or []
            for x, y in zip(va, vb):
                if x is None or y is None:
                    continue
                d = abs(float(x) - float(y))
                if d > out["max_abs_diff"]:
                    out["max_abs_diff"] = d
                    out["worst"] = f"{name}:{key}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(rg.ART / "it36l2i8n2i8_124.safetensors"))
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--users", default="107,136,156")
    ap.add_argument("--configs", default="l2i8s4,l2i8n2i8,l3i8n2i8")
    ap.add_argument("--iters", default="2")
    ap.add_argument("--modes", default="ref,check,fast")
    ap.add_argument("--tag", default="check")
    args = ap.parse_args()

    users = [int(u) for u in args.users.split(",")]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    iters_list = [int(i) for i in args.iters.split(",")]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    trace_dir = ROOT / args.trace_dir
    snap_root = ROOT / "scratchpad" / "quant_plan" / f"rust_fastrank_preds_{args.tag}"
    # the engine writes `{RWKV_PRED_DIR}/rust_pred_<user>.json` and ignores any nesting below it, so
    # every run gets its OWN output dir (a shared one silently overwrites run after run).
    preds_root = ROOT / "scratchpad" / "quant_plan" / f"rust_fastrank_preds_live_{args.tag}"

    print(f"engine  {rg.ENGINE}\nweights {args.weights}\nusers   {users}\nconfigs {configs} "
          f"iters {iters_list} modes {modes}\n")
    out = {"weights": args.weights, "users": users, "configs": configs, "iters": iters_list,
           "modes": modes, "runs": {}, "compare": {}}

    for cfg in configs:
        for iters in iters_list:
            for mode in modes:
                tag = f"{cfg}_it{iters}_{mode}"
                run_dir = preds_root / tag
                info = run(Path(args.weights), trace_dir, users, cfg, iters, mode, run_dir)
                if info["rc"] != 0:
                    print(f"{tag:26s} ENGINE FAILED rc={info['rc']} :: {info['tail'][-200:]}")
                    out["runs"][tag] = info
                    continue
                dst = snap_root / mode / f"{cfg}_it{iters}"
                names = snapshot(run_dir, dst)
                out["runs"][tag] = {**info, "preds": len(names)}
                f = info["fast"]
                rel = "n/a" if f is None else f"{f['max_rel_frobenius']:.3e}"
                bks = "-" if f is None else "/".join(str(x) for x in f["rel_buckets"])
                print(f"{tag:26s} fast={f['mode'] if f else 'off':18s} calls="
                      f"{f['calls'] if f else 0:7d} fb={f['fallbacks'] if f else 0:5d} "
                      f"rel={rel:>9s} buckets={bks:>28s} preds={len(names)}")

    # every (config, iters) is judged against its own reference run
    for cfg in configs:
        for iters in iters_list:
            ref_dir = snap_root / "ref" / f"{cfg}_it{iters}"
            if not ref_dir.exists():
                continue
            names = sorted(p.name for p in ref_dir.glob("rust_pred_*.json"))
            for mode in ("check", "fast"):
                d = snap_root / mode / f"{cfg}_it{iters}"
                if not d.exists():
                    continue
                c = compare(ref_dir, d, names)
                out["compare"][f"{cfg}_it{iters}_{mode}"] = c
                print(f"compare {cfg} it{iters} {mode:5s} vs ref: files={c['files']} "
                      f"identical={c['text_identical']} max|dp|={c['max_abs_diff']:.3e} "
                      f"worst={c['worst']}")

    p = ROOT / "scratchpad" / "quant_plan" / f"rust_fastrank_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
