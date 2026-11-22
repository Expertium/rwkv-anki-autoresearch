"""Is `RWKV_FINITE_CHECK` free when it is OFF, and is it truly observation-only when it is ON?

Two claims about `rust/rwkv-infer/src/finite.rs` that must be MEASURED, not assumed:

  1. OFF IS FREE. The mode is read once into a `OnceLock` and every entry point starts with
     `if !on() { return; }`, so an unset `RWKV_FINITE_CHECK` should cost one predictable branch per
     call. This script runs each config with the check off and on, INTERLEAVED (the GPU is usually busy
     while this runs, so interleaving cancels load drift), and reports the speed ratio.
  2. OBSERVATION-ONLY. Every entry point only reads and counts; no value is written. So the per-review
     predictions of a checked run must be BIT-IDENTICAL to an unchecked one. This script keeps the
     engine's own `rust_pred_<user>.json` dumps from both modes and compares them as text (and
     numerically, for a readable maximum difference).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/measure_finite_offpath.py [--reps 2] [--tag offpath]

Writes `scratchpad/quant_plan/rust_finite_offpath.json` and the prediction snapshots under
`scratchpad/quant_plan/rust_finite_offpath_preds/{off,on}/<config>/`.
"""
import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402
import rust_speed as rs  # noqa: E402

WEIGHTS_DEFAULT = ROOT / "scratchpad" / "quant_plan" / "rust_artifacts" / "it36l2i8_124.safetensors"
MODES = ("off", "on")
CONFIGS = ("plain", "l2i8s4")


def set_mode(mode):
    if mode == "on":
        os.environ["RWKV_FINITE_CHECK"] = "1"
    else:
        # a leftover RWKV_FINITE_SELFTEST in the environment would plant a fake failure
        os.environ.pop("RWKV_FINITE_CHECK", None)
    os.environ.pop("RWKV_FINITE_SELFTEST", None)


def snapshot(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("rust_pred_*.json")):
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def compare(dir_a, dir_b):
    """Text-identity plus the largest numeric difference over every saved prediction."""
    out = {"files": 0, "text_identical": True, "max_abs_diff": None, "worst_user": None,
           "missing": []}
    for p in sorted(dir_a.glob("rust_pred_*.json")):
        q = dir_b / p.name
        if not q.exists():
            out["missing"].append(p.name)
            continue
        out["files"] += 1
        ta, tb = p.read_text(encoding="utf-8"), q.read_text(encoding="utf-8")
        if ta != tb:
            out["text_identical"] = False
        a, b = json.loads(ta), json.loads(tb)
        for key in ("pred_imm", "pred_ahead"):
            va, vb = a.get(key) or [], b.get(key) or []
            if len(va) != len(vb):
                out["max_abs_diff"] = float("inf")
                out["worst_user"] = p.stem
                continue
            for x, y in zip(va, vb):
                if x is None or y is None:
                    if x is not y:
                        out["max_abs_diff"] = float("inf")
                        out["worst_user"] = p.stem
                    continue
                d = abs(float(x) - float(y))
                if out["max_abs_diff"] is None or d > out["max_abs_diff"]:
                    out["max_abs_diff"] = d
                    out["worst_user"] = p.stem
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_DEFAULT))
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--users", default="107,136,156")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--reps", type=int, default=2, help="rounds; each round runs every config in BOTH modes")
    ap.add_argument("--tag", default="offpath")
    args = ap.parse_args()

    users = [int(u) for u in args.users.split(",")]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    trace_dir = ROOT / args.trace_dir
    preds_root = ROOT / "scratchpad" / "quant_plan" / f"rust_speed_preds_{args.tag}"
    snap_root = ROOT / "scratchpad" / "quant_plan" / f"rust_finite_offpath_preds_{args.tag}"
    best = {(c, m): None for c in configs for m in MODES}
    samples = {(c, m): [] for c in configs for m in MODES}

    print(f"engine {rg.ENGINE}\nweights {args.weights}\nusers {users} configs {configs} "
          f"reps {args.reps} (modes interleaved)\n")
    for _ in range(max(1, args.reps)):
        for cfg in configs:
            for mode in MODES:
                set_mode(mode)
                pu, err = rs.run_once(Path(args.weights), trace_dir, users, cfg, preds_root)
                if pu is None:
                    print(f"{cfg:8s} {mode:3s} ENGINE FAILED: {err}")
                    continue
                s = rs.summarise(pu)
                samples[(cfg, mode)].append(s)
                cur = best[(cfg, mode)]
                if cur is None or s["rev_s"] > cur["rev_s"]:
                    best[(cfg, mode)] = s
                snap = snap_root / mode / cfg
                snap.mkdir(parents=True, exist_ok=True)
                snapshot(preds_root / cfg, snap)
                print(f"{cfg:8s} {mode:3s} rev/s={s['rev_s']:9.1f} states/s={s['states_s']:8.1f}")

    print(f"\nNOTE the engine prints each user's elapsed time with one decimal, so an aggregate rev/s"
          f" moves in ~1% steps; medians of {args.reps} interleaved samples are the honest number.")
    print(f"{'config':9s} {'off rev/s':>10s} {'on rev/s':>10s} {'on/off':>7s} "
          f"{'off med':>9s} {'on med':>9s} {'med on/off':>10s}")
    ratios = {}
    for cfg in configs:
        off, on = best[(cfg, "off")], best[(cfg, "on")]
        if not off or not on:
            continue
        xr = on["rev_s"] / off["rev_s"]
        med_off = statistics.median([s["rev_s"] for s in samples[(cfg, "off")]])
        med_on = statistics.median([s["rev_s"] for s in samples[(cfg, "on")]])
        xs = on["states_s"] / off["states_s"]
        ratios[cfg] = {"off_rev_s": off["rev_s"], "on_rev_s": on["rev_s"], "rev_ratio": xr,
                       "off_states_s": off["states_s"], "on_states_s": on["states_s"],
                       "states_ratio": xs, "off_samples": [s["rev_s"] for s in samples[(cfg, "off")]],
                       "on_samples": [s["rev_s"] for s in samples[(cfg, "on")]],
                       "off_median_rev_s": med_off, "on_median_rev_s": med_on,
                       "median_ratio": med_on / med_off}
        print(f"{cfg:9s} {off['rev_s']:10.1f} {on['rev_s']:10.1f} {xr:7.3f} "
              f"{med_off:9.1f} {med_on:9.1f} {med_on / med_off:10.3f}")

    print("\nbit-identity of the predictions (check OFF vs ON):")
    ident = {}
    for cfg in configs:
        cmp_out = compare(snap_root / "off" / cfg, snap_root / "on" / cfg)
        ident[cfg] = cmp_out
        print(f"  {cfg:8s} files={cmp_out['files']} text_identical={cmp_out['text_identical']} "
              f"max_abs_diff={cmp_out['max_abs_diff']} missing={cmp_out['missing']}")

    free = [c for c, r in ratios.items() if r["median_ratio"] >= 0.97]
    same = [c for c, r in ident.items() if r["text_identical"] and not r["missing"]]
    if len(same) == len(configs) and len(free) == len(ratios) == len(configs):
        verdict = "OFF-PATH FREE (within noise) and OBSERVATION-ONLY"
    elif len(same) == len(configs):
        verdict = "OBSERVATION-ONLY, but the ON path may cost speed on " + ",".join(
            c for c in ratios if c not in free)
    else:
        verdict = "REVIEW NEEDED"
    print(f"\nverdict: {verdict}")
    print(f"  bit-identical prediction dumps on {len(same)}/{len(configs)} configs (the hard claim);")
    print(f"  median ON/OFF rev/s >= 0.97 on {len(free)}/{len(configs)} configs "
          f"(this machine runs a training job, so a sub-2% effect is not resolvable here)")

    dst = ROOT / "scratchpad" / "quant_plan" / f"rust_finite_{args.tag}.json"
    dst.write_text(json.dumps({"weights": args.weights, "users": users, "configs": configs,
                               "reps": args.reps, "speed": ratios, "bit_identity": ident,
                               "verdict": verdict}, indent=1), encoding="utf-8")
    print(f"written to {dst}")
    return 0 if verdict.startswith("OFF-PATH") else 1


if __name__ == "__main__":
    sys.exit(main())
