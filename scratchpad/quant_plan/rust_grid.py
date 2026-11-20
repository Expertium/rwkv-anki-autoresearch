"""Grid the Rust deploy quantizer: run the engine once per config on the same traces and score each
against the unquantized run, so the deploy quantization tax is measurable per config.

Rust env names differ from the Python training-side names (`RWKV_STATE_QUANT_SCOPE` vs
`RWKV_QAT_SCOPE`, `RWKV_STATE_LOWRANK_SCOPE` vs `RWKV_QAT_LOWRANK_SCOPE`), and the engine speaks
int2/int3/int4/int5/int6/int8 while the Python training path only knows int2/int3/int4/int8.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/rust_grid.py --weights <sft> \
      --trace-dir reference --users 107,136,156 --configs l1i2,l1i4,l1i8 [--reps 1]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "rust" / "rwkv-infer" / "target" / "release" / "rwkv-infer.exe"
ART = ROOT / "scratchpad" / "quant_plan" / "rust_artifacts"
QBITS = {"int8": 8, "int6": 6, "int5": 5, "int4": 4, "int3": 3, "int2": 2, "fp32": 32}
LAYERS = {"card": 1, "note": 3}
BAR = {"card": 2176, "note": 6528}

# name -> env. "plain" is the reference and is always run first.
CONFIGS = {
    "plain": {},
    # the QAT redo's config (Andrew's pick): card keeps the FULL W matrix at int2, note is rank-1
    # int8. bits_for: card 1x(1024x2 + 64x2) = 2,176 (exactly at the bar), note 3x(2x32x8 + 64x8)
    # = 3,072 (half the bar).
    "c2_nl1i8": {"RWKV_STATE_QUANT_SCOPE": "card:int2", "RWKV_STATE_LOWRANK_SCOPE": "note:1:int8",
                 "RWKV_QUANT_SHIFTS": "1", "RWKV_STATE_SHIFT_LEVEL": "card:int2,note:int8"},
    "cn2": {"RWKV_STATE_QUANT_SCOPE": "card:int2,note:int2", "RWKV_QUANT_SHIFTS": "1",
            "RWKV_STATE_SHIFT_LEVEL": "int2"},
    "cn3": {"RWKV_STATE_QUANT_SCOPE": "card:int3,note:int3", "RWKV_QUANT_SHIFTS": "1",
            "RWKV_STATE_SHIFT_LEVEL": "int3"},
    "l1i2": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int2,note:1:int2", "RWKV_QUANT_SHIFTS": "1",
             "RWKV_STATE_SHIFT_LEVEL": "int2"},
    "l1i3": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int3,note:1:int3", "RWKV_QUANT_SHIFTS": "1",
             "RWKV_STATE_SHIFT_LEVEL": "int3"},
    "l1i4": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1"},
    "l1i5": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int5,note:1:int5", "RWKV_QUANT_SHIFTS": "1",
             "RWKV_STATE_SHIFT_LEVEL": "int5"},
    "l1i6": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int6,note:1:int6", "RWKV_QUANT_SHIFTS": "1",
             "RWKV_STATE_SHIFT_LEVEL": "int6"},
    "l1i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
             "RWKV_STATE_SHIFT_LEVEL": "int8"},
    # attribution pair: one stream quantized, the other left fp32 (not valid candidates)
    "l1i4c": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4", "RWKV_QUANT_SHIFTS": "1"},
    "l1i4n": {"RWKV_STATE_LOWRANK_SCOPE": "note:1:int4", "RWKV_QUANT_SHIFTS": "1"},
    "l2i4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:2:int4", "RWKV_QUANT_SHIFTS": "1"},
    "l1i4h": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",
              "RWKV_LOWRANK_HADAMARD": "1"},
    "l1i4p": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",
              "RWKV_LOWRANK_PERCOL": "1"},
    "l1i4hp": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_LOWRANK_HADAMARD": "1", "RWKV_LOWRANK_PERCOL": "1"},
    "l1i2hp": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int2,note:1:int2", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_LOWRANK_HADAMARD": "1", "RWKV_LOWRANK_PERCOL": "1"},
    "l1i3hp": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int3,note:1:int3", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_LOWRANK_HADAMARD": "1", "RWKV_LOWRANK_PERCOL": "1"},
    "l1i4als": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",
                "RWKV_LOWRANK_ALS": "3"},
}

sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_score  # noqa: E402


def bits_for(env):
    shift = {}
    for entry in env.get("RWKV_STATE_SHIFT_LEVEL", "").split(","):
        if not entry.strip():
            continue  # "" splits to [""] and would otherwise register a bogus fp32 shift level
        key = "card" if "card" in entry else "note"
        level = entry.split(":")[-1] if ":" in entry else entry
        shift[key] = QBITS.get(level, 32)
    low = {}
    for entry in env.get("RWKV_STATE_LOWRANK_SCOPE", "").split(","):
        parts = entry.split(":")
        if len(parts) >= 3:
            low[parts[0]] = (int(parts[1]), QBITS[parts[2]])
    full = {}
    for entry in env.get("RWKV_STATE_QUANT_SCOPE", "").split(","):
        parts = entry.split(":")
        if parts[0] in LAYERS:
            full[parts[0]] = QBITS.get(parts[-1], 32)
    out = {}
    for s, n in LAYERS.items():
        # shifts follow the WKV level when the override is absent or fp32
        sb = shift.get(s, low[s][1] if s in low else full.get(s, 32))
        if s in low:
            r, qb = low[s]
            out[s] = n * (2 * r * 32 * qb + 64 * sb)
        else:
            out[s] = n * (1024 * full.get(s, 32) + 64 * sb)
    return out


def run(weights, trace_dir, users, name, preds_root):
    env = dict(__import__("os").environ)
    env["RWKV_WEIGHTS"] = str(weights)
    env["RWKV_TRACE_DIR"] = str(trace_dir)
    pdir = preds_root / name
    env["RWKV_PRED_DIR"] = str(pdir)
    for k in ("RWKV_STATE_QUANT_SCOPE", "RWKV_STATE_LOWRANK_SCOPE", "RWKV_QUANT_SHIFTS",
              "RWKV_STATE_SHIFT_LEVEL", "RWKV_LOWRANK_HADAMARD", "RWKV_LOWRANK_PERCOL",
              "RWKV_LOWRANK_ALS", "RWKV_LOWRANK_4LEVEL", "RWKV_LOWRANK_MIXED53",
              "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in CONFIGS[name].items()})
    r = subprocess.run([str(ENGINE)] + [str(u) for u in users], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-400:]
    return pdir, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ART / "adaptive173416.safetensors"))
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--users", default="107,136,156")
    ap.add_argument("--configs", default="plain,cn2,l1i2,l1i3,l1i4,l1i5,l1i6,l1i8")
    ap.add_argument("--preds-root", default=str(ART.parent / "rust_grid_preds"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    users = [int(u) for u in args.users.split(",")]
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    if "plain" not in names:
        names = ["plain"] + names
    preds_root = Path(args.preds_root) / (args.tag or Path(args.trace_dir).name)
    preds_root.mkdir(parents=True, exist_ok=True)
    trace_dir = ROOT / args.trace_dir

    rows = {}
    for name in names:
        pdir, err = run(Path(args.weights), trace_dir, users, name, preds_root)
        if err:
            print(f"{name:8s} ENGINE FAILED: {err}")
            continue
        per = {u: rust_score.score_user(trace_dir, pdir, u) for u in users}
        got = {u: v for u, v in per.items() if v}
        if not got:
            print(f"{name:8s} no preds")
            continue
        mean = {m: sum(v[m] for v in got.values() if v[m] is not None) / len(got)
                for m in ("ahead", "imm")}
        b = bits_for(CONFIGS[name])
        rows[name] = {"bits": b, "means": mean, "n_users": len(got)}
        print(f"{name:8s} n={len(got)}  ahead={mean['ahead']:.6f}  imm={mean['imm']:.6f}  "
              f"card={b['card']}b note={b['note']}b")

    if "plain" in rows:
        base = rows["plain"]["means"]
        print("\nconfig    bits card/note   tax ahead    tax imm   bar card/note")
        for name, r in rows.items():
            if name == "plain":
                continue
            ta = r["means"]["ahead"] - base["ahead"]
            ti = r["means"]["imm"] - base["imm"]
            ok = "OK " if (r["bits"]["card"] <= BAR["card"] and r["bits"]["note"] <= BAR["note"]) else "OVER"
            gate = "PASS" if (ta <= 0.0005 and ti <= 0.0005) else "fail"
            print(f"{name:8s} {r['bits']['card']:5d}/{r['bits']['note']:5d}  {ta:+.6f}  {ti:+.6f}   "
                  f"{ok}  {gate}")
    out = ART.parent / f"rust_grid_{args.tag or Path(args.trace_dir).name}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
