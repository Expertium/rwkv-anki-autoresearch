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
    # card-only rank-2: the shape the loss winner actually uses (RESULTS row 15). Added 2026-11-21.
    "l2i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
             "RWKV_STATE_SHIFT_LEVEL": "int8"},
    "l2i8s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "int4"},
    "l2i8s4f": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
                "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1"},  # row 17 + E2 solver
    "l2i8s4f1": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1",
                 "RWKV_LOWRANK_FAST_ITERS": "1"},  # row 17 + E2 solver, 1 iteration
    "l3i8s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:3:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "int4"},
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
    # Rank-2 (and 3) on the NOTE, the new chain-14 candidates (Andrew, 2026-11-22 12:01). The engine's
    # low-rank path is generic over the stream, so these exist to prove it parses a rank>=2 note and to
    # price the speed. l2i8n2i8 is the direct twin of l2i8s4 with the note's rank raised 1 -> 2.
    # RWKV_LOWRANK_FAST=1 is the FAST rank-r SOLVER (E2, 2026-11-22): a deploy-side approximation, not a
    # bit-budget change (bits_for ignores it). Without it a rank>=2 NOTE measures 0.618x rev/s (FAIL on
    # the 0.75x floor); with it, 0.803x for l2i8n2i8 and 0.791x for l3i8n2i8, at a deploy log-loss cost
    # of imm +0.000051 / ahead +0.000000 on users 107/136/156 (rust_fastrank_check.json,
    # rust_score.json, rust_speed_fastrank_{on,off}.json). ITERS defaults to 2 and 4 was measured WORSE
    # (rust_fastrank_iters2.json), so the default is the deploy choice. NOTE the flag is IN this dict:
    # to measure the REFERENCE solver, remove it (an inherited RWKV_LOWRANK_FAST cannot override it).
    # Do NOT set it for a Python<->Rust PARITY run:
    # the fast factors are not bit-identical to the reference truncation by construction.
    "l2i8n2i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1"},
    "l3i8n2i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:3:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1"},
    # NOTE: the engine has NO per-stream shift level - parse_level() takes a bare "int4"/"int8" and any
    # "card:x,note:y" string yields None (= no override). So `l2i8n2i8s8` (card int4 / note int8 shifts)
    # cannot be expressed on the deploy side yet; that is a small engine gap, recorded in GPU_RUN_BOARD.
    "l2i8n3i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:3:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1"},

    # THE int6/int5 FACTOR RUNGS (2026-11-22). Every rung below row 17 that has been tested so
    # far used int4 factors (rows 19-26); int6 and int5 were NEVER tried, although the ENGINE has
    # always supported them (parse_level, model.rs:2104) and only the Python QAT path lacks the
    # _QMAX keys. The card keeps rank 2 and the note rank 1 (row 17's exact layout), so the ONLY
    # change against l2i8s4 is factor precision: 6 bits instead of 8. bits_for:
    #   l2i6s4  card 2x2x32x6 + 64x4 = 1,024   note 3x(2x1x32x6 + 64x4) = 1,920   total 2,944
    #   l2i5s4  card 2x2x32x5 + 64x4 =   896   note 3x(2x1x32x5 + 64x4) = 1,728   total 2,624
    # i.e. 640 and 960 bits below row 17. The old rank-1 PTQ grid (l1i4/l1i5/l1i6/l1i8,
    # RESULTS.md) said precision above int4 does not pay AT RANK 1, because the residual there IS
    # the rank truncation; but that was rank 1, and PTQ on a checkpoint adapted to int8. At rank 2
    # on a QAT-trained checkpoint the question is open.
    "l2i6s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "int4"},
    "l2i5s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int5", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "int4"},
    # PROBE CALIBRATION (2026-11-22): l2i4s4 is row 23's layout and l2i8s4n4 is row 21's, and both
    # have 750-user QAT numbers in RESULTS.md, so running all three in one grid makes the 2-user
    # Rust PTQ probe checkable against a known delta.
    "l2i4s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "int4"},
    "l2i8s4n4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int4", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4"},
    # ASYMMETRIC FACTOR BUDGETS (2026-11-22): U keeps the scope level, V drops to VLEVEL. At
    # row 17's layout U int8 / V int4 is 2,944 bits and V int3 / V int2 give 2,784 / 2,624,
    # i.e. 640 / 800 / 960 bits below row 17. U and V are not interchangeable (U holds the
    # state-space directions, V the read-out weights), which is why this is worth asking even
    # though the symmetric ladder said precision above int4 does not pay at rank 1.
    "l2i8s4v4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int4"},
    "l2i8s4v3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int3"},
    "l2i8s4v2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int2"},

}

sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_score  # noqa: E402


def bits_for(env):
    """State bits per stream (WKV + token shifts). Must agree with `quant_sweep.bits_for`.

    A BARE `RWKV_STATE_SHIFT_LEVEL` is GLOBAL: the engine's `parse_level` takes ONE level and applies
    it to every stream. Reading a bare level as "note-only" (the old behaviour here) made the card
    fall back to its WKV width and reported 1,536 bits for `l2i8s4` instead of the true 1,280, so the
    deployed row was scored against the wrong card bit count. A `stream:level` list is accepted too,
    for symmetry with the training side, but the engine ignores such a string.
    """
    shift = {}
    bare = None
    for entry in env.get("RWKV_STATE_SHIFT_LEVEL", "").split(","):
        entry = entry.strip()
        if not entry:
            continue  # "" splits to [""] and would otherwise register a bogus fp32 shift level
        head, level = entry.split(":")[0], entry.split(":")[-1]
        if ":" in entry:
            shift[head] = QBITS.get(level, 32)
        else:
            bare = QBITS.get(level, 32)
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
    # RWKV_LOWRANK_VLEVEL overrides the SECOND factor (V) only, so the pair costs r*32*(U+V)
    # bits instead of 2*r*32*U. Without this the reported bit counts would be the symmetric
    # ones, which is exactly the kind of mis-count that made `l2i8s4` read 1,536 instead of 1,280.
    vbits = QBITS.get(str(env.get("RWKV_LOWRANK_VLEVEL", "")).strip())
    out = {}
    for s, n in LAYERS.items():
        # the shifts follow the WKV level when no shift override is given
        if bare is not None:
            sb = bare
        else:
            sb = shift.get(s, low[s][1] if s in low else full.get(s, 32))
        if s in low:
            r, qb = low[s]
            if vbits is None:
                out[s] = n * (2 * r * 32 * qb + 64 * sb)
            else:
                out[s] = n * (r * 32 * qb + r * 32 * vbits + 64 * sb)
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
