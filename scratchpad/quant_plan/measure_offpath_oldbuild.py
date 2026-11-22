"""Does the OFF path of `finite.rs` cost anything? Compare against the PRE-CHANGE engine.

`measure_finite_offpath.py` answers ON vs OFF inside one build: ON costs about 1% of the state pass
(it scans the state on purpose), and that is the price of the diagnostic. This script answers the
other half of the question: does an OFF run of the CURRENT build match an OFF run of the build from
BEFORE `finite.rs` existed? The only difference left is one `if !on() { return; }` branch per call, so
the two speeds must agree within noise - and that is what keeps the already-published constraint-6
numbers valid without re-measuring them all.

How the old binary is obtained: the pre-change `main.rs` / `model.rs` are read from HEAD (the finite
wiring is uncommitted), written over the working copies, built into a THROW-AWAY target dir, and the
working copies are restored in a `finally`. The normal `target/release` binary is never touched.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/measure_offpath_oldbuild.py [--reps 3] [--configs plain,l2i8s4]

Writes `scratchpad/quant_plan/rust_finite_offpath_oldbuild.json`.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402
import rust_speed as rs  # noqa: E402

SRC = ROOT / "rust" / "rwkv-infer" / "src"
CARGO = ROOT / "rust" / "rwkv-infer"
OLD_TARGET = ROOT / "rust" / "rwkv-infer" / "target_offpath_old"
WIRED = ("crate::finite::p(p);", "crate::finite::vec_raw(v);", "crate::finite::state_raw(data);")
WEIGHTS_DEFAULT = ROOT / "scratchpad" / "quant_plan" / "rust_artifacts" / "it36l2i8_124.safetensors"
ALL_USERS = ("107,110,116,120,121,128,136,146,150,151,156,159,162,165,175,176,187,198")


def head_blob(rel, dst):
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True)
    dst.write_bytes(blob.stdout)
    return len(blob.stdout)


def build_old_engine(work):
    """Build the pre-change engine; ALWAYS restore the working tree."""
    files = ["main.rs", "model.rs", "finite.rs"]
    backup = {f: (SRC / f).read_bytes() for f in files if (SRC / f).exists()}
    try:
        for f in ("main.rs", "model.rs"):
            n = head_blob(f"rust/rwkv-infer/src/{f}", SRC / f)
            print(f"  wrote HEAD:{f} ({n} bytes)")
        r = subprocess.run(["cargo", "build", "--release", "-j", "6", "--target-dir", str(OLD_TARGET)],
                           cwd=CARGO, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""
        print(f"  old build rc={r.returncode}: {tail}")
        if r.returncode != 0:
            raise SystemExit("old engine build failed")
        exe = OLD_TARGET / "release" / "rwkv-infer.exe"
        dst = work / "rwkv-infer-precheck.exe"
        shutil.copy2(exe, dst)
        return dst
    finally:
        for f, data in backup.items():
            (SRC / f).write_bytes(data)
        print("  working tree restored "
              f"(finite wiring back in main.rs: {'crate::finite::p(p);' in (SRC / 'main.rs').read_text(encoding='utf-8', errors='replace')})")


def snapshot(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("rust_pred_*.json")):
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def same_text(a, b):
    out = {"files": 0, "text_identical": True, "missing": []}
    for p in sorted(a.glob("rust_pred_*.json")):
        q = b / p.name
        if not q.exists():
            out["missing"].append(p.name)
            continue
        out["files"] += 1
        if p.read_bytes() != q.read_bytes():
            out["text_identical"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS_DEFAULT))
    ap.add_argument("--trace-dir", default="reference")
    ap.add_argument("--users", default=ALL_USERS)
    ap.add_argument("--configs", default="plain,l2i8s4")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    main_src = (SRC / "main.rs").read_text(encoding="utf-8", errors="replace")
    if not all(w in main_src or w in (SRC / "model.rs").read_text(encoding="utf-8", errors="replace")
               for w in WIRED):
        raise SystemExit("the working tree does not look like it has the finite wiring; nothing to compare")

    users = [int(u) for u in args.users.split(",")]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    trace_dir = ROOT / args.trace_dir
    work = ROOT / "scratchpad" / "quant_plan" / "rust_bin"
    work.mkdir(parents=True, exist_ok=True)
    preds_root = ROOT / "scratchpad" / "quant_plan" / "rust_speed_preds_oldbuild"
    snap_root = ROOT / "scratchpad" / "quant_plan" / "rust_finite_oldbuild_preds"

    print("building the PRE-CHANGE engine (HEAD main.rs + model.rs) into a throw-away target dir:")
    old_exe = build_old_engine(work)
    cur_exe = Path(rg.ENGINE)
    print(f"  current engine {cur_exe}\n  old engine     {old_exe}\n")

    os.environ.pop("RWKV_FINITE_CHECK", None)   # both sides run with the check OFF
    os.environ.pop("RWKV_FINITE_SELFTEST", None)
    engines = {"current": cur_exe, "old": old_exe}
    samples = {(c, e): [] for c in configs for e in engines}
    print(f"users {len(users)} configs {configs} reps {args.reps} (engines interleaved, check OFF)")
    for _ in range(max(1, args.reps)):
        for cfg in configs:
            for ename, exe in engines.items():
                rg.ENGINE = exe
                pu, err = rs.run_once(Path(args.weights), trace_dir, users, cfg, preds_root)
                if pu is None:
                    print(f"{cfg:8s} {ename:8s} ENGINE FAILED: {err}")
                    continue
                s = rs.summarise(pu)
                samples[(cfg, ename)].append(s["rev_s"])
                snap = snap_root / ename / cfg
                snap.mkdir(parents=True, exist_ok=True)
                snapshot(preds_root / cfg, snap)
                print(f"{cfg:8s} {ename:8s} rev/s={s['rev_s']:9.1f} states/s={s['states_s']:8.1f}")
    rg.ENGINE = cur_exe

    print(f"\n{'config':9s} {'current med':>11s} {'old med':>9s} {'ratio':>7s} {'spread cur':>10s} "
          f"{'spread old':>10s}")
    speed, ident = {}, {}
    for cfg in configs:
        cur, old = samples[(cfg, "current")], samples[(cfg, "old")]
        if not cur or not old:
            continue
        mc, mo = statistics.median(cur), statistics.median(old)
        speed[cfg] = {"current_samples": cur, "old_samples": old, "current_median": mc,
                      "old_median": mo, "ratio": mc / mo}
        ident[cfg] = same_text(snap_root / "current" / cfg, snap_root / "old" / cfg)
        print(f"{cfg:9s} {mc:11.1f} {mo:9.1f} {mc / mo:7.3f} "
              f"{max(cur) - min(cur):10.1f} {max(old) - min(old):10.1f}")
    print("\nprediction dumps, current build vs old build (both with the check OFF):")
    for cfg, r in ident.items():
        print(f"  {cfg:8s} files={r['files']} text_identical={r['text_identical']} missing={r['missing']}")

    ok_speed = all(abs(r["ratio"] - 1.0) <= 0.03 for r in speed.values())
    ok_ident = all(r["text_identical"] and not r["missing"] for r in ident.values())
    verdict = ("OFF PATH COSTS NOTHING (and both builds agree bit-for-bit)" if ok_speed and ok_ident
               else "REVIEW NEEDED")
    print(f"\nverdict: {verdict}\n  |ratio - 1| <= 0.03 on {len(speed)} configs: {ok_speed}; "
          f"identical dumps: {ok_ident}")

    dst = ROOT / "scratchpad" / "quant_plan" / "rust_finite_offpath_oldbuild.json"
    dst.write_text(json.dumps({"current_engine": str(cur_exe), "old_engine": str(old_exe),
                               "users": users, "configs": configs, "reps": args.reps,
                               "speed": speed, "bit_identity": ident, "verdict": verdict},
                              indent=1), encoding="utf-8")
    print(f"written to {dst}")
    return 0 if verdict.startswith("OFF PATH") else 1


if __name__ == "__main__":
    sys.exit(main())
