"""Apples-to-apples CPU-inference comparison: rwkv-srs Anki addon vs our Rust engine.

Runs, for one user, back-to-back (matched machine contention):
  ADDON (d=128, _native.pyd):  build = process_many(1T); predict = predict_many(B, {1,8}T)
  OURS  (d=128 AND d=32 champ, rust/rwkv-infer):
      build = --bench (candle sequential, 1T)
      predict 1T = --bench-synth (candle)  /  --bench-synth-fast (fast.rs)
      predict 8T = --bench-mt (candle)
Best-of-N for single-thread rows (max = least-contended slice) since the box runs a permanent
FSRS benchmark (~8 cores) + an export. Prints a markdown table.

Usage: python run_compare.py <user> [--reps 3] [--batch 192] [--secs 4] [--mt 8]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
ADDON_VENDOR = Path(
    r"C:\Temp\claude\C--Users-Andrew-rwkv-anki-autoresearch"
    r"\fc27c4c3-6e8f-4963-b71c-890cea4b4781\scratchpad\ankiaddon\vendor"
)
D128 = str(ADDON_VENDOR / "rwkv_srs" / "pretrained" / "RWKV_trained_on_5000_10000.safetensors")
D32 = str(ROOT / "reference" / "champ_h2k16.safetensors")
BIN = str(ROOT / "rust" / "rwkv-infer" / "target" / "release" / "rwkv-infer.exe")
WD = ROOT / "scratchpad" / "cpu_compare" / "wd"  # has reference/trace_user_<u>.safetensors
DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
REQUIRED = ("review_id", "card_id", "note_id", "deck_id", "preset_id",
            "day_offset", "elapsed_days", "elapsed_seconds", "rating", "duration", "state")

sys.path.insert(0, str(ADDON_VENDOR))


def load_reviews(user):
    rl = pd.read_parquet(DATA / "revlogs" / f"user_id={user}")
    cr = pd.read_parquet(DATA / "cards" / f"user_id={user}")
    dk = pd.read_parquet(DATA / "decks" / f"user_id={user}").drop(columns=["parent_id"])
    df = rl.merge(cr, on="card_id", how="left").merge(dk, on="deck_id", how="left")
    df["review_id"] = range(1, len(df) + 1)
    return [{k: rec[k] for k in REQUIRED} for rec in df.to_dict(orient="records")]


def run_bin(args, env_extra):
    import os
    env = os.environ.copy()
    env.update({"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    env.update(env_extra)
    out = subprocess.run([BIN] + args, cwd=str(WD), env=env,
                         capture_output=True, text=True).stdout
    return out


def parse_rate(out, key):
    m = re.search(key + r"[ =]([0-9.]+)", out)
    return float(m.group(1)) if m else None


def ours_build(model, user, secs):
    out = run_bin(["--bench", str(secs), str(user)], {"RWKV_WEIGHTS": model})
    return parse_rate(out, "rev_s")


def ours_predict(model, secs, batch, mode):
    flag = {"candle": "--bench-synth", "fast": "--bench-synth-fast"}[mode]
    out = run_bin([flag, str(secs), str(batch)], {"RWKV_WEIGHTS": model})
    return parse_rate(out, "rev_s")


def ours_predict_mt(model, secs, batch, threads):
    out = run_bin(["--bench-mt", str(secs), str(batch), str(threads)], {"RWKV_WEIGHTS": model})
    return parse_rate(out, "rev_s")


def addon_build(user, rows, reps):
    from rwkv_srs.backends.rust import RWKV_SRS
    best = 0.0
    for _ in range(reps):
        srs = RWKV_SRS(model=D128, seed=0)
        t0 = time.perf_counter()
        srs.process_many(rows, num_threads=1)
        best = max(best, len(rows) / (time.perf_counter() - t0))
        srs.close()
    return best


def addon_predict(user, rows, batch, threads, secs):
    from rwkv_srs.backends.rust import RWKV_SRS
    srs = RWKV_SRS(model=D128, seed=0)
    srs.process_many(rows, num_threads=1)  # warm state
    th = None if threads <= 1 else threads
    _ = srs.predict_many(rows[:batch], batch_size=batch, num_threads=th)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        srs.predict_many(rows, batch_size=batch, num_threads=th)
        n += len(rows)
    rate = n / (time.perf_counter() - t0)
    srs.close()
    return rate


def best_of(fn, reps):
    return max(fn() for _ in range(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user", type=int)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--mt", type=int, default=8)
    args = ap.parse_args()

    import os
    os.environ["RWKV_SRS_BACKEND"] = "rust"
    rows = load_reviews(args.user)
    n = len(rows)
    print(f"# user {args.user}: {n} reviews | batch {args.batch} | reps {args.reps} | mt {args.mt}", flush=True)

    r = {}
    # ----- ADDON (d128) -----
    r["addon_build"] = addon_build(args.user, rows, args.reps)
    r["addon_pred_1t"] = addon_predict(args.user, rows, args.batch, 1, args.secs)
    r["addon_pred_mt"] = addon_predict(args.user, rows, args.batch, args.mt, args.secs)
    # ----- OURS d128 (matched) -----
    r["d128_build"] = best_of(lambda: ours_build(D128, args.user, args.secs), args.reps)
    r["d128_pred_candle"] = best_of(lambda: ours_predict(D128, args.secs, args.batch, "candle"), args.reps)
    r["d128_pred_fast"] = best_of(lambda: ours_predict(D128, args.secs, args.batch, "fast"), args.reps)
    r["d128_pred_mt"] = ours_predict_mt(D128, args.secs, args.batch, args.mt)
    # ----- OURS d32 champion (full-stack) -----
    r["d32_build"] = best_of(lambda: ours_build(D32, args.user, args.secs), args.reps)
    r["d32_pred_candle"] = best_of(lambda: ours_predict(D32, args.secs, args.batch, "candle"), args.reps)
    r["d32_pred_fast"] = best_of(lambda: ours_predict(D32, args.secs, args.batch, "fast"), args.reps)
    r["d32_pred_mt"] = ours_predict_mt(D32, args.secs, args.batch, args.mt)

    r = {k: (round(v, 1) if v else v) for k, v in r.items()}
    print("RESULT " + json.dumps({"user": args.user, "n": n, **r}), flush=True)

    def row(name, b, p1, pf, pm):
        f = lambda x: f"{x:>9.0f}" if x else "    n/a"
        print(f"| {name:<26} |{f(b)} |{f(p1)} |{f(pf)} |{f(pm)} |")
    print(f"\n## user {args.user} ({n} reviews)  [B={args.batch}, MT={args.mt}t, best-of-{args.reps}]")
    print("| engine / model            |  build/s | pred 1T candle | pred 1T fast | pred MT |")
    print("|---|---|---|---|---|")
    row("ADDON d128 (native)", r["addon_build"], r["addon_pred_1t"], None, r["addon_pred_mt"])
    row("OURS d128 (matched)", r["d128_build"], r["d128_pred_candle"], r["d128_pred_fast"], r["d128_pred_mt"])
    row("OURS d32 champion", r["d32_build"], r["d32_pred_candle"], r["d32_pred_fast"], r["d32_pred_mt"])


if __name__ == "__main__":
    main()
