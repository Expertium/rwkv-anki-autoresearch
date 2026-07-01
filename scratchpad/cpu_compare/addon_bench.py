"""Drive the rwkv-srs Anki addon's Rust predict engine for the apples-to-apples CPU comparison.

The addon (vendor/rwkv_srs, _native.pyd, candle+Rayon) runs the published d=128 / 2.76M model.
Its predict = immediate recall prob (1 - P(again)); process = sequential state build.

For a user we build the addon's review-dicts from anki-revlogs-10k (revlogs JOIN cards JOIN decks,
exactly the merge our export does), then time:
  - BUILD  (sequential): RWKV_SRS.process_many(rows, num_threads=1)        -> reviews/s
  - PREDICT (batched, read-only): RWKV_SRS.predict_many(rows, B, threads)  -> states/s

Usage:
  python addon_bench.py <user> <d128_safetensors> [--predict-secs S] [--batch B] [--threads T]
Prints JSON lines tagged RESULT for the orchestrator + a PARITY block (first-K imm) for validation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ADDON_VENDOR = Path(
    r"C:\Temp\claude\C--Users-Andrew-rwkv-anki-autoresearch"
    r"\fc27c4c3-6e8f-4963-b71c-890cea4b4781\scratchpad\ankiaddon\vendor"
)
sys.path.insert(0, str(ADDON_VENDOR))
DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")

REQUIRED = (
    "review_id", "card_id", "note_id", "deck_id", "preset_id",
    "day_offset", "elapsed_days", "elapsed_seconds", "rating", "duration", "state",
)


def load_reviews(user: int):
    rl = pd.read_parquet(DATA / "revlogs" / f"user_id={user}")
    cr = pd.read_parquet(DATA / "cards" / f"user_id={user}")
    dk = pd.read_parquet(DATA / "decks" / f"user_id={user}").drop(columns=["parent_id"])
    df = rl.merge(cr, on="card_id", how="left").merge(dk, on="deck_id", how="left")
    df["review_id"] = range(1, len(df) + 1)
    # to plain python dicts (the addon coerces numpy scalars, but native dicts are cleanest)
    rows = []
    for rec in df.to_dict(orient="records"):
        rows.append({k: rec[k] for k in REQUIRED})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user", type=int)
    ap.add_argument("model", type=str, help="d=128 safetensors path")
    ap.add_argument("--predict-secs", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--build-passes", type=int, default=1)
    ap.add_argument("--parity-k", type=int, default=8)
    args = ap.parse_args()

    import os
    os.environ["RWKV_SRS_BACKEND"] = "rust"
    # The addon respects RAYON_NUM_THREADS as the global pool default; predict_many can override.
    from rwkv_srs.backends.rust import RWKV_SRS

    rows = load_reviews(args.user)
    n = len(rows)
    print(f"# user {args.user}: {n} reviews loaded", file=sys.stderr)

    # ---- BUILD (sequential, single-thread): process the whole history, fresh object per pass ----
    build_rates = []
    parity_imm = None
    for p in range(args.build_passes):
        srs = RWKV_SRS(model=args.model, seed=0)
        t0 = time.perf_counter()
        preds = srs.process_many(rows, num_threads=1)
        dt = time.perf_counter() - t0
        build_rates.append(n / dt)
        if parity_imm is None:
            parity_imm = [float(x) for x in preds[: args.parity_k]]
        if p < args.build_passes - 1:
            srs.close()
    # keep the last srs warmed (state fully built) for the predict phase

    build_rate = max(build_rates)  # best (least contended) pass

    # ---- PREDICT (batched, read-only): predict_many over the user's rows, looped for predict-secs ----
    # State is fully built; all cards/notes/decks/presets known -> predict_many uses the batched path.
    threads = None if args.threads <= 0 else args.threads
    # warm
    _ = srs.predict_many(rows[: args.batch], batch_size=args.batch, num_threads=threads)
    total = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.predict_secs:
        _ = srs.predict_many(rows, batch_size=args.batch, num_threads=threads)
        total += n
    dt = time.perf_counter() - t0
    predict_rate = total / dt
    srs.close()

    result = {
        "engine": "addon_d128",
        "user": args.user,
        "n_reviews": n,
        "build_reviews_per_s": round(build_rate, 1),
        "build_passes": args.build_passes,
        "predict_states_per_s": round(predict_rate, 1),
        "predict_batch": args.batch,
        "predict_threads": args.threads,
        "predict_total": total,
        "predict_secs": round(dt, 2),
    }
    print("RESULT " + json.dumps(result))
    print("PARITY " + json.dumps({"user": args.user, "imm_first_k": parity_imm}))


if __name__ == "__main__":
    main()
