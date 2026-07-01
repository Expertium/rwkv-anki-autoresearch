"""FAST feature-only trace export (no Python net forward).

The 92-dim features fed to the net are built by the stateful feature pipeline
(RNNProcess.add_same + get_tensor) BEFORE the net runs, and none of that pipeline state
depends on the net output. So we override run() to capture the feature tensor + routing ids
and skip the net entirely -- the Rust engine does the (much faster) net forward later.

Produces, per user: reference/trace_user_{u}.safetensors (feats_imm/feats_proc/route/elapsed/
review_th) + reference/trace_user_{u}.json (equalize set + label_rating + size; NO py preds --
the fp32 baseline comes from a rust fp32 run, which parity proved == python bit-exactly).

Usage: python scratchpad/export_features_fast.py U1 U2 U3 ...   (explicit user ids)
       python scratchpad/export_features_fast.py --range START END
Skips users whose trace_user_{u}.safetensors already exists (resumable across teardowns).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from safetensors.numpy import save_file as save_np
import rwkv.run_as_rnn as rnn_mod
from rwkv.get_result import get_benchmark_info

import os

# Low default per-process thread count so cross-user multiprocessing (export_mp.py) doesn't
# oversubscribe. The export is parquet-I/O + 1-element torch ops, not torch-compute bound, so
# few threads costs ~nothing; the parallelism win is across USERS (processes), not within one.
torch.set_num_threads(int(os.environ.get("RWKV_TORCH_THREADS", "7")))
DATA = Path("../anki-revlogs-10k")
LABEL_DB = "label_filter_db"
LABEL_DB_SIZE = 2_000_000_000
OUT = Path(os.environ.get("RWKV_TRACE_OUT", "reference"))
_DUMMY = (torch.zeros(1, 1), torch.zeros(1, 1)), torch.zeros(1)


class FeatRNN(rnn_mod.RNNProcess):
    """RNNProcess that captures the 92-dim feature + routing ids and SKIPS the net."""

    def run(self, row, skip):
        feats = self.get_tensor(row)
        self._last_features = (
            feats.detach().cpu().numpy().reshape(-1).astype(np.float32).copy()
        )
        self._route = (row["card_id"], row["note_id"], row["deck_id"], row["preset_id"])
        return _DUMMY  # (curve_tuple, imm_prob) -- unused; state updates run regardless


def export_user(user_id):
    sft = OUT / f"trace_user_{user_id}.safetensors"
    if sft.exists():
        print(f"user {user_id}: SKIP (exists)", flush=True)
        return
    # Skip users absent from the dataset (ids in a range are NOT contiguous): a missing partition
    # raises FileNotFoundError which, unguarded, killed the rest of the worker's round-robin chunk --
    # the cause of the 6000-6999 export landing 836/995 traces. Guard here + try/except in main().
    if not (DATA / "revlogs" / f"{user_id=}").exists():
        print(f"user {user_id}: SKIP (not in dataset)", flush=True)
        return
    torch.manual_seed(user_id)  # matches export_rnn_trace id-encoding seed

    df = rnn_mod.RNNProcess.__dict__  # noqa (silence linters); real df below
    import pandas as pd

    # Read the user's partition DIRECTLY (like revlogs) instead of read_parquet(dir, filters=...),
    # which re-discovers all ~10k partition dirs every call (the 8.4s "_filesystem_dataset" in the
    # profile). Direct path = no dataset discovery, no user_id column (it's the partition key).
    df = pd.read_parquet(DATA / "revlogs" / f"{user_id=}")
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(DATA / "cards" / f"{user_id=}")
    df_decks = pd.read_parquet(DATA / "decks" / f"{user_id=}")
    df_decks.drop(columns=["parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)

    equalize_review_ths, rmse_bins = get_benchmark_info(LABEL_DB, LABEL_DB_SIZE, user_id)

    srs = FeatRNN(path=None, device=torch.device("cpu"), dtype=torch.float32)

    maps = {"card": {}, "note": {}, "deck": {}, "preset": {}}

    def densify(kind, raw):
        m = maps[kind]
        if raw not in m:
            m[raw] = len(m)
        return m[raw]

    feats_imm, feats_proc, route = [], [], []
    elapsed_list, review_th_list, rating_list, has_ahead_list = [], [], [], []
    label_rating = {}
    seen_cards = set()

    for i, row in df.iterrows():
        i = int(i)
        if (i + 1) % 2000 == 0:
            print(f"  user {user_id}: {i + 1}/{len(df)}", flush=True)
        card_id = row["card_id"]
        review_th = int(row["review_th"])

        has_ahead = card_id in seen_cards

        imm_info = row.copy()
        imm_info.drop(columns=["rating", "duration"], inplace=True)
        srs.imm_predict(imm_info)
        fi = srs._last_features.copy()
        route_raw = srs._route

        srs.process_row(row)
        fp = srs._last_features.copy()
        seen_cards.add(card_id)

        label_rating[review_th] = int(row["rating"]) - 1
        feats_imm.append(fi)
        feats_proc.append(fp)
        route.append([
            densify("card", route_raw[0]),
            densify("note", route_raw[1]),
            densify("deck", route_raw[2]),
            densify("preset", route_raw[3]),
        ])
        elapsed_list.append(float(row["elapsed_seconds"]))
        review_th_list.append(review_th)
        rating_list.append(int(row["rating"]))
        has_ahead_list.append(1 if has_ahead else 0)

    tensors = {
        "feats_imm": np.stack(feats_imm).astype(np.float32),
        "feats_proc": np.stack(feats_proc).astype(np.float32),
        "route": np.asarray(route, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_list, dtype=np.float32),
        "review_th": np.asarray(review_th_list, dtype=np.int64),
        "rating": np.asarray(rating_list, dtype=np.int64),
        "has_ahead": np.asarray(has_ahead_list, dtype=np.int64),
    }
    save_np(tensors, str(sft))
    meta = {
        "user": user_id,
        "n_reviews": len(df),
        "equalize_review_ths": [int(x) for x in equalize_review_ths],
        "rmse_bins": [int(x) for x in rmse_bins],
        "label_rating": {int(k): int(v) for k, v in label_rating.items()},
        "size": len(equalize_review_ths),
    }
    json.dump(meta, open(OUT / f"trace_user_{user_id}.json", "w"))
    print(f"user {user_id}: n {len(df)}, size {meta['size']} -> {sft.name}", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--range":
        users = list(range(int(sys.argv[2]), int(sys.argv[3])))
    else:
        users = [int(x) for x in sys.argv[1:]]
    for u in users:
        try:
            export_user(u)
        except Exception as e:
            # Never let one user kill the rest of a worker's chunk (belt-and-suspenders vs the guard
            # in export_user). Log + continue so a resumable re-run recovers everything reachable.
            print(f"user {u}: ERROR {type(e).__name__}: {e} -- skipping", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
