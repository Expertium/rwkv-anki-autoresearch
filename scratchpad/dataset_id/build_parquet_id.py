"""Build the anki-revlogs-10k-id dataset (real IDs + corrected review times).

Adapted from open-spaced-repetition/anki-revlogs-dataset-builder/build_parquet.py
(the script that produced the published, ANONYMIZED anki-revlogs-10k). Output layout is
identical ({revlogs,cards,decks}/user_id=N/data.parquet, user_id = .revlog file stem, so
user numbering matches the published dataset 1:1), but with two deliberate differences:

1. REAL IDS ARE KEPT (the whole point of the "-id" dataset). The upstream script
   factorized card/note/deck/parent/preset ids to small per-user integers; here they stay
   the raw Anki epoch-millisecond IDs (Anki uses creation-time-in-ms as the ID), so
   creation-time features (card age, creation-batch size, note/deck age, ...) become
   derivable. See optimization/FUTURE_FEATURES.md.

2. ⚠ REVIEW-TIME CORRECTION (Andrew 2026-07-15): a revlog entry's `id` is the epoch-ms
   timestamp of when the review was ANSWERED (the row is written on answer). We store
       review_time = id - taken_millis
   i.e. the moment the card was SHOWN — the true start of the review, which is the right
   time base for elapsed-time and time-of-day features. EVERYTHING downstream is computed
   from the corrected time: day_offset, elapsed_days, elapsed_seconds, and the final sort
   order. The correction applies ONLY to review ids — card/note/deck/preset ids are
   creation timestamps and involve no duration. The raw answer-time id is exactly
   recoverable as review_time + duration (both int64 ms), so it is not stored separately.
   Consequences to be aware of:
     - a review whose answer timestamp fell just after the Anki day rollover but whose
       SHOW time fell before it moves one day_offset earlier than in the published
       dataset (intended — the published set used raw answer times);
     - elapsed_seconds/elapsed_days between two reviews now measure show-to-show;
     - in pathological sync/clock-skew cases corrected times could produce negative
       elapsed_seconds; the upstream script had the same exposure with raw ids and we
       add no extra guard here.

Everything else (revlog filtering, learn-start masking, state renumbering, the
first-row-must-be-learn group filter) is copied from upstream byte-for-byte so the row
set matches the published dataset.

Usage: python build_parquet_id.py <revlogs_dir> <output_root> [num_procs]
Resumable: users whose revlogs/user_id=N/data.parquet already exists are skipped.
"""

import sys
from pathlib import Path
from typing import Iterable
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from stats_pb2 import Dataset, RevlogEntry, CardEntry, DeckEntry  # locally compiled from stats.proto

OUTPUT_ROOT = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")


def filter_revlog(entries: Iterable[RevlogEntry]):
    # verbatim from upstream: keep rated reviews; drop filtered-deck rescheduling rows
    return filter(
        lambda entry: entry.button_chosen >= 1
        and (entry.review_kind != 3 or entry.ease_factor != 0),
        entries,
    )


def convert_revlog(entries: Iterable[RevlogEntry]):
    return map(
        lambda entry: {
            # ⚠ CORRECTION (see module docstring): revlog id = answer time (epoch ms);
            # subtract the answer duration to get the SHOW time. All downstream
            # quantities (day_offset, elapsed_*, sort) use this corrected time.
            "review_time": entry.id - entry.taken_millis,
            "card_id": entry.cid,  # real Anki id (creation epoch-ms) — NOT corrected, NOT factorized
            "rating": entry.button_chosen,
            "state": entry.review_kind,
            "duration": entry.taken_millis,
        },
        filter_revlog(entries),
    )


def convert_card(entries: Iterable[CardEntry]):
    # real ids kept (creation epoch-ms timestamps) — upstream factorized these
    return map(
        lambda entry: {
            "card_id": entry.id,
            "note_id": entry.note_id,
            "deck_id": entry.deck_id,
        },
        entries,
    )


def convert_deck(entries: Iterable[DeckEntry]):
    # real ids kept — upstream factorized these
    return map(
        lambda entry: {
            "deck_id": entry.id,
            "parent_id": entry.parent_id,
            "preset_id": entry.preset_id,
        },
        entries,
    )


def process_revlogs(dataset, df):
    if df.empty:
        return df

    # --- verbatim upstream masking logic (operates on card_id + state only) ---
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["is_learn_start"] = (df["state"] == 0) & (
        (df["state"].shift() != 0) | (df["i"] == 1)
    )
    df["sequence_group"] = df["is_learn_start"].cumsum()
    last_learn_start = (
        df[df["is_learn_start"]].groupby("card_id")["sequence_group"].last()
    )
    df["last_learn_start"] = (
        df["card_id"].map(last_learn_start).fillna(0).astype("int64")
    )
    df["mask"] = df["last_learn_start"] <= df["sequence_group"]
    df = df[df["mask"] == True]
    df.loc[:, "state"] += 1
    df.loc[df["is_learn_start"], "state"] = 0
    df = df.groupby("card_id").filter(lambda group: group["state"].iloc[0] == 0)

    # --- time-derived columns: identical formulas to upstream, but review_time is the
    # CORRECTED (show-time) value, so day_offset/elapsed_* shift accordingly ---
    df["review_time"] = df["review_time"].astype("int64")
    df["day_offset"] = df["review_time"].apply(
        lambda x: int((x / 1000 - dataset.next_day_at) / 86400)
    )
    df["day_offset"] = df["day_offset"] - df["day_offset"].min()
    # NOTE (inherited from upstream): these diffs are per-ROW in protobuf order, which is
    # per-card blocks in Anki's export; each card's first surviving row is state==0 and
    # gets -1 below, which masks the cross-card boundary contamination.
    df["elapsed_days"] = df["day_offset"].diff().fillna(0).astype("int64")
    df["elapsed_seconds"] = (df["review_time"].diff().fillna(0) / 1000).astype("int64")
    df.loc[df["state"] == 0, "elapsed_days"] = -1
    df.loc[df["state"] == 0, "elapsed_seconds"] = -1
    # NO factorize: card_id stays the real Anki id
    # sort by the corrected time (upstream sorted by raw answer time)
    df.sort_values(by="review_time", inplace=True)
    return df[
        [
            "review_time",  # corrected show-time, epoch ms (raw answer id = review_time + duration)
            "card_id",
            "day_offset",
            "rating",
            "state",
            "duration",
            "elapsed_days",
            "elapsed_seconds",
        ]
    ]


def save_to_parquet(df, table_name, user_id, output_root):
    if df.empty:
        return
    df["user_id"] = user_id
    table = pa.Table.from_pandas(df, preserve_index=False)
    output_path = output_root / table_name
    pq.write_to_dataset(
        table,
        output_path,
        partition_cols=["user_id"],
        existing_data_behavior="delete_matching",
    )
    for file in (output_path / f"user_id={user_id}").glob("*.parquet"):
        file.rename(file.with_name("data.parquet"))


def process_and_save(args):
    file_path, output_root = args
    user_id = int(file_path.stem)
    # resume marker: revlogs output exists -> user done (every user has revlogs)
    if (output_root / "revlogs" / f"user_id={user_id}" / "data.parquet").exists():
        return user_id
    dataset = Dataset()
    dataset.ParseFromString(open(file_path, "rb").read())

    df_revlogs = process_revlogs(dataset, pd.DataFrame(convert_revlog(dataset.revlogs)))
    df_cards = pd.DataFrame(convert_card(dataset.cards))
    df_decks = pd.DataFrame(convert_deck(dataset.decks))

    save_to_parquet(df_revlogs, "revlogs", user_id, output_root)
    save_to_parquet(df_cards, "cards", user_id, output_root)
    save_to_parquet(df_decks, "decks", user_id, output_root)
    return user_id


def main():
    revlogs_dir = Path(sys.argv[1])
    output_root = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_ROOT
    num_procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6  # machine limit: <=7 CPU threads
    files = sorted(revlogs_dir.glob("*.revlog"), key=lambda p: int(p.stem))
    print(f"{len(files)} .revlog files -> {output_root} ({num_procs} procs)", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    done = 0
    with Pool(num_procs) as pool:
        for _uid in pool.imap_unordered(
            process_and_save, ((f, output_root) for f in files), chunksize=8
        ):
            done += 1
            if done % 250 == 0:
                print(f"processed {done}/{len(files)}", flush=True)
    print(f"ALL_DONE {done}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
