"""Draw the seeded RANDOM 750-user evaluation roster for the quantization work.

Andrew (2026-11-19, constraint 5): "Select a random 750 users subset of the 2.5 eval set we use,
and test your ideas on that." The 2.5k eval set = VAL users 5001-7500.

Eligibility rule = the project's existing screen rule (see scratchpad/adaptive_wd/adaptive_train.py
`choose_screen_users`): a user counts only if the VAL DB stores EXACTLY ONE batch for it, which also
drops the four known multi-batch VAL users (5626, 6630, 6701, 6810). Unlike the old deterministic
screen roster (every other eligible user), this draw is uniformly random over the eligible set.

Writes scratchpad/quant_plan/roster750_seed<SEED>.json and prints a summary, including the overlap
with the deterministic roster recorded in scratchpad/ta3_fullrun_20261116/ta3_state.json so old
numbers can be cross-checked.

Read-only on the LMDB. No GPU.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL_DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
START, END = 5001, 7500
COUNT = 750
SEED = 20261119
OUT = ROOT / "scratchpad" / "quant_plan" / f"roster750_seed{SEED}.json"
OLD_STATE = ROOT / "scratchpad" / "ta3_fullrun_20261116" / "ta3_state.json"


def eligible_users():
    import lmdb
    env = lmdb.open(VAL_DB, map_size=250_000_000_000, readonly=True, lock=False,
                    readahead=False, max_readers=512)
    ok, excluded = [], []
    with env.begin() as txn:
        for user in range(START, END + 1):
            raw = txn.get(f"{user}_batches".encode())
            if raw is None or len(json.loads(raw)) != 1:
                excluded.append(user)
            else:
                ok.append(user)
    env.close()
    return ok, excluded


def main():
    ok, excluded = eligible_users()
    if len(ok) < COUNT:
        raise SystemExit(f"only {len(ok)} eligible users; need {COUNT}")
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(ok, COUNT))
    old = []
    if OLD_STATE.exists():
        state = json.loads(OLD_STATE.read_text(encoding="utf-8"))
        old = list(state.get("plan", {}).get("screen_users", []))
    overlap = len(set(chosen) & set(old)) if old else None
    doc = {
        "purpose": "random 750-user evaluation subset for the card/note state-quantization work",
        "source_db": VAL_DB,
        "range": [START, END],
        "rule": "users with exactly one stored batch in the VAL db (drops the 4 multi-batch VAL users)",
        "seed": SEED,
        "count": len(chosen),
        "eligible": len(ok),
        "excluded": len(excluded),
        "overlap_with_deterministic_screen_roster": overlap,
        "users": chosen,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"eligible one-batch VAL users: {len(ok)}   excluded: {len(excluded)}")
    print(f"drew {len(chosen)} users with seed {SEED} -> {OUT}")
    print(f"first 12: {chosen[:12]}")
    print(f"last  6: {chosen[-6:]}")
    if overlap is not None:
        print(f"overlap with the old deterministic 750-roster: {overlap}/750 "
              f"({100.0 * overlap / 750:.1f}%)")


if __name__ == "__main__":
    main()
