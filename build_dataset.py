"""Robust wrapper around `python -m rwkv.data_processing`.

data_processing uses a multiprocessing Pool + Manager queues. If any worker raises
(e.g. a transient Windows LMDB file race), the Pool tears down and in-flight users are
SILENTLY dropped while the process still exits 0. This wrapper re-runs data_processing
(its built-in `_done` resume logic makes re-runs cheap) until every expected user has a
`_batches` entry, or until no further progress is made.

Usage:
    python build_dataset.py --config rwkv/data_processing_config_test_subset.toml
"""

import argparse
import subprocess
import sys

import lmdb
import tomli


def expected_users(cfg):
    if "USER_IDS" in cfg:
        return list(cfg["USER_IDS"])
    return list(range(cfg["USER_START"], cfg["USER_END"] + 1))


def coverage(lmdb_path, lmdb_size, users):
    env = lmdb.open(lmdb_path, map_size=lmdb_size, readonly=True, lock=False)
    present = set()
    with env.begin() as txn:
        for u in users:
            if txn.get(f"{u}_batches".encode()) is not None:
                present.add(u)
    env.close()
    return present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-retries", type=int, default=6)
    args = ap.parse_args()

    cfg = tomli.load(open(args.config, "rb"))
    users = expected_users(cfg)
    lmdb_path, lmdb_size = cfg["LMDB_PATH"], cfg["LMDB_SIZE"]

    prev_missing = None
    for attempt in range(1, args.max_retries + 1):
        print(f"\n=== data_processing attempt {attempt}/{args.max_retries} ===", flush=True)
        subprocess.run(
            [sys.executable, "-m", "rwkv.data_processing", "--config", args.config]
        )
        present = coverage(lmdb_path, lmdb_size, users)
        missing = [u for u in users if u not in present]
        print(
            f"coverage: {len(present)}/{len(users)} present; {len(missing)} missing",
            flush=True,
        )
        if not missing:
            print("BUILD COMPLETE: all expected users present.")
            return 0
        print(f"missing (first 20): {missing[:20]}", flush=True)
        if prev_missing is not None and set(missing) == set(prev_missing):
            print(
                "No progress vs previous attempt — these users may be genuinely "
                "unprocessable. Stopping; investigate them individually.",
                flush=True,
            )
            return 1
        prev_missing = missing

    print(f"INCOMPLETE after {args.max_retries} attempts; still missing: {missing}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
