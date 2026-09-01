"""Does insert_probes survive a first review that is not first in frame order?

THE BUG. `insert_probes` paired every probe target with its imm query row by indexing
`q_map[review_th]` directly, on the stated assumption that "eligibility implies non-first". On
the -id databases that assumption is false: `is_first_review` is `elapsed_days == -1`, which the
builder sets from `state == 0` and only then sorts the frame by the corrected show time, so
Anki's 60 s duration cap can float a neighbouring review ahead of a card's genuine first review.
The first review then passes the positional eligibility mask, has no query row, and raises
KeyError -- killing a fetch worker and deadlocking the run. It did exactly that to featB twice.

WHAT THIS SMOKE ASSERTS, and why each part is here:

  1. NON-VACUITY. The legacy indexing is replayed on the same sample and MUST raise. Without
     this the other checks would pass just as happily against a database that never triggers the
     bug, which is the false-green shape this project has hit repeatedly (a control arm that
     inherits the treatment; an inertness check comparing two identical models).
  2. THE FIX WORKS. The real insert_probes runs on that sample without raising, and drops
     exactly the unpairable targets -- no more.
  3. BIT-IDENTITY WHERE THE ASSUMPTION HELD. On the published/e2s databases every picked target
     is pairable, so the filter removes nothing and the emitted probe geometry is unchanged.
     This is what licenses "no existing number moves"; it is asserted, not argued.

Probe density is forced to 1.0 so every eligible row is picked. At the production 0.08 the crash
is a coin flip per (seed, user, chunk), which is why it surfaced as an intermittent death rather
than a reproducible one.

Usage: smoke_probe_pairing.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ID_DB = "F:/rwkv_lmdb/train_db_5k_h1_id3"
ID_DB_SIZE = 400_000_000_000
ID_TEST_DB = "F:/rwkv_lmdb/test_db_5k_id3"
ID_TEST_DB_SIZE = 250_000_000_000
E2S_DB = "F:/rwkv_lmdb/train_db_5k_h1_e2s"
E2S_DB_SIZE = 400_000_000_000

# Located by scratchpad/features_rebuild/probe_query_mismatch.py and confirmed against the raw
# parquet: user 477 card 1708127478116, review_th 73724 is the first review (elapsed_days -1,
# 11.5 s) but 73723 carries duration exactly 60000 -- the cap -- and sorts ahead of it.
ID_CASES = [(ID_DB, ID_DB_SIZE, 477, 1), (ID_DB, ID_DB_SIZE, 344, 1)]
ID_TEST_CASES = [(ID_TEST_DB, ID_TEST_DB_SIZE, 6477, 1)]

_LBL_HAS_LABEL = 4
_LBL_IS_QUERY = 6

failures = []


def check(cond, label):
    print("  %-64s %s" % (label, "PASS" if cond else "*** FAIL ***"))
    if not cond:
        failures.append(label)


def eligible_and_qmap(data):
    """The eligibility set and query map, exactly as insert_probes computes them."""
    sk = data.skips.numpy()
    lab = data.global_labels.float().numpy()
    cards = data.ids["card_id"].numpy()
    review_ths = data.review_ths.numpy()
    n = sk.shape[0]

    real = ~sk
    has_lab = lab[:, _LBL_HAS_LABEL] > 0.5
    real_idx = np.nonzero(real)[0]
    _, first_pos = np.unique(cards[real_idx], return_index=True)
    first_mask = np.zeros(n, dtype=bool)
    first_mask[real_idx[first_pos]] = True
    elig_rows = np.nonzero(real & has_lab & ~first_mask)[0]

    qmask = sk & (lab[:, _LBL_IS_QUERY] > 0.5)
    q_map = {int(review_ths[q]): int(q) for q in np.nonzero(qmask)[0]}
    return elig_rows, q_map, review_ths


def legacy_pair(elig_rows, q_map, review_ths):
    """The pre-fix line. Raises KeyError on an unpairable target, as it did in production."""
    return np.array([q_map[int(review_ths[r])] for r in elig_rows], dtype=np.int64)


def iter_chunks(txn, get_data, user_id):
    raw = txn.get(f"{user_id}_batches".encode())
    if raw is None:
        return
    for batch in json.loads(raw):
        yield batch, get_data(txn, (user_id, batch[0], batch[1], batch[2]), device="cpu")


def run_id_case(db, size, user_id, expect_at_least, insert_probes, get_data, lmdb):
    env = lmdb.open(db, map_size=size, readonly=True, lock=False)
    n_bad_total, n_raised, n_ok = 0, 0, 0
    with env.begin(write=False) as txn:
        for batch, data in iter_chunks(txn, get_data, user_id):
            elig, q_map, ths = eligible_and_qmap(data)
            bad = [r for r in elig if int(ths[r]) not in q_map]
            if not bad:
                continue
            n_bad_total += len(bad)

            # 1. NON-VACUITY -- the legacy line must fail here.
            try:
                legacy_pair(elig, q_map, ths)
            except KeyError:
                n_raised += 1

            # 2. THE FIX -- the real function must survive and drop exactly the unpairable rows.
            out, meta = insert_probes(data, 1.0, 12345)
            n_ok += 1
            if meta is not None:
                n_probe_targets = meta.target.shape[0]
                if n_probe_targets != len(elig) - len(bad):
                    failures.append(
                        "user %s chunk %s: kept %d targets, expected %d"
                        % (user_id, batch, n_probe_targets, len(elig) - len(bad))
                    )
                # every kept target must genuinely have a query row
                if int(meta.query.min()) < 0:
                    failures.append("user %s chunk %s: negative query index" % (user_id, batch))
    print("  user %-6d unpairable targets=%d  legacy raised on %d chunk(s)  fixed ran on %d"
          % (user_id, n_bad_total, n_raised, n_ok))
    check(n_bad_total >= expect_at_least, "user %d: the bug is present (non-vacuous)" % user_id)
    check(n_raised == n_ok and n_ok > 0, "user %d: legacy raises exactly where fixed survives" % user_id)


def main():
    os.environ["RWKV_ID_FEATURES"] = "1"
    import lmdb
    from rwkv.prepare_batch import get_data, insert_probes

    print("=" * 78)
    print("PART 1+2  -id databases: the bug is real, and the fix absorbs it")
    print("=" * 78)
    for db, size, user, expect in ID_CASES + ID_TEST_CASES:
        run_id_case(db, size, user, expect, insert_probes, get_data, lmdb)

    # The published layout has 24 feature columns, the -id layout 46, and CARD_FEATURE_COLUMNS is
    # bound at import. A single process cannot hold both, so part 3 runs in its own.
    print()
    print("=" * 78)
    print("PART 3  published/e2s: nothing is dropped, so the change is bit-identical there")
    print("=" * 78)
    import subprocess

    env = dict(os.environ)
    env.pop("RWKV_ID_FEATURES", None)
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--e2s-noop"],
        env=env, capture_output=True, text=True,
    )
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stdout.write(r.stderr)
    check(r.returncode == 0, "e2s: every picked target is pairable (filter is a no-op)")

    print()
    print("=" * 78)
    if failures:
        print("SMOKE FAILED (%d)" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("SMOKE PASSED -- bug reproduced, fix absorbs it, e2s path provably unchanged")
    return 0


def e2s_noop():
    """Assert that on the published/e2s lineage the new filter never removes anything."""
    import lmdb
    from rwkv.data_processing import CARD_FEATURE_COLUMNS
    from rwkv.prepare_batch import get_data

    print("  feature columns = %d (published layout)" % len(CARD_FEATURE_COLUMNS))
    env = lmdb.open(E2S_DB, map_size=E2S_DB_SIZE, readonly=True, lock=False)
    n_chunks, n_elig, n_bad = 0, 0, 0
    with env.begin(write=False) as txn:
        for k in range(60):
            user_id = 1 + k * 83
            for _batch, data in iter_chunks(txn, get_data, user_id):
                elig, q_map, ths = eligible_and_qmap(data)
                n_chunks += 1
                n_elig += len(elig)
                n_bad += sum(1 for r in elig if int(ths[r]) not in q_map)
    print("  scanned %d chunks / %d eligible targets -> %d unpairable" % (n_chunks, n_elig, n_bad))
    if n_chunks == 0 or n_elig == 0:
        print("  *** the scan found nothing to check -- that is a FAILED TEST, not a pass")
        return 2
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    if "--e2s-noop" in sys.argv:
        raise SystemExit(e2s_noop())
    raise SystemExit(main())
