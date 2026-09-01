"""Why does insert_probes raise KeyError on the -id databases?

THE TWO MASKS THAT MUST AGREE.

  insert_probes (prepare_batch.py:88-99) decides a row may carry a probe if it is
      real & has_label & NOT the in-chunk-first REAL row of its card_id
  add_queries (data_processing.py:579) creates the paired imm query row for every row with
      is_first_review == False
  and then insert_probes assumes the first implies the second:
      "imm query row of each target: same review_th, is_query row (exists for every
       non-first review; eligibility implies non-first)"

A row that is eligible but has NO query row raises KeyError on q_map[review_th] and kills the
fetch worker. That is what happened to featB on 2026-08-21 AND again on 2026-09-01.

The two masks are computed from DIFFERENT quantities, so there are two distinct ways to
disagree, and this script reports which one fires:

  MECHANISM A -- CARD_ID COLLISION. Two genuinely different cards share a card_id value inside
      the chunk. Card B's own first review is then not the in-chunk-first row for that VALUE, so
      it is eligible, but is_first_review is True for it so it has no query row.
      Tell: the row has is_first_review == 1.

  MECHANISM B -- A NON-FIRST ROW LABELLED is_first_review. is_first_review is derived, not
      counted (CLAUDE.md: it IS elapsed_days == -1), so a mid-card review can carry it.
      Tell: the row has is_first_review == 1 AND its card_id appears earlier with
      is_first_review == 1 as well, i.e. the same card claims "first" twice.

  Both present as is_first_review == 1 on an eligible row, so they are separated by counting how
  many rows of that card_id claim to be first, and by whether the card_id block is contiguous.

A third possibility the script must be able to report, because it would refute both stories:
  MECHANISM C -- the row is NOT is_first_review and still has no query row. That would mean
      add_queries did not emit a row it should have, which is a different bug entirely.

Usage: probe_query_mismatch.py <db_path> <db_size> [n_users] [stride] [start_user]
Run with the SAME RWKV_ID_FEATURES value as the run being diagnosed -- the column layout, and so
the meaning of every index below, depends on it.
"""
import json
import os
import sys
from collections import Counter

import lmdb
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rwkv.data_processing import CARD_FEATURE_COLUMNS  # noqa: E402
from rwkv.prepare_batch import get_data  # noqa: E402

_LBL_HAS_LABEL = 4
_LBL_IS_QUERY = 6

# is_first_review is NOT a stored feature column -- it lives only in the build-time frame.
# So the LMDB can show WHICH rows mismatch and the STRUCTURE around them, and the mechanism
# is then settled against the raw parquet, which is ground truth.


def scan_sample(data):
    """Replay insert_probes' eligibility and return the rows whose query row is missing."""
    sk = data.skips.numpy()
    lab = data.global_labels.float().numpy()
    cards = data.ids["card_id"].numpy()
    review_ths = data.review_ths.numpy()
    n = sk.shape[0]

    real = ~sk
    has_lab = lab[:, _LBL_HAS_LABEL] > 0.5
    real_idx = np.nonzero(real)[0]
    if real_idx.size == 0:
        return []
    _, first_pos = np.unique(cards[real_idx], return_index=True)
    first_mask = np.zeros(n, dtype=bool)
    first_mask[real_idx[first_pos]] = True
    elig = real & has_lab & ~first_mask
    elig_rows = np.nonzero(elig)[0]
    if elig_rows.size == 0:
        return []

    qmask = sk & (lab[:, _LBL_IS_QUERY] > 0.5)
    q_map = {int(review_ths[q]): int(q) for q in q_rows} if (q_rows := np.nonzero(qmask)[0]).size else {}

    bad = []
    q_ths = set(q_map)
    for r in elig_rows:
        if int(review_ths[r]) in q_map:
            continue
        cid = int(cards[r])
        same = np.nonzero((cards == cid) & real)[0]
        ths = sorted(int(review_ths[i]) for i in same)
        n_q = sum(1 for t in ths if t in q_ths)
        # A single genuine card with R real rows yields exactly R-1 query rows. A shortfall of 2
        # means TWO rows of this card_id claim to be first -- either a collision between two
        # cards sharing the value, or one card labelled first twice.
        bad.append(
            dict(
                row=int(r),
                review_th=int(review_ths[r]),
                card_id=cid,
                n_real=int(same.size),
                n_query=n_q,
                shortfall=int(same.size) - 1 - n_q,
                contiguous=bool(same.size == (same.max() - same.min() + 1)),
                th_span=(ths[0], ths[-1]),
            )
        )
    return bad


def main():
    db_path = sys.argv[1]
    db_size = int(sys.argv[2])
    n_users = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 127
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    print("db          : %s" % db_path)
    print("RWKV_ID_FEATURES = %s   -> %d feature columns"
          % (os.environ.get("RWKV_ID_FEATURES", "<unset>"), len(CARD_FEATURE_COLUMNS)))
    print()

    env = lmdb.open(db_path, map_size=db_size, readonly=True, lock=False)
    tally = Counter()
    n_chunks = 0
    n_elig_users = 0
    shown = 0

    with env.begin(write=False) as txn:
        for k in range(n_users):
            user_id = start + k * stride
            raw = txn.get(f"{user_id}_batches".encode())
            if raw is None:
                continue
            n_elig_users += 1
            for batch in json.loads(raw):
                key = (user_id, batch[0], batch[1], batch[2])
                data = get_data(txn, key, device="cpu")
                n_chunks += 1
                for b in scan_sample(data):
                    mech = ("shortfall=%d (a lone genuine card must give 0)" % b["shortfall"])
                    tally[mech] += 1
                    tally["users_hit:%d" % user_id] += 0  # register the user
                    if shown < 12:
                        shown += 1
                        print("MISMATCH  user %d  chunk %s" % (user_id, batch))
                        print("   review_th=%d  card_id=%d" % (b["review_th"], b["card_id"]))
                        print("   this card_id in chunk: %d real rows, %d query rows, shortfall %d"
                              % (b["n_real"], b["n_query"], b["shortfall"]))
                        print("   contiguous=%s  review_th span=%s"
                              % (b["contiguous"], b["th_span"]))
                        print()

    print("=" * 78)
    print("scanned %d users / %d chunks" % (n_elig_users, n_chunks))
    total = sum(v for k, v in tally.items() if not k.startswith("users_hit:"))
    if total == 0:
        print("NO MISMATCH FOUND -- this db does not reproduce the KeyError on this sample.")
        print("That is not proof of absence: widen n_users/stride before concluding.")
        return 0
    for k, v in sorted(tally.items()):
        if not k.startswith("users_hit:"):
            print("  %-52s %6d" % (k, v))
    hits = sorted(int(k.split(":")[1]) for k in tally if k.startswith("users_hit:"))
    print("users affected: %s" % hits)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
