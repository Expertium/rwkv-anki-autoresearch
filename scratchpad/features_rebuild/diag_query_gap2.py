"""Find the chunk that killed featB's fetch worker -- by replicating insert_probes' own lookup.

The first hypothesis (a card with two elapsed_days == -1 rows) was REFUTED: n_first == n_cards
exactly in both datasets, 24 users each. So stop theorising and read the chunks.

This replicates exactly what insert_probes:118-123 does --

    qmask  = skips & is_query
    q_map  = {review_th -> query row}
    elig   = real & has_label & ~(in-chunk FIRST REAL occurrence of the card)
    lookup = q_map[review_th[r]] for every eligible r

-- and reports every chunk containing an eligible row whose review_th is missing from q_map. It does
NOT sample probes: it checks ALL eligible rows, which is strictly stronger than what the run does
(the run picks ~8% at random, so it hit the bug 950 steps in; the bug may be present far earlier).

Run it on BOTH dbs. featA survived 21,870 steps on the old one, so if the old db also has offending
chunks, this is a LATENT bug that featA merely got lucky on -- a different and more serious finding
than "the rebuild broke it".

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/diag_query_gap2.py <db> <n_users>
"""
import json
import os
import sys

import lmdb
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "0")

from rwkv.prepare_batch import _LBL_HAS_LABEL, _LBL_IS_QUERY, get_data  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else "F:/rwkv_lmdb/train_db_5k_h1_id2"
N_USERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SIZE = 400_000_000_000

env = lmdb.open(DB, map_size=SIZE, readonly=True, lock=False)
bad_chunks = 0
scanned = 0
elig_total = 0
examples = []

with env.begin(write=False) as txn:
    uid = 0
    checked_users = 0
    while checked_users < N_USERS and uid < 5000:
        uid += 1
        raw = txn.get(("%d_batches" % uid).encode())
        if raw is None:
            continue
        checked_users += 1
        for batch in json.loads(raw):
            key = (uid, batch[0], batch[1], batch[2])
            try:
                data = get_data(txn, key, device="cpu")
            except Exception as exc:  # noqa: BLE001
                print("  chunk %s unreadable: %s" % (key, exc))
                continue
            scanned += 1
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
            elig = real & has_lab & ~first_mask

            qmask = sk & (lab[:, _LBL_IS_QUERY] > 0.5)
            q_ths = set(int(review_ths[q]) for q in np.nonzero(qmask)[0])

            elig_rows = np.nonzero(elig)[0]
            elig_total += elig_rows.size
            missing = [int(review_ths[r]) for r in elig_rows
                       if int(review_ths[r]) not in q_ths]
            if missing:
                bad_chunks += 1
                if len(examples) < 6:
                    examples.append((key, len(missing), elig_rows.size, missing[:5]))

print("")
print("db      : %s" % DB)
print("users   : %d   chunks scanned: %d   eligible rows: %d" % (N_USERS, scanned, elig_total))
print("chunks with an eligible row lacking a query row: %d" % bad_chunks)
for key, nm, ne, sample in examples:
    print("  chunk %s : %d of %d eligible rows missing, review_ths %s" % (key, nm, ne, sample))
print("")
print("CLEAN" if bad_chunks == 0 else "OFFENDING CHUNKS FOUND")
