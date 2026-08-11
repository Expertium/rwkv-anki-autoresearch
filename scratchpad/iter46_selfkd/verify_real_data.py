"""Iter 46: does the self-distillation teacher index hold up on REAL LMDB rows, and how much of
the ahead objective does it actually cover? CPU + disk only, ~1 min, safe beside a running train
(LMDB readers are MVCC and the training run already has several).

The synthetic fixture in smoke_selfkd.py proves the JOIN LOGIC. It cannot prove:
  * that real dtypes/NaN patterns survive it (label_review_th is NOT fillna'd upstream),
  * COVERAGE -- what fraction of ahead-scored rows actually get a teacher. This is the number
    that decides whether the dose can move a 1e-4 gate at all, and it is the reason this iteration
    moved off the probe path in the first place (probes cover 8% of reviews at lambda 0.2,
    i.e. ~3% of the ahead objective's weight -- far under the 7.5e-5 noise floor).
  * that the teacher differs from the probe channel's own-review join ON REAL DATA.

Reads the TEST db so it does not add load to the train db the live run is streaming.
Run:  .venv/Scripts/python.exe scratchpad/iter46_selfkd/verify_real_data.py
"""
import json
import os
import sys

os.environ.setdefault("RWKV_SELFKD_BETA", "0.5")  # gate is on the flag; set before importing

import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv.prepare_batch import (  # noqa: E402
    _LBL_HAS_LABEL, _LBL_IS_QUERY, build_ahead_query, get_data, prepare,
)

DB = sys.argv[1] if len(sys.argv) > 1 else "test_db_5k"
SIZE = 300 * 1024 ** 3
USERS = [5001, 5002, 5010, 5123, 5500]

env = lmdb.open(DB, map_size=SIZE, readonly=True, lock=False)
tot_rows = tot_ahead = tot_hit = tot_q = 0
samples = []
with env.begin(write=False) as txn:
    for uid in USERS:
        raw = txn.get(f"{uid}_batches".encode())
        if raw is None:
            print(f"  user {uid}: no data, skipped")
            continue
        batches = json.loads(raw)
        b = batches[0]
        data = get_data(txn, (uid, b[0], b[1], b[2]), device="cpu")
        samples.append(data)
        lab = data.global_labels.float().numpy()
        isq = lab[:, _LBL_IS_QUERY] > 0.5
        has = lab[:, _LBL_HAS_LABEL] > 0.5
        ahead_rows = has & (~isq)          # exactly srs_model's ahead_mask
        aq = build_ahead_query(data, base=0)
        hit = (aq >= 0)
        # every hit must be an ahead row, land on a query row, and match review identity
        lrt = data.label_review_ths.numpy().astype(np.float64)
        rt = data.review_ths.numpy()
        bad = 0
        for r in np.nonzero(hit)[0]:
            q = int(aq[r])
            if not (ahead_rows[r] and isq[q] and rt[q] == lrt[r]):
                bad += 1
        n = len(rt)
        tot_rows += n
        tot_ahead += int(ahead_rows.sum())
        tot_hit += int(hit.sum())
        tot_q += int(isq.sum())
        print(f"  user {uid}: rows={n:6d} ahead_rows={int(ahead_rows.sum()):6d} "
              f"query_rows={int(isq.sum()):6d} teachers={int(hit.sum()):6d} "
              f"coverage={100 * hit.sum() / max(1, ahead_rows.sum()):5.1f}%  bad={bad}")
        assert bad == 0, f"user {uid}: {bad} teacher indices violate the pairing property"

print(f"\nTOTAL rows={tot_rows:,} ahead_rows={tot_ahead:,} query_rows={tot_q:,} "
      f"teachers={tot_hit:,}")
print(f"COVERAGE of the ahead objective = {100 * tot_hit / max(1, tot_ahead):.2f}% "
      f"(the probe path, for contrast, carries ~8% of reviews at lambda 0.2)")

# ---- through the REAL prepare(), including padding and the b*global_T offset ----
batch = prepare(samples[:3], seed=1234, probe_density=0.08)
aq = batch.ahead_query
assert aq is not None, "prepare() did not build ahead_query with the flag set"
B, T = aq.shape
flat = aq.reshape(-1)
lab = batch.labels.float()
isq_b = (lab[:, :, _LBL_IS_QUERY] > 0.5).reshape(-1)
has_b = (lab[:, :, _LBL_HAS_LABEL] > 0.5).reshape(-1)
hit = flat >= 0
print(f"\nprepare(): ahead_query {tuple(aq.shape)} dtype={aq.dtype} "
      f"teachers={int(hit.sum()):,} / ahead_rows={int((has_b & ~isq_b).sum()):,}")
assert int(hit.sum()) > 0
# in range, and every teacher is a query row -- checked on the PADDED (B,T) layout
assert int(flat[hit].max()) < B * T, "teacher index out of range for the padded layout"
assert bool(isq_b[flat[hit]].all()), "a teacher index does not land on a query row"
assert not bool(hit[isq_b].any()), "a query row was given a teacher (self-reference)"
# teachers must sit in the SAME sample block as their source row (no cross-user leakage)
rows = torch.arange(B * T)
assert bool(((flat[hit] // T) == (rows[hit] // T)).all()), "teacher crosses a sample boundary"
print("all teachers: in range, on query rows, same sample block, no query-row self-reference")

# ---- the probe channel's join must DIFFER on real data (it is a different question) ----
if batch.probe_target is not None and batch.probe_target.numel() > 0:
    pt, pq = batch.probe_target, batch.probe_query
    mine = flat[pt]
    both = mine >= 0
    diff = int((mine[both] != pq[both]).sum())
    print(f"probe rows: {int(both.sum())} with a teacher; differs from probe_query on {diff} "
          f"({100 * diff / max(1, int(both.sum())):.1f}%) -- must be ~100%, they answer "
          f"different questions")
    assert diff == int(both.sum()), "the two joins coincide -- one of them is wrong"

print("\nREAL-DATA VERIFY OK")
