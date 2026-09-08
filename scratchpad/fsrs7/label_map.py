"""Build the (review_th -> label_review_th) map so our dumps can be joined to FSRS.

THE BUG THIS FIXES, and the alignment guard is what found it. decorrelate.py keys each row on
`review_th` -- the review the prediction is made FROM -- while `label_y` is the outcome of the
NEXT review of that card, whose index is `label_review_th`. FSRS keys on the review being
PREDICTED. So the two are off by one review of the same card, and joining them directly gave a
label agreement of 0.7726 against a chance rate of ~0.745, i.e. no alignment at all.

The RWKV-vs-RWKV comparison is unaffected: both arms used the same convention, so they aligned
with each other correctly (0.9998). Only the cross-family join needs the other key.

WHY THIS IS CHEAP. The map is a property of the DATA, not of any model, so it does not need the
2.7 h of RNN forward passes the dumps cost. It rebuilds the same frame through the same
`get_rwkv_data` call and reads two columns.

⚠ The map is per DATASET: arm A read `-id`, arm B read the published set. They are built
separately and applied to the matching arm.
"""

import os
import sys
from pathlib import Path

USERS = [5001, 5002, 5003, 5004]
OUT = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fsrs7")

arm = sys.argv[1]
if arm == "A":
    os.environ["RWKV_ID_FEATURES"] = "1"
    os.environ["RWKV_REAL_CYCLES"] = "1"
    data = r"C:\Users\Andrew\anki-revlogs-10k-id"
else:
    data = r"C:\Users\Andrew\anki-revlogs-10k"

sys.path.insert(0, "C:/Users/Andrew/rwkv-anki-autoresearch")

import numpy as np  # noqa: E402
from rwkv.data_processing import get_rwkv_data  # noqa: E402

RTH, LRTH, U = [], [], []
for uid in USERS:
    df = get_rwkv_data(Path(data), uid).sort_values("review_th", kind="stable")
    for col in ("has_label", "review_th", "label_review_th"):
        assert col in df.columns, f"missing column {col}"
    m = df["has_label"].astype(int) == 1
    RTH.append(df.loc[m, "review_th"].to_numpy(np.int64))
    LRTH.append(df.loc[m, "label_review_th"].to_numpy(np.int64))
    U.append(np.full(int(m.sum()), uid, np.int64))
    print(f"arm {arm} user {uid}: {len(df):,} rows -> {int(m.sum()):,} labelled", flush=True)

rth, lrth, u = np.concatenate(RTH), np.concatenate(LRTH), np.concatenate(U)
# label_review_th must be a LATER review of the same card, so strictly greater; a violated
# assert means the column is not what this script assumes and the map must not be used.
assert (lrth > rth).all(), "label_review_th is not strictly after review_th"
np.savez_compressed(OUT / f"labelmap_{arm}.npz", u=u, rth=rth, lrth=lrth)
print(f"wrote labelmap_{arm}.npz  rows={len(rth):,}")
