"""Is storing label_elapsed_seconds in bf16 a train-vs-deploy divergence that MATTERS?

WHY ASK. create_sample puts the label columns in ONE tensor at the config dtype, which is bfloat16:
    global_labels_df = section_df[["label_elapsed_seconds", "label_elapsed_days", "label_y",
                                   "label_rating", "has_label", "label_is_equalize", "is_query"]]
    global_labels_tensor = torch.tensor(global_labels_df.to_numpy(), dtype=dtype)
bf16 has 8 mantissa bits, i.e. ~0.4% relative precision. The five flag/rating columns are small
integers and bf16 represents them EXACTLY, so they are not at risk. The two ELAPSED columns are
not: label_elapsed_seconds reaches ~1e8 on long gaps.

This is the same SHAPE as the int32 id bug found this morning -- training and eval read the stored
value, deploy computes it fresh at full precision -- so it deserves the same treatment: measure the
magnitude instead of assuming, and decide on the number.

WHAT WOULD MAKE IT HARMLESS. If the model consumes log(t), a 0.4% error in t is a ~0.004 absolute
error in log t, which is nothing next to the spread of log-gaps. If it consumes t directly against
a learned scale, the error stays relative and is still small. The number that matters is therefore
the error in whatever space the curve is evaluated in.

Reports, over real users:
  * fraction of rows where the bf16 round-trip changes the value at all;
  * the relative error distribution (median / p99 / max);
  * the ABSOLUTE error in log-space, which is what the forgetting curve actually sees;
  * the same for the flag columns, which must be EXACTLY zero.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/diag_label_bf16.py [n_users]
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "0")

from rwkv.data_processing import get_rwkv_data  # noqa: E402

PUB = Path(r"C:\Users\Andrew\anki-revlogs-10k")
n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 6
uids = [1, 101, 209, 417, 625, 833][:n_users]

ELAPSED = ["label_elapsed_seconds", "label_elapsed_days"]
FLAGS = ["label_y", "label_rating", "has_label", "label_is_equalize", "is_query"]

agg = {c: [] for c in ELAPSED}
flag_bad = {c: 0 for c in FLAGS}
tot = 0
for uid in uids:
    df = get_rwkv_data(PUB, uid)
    # add_queries creates the label columns; get_rwkv_data alone may not have them all
    from rwkv.data_processing import add_queries
    df = add_queries(df, [])
    tot += len(df)
    for c in ELAPSED:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=np.float64)
        rt = torch.tensor(v, dtype=torch.bfloat16).float().numpy().astype(np.float64)
        m = v > 0
        if m.any():
            agg[c].append(np.abs(rt[m] - v[m]) / v[m])
    for c in FLAGS:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=np.float64)
        rt = torch.tensor(v, dtype=torch.bfloat16).float().numpy().astype(np.float64)
        flag_bad[c] += int((rt != v).sum())

print("rows examined: %d over users %s" % (tot, uids))
print("")
print("--- FLAG / RATING columns (must be EXACT in bf16)")
for c in FLAGS:
    print("  %-20s rows changed by the bf16 round-trip: %d" % (c, flag_bad[c]))
print("")
print("--- ELAPSED columns (the ones at risk)")
for c in ELAPSED:
    if not agg[c]:
        print("  %-24s no positive values" % c)
        continue
    r = np.concatenate(agg[c])
    print("  %-24s n=%d  rel err  median %.2e  p99 %.2e  max %.2e"
          % (c, r.size, np.median(r), np.percentile(r, 99), r.max()))
    # what the curve actually sees: log1p(t) error for the worst relative case
    print("  %-24s => |delta log(1+t)| at the max rel err: %.5f"
          % ("", abs(np.log1p(1.0 + r.max()) - np.log1p(1.0))))
print("")
print("READ: the flags must be exactly 0 changed. For the elapsed columns the question is not")
print("whether bf16 rounds -- it does -- but whether the error in LOG space is large next to the")
print("gate, which resolves differences of 1e-4 in logloss.")
