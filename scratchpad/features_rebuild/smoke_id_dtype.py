"""Do the stored id streams survive the cast, on BOTH datasets?

Guards the int32 -> int64 fix in data_processing.create_sample. Two independent things must hold,
and the second is the one that was silently false for the entire project:

  1. NO VALUE CHANGES in the cast (the new assert inside create_sample enforces this and would
     raise; this smoke proves the assert is reached and passes on real data);
  2. ENTITY IDENTITY SURVIVES -- the number of distinct note ids in the built sample must match the
     number in the frame it was built from. Under int32 the -id set gave n_unique == 1 for every
     user, and the PUBLISHED set gave n_unique == 1 for users whose cards lack note metadata,
     because ID_PLACEHOLDER (3.14e17) saturated to INT32_MIN and merged them all into one note.

⚠ The second check is the point. A dtype smoke that only asserts "no exception" would have passed
happily on the broken build: saturation is not an error, it is a silent value change.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/smoke_id_dtype.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "0")

from rwkv.data_processing import create_sample, get_rwkv_data  # noqa: E402

PUB = Path(r"C:\Users\Andrew\anki-revlogs-10k")
IDD = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")

fails = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)


# user 1   : 0.0% NaN note metadata -- was healthy even under int32 on the published set
# user 101 : 66.8% NaN -- the pre-existing ID_PLACEHOLDER saturation case
# user 417 : 99.6% NaN -- almost entirely placeholder
for label, root, users in (("PUBLISHED", PUB, (1, 101, 417)), ("-id", IDD, (1, 101))):
    print("=== %s" % label)
    for uid in users:
        df = get_rwkv_data(root, uid)
        # create_sample calls add_queries itself; calling it here first trips the exhaustive
        # column-partition assert on the `index` column that the first pass adds.
        smp = create_sample(uid, df, [], torch.float32, "cpu")
        for name in ("card_id", "note_id", "deck_id"):
            t = smp.ids[name]
            a = t.numpy()
            want = int(df[name].nunique())
            got = int(len(np.unique(a)))
            neg = int((a < 0).sum())
            check("%s user %-4d %-8s dtype=%s uniq %d==%d, no negatives"
                  % (label, uid, name, t.dtype, got, want),
                  t.dtype == torch.int64 and got == want and neg == 0,
                  "got uniq=%d want=%d neg=%d" % (got, want, neg))

print("")
print("ALL PASS" if not fails else "FAILURES: %d" % len(fails))
sys.exit(1 if fails else 0)
