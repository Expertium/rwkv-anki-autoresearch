"""Exercise the BUG C fixes on real users at scale.

The new assert in data_processing (`note_id placeholders collided`) only fires on data that
actually has NaN note_ids, and the identity smoke deliberately samples 5 users chosen by NaN
rate. This runs the real `get_rwkv_data` over a broad stride sample of BOTH datasets and
reports what the fill produced, so the fix is confirmed on the distribution rather than on the
handful of users picked to be extreme.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RWKV_ID_FEATURES", "0")
from rwkv.data_processing import get_rwkv_data  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for label, root in (("PUBLISHED", Path(r"C:\Users\Andrew\anki-revlogs-10k")),
                    ("-id", Path(r"C:\Users\Andrew\anki-revlogs-10k-id"))):
    if not root.exists():
        continue
    users = sorted(int(x.split("=")[1]) for x in os.listdir(root / "revlogs"))
    sample = users[::max(1, len(users) // N)][:N]
    ok = bad = 0
    nan_rates, tot_rows, tot_nan = [], 0, 0
    for u in sample:
        try:
            df = get_rwkv_data(root, u)
        except AssertionError as e:
            bad += 1
            print("  *** user %d ASSERT: %s" % (u, str(e)[:90]))
            continue
        except Exception as e:  # noqa: BLE001
            print("  user %d skipped (%s)" % (u, type(e).__name__))
            continue
        ok += 1
        flag = df["note_id_is_nan"].to_numpy().astype(bool)
        tot_rows += len(df)
        tot_nan += int(flag.sum())
        nan_rates.append(100.0 * flag.mean())
        if flag.any():
            # the property the fill exists to provide, on real data
            nid = df.loc[flag, "note_id"]
            cid = df.loc[flag, "card_id"]
            assert nid.nunique() == cid.nunique(), "user %d: %d placeholders for %d cards" % (
                u, nid.nunique(), cid.nunique())
    print("%-10s %d users OK, %d assert failures" % (label, ok, bad))
    if nan_rates:
        print("           NaN note_id: %.2f%% of rows pooled; per-user mean %.2f%% median %.2f%% max %.2f%%"
              % (100.0 * tot_nan / tot_rows, np.mean(nan_rates), np.median(nan_rates),
                 max(nan_rates)))
        print("           per-card placeholder uniqueness held on every user with NaN rows")
