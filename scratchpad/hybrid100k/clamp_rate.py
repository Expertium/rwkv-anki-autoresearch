"""How much of the -id dataset's review_time correction is spoiled by a clamped taken_millis?

THE OBJECTION (OSR, HF discussion #3): review_time = revlog.id - taken_millis is only exact
when taken_millis is the true answer duration. Anki caps it at the deck preset's "maximum
answer seconds" (maxTaken, default 60 s), so for any review the user took longer on, the
subtraction is too small and the computed show time lands LATE.

WHY THIS IS MEASURABLE RATHER THAN UNKNOWN. The clamped value is what gets stored, and it is
stored in our `duration` column. So a clamped review is not hidden -- it sits exactly at the
cap, and the cap shows up as a spike at the top of the duration histogram. This counts them.

WHAT THE ERROR ACTUALLY IS. If clamped, true_show = answer - actual and computed_show =
answer - cap with actual >= cap, so computed_show >= true_show: the computed time is an UPPER
bound, late by (actual - cap), which is unbounded above but concentrated just past the cap.
Note the correction is still in the right direction and still an improvement -- the
uncorrected alternative is answer time itself, which is late by the FULL duration on every row.

Read-only over ../anki-revlogs-10k-id.
"""
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join("..", "anki-revlogs-10k-id", "revlogs")
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

users = sorted(int(d.split("=")[1]) for d in os.listdir(ROOT) if d.startswith("user_id="))
step = max(1, len(users) // N_USERS)
sample = users[::step][:N_USERS]

tot_rows = tot_clamped = 0
caps = Counter()
per_user = []
for u in sample:
    d = os.path.join(ROOT, "user_id=%d" % u)
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet")]
    if not files:
        continue
    dur = np.concatenate([pq.read_table(f, columns=["duration"])["duration"].to_numpy()
                          for f in files])
    if dur.size == 0:
        continue
    mx = int(dur.max())
    # A cap is a value with anomalous mass at the very top. Require the max to be shared by
    # more than one row AND to be a whole number of seconds -- a genuine longest-review time
    # landing on an exact second, repeatedly, is not plausible.
    n_at_max = int((dur == mx).sum())
    is_cap = n_at_max > 1 and mx % 1000 == 0
    tot_rows += dur.size
    if is_cap:
        tot_clamped += n_at_max
        caps[mx] += 1
    per_user.append((u, dur.size, mx, n_at_max, 100.0 * n_at_max / dur.size, is_cap))

print("sampled %d users, %d reviews\n" % (len(per_user), tot_rows))
print("cap values seen (ms -> how many users): %s"
      % ", ".join("%d(%.0fs)->%d" % (k, k / 1000, v) for k, v in caps.most_common(8)))
print()
rates = [r for (_, _, _, _, r, c) in per_user if c]
if rates:
    print("clamped-row rate per user, over the %d users with a detected cap:" % len(rates))
    print("   mean %.3f%%   median %.3f%%   p90 %.3f%%   max %.3f%%"
          % (np.mean(rates), np.median(rates), np.percentile(rates, 90), max(rates)))
print("POOLED clamped rows: %d of %d = %.3f%%"
      % (tot_clamped, tot_rows, 100.0 * tot_clamped / tot_rows))
print()
print("worst 8 users by clamped rate:")
for u, n, mx, k, r, c in sorted(per_user, key=lambda t: -t[4])[:8]:
    print("   user %-5d rows %-8d cap %-7d at-cap %-6d %.3f%% %s"
          % (u, n, mx, k, r, "" if c else "(max not a whole second -- not counted)"))
