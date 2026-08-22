"""What does ONE prefix re-optimization of FSRS-7 cost?

Andrew 2026-08-22: the stored per-user parameters are the FINAL ones, fitted on all
TimeSeriesSplit folds but the last, so they have seen most of the history -- including the
future relative to almost every replay day. A realistic replay has to re-optimize at each
checkpoint on the prefix available THEN. This measures whether that is affordable.

Uses srs-benchmark's own training path (script._fit_trainable_weights), with
--recency, so the parameters are the ones the leaderboard's best FSRS-7 variant would have.
"""
import sys, os, time
from pathlib import Path

SRSB = Path(r"C:\Users\Andrew\srs-benchmark")
sys.path.insert(0, str(SRSB))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.chdir(SRSB)

sys.argv = ["script.py", "--algo", "FSRS-7", "--short", "--secs", "--recency",
            "--processes", "1"]
import torch
torch.set_num_threads(1)
import script                                   # builds script.config from sys.argv
from data_loader import UserDataLoader

script.config.device = torch.device("cpu")
print("device =", script.config.device, " batch =", script.config.batch_size,
      " recency =", script.config.use_recency_weighting)

uid = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 5100
ds = UserDataLoader(script.config).load_user_data(uid)
ds = ds.sort_values("review_th", kind="stable").reset_index(drop=True)
print("user %d: %d rows in the benchmark frame, days %d..%d"
      % (uid, len(ds), ds["day_offset"].min(), ds["day_offset"].max()))

for frac in (0.25, 0.5, 1.0):
    n = int(len(ds) * frac)
    pre = ds.iloc[:n]
    t0 = time.perf_counter()
    w = script._fit_trainable_weights(script._apply_recency_weighting(pre))
    dt = time.perf_counter() - t0
    print("  prefix %5.0f%%  %6d rows  ->  %6.1f s   w[0:4]=%s"
          % (100 * frac, n, dt, [round(float(x), 4) for x in w[:4]]))
