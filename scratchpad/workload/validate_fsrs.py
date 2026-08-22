"""Does the FSRS-7 arm reproduce the benchmark's OWN per-row predictions, exactly?

WHY THIS EXISTS. The first validation compared LogLoss against the recorded per-user value
and got -0.0006 on user 5100 but +0.029 on user 5530. That gap is almost certainly the ROW
SET (this replay keeps rows the benchmark's pipeline drops, so the states differ), but
"almost certainly" is not a check -- and a LogLoss comparison across different row sets can
hide a real implementation error behind a plausible excuse.

So this removes the row set from the comparison entirely: run srs-benchmark's own
create_features + Collection.batch_predict on a user, then run THIS replay on THAT SAME
frame, and compare per-row retention. Same inputs, same model, so the two must agree to
float precision. A disagreement is an implementation bug in the arm; agreement means the
earlier LogLoss deltas are purely a row-set effect and can be reported as such.

INDEX ALIGNMENT, which is the whole subtlety. The benchmark predicts row j's outcome from
the rows BEFORE j (its `tensor` column is the prefix, `delta_ts` is row j's own gap). This
replay produces the state AFTER row j and predicts row j+1. So the benchmark's prediction
at row j corresponds to this replay's state at the previous review of the same card.

Usage: <srs-benchmark venv python> validate_fsrs.py <user_id> [more user ids...]
"""
import sys
import os
from pathlib import Path

SRSB = Path(r"C:\Users\Andrew\srs-benchmark")
sys.path.insert(0, str(SRSB))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from models.fsrs_v7 import FSRS7
from data_loader import UserDataLoader
from utils import Collection

from fsrs_arm import make_config, load_params, replay_states, SECONDS_PER_DAY

TABLES = Path(__file__).resolve().parent / "tables"


def check(uid, cfg):
    w, bench_ll, bench_size = load_params(uid)
    model = FSRS7(cfg, w=w).to(cfg.device)
    model.eval()

    ds = UserDataLoader(cfg).load_user_data(uid)
    ds = ds.sort_values("review_th", kind="stable").reset_index(drop=True)
    ret, _, _ = Collection(model, cfg).batch_predict(ds)
    ret = np.asarray(ret, dtype=np.float64)

    # ⚠ REPLAY THE FULL RAW STREAM, NOT `ds`. The benchmark builds its history prefixes
    # BEFORE _common_postprocessing drops the delta_t==0 rows, so a scored row's `tensor`
    # still contains those dropped reviews -- verified directly: user 5100 review_th 16 is
    # the card's SECOND review, is labelled i=1 in `ds`, and carries tensor [[0., 1.]], the
    # dropped first review. Replaying only the surviving rows gives a DIFFERENT state and
    # was the entire mismatch in this script's first version (mean |diff| 0.039).
    tbl = pd.read_parquet(TABLES / ("u%d.parquet" % uid))
    tb, st = replay_states(model, tbl)

    # tb is sorted by (card_id, review_th); the benchmark's prediction at row j is made
    # from the state after the PREVIOUS row of the same card, i.e. st[j-1] within a block.
    card = tb["card_id"].to_numpy()
    dt = np.maximum(0.0, tb["elapsed_seconds"].to_numpy(dtype=np.float64)) / SECONDS_PER_DAY
    prev_ok = np.zeros(len(tb), dtype=bool)
    prev_ok[1:] = card[1:] == card[:-1]
    w_backup = model.w
    model.w = torch.nn.Parameter(model.w.data.double(), requires_grad=False)
    try:
        prev = np.roll(np.arange(len(tb)), 1)
        mine = model.forgetting_curve(
            torch.from_numpy(dt), st[prev, 0].double(), st[prev, 1].double(),
            st[prev, 2].double()).numpy()
    finally:
        model.w = w_backup

    out = pd.DataFrame({"review_th": tb["review_th"].to_numpy(), "mine": mine,
                        "ok": prev_ok}).set_index("review_th")
    ref = pd.DataFrame({"review_th": ds["review_th"].to_numpy(), "bench": ret}
                       ).set_index("review_th")
    j = ref.join(out, how="inner")
    # The benchmark's FIRST scored row per card has no previous row here either; it reports
    # the init-state retention for it. Compare only where both sides have a real predecessor.
    j = j[j["ok"]]
    d = np.abs(j["mine"].to_numpy() - j["bench"].to_numpy())
    print("u%-6d rows compared %-7d   max |diff| %.3e   mean %.3e   %s"
          % (uid, len(j), d.max(), d.mean(),
             "MATCH" if d.max() < 1e-4 else "*** MISMATCH ***"))
    return d.max()


def main():
    cfg = make_config()
    worst = 0.0
    for a in sys.argv[1:]:
        worst = max(worst, check(int(a), cfg))
    print("")
    print("worst max|diff| across users: %.3e" % worst)
    print("MATCH means the arm's recurrence and curve ARE the benchmark's, so the earlier")
    print("per-user LogLoss deltas come from the row set, not from the implementation.")


if __name__ == "__main__":
    main()
