"""RWKV_REAL_CYCLES: is it inert when off, and structurally right when on? CPU, ~1 min.

The §9 three-way-parity rule: every new arch env flag gets a case somewhere that exercises the
paths it touches. This flag touches FOUR sites -- id_features (the 24 columns + the width
contract), data_processing (CARD_FEATURE_COLUMNS), prepare_batch.add_encodings (the pseudo cycles
it removes) and run_as_rnn.get_tensor (the deploy twin of add_encodings) -- so the single-stack
harness cannot see it. This is its case. Three subprocesses, because the flag is read at import.

  off  : RWKV_ID_FEATURES=1 alone -- the champion's -id path. Must be byte-identical to before
         the flag existed. The one-time proof was a prepare().start snapshot taken BEFORE any
         edit on 2026-09-02 (users 5001/5137, gen-3 test db), reproduced bit-for-bit after all
         edits. This smoke keeps the structural half: width 114, encoding block 68, and the
         pseudo cycles present as 14 unit-circle pairs at dims 86..113.
  on   : width 109 = 69 card features + 40 ID dims; the encoding block is IDs ONLY (dims 46..85 of
         a gen-3 batch, since the cycle columns arrive with a gen-5 db); the 24 cycle columns are
         the tail of CARD_FEATURE_COLUMNS; day_of_week AND scaled_state are dropped; the
         review-time 7 d / 365 d halves are NOT duplicated (dow/doy already exist).
  data : get_rwkv_data on two -id users under the flag emits all 24 columns as unit-circle pairs,
         two users on the same UTC day agree on the phase to 1e-12 (epoch-anchored, not
         user-relative -- the whole point), and each first-review half is constant within a card.

NON-VACUITY: the unit-circle test is also applied to a window shifted by one dim and must FAIL
there, or a pass says nothing about position. bf16 tolerance (0.02) on the stored batch; the
first version used 1e-3 and "failed" the true layout on roundoff.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = "F:/rwkv_lmdb/test_db_5k_id3"

CHILD = r"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.environ["PYTHONPATH"])
mode = os.environ["MODE"]
import rwkv.id_features as idf
import rwkv.data_processing as dp
from rwkv.prepare_batch import get_data, prepare
import rwkv.run_as_rnn, rwkv.model.srs_model, rwkv.model.srs_model_rnn  # must import under both flag states
import lmdb

def circ(x, lo, hi):
    seg = x[:, lo:hi].reshape(x.shape[0], -1, 2)
    return float(((seg ** 2).sum(-1) - 1).abs().max())

on = mode == "on"
want = (69, 109, 40) if on else (46, 114, 68)
got = (len(dp.CARD_FEATURE_COLUMNS), idf.input_width(), idf.id_encoding_dims())
assert got == want, (got, want)
print("widths", got)
if on:
    C = dp.CARD_FEATURE_COLUMNS
    assert C[-24:] == list(idf.CYCLE_COLUMNS), "cycle columns are not the tail"
    assert "day_of_week" not in C and "scaled_state" not in C and "dow_sin" in C and "cyc7_sin" not in C
    print("layout: cycles are the tail; day_of_week + scaled_state dropped; 7 d review half not duplicated")

env = lmdb.open(os.environ["DB"], map_size=250_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    b = json.loads(txn.get(b"5001_batches"))[0]
    d = get_data(txn, (5001, b[0], b[1], b[2]), device="cpu")
x = prepare([d], target_len=int(d.length), seed=4321).start.float().reshape(-1, idf.input_width() if not on else 86)
x = x[x.abs().sum(1) > 0]
codes = torch.tensor([-1.5, -0.5, 0.5, 1.5])
assert torch.isin(x[:, 46:86], codes).float().mean().item() > 0.999, "dims 46..85 are not ID codes"
if on:
    assert x.shape[1] == 86, x.shape   # gen-3 db (46 card cols) + IDs only: the pseudo cycles are GONE
    print("encoding block is IDs only (assembled width 86 on a gen-3 db)")
else:
    e86, e85 = circ(x, 86, 114), circ(x, 85, 113)
    assert e86 < 0.02 and e85 > 0.02, (e86, e85)
    print("pseudo cycles present at 86..113 (unit-circle %.4f; shifted control %.3f)" % (e86, e85))

if on:
    from pathlib import Path
    import pandas as pd
    def load(u):
        df = dp.get_rwkv_data(Path("../anki-revlogs-10k-id"), u)
        raw = pd.read_parquet("../anki-revlogs-10k-id/revlogs/user_id=%d" % u)
        assert len(raw) == len(df)
        return df, (raw["review_time"].to_numpy() // 86400000).astype("int64")
    (d1, day1), (d2, day2) = load(5001), load(5137)
    worst = 0.0
    for c in idf.CYCLE_COLUMNS:
        if c.endswith("_sin"):
            s, cc = d1[c].to_numpy(), d1[c[:-4] + "_cos"].to_numpy()
            worst = max(worst, float(np.abs(s * s + cc * cc - 1).max()))
    assert worst < 1e-9, worst
    common = sorted(set(day1) & set(day2)); day = common[len(common) // 2]
    for c in ("cyc3_sin", "cyc36500_cos"):
        v1 = float(d1[c].to_numpy()[day1 == day][0]); v2 = float(d2[c].to_numpy()[day2 == day][0])
        assert abs(v1 - v2) < 1e-12, (c, v1, v2)
    g = d1.groupby("card_id")["cyc30_first_sin"].nunique()
    assert (g == 1).all() and d1["cyc30_first_sin"].nunique() > 1
    print("data: 24 unit-circle columns (%.1e); shared epoch phase across users; first halves constant per card" % worst)
print("REAL_CYCLES_%s_OK" % mode.upper())
"""


def run(mode):
    env = dict(os.environ, PYTHONPATH=REPO, RWKV_ID_FEATURES="1", RWKV_AUGMENT_SEED="4321",
               RWKV_REAL_CYCLES="1" if mode == "on" else "0", MODE=mode, DB=DB)
    env.pop("RWKV_ZERO_FEATURES", None)
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env, capture_output=True, text=True)
    out = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and "Warning" not in l]
    print("--- RWKV_REAL_CYCLES=%s" % ("1" if mode == "on" else "0"))
    for ln in out[-6:]:
        print("    " + ln)
    return p.returncode == 0 and any("REAL_CYCLES_%s_OK" % mode.upper() in ln for ln in out)


def main():
    ok = [run("off"), run("on")]
    print("\n" + ("REAL_CYCLES_ALL_PASS" if all(ok) else "REAL_CYCLES_FAILED"))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
