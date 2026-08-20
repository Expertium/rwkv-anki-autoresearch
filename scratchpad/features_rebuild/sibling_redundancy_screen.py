"""Does the champion's NOTE-STREAM state already encode sibling recency?

THE QUESTION, and why it is worth asking even though the column already ships. The note_id stream
pools every review of a note, so the note state has SEEN the sibling reviews. Whether its recurrence
encodes THE GAP TO THE NEAREST ONE is a different claim -- the state would have to reconstruct a
time difference. iter 50 is the precedent that makes this worth measuring: the deck tree was an exact
tie because the 5-stream hierarchy already BRACKETED the scope it added.

WHAT THIS CHANGES. Not whether the column ships -- it does, because a column that is IN the db can be
ablated later while one that is OUT costs a full rebuild. It changes how a NULL on that column is
READ:
  * high R2 -> the note stream already carries it, so a null is EXPECTED and says nothing about
    whether sibling recency matters to recall;
  * low R2  -> the recurrence genuinely does not represent it, so a null means the information does
    not help, which is a real finding.

METHOD. Walk users through the DEPLOY RNN path with the iter-53 champion and capture the INCOMING
note state at each review -- incoming, not outgoing, because the question is what the model holds
when it predicts THIS review. Then ridge-regress the true sibling gap on that state.

DAY RESOLUTION, DELIBERATELY. The gap is computed from the PUBLISHED dataset's own day_offset, not
from -id timestamps, so regressor and target come from the SAME frame and no cross-dataset row
alignment is needed. The two sets disagree on 0.001% of rows at day rollovers -- small, and exactly
the kind of silent misalignment that invents a result. The shipped column is seconds-resolution
end-to-start; for "does the state encode sibling recency at all", day resolution answers it.

THE CONTROLS ARE THE POINT. A 1440-dim regressor on a few thousand rows will fit SOMETHING, so this
reports three numbers and should not be read without them:
  * R2 of the note state,
  * R2 of a SHUFFLED target at the same dimensionality (the overfitting floor),
  * R2 of the review index alone (a trivial 1-dim baseline).
A note-state R2 that does not clearly beat the shuffled control is not evidence of anything. Same
discipline as the vacuous brute-force check earlier today, which "agreed" at 0.000e+00 while
comparing two all-sentinel arrays.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/sibling_redundancy_screen.py [users...]
"""
import os
import sys
from pathlib import Path

# The arch env MUST be set before the import: old-style ScriptModule bakes the first construction's
# flags into the compiled class, so a late setenv is silently ignored.
ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1",
    "RWKV_GRU_HEAD": "3",
    "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_STATE_CLAMP_TAU": "300",
    "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_MUON": "1",
    "RWKV_MUON_INCLUDE_LORA": "1",
    "RWKV_NO_JIT": "1",
    "RWKV_ID_FEATURES": "0",
}
for _k, _v in ENV.items():
    os.environ[_k] = _v

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.getcwd())
from rwkv import run_as_rnn  # noqa: E402

CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"
DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
USERS = [int(a) for a in sys.argv[1:]] or [101]

cap = {"states": []}
_orig_run = run_as_rnn.RNNProcess.run


def flatten_state(st):
    """The RNN state is NOT a tensor. RWKV7RNN.init_state returns {layer_idx: block_state} and each
    block returns `(time_state, channel_state)` -- the WKV matrix and the token-shift vector. The
    first version of this hook called .detach() on the dict and died on review 3; the vacuity assert
    at the bottom then refused to regress on the 2 rows it had, which is the only reason this did
    not silently report a number computed from nothing.

    Collect every tensor in the nested structure, in a deterministic order, and concatenate."""
    out = []

    def walk(x):
        if x is None:
            return
        if hasattr(x, "detach"):
            # .copy() is REQUIRED, not defensive: .numpy() shares memory with the tensor, and
            # rwkv_rnn_model mutates the state dict IN PLACE (`state[layer_idx] = block_state`,
            # with a comment that the caller deepcopies once per review). Without the copy every
            # captured row would alias whatever the state became later.
            out.append(x.detach().flatten().float().numpy().copy())
            return
        if isinstance(x, dict):
            for k in sorted(x.keys()):
                walk(x[k])
            return
        if isinstance(x, (tuple, list)):
            for v in x:
                walk(v)
            return

    walk(st)
    return np.concatenate(out) if out else None


def hooked(self, row, skip):
    st = self.note_states.get(row["note_id"])
    cap["states"].append(None if st is None else flatten_state(st))
    return _orig_run(self, row, skip)


run_as_rnn.RNNProcess.run = hooked


def sibling_gap_days(df):
    """Day-resolution sibling gap, by the same block trick as id_features.sibling_gap_seconds:
    sort by (note, time); adjacent blocks differ by card, so for every row of block b the answer is
    the last row of block b-1. Returns -1 where the note has no preceding sibling review."""
    n = len(df)
    out = np.full(n, -1.0)
    nid = pd.to_numeric(df["note_id"], errors="coerce").to_numpy(dtype=np.float64)
    cid = pd.to_numeric(df["card_id"], errors="coerce").to_numpy(dtype=np.float64)
    day = df["day_offset"].to_numpy(dtype=np.float64)
    ok = np.isfinite(nid) & np.isfinite(cid)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(nid[idx], kind="stable")]
    ns, cs, ds = nid[order], cid[order], day[order]
    m = order.size
    newnote = np.empty(m, dtype=bool)
    newnote[0] = True
    newnote[1:] = ns[1:] != ns[:-1]
    blk = newnote.copy()
    blk[1:] |= cs[1:] != cs[:-1]
    b = np.cumsum(blk) - 1
    nb = int(b[-1]) + 1
    last = np.zeros(nb, dtype=np.int64)
    last[b] = np.arange(m)
    bnote = np.zeros(nb)
    bnote[b] = ns
    bday = ds[last]
    prev = np.full(nb, np.nan)
    prev[1:] = np.where(bnote[1:] == bnote[:-1], bday[:-1], np.nan)
    rp = prev[b]
    out[order] = np.where(np.isnan(rp), -1.0, np.maximum(ds - rp, 0.0))
    return out


def ridge_r2(X, y, seed=0, lam=10.0):
    """Ridge with a 70/30 split. Returns HELD-OUT R2, which is the only kind that means anything
    with a 1440-dim regressor."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    k = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:k], X[k:], y[:k], y[k:]
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ym = ytr.mean()
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


allX, ally, allidx = [], [], []
for uid in USERS:
    cap["states"].clear()
    df = pd.read_parquet(DATA / "revlogs" / ("user_id=%d" % uid))
    dfc = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", uid)])
    df = df.merge(dfc.drop(columns=["user_id"], errors="ignore"), on="card_id", how="left")
    gap = sibling_gap_days(df)
    n_def = int((gap >= 0).sum())
    print("user %d: %d rows, %d with a preceding sibling review (%.2f%%)"
          % (uid, len(df), n_def, 100.0 * n_def / max(len(df), 1)), flush=True)
    if n_def < 100:
        print("  skipping -- too few defined rows to regress on", flush=True)
        continue
    try:
        run_as_rnn.run(data_path=DATA, model_path=CKPT, label_db_path="label_filter_db",
                       label_db_size=40_000_000_000, user_id=uid, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print("  walk ended: %s: %s" % (type(exc).__name__, exc), flush=True)
    got = min(len(cap["states"]), len(gap))
    print("  captured %d incoming note states" % got, flush=True)
    widths = set()
    for i in range(got):
        if cap["states"][i] is not None and gap[i] >= 0:
            allX.append(cap["states"][i])
            ally.append(np.log1p(gap[i]))
            allidx.append(i)
            widths.add(cap["states"][i].shape[0])
    assert len(widths) <= 1, "captured states have inconsistent widths %s" % sorted(widths)

assert len(ally) > 200, "VACUOUS: only %d regression rows -- not enough to conclude" % len(ally)
X = np.asarray(allX, dtype=np.float64)
y = np.asarray(ally, dtype=np.float64)
idx = np.asarray(allidx, dtype=np.float64).reshape(-1, 1)
assert float(y.std()) > 1e-6, "VACUOUS: the target is constant"

print("")
print("--- REGRESSION on %d rows, note-state dim %d, target std %.3f"
      % (X.shape[0], X.shape[1], float(y.std())))
r2_state = ridge_r2(X, y)
r2_shuf = ridge_r2(X, np.random.default_rng(1).permutation(y))
r2_idx = ridge_r2(idx, y)
print("R2 note state           : %+.4f" % r2_state)
print("R2 SHUFFLED control     : %+.4f   (the overfitting floor at this dimensionality)" % r2_shuf)
print("R2 review index (1 dim) : %+.4f   (trivial baseline)" % r2_idx)
print("")
if r2_state <= r2_shuf + 0.02:
    print("VERDICT: the note state does NOT encode sibling recency beyond the overfitting floor.")
    print("  => a null on scaled_sibling_gap would mean the information does not help,")
    print("     NOT that the recurrence already had it.")
else:
    print("VERDICT: the note state DOES carry sibling recency (%+.4f over the floor)."
          % (r2_state - r2_shuf))
    print("  => a null on scaled_sibling_gap is EXPECTED and uninformative -- iter 50's shape.")
