"""How should a <=100k parameter budget be SPLIT across the five streams?

THE QUESTION ANDREW'S PROPOSAL RAISES. "Use a much simpler recurrent core for interval
lengths + grades, and allocate >=99% of parameters to how other input features are
processed." Before designing that, three things have to be measured on the champion,
because the parameter map alone does not say which pools are EARNING their size:

  1. CONTRIBUTION  -- how much does each stream actually move the representation?
     The streams are a residual chain under RWKV_INTERLEAVE, so a stream's contribution
     is the sum of the deltas its own layers add to the running vector x.
  2. EFFECTIVE RANK -- in how many dimensions does that contribution live? A stream whose
     delta occupies 6 of 80 dimensions is paying for 80. Participation ratio of the PCA
     spectrum, plus the dimension count reaching 95% of variance.
  3. WITHIN-ENTITY FRACTION -- of that contribution's variance, how much varies over time
     WITHIN one entity, versus being a constant offset per entity? This is the one that
     speaks directly to the proposal. A stream that is ~all between-entity is doing
     in-context ENTITY IDENTIFICATION, which a small state can carry; a stream that is
     mostly within-entity is tracking DYNAMICS and needs its recurrence.

WHAT THIS IS NOT. Contribution magnitude is not accuracy. A small, well-aimed delta can
matter more than a large one -- exactly the lesson the delta-rule ablation taught (0.15 of
eigenvalue movement, +0.208 imm when removed). Read these as a SIZING prior for a redesign,
never as permission to delete a stream. Only an ablation measures accuracy.

WHERE THE HOOKS GO, AND WHY IT IS NOT SrsRWKVRnn.run. run_as_rnn has its OWN driver
(RNNProcess) which calls rnn.review() TWICE per review: once from imm_predict (a query
probe, skip=True) and once from process_row (the real state-advancing pass). Capturing
both would mix a counterfactual probe into the statistics, so capture is gated to
process_row -- which is also the only place the row's entity ids are in scope.

CPU-only, one thread, minutes.
Usage: .venv/Scripts/python.exe scratchpad/hybrid100k/stream_budget.py [n_users]
"""
import contextlib
import io
import os
import sys
from pathlib import Path

ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1", "RWKV_ZERO_FEATURES": "22",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_MUON": "1", "RWKV_MUON_INCLUDE_LORA": "1", "RWKV_NO_JIT": "1",
    "OMP_NUM_THREADS": "1",
}
for k, v in ENV.items():
    os.environ.setdefault(k, v)

import numpy as np   # noqa: E402
import torch         # noqa: E402

torch.set_num_threads(1)
sys.path.insert(0, os.getcwd())
import rwkv.run_as_rnn as ras                       # noqa: E402
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG   # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn     # noqa: E402

CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"
USERS = [5044, 5100, 5063, 5097, 5048, 5030]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NAMES = [n for n, _ in CFG.modules]
ID_COLS = ["card_id", "note_id", "deck_id", "preset_id"]


def fresh():
    return {"on": False, "acc": None, "last": None,
            "delta": [[] for _ in NAMES], "ids": {c: [] for c in ID_COLS},
            "xnorm": [], "xvec": []}


CAP = fresh()

_orig_review = SrsRWKVRnn.review
_orig_process = ras.RNNProcess.process_row


def _process_row(self, row):
    CAP["on"] = True
    try:
        out = _orig_process(self, row)
    finally:
        CAP["on"] = False
    for c in ID_COLS:
        CAP["ids"][c].append(row[c])
    return out


def _review(self, *a, **kw):
    if not CAP["on"]:
        return _orig_review(self, *a, **kw)
    if not getattr(self, "_probed", False):
        self._probed = True
        for i, mod in enumerate(self.rwkv_modules):
            mod._sidx = i
            _fl = mod.forward_layer

            def wrapped(lj, x_in, v0_in, state, extra=None, _fl=_fl, _m=mod):
                out = _fl(lj, x_in, v0_in, state, extra)
                d = (out[0] - x_in).detach().to(torch.float32).reshape(-1).numpy()
                CAP["acc"][_m._sidx] = (d.copy() if CAP["acc"][_m._sidx] is None
                                        else CAP["acc"][_m._sidx] + d)
                CAP["last"] = out[0].detach().to(torch.float32)
                return out

            mod.forward_layer = wrapped
    CAP["acc"] = [None] * len(NAMES)
    CAP["last"] = None
    out = _orig_review(self, *a, **kw)
    width = CFG.d_model
    for i, d in enumerate(CAP["acc"]):
        CAP["delta"][i].append(d if d is not None else np.zeros(width, dtype=np.float32))
    if CAP["last"] is not None:
        CAP["xnorm"].append(float(torch.linalg.vector_norm(CAP["last"])))
        CAP["xvec"].append(CAP["last"].reshape(-1).numpy().copy())
    else:
        CAP["xnorm"].append(float("nan"))
        CAP["xvec"].append(np.full(CFG.d_model, np.nan, dtype=np.float32))
    return out


SrsRWKVRnn.review = _review
ras.RNNProcess.process_row = _process_row


def eff_rank(X):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2. It is 1 for a
    rank-1 cloud and D for an isotropic one, and unlike a 95%-variance cutoff it is not a
    step function of the threshold. Both are reported."""
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return float("nan"), -1
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    if lam.sum() <= 0:
        return float("nan"), -1
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    d95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
    return pr, d95


def within_frac(X, ids):
    """Fraction of total variance that is WITHIN entity. Between = the spread of per-entity
    means; within = pooled spread inside entities. Entities seen once carry no within
    information; they still contribute to the total, which is the honest denominator."""
    X = X.astype(np.float64)
    tot = ((X - X.mean(0)) ** 2).sum()
    if tot <= 0:
        return float("nan"), 0
    uniq, inv = np.unique(np.asarray(ids), return_inverse=True)
    within, n_multi = 0.0, 0
    for k in range(len(uniq)):
        m = inv == k
        if m.sum() < 2:
            continue
        n_multi += 1
        Xi = X[m]
        within += ((Xi - Xi.mean(0)) ** 2).sum()
    return float(within / tot), n_multi


print("STREAM BUDGET PROBE -- champion %s" % CKPT)
print("%d smallest VAL users; contribution = sum of that stream's own residual deltas\n" % N)

agg = {}
user_means = {nm: [] for nm in NAMES}
for u in USERS[:N]:
    CAP.update(fresh())
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ras.run(data_path=Path("../anki-revlogs-10k"), model_path=CKPT,
                    label_db_path="label_filter_db", label_db_size=40_000_000_000,
                    user_id=u, verbose=False)
    except Exception as e:  # noqa: BLE001
        print("user %d skipped (%s: %s)" % (u, type(e).__name__, str(e)[:70]))
        continue
    n = len(CAP["delta"][0])
    xn = float(np.nanmean(CAP["xnorm"]))
    print("user %-6d state-advancing reviews %-6d mean||x||=%.3f" % (u, n, xn))
    print("   %-9s %11s %9s %6s %12s %9s"
          % ("stream", "rel|delta|", "eff_rank", "d95", "within_var", "entities"))
    for i, nm in enumerate(NAMES):
        X = np.stack(CAP["delta"][i])
        rel = float(np.mean(np.linalg.norm(X, axis=1))) / xn
        pr, d95 = eff_rank(X)
        col = nm if nm in ID_COLS else None
        if col is None:
            wf, ne = float("nan"), -1
        else:
            wf, ne = within_frac(X, CAP["ids"][col][:len(X)])
        print("   %-9s %11.4f %9.2f %6d %12s %9s"
              % (nm, rel, pr, d95,
                 "n/a" if wf != wf else "%.3f" % wf,
                 "-" if ne < 0 else ne))
        a = agg.setdefault(nm, {"rel": [], "pr": [], "d95": [], "wf": []})
        a["rel"].append(rel)
        a["pr"].append(pr)
        a["d95"].append(d95)
        if wf == wf:
            a["wf"].append(wf)
        # the user stream has ONE entity per user, so its between-entity term only exists
        # across users -- collect the per-user mean delta and decompose it at the end.
        user_means[nm].append(X.mean(0))
    np.savez_compressed(
        os.path.join("scratchpad", "hybrid100k", "deltas_u%d.npz" % u),
        x=np.stack(CAP["xvec"]),
        **{nm: np.stack(CAP["delta"][i]) for i, nm in enumerate(NAMES)},
        **{"id_" + c: np.asarray(CAP["ids"][c]) for c in ID_COLS})
    print("   saved deltas_u%d.npz" % u)
    print()

print("=" * 78)
print("POOLED across users")
print("   %-9s %11s %9s %6s %12s" % ("stream", "rel|delta|", "eff_rank", "d95", "within_var"))
for nm in NAMES:
    a = agg.get(nm)
    if not a:
        continue
    print("   %-9s %11.4f %9.2f %6.1f %12s"
          % (nm, np.mean(a["rel"]), np.mean(a["pr"]), np.mean(a["d95"]),
             "n/a" if not a["wf"] else "%.3f" % np.mean(a["wf"])))

if len(user_means[NAMES[0]]) >= 2:
    print()
    print("user_id stream: between-USER spread of its mean delta (its entity is the user,")
    print("so a within-user number cannot exist; this is the analogue of within_var for it)")
    M = np.stack(user_means["user_id"])
    across = float(np.linalg.norm(M - M.mean(0), axis=1).mean())
    typ = float(np.linalg.norm(M, axis=1).mean())
    print("   mean||mean delta|| = %.4f, spread across users = %.4f  (ratio %.3f)"
          % (typ, across, across / typ if typ else float("nan")))
print()
print("READ: eff_rank << 80 means the stream's OUTPUT is far narrower than d_model=80.")
print("      within_var near 0 means the stream IDENTIFIES the entity, it does not track it.")
print("      Neither is an accuracy claim -- see the header caveat.")
