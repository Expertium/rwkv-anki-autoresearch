"""SELF-FEEDBACK: the model reads its own previous readout for the same card (Andrew 2026-09-21).

    "can't we add FSRS-style recurrence? ... use R(ahead, n-1), R(imm, n-1), S/weight/decay per
     curve as input features for the nth review. That way RWKV 'remembers' its own past
     predictions" ... "let's do the version that costs ~0" ... "let's use a proper scoring function
     aka log loss" ... "let's do it per-card"

WHY IT IS NOT AN ACCUMULATION FEATURE. The state update is LINEAR (delta-rule linear attention);
the head is NONLINEAR (MLP + softmax + the power-law mixture). A linear recurrence cannot recompute
its own nonlinear readout, so this is the one thing it structurally cannot do for itself. That also
separates it from iter 48 (`rcouple_w`), which fed R(t) from the SAME hidden vector -- something the
rating head could already compute -- and is part of the 0/2-closed routing family.

WHY THE HEAD AND NOT THE INPUT FC. If the readout entered `features2card`, then input(n) would
depend on trunk_out(n-1): a sequential dependency across rows that the chunked WKV scan cannot
express, forcing a per-review Python loop. That is exactly what shelved V1 (0.166 steps/s, ~9x
slower). Entering at the HEAD keeps one trunk pass and costs one extra head pass:

    base readout  = head(x)                       # all rows, parallel
    feats(r)      = f( base readout at src(r) )   # a gather, parallel, DETACHED
    final readout = head(x + fb2h(feats))         # all rows, parallel

`fb2h` is zero-init, so at step 0 `final == base` exactly and the lever is function-preserving.

THE 14 FEATURES, and the leak analysis per row. `src(r)` = the previous REAL row of the SAME CARD;
`srcq(r)` = that review's QUERY row (grade hidden), which is where its imm prediction lives.

    0-2   w_i        at src          the mixture weights
    3-5   s_raw_i/25 at src          log S -- the curve's time scale
    6-8   d_raw_i/25 at src          log d -- the curve's decay exponent
    9     logit R_ahead              src's curve at src's OWN label gap = the gap to THIS review
    10    logit R_imm                logit(1 - P(Again)) at srcq
    11    -log P(rating(src))        THE SURPRISE, as a proper score, from srcq
    12    has_src
    13    has_srcq

⚠⚠ THE LEAK IS ONE INDEX AWAY, AND FEATURE 9 IS THE ONE TO WATCH. `label_elapsed_seconds[r]` is the
gap from r to r's OWN ahead target, i.e. to the NEXT review -- so a curve evaluated with r's gap at
row r predicts the event row r is being scored on. Feature 9 is evaluated at **src's** gap, which is
the gap from src to r. Nothing here reads any row at or after r.
Feature 11 is safe for the same reason in a different direction: rating(src) was ALREADY an input at
src's real row, so the feature adds the model's own MISCALIBRATION, never new label information.
=> `scratchpad/selffeed/smoke_selffeed.py` shifts the index forward by one and REQUIRES the loss to
collapse; a lever whose leak control does not fire is not being tested.

WHAT WAS DROPPED FROM THE FIRST DESIGN, AND WHY. A running mean of the log score over the card's
earlier reviews. It is REDUNDANT BY CONSTRUCTION once feature 11 is an input: accumulating an input
over time is precisely what a linear recurrence CAN do. The frozen-model screen also measured it at
zero (`scratchpad/selffeed/FINDINGS.md`).

DEPLOY. Carrying the readout from review n-1 to review n grows the PER-CARD state, which gate #3
freezes -- Andrew's call, to be asked with a measured number. The 2026-09-07 screen found the curve
head's hidden is ~2.4-dimensional and a sufficient statistic of its input, so a learned 3-scalar
bottleneck should carry nearly all of it; the realistic ask is ~3 scalars per card, not 14.
The Python RNN + Rust paths are NOT built yet (same staging as R2: only if adopted).

Env: RWKV_SELFFEED=1 turns it on. Unset => not one line here runs and the model is bit-identical.
"""
import os

import numpy as np

N_FEATURES = 14


def enabled() -> bool:
    return os.environ.get("RWKV_SELFFEED", "0") == "1"


def detach_feedback() -> bool:
    """Default ON. `0` lets the loss at row n reach row n-1's readout -- two-step BPTT through the
    head, the in-family variant to try if the detached arm shows partial life."""
    return os.environ.get("RWKV_SELFFEED_DETACH", "1") == "1"


def base_only() -> bool:
    """RWKV_SELFFEED_BASE_ONLY=1: return the BASE readout and skip the second head pass.

    This is an EVAL-ONLY instrument and it answers PREREG P4. The base readout is detached and
    unsupervised, so the head can drift to expect the `fb2h` offset and degrade on bare `x` -- the
    exposure-bias risk (Bengio et al. 2015). If it has, the feedback the base pass produces is
    noise and a null result measures nothing. Scoring the same checkpoint with this flag gives the
    base readout's own by-user LogLoss, which is the number P4 is stated in.

    ⚠ It is read at CONSTRUCTION, like every other flag here: the model is a ScriptModule and an
    env read inside forward would be baked from whichever value the first construction saw.
    """
    return os.environ.get("RWKV_SELFFEED_BASE_ONLY", "0") == "1"


def build_indices(data, base: int, col_has_label: int, col_is_query: int):
    """One chunk -> (prev_row, src_query), each (n,) int64 flat indices offset by `base`, -1 = none.

    prev_row[r]   the previous REAL labelled row of r's card, strictly earlier in review order
    src_query[r]  the QUERY row of that same review (review_th is unique per review, so the join
                  cannot cross cards)

    Mirrors `build_ahead_query`: chunk-canonical row order, `+ base`, -1 where absent. Card ids are
    FACTORIZED first -- they are raw Anki epoch-ms (~1.6e12) and a naive `card * M + rt` composite
    key would overflow int64 on a long user.
    """
    sk = data.skips.numpy().astype(bool)
    lab = data.global_labels.float().numpy()
    n = sk.shape[0]
    prev = np.full(n, -1, dtype=np.int64)
    srcq = np.full(n, -1, dtype=np.int64)
    # the column indices are PASSED IN from prepare_batch's own _LBL_* constants rather than
    # hardcoded here, so the two copies cannot drift apart silently
    isq = lab[:, col_is_query] > 0.5
    has_lab = lab[:, col_has_label] > 0.5
    rt = data.review_ths.numpy().astype(np.int64)
    cards = data.ids["card_id"].numpy()
    _, cinv = np.unique(cards, return_inverse=True)
    cinv = cinv.astype(np.int64)

    # ⚠ `skips` is TRUE for QUERY and PROBE rows and FALSE for real reviews (insert_probes:128
    # `real = ~sk`; build_ahead_query reads query rows as `sk & isq`). Writing `sk & has_lab &
    # ~isq` here selects only probe rows, which carry has_label=0 -- so the mask comes back EMPTY,
    # every `prev` stays -1 and the whole lever is a silent no-op that trains, evaluates and
    # reports a perfectly clean null. That is why the smoke asserts a non-trivial share of rows
    # HAVE a predecessor rather than only asserting correctness where one exists.
    real = (~sk) & has_lab & ~isq
    ridx = np.nonzero(real)[0]
    if ridx.size == 0:
        return prev, srcq
    M = int(rt.max()) + 2
    key_real = cinv[ridx] * M + rt[ridx]
    o = np.argsort(key_real, kind="stable")
    ridx, key_real = ridx[o], key_real[o]
    key_row = cinv * M + rt
    # `left`: a real row finds ITSELF, so pos-1 is its card-predecessor; a query row finds its own
    # review's real row (same key), so pos-1 is the same predecessor. One rule, both row kinds.
    # RWKV_SELFFEED_SHIFT is the SMOKE'S LEAK CONTROL and must never be set by a real run: it
    # takes the card's NEXT real review instead of its previous one, which hands the row the
    # outcome it is being scored on. The smoke requires that control to show a near-perfect
    # feature-to-label relationship; a lever whose deliberate leak does not fire is not tested.
    if os.environ.get("RWKV_SELFFEED_SHIFT") == "1":
        pos = np.searchsorted(key_real, key_row, side="right")
        ok = pos < ridx.size
        cand = ridx[np.clip(pos, 0, ridx.size - 1)]
        ok &= cinv[cand] == cinv
        ok &= rt[cand] > rt
    else:
        pos = np.searchsorted(key_real, key_row, side="left") - 1
        ok = pos >= 0
        cand = ridx[np.clip(pos, 0, ridx.size - 1)]
        ok &= cinv[cand] == cinv                # the predecessor must be the SAME card
        ok &= rt[cand] < rt                     # and STRICTLY earlier -- never the row itself
    prev[ok] = cand[ok] + base

    q_rows = np.nonzero(sk & isq)[0]
    if q_rows.size:
        q_rt = rt[q_rows]
        qo = np.argsort(q_rt, kind="stable")
        q_rt_s, q_rows_s = q_rt[qo], q_rows[qo]
        want = np.where(ok, rt[cand], -1)
        p2 = np.searchsorted(q_rt_s, want)
        p2c = np.clip(p2, 0, q_rt_s.size - 1)
        hit = ok & (p2 < q_rt_s.size) & (q_rt_s[p2c] == want)
        srcq[hit] = q_rows_s[p2c][hit] + base
    return prev, srcq
