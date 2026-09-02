"""Real-timestamp input features from `anki-revlogs-10k-id`. OFF unless RWKV_ID_FEATURES=1.

This is code site 1-3 of `optimization/FUTURE_FEATURES.md`'s implementation plan. The published
`anki-revlogs-10k` is anonymized (integer `day_offset`, no absolute time), so none of this can be
derived from it; the `-id` set keeps raw Anki epoch-ms ids and adds a corrected `review_time`
(= `revlog.id - taken_millis`, i.e. the SHOW time), which is what every feature here is built on.

**The flag is OFF by default and the module is inert when off** -- `data_processing.py` neither
extends its column list nor calls anything here. That matters because a feature change requires an
LMDB rebuild: a run against the existing DBs must produce byte-identical input tensors, and the
cheapest way to guarantee that is for the new code to not execute at all.

★ THE LEAKAGE RULE IS THE ONLY THING HERE THAT IS EASY TO GET SILENTLY WRONG. Every count and every
batch statistic must be computed AS OF REVIEW TIME, not from the finished table. The reference
derivations in `optimization/feature_stats_id.py` (which measured the normalization constants over
300 users) did NOT do this -- they counted a user's whole card collection -- because they only ever
needed marginal distributions. Ported here they would leak: `creation_batch_1d` would tell the model
how many cards the user WOULD create later that day. The counts below are clipped at `review_time`.

⚠ TIMEZONES ARE UNKNOWN. `review_time` is epoch ms with no offset, so a user's local midnight is an
unknown constant phase shift. That is exactly why the high-value time-of-day feature is the
DEVIATION from the user's own running circular mean (the shift cancels), with the raw phase fed
alongside for the ~10% of users whose review hours are near-uniform (measured mean resultant length
R = 0.415, p10 = 0.197). The calendar features (day-of-week, day-of-year) cannot cancel it and are
UTC-based; for a reviewer near local midnight the weekday can be off by one. Accepted, not fixed.
"""

import os

import numpy as np

# pandas is imported lazily inside the two functions that need it: this module is imported by
# `srs_model.py` / `srs_model_rnn.py` purely for the input WIDTH, and the deploy path should not
# pull pandas in to answer an integer question.

# Anki ids are epoch-MILLISECONDS. The default deck and the default preset both use the sentinel
# id 1, and FUTURE_FEATURES.md measured the population as cleanly bimodal (a real stamp, or the
# sentinel) -- so any threshold between them works. 1e11 ms = 1973: far below Anki's existence and
# far above 1.
TIMESTAMP_MIN_MS = 1e11
_MS_PER_DAY = 86_400_000.0
_DEFAULT_ID = 1

# The column the rebuild DROPS (Andrew 2026-08-09): Anki's card state. Already zeroed everywhere
# via RWKV_ZERO_FEATURES=22 and baked out of the Rust export; the rebuild stops emitting it at the
# source so no consumer needs to know a mask ever existed. Dropped from the INPUT vector only --
# the raw `state` column stays in the frame because the filtering/label machinery reads it.
DROPPED_COLUMN = "scaled_state"

# Appended to CARD_FEATURE_COLUMNS, in this order. All 21 are `keep_columns`: none of them carries
# the outcome of the review being predicted, only WHEN it happens and what the card/deck are -- the
# same status as `day_offset_diff` and `deck_id_is_nan`, which are kept today.
NEW_COLUMNS = [
    # -- time of day (FUTURE_FEATURES priority: high, Andrew's #1) --
    "tod_sin",
    "tod_cos",
    "tod_dev_sin",
    "tod_dev_cos",
    # -- TRUE-phase calendar cycles; upgrades the pseudo-phase day_offset cycles --
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    # -- sub-day recency and ages --
    "scaled_t_since_any_review",
    "scaled_user_tenure",
    "scaled_creation_to_first_review",
    # -- deck-derived --
    "scaled_deck_age_at_review",
    "card_predates_deck",
    "is_default_deck",
    "scaled_deck_depth",
    # -- creation batch (import-vs-handmade signal) --
    "scaled_creation_batch_1min",
    "scaled_creation_batch_1h",
    "scaled_creation_batch_1d",
    "scaled_creation_batch_pos_1h",
    # -- preset: the AGE is dead (defined for 1 row in 14); ship only the flag --
    "is_default_preset",
    # -- THE TWO OMISSIONS, added 2026-08-20 after Andrew's coverage audit --
    # Andrew's sibling gap, in its deployable (non-leaking) form: seconds since the END of the
    # most recent review of a DIFFERENT card of the same note. See sibling_gap_seconds().
    "scaled_sibling_gap",
    # Andrew's "probably not important, but we can try": was this card created before the user
    # ever reviewed anything? Separates imported/pre-existing collections from cards made by a
    # user who was already studying.
    "card_predates_first_review",
]

# Measured 2026-08-03 over 300 users / 24,306,799 reviews, stride-16 across the TRAIN half 1-5000
# ONLY -- deriving normalization constants from 5001-10000 would leak eval-set statistics into
# every candidate's inputs. Source: optimization/feature_stats_id.py, full output
# optimization/feature_stats_id.json.
#
# ⚠ THE EXISTING NINE CONSTANTS ARE DELIBERATELY NOT TOUCHED. The same pass recomputed them and
# missed by up to 24%, but the pattern says SAMPLE DIFFERENCE, not error: every badly-missed
# constant scales with how much the user reviews, and all of them come out higher, i.e. upstream
# sampled smaller users. They are part of the model as trained and each column is standardized to
# roughly unit scale anyway. The new columns are therefore centred on a slightly different sample
# than the old ones, which is a cosmetic inconsistency and not a modelling problem.
STATISTICS_ID = {
    "t_since_any_review_mean": 2.2682,
    "t_since_any_review_std": 1.3649,
    "user_tenure_mean": 17.8880,
    "user_tenure_std": 1.1326,
    "creation_to_first_review_mean": 14.8600,
    "creation_to_first_review_std": 4.3134,
    "deck_age_at_review_mean": 14.3750,
    "deck_age_at_review_std": 5.5264,
    "creation_batch_1min_mean": 2.5430,
    "creation_batch_1min_std": 2.1196,
    "creation_batch_1h_mean": 3.6674,
    "creation_batch_1h_std": 2.1707,
    "creation_batch_1d_mean": 4.4760,
    "creation_batch_1d_std": 2.2149,
    "creation_batch_pos_1h_mean": 3.0207,
    "creation_batch_pos_1h_std": 1.9252,
    "deck_depth_mean": 1.2703,
    "deck_depth_std": 1.6451,
    # Added 2026-08-20 with the sibling gap. Same TRAIN-half-only stride sampling as the other
    # constants, measured by scratchpad/features_rebuild/sibling_stats.py.
    # ⚠ PROVENANCE: 40 users / 269,346 DEFINED rows, not the 300 users the others used. The
    # 300-user job was still running when the rebuild had to launch, and waiting was not worth
    # it: this constant is a fixed affine transform of one column, which the input FC absorbs,
    # and the block comment above records the existing nine being off by up to 24% on resample
    # and kept deliberately. Sampling error on the mean is ~0.7 (SE over users), i.e. well inside
    # that. Re-derive at the next rebuild if a 300-user number is wanted.
    "sibling_gap_mean": 9.3354,
    "sibling_gap_std": 4.2198,
}


def enabled():
    return os.environ.get("RWKV_ID_FEATURES", "0") == "1"


# The model input is `card_feature_width() + ID_ENCODING_DIMS`. The ID-encoding half (the cyclic
# codes for card/note/deck/preset/user) is untouched by the rebuild, so it stays a constant; the
# card-feature half is what the flag changes. Derived rather than hardcoded per [[keep-optimizations-
# arch-agnostic]] -- `card_features_dim = 92` was written in two model files and would have been a
# silent shape mismatch the moment the rebuild landed.
# The encoding block prepare_batch.add_encodings appends after the card features: the four ID
# codes (12+12+8+8 = 40, rwkv/config.py ID_ENCODE_DIMS) plus, unless RWKV_REAL_CYCLES=1, the
# seven pseudo day-offset cycles (7 periods x {review day, first-review day} x {sin, cos} = 28).
_ID_CODE_DIMS = 40
_PSEUDO_CYCLE_DIMS = 28
ID_ENCODING_DIMS = _ID_CODE_DIMS + _PSEUDO_CYCLE_DIMS  # 68: the historical constant, flag off


# ---- RWKV_REAL_CYCLES=1 (default OFF): replace the pseudo day-offset cycles with real ones ----
# Andrew 2026-09-02: "use real features for 3 days/week/month/year/decade/century, so that every
# pseudo feature is replaced with its real counterpart". The pseudo cycles (prepare_batch
# add_encodings) are sin/cos of (baseline + day_offset) mod N, where day_offset counts from the
# USER's first review and baseline is an arbitrary phase -- so they encode relative position only.
# The real counterpart uses the same math on the epoch-anchored UTC day index of `review_time`
# (integer days, like day_offset), so the phase means the same thing for every user. Each period
# keeps its first-review-day half, as the pseudo ones had. The review-time 7 d and 365 d halves
# already exist as dow/doy and are NOT duplicated, so 24 new card-feature columns replace 28
# encoding dims: input 114 -> 110. Because they live in the card-feature block they are reachable
# by RWKV_ABLATE_FEATURES by name, which the pseudo ones never were.
# Requires RWKV_ID_FEATURES=1 (needs review_time) and a rebuild (they are computed at build time).
CYCLE_PERIODS = [3, 7, 30, 100, 365.25, 3650, 36500]
_CYCLES_WITH_REAL_REVIEW_HALF = (7, 365.25)   # dow / doy already cover the review-time half


def _cycle_tag(p):
    return "365" if p == 365.25 else str(int(p))


def _build_cycle_columns():
    cols = []
    for p in CYCLE_PERIODS:
        t = _cycle_tag(p)
        if p not in _CYCLES_WITH_REAL_REVIEW_HALF:
            cols += [f"cyc{t}_sin", f"cyc{t}_cos"]
        cols += [f"cyc{t}_first_sin", f"cyc{t}_first_cos"]
    return cols


CYCLE_COLUMNS = _build_cycle_columns()
assert len(CYCLE_COLUMNS) == 24, len(CYCLE_COLUMNS)


def real_cycles_enabled():
    on = os.environ.get("RWKV_REAL_CYCLES", "0") == "1"
    if on and not enabled():
        raise RuntimeError(
            "RWKV_REAL_CYCLES=1 requires RWKV_ID_FEATURES=1: the real cycles are functions of "
            "review_time, which only the -id layout carries."
        )
    return on


def active_new_columns():
    """Every column the -id layout appends to the base card features, in vector order."""
    if not enabled():
        return []
    return list(NEW_COLUMNS) + (list(CYCLE_COLUMNS) if real_cycles_enabled() else [])


def id_encoding_dims():
    return _ID_CODE_DIMS if real_cycles_enabled() else ID_ENCODING_DIMS
BASE_CARD_FEATURES = 24


# Andrew 2026-09-02, after the cycles: "11 is also a pseudo-calendar feature, so make sure it
# also gets replaced." Row 11 is `day_of_week`, the ((day_offset mod 7) - 3)/3 sawtooth counted
# from the user's first day. Its real counterpart (dow_sin/dow_cos) is already in NEW_COLUMNS,
# so "replace" means DROP it from the input vector under the same flag. The raw column stays in
# the frame, like `state` does. Index 17 sits after COL_DUR/COL_R1 (8, 9), so those stay put.
DROPPED_UNDER_REAL_CYCLES = "day_of_week"


def dropped_columns():
    """Base card-feature columns the -id layout removes from the input vector, in order."""
    if not enabled():
        return []
    return [DROPPED_COLUMN] + ([DROPPED_UNDER_REAL_CYCLES] if real_cycles_enabled() else [])


def card_feature_width():
    if not enabled():
        return BASE_CARD_FEATURES
    return BASE_CARD_FEATURES - len(dropped_columns()) + len(active_new_columns())


def input_width():
    return id_encoding_dims() + card_feature_width()


def _log_t(x):
    """The transform `data_processing` uses for the elapsed_*/duration family."""
    return np.log(1.0 + 1e-5 + np.asarray(x, dtype=np.float64))


def _log3(x):
    """The transform `data_processing` uses for the count family (diff_new_cards etc.)."""
    return np.log(3.0 + np.asarray(x, dtype=np.float64))


def _std(name, v):
    return (v - STATISTICS_ID[f"{name}_mean"]) / STATISTICS_ID[f"{name}_std"]


def elapsed_end_to_start(df):
    """Re-derive `elapsed_seconds` as END-of-previous-review to START-of-this-review.

    ANDREW 2026-08-19: "make sure that all the stuff like elapsed_days, elapsed_seconds, etc. is
    based on review ID *after* subtracting review duration, so that everything interval-related is
    'from the end of the prior review to the beginning of next review', NOT 'from the end of the
    prior review to the end of next review'."

    WHAT WAS STORED, AND WHY IT LOOKED FIXED. `build_parquet_id.py` sets
    `review_time = revlog.id - taken_millis`, i.e. the SHOW time, and then derives
    `elapsed_seconds = review_time.diff()` -- its own docstring says "show-to-show". That
    correction is real and worth having: it makes the timestamp the moment the user actually saw
    the card, which is what every time-of-day feature needs. But the INTERVAL is still
    start-to-start, so it carries the previous review's duration inside it:

        show(k) - show(k-1)  =  duration(k-1) + [ show(k) - answer(k-1) ]

    Verified numerically on user 333: stored 67/20/215/23/21 s against end-to-start
    40.7/9.9/206.9/13.0/11.9 s, the difference equal to the prior duration to the millisecond.
    On learning steps that is 40-50% of the gap; on multi-day intervals it is noise.

    THE FIX: subtract the PREVIOUS review's duration, per card.

        elapsed_seconds[k] = ( review_time[k] - (review_time[k-1] + duration[k-1]) ) / 1000

    The -1 sentinel is preserved on first reviews, and negatives are clamped to 0 BEFORE the
    int cast -- flooring -0.4 would produce -1 and silently mint a fake "first review", which is
    exactly the sentinel-collision class fixed earlier the same day in the cumulative columns.

    ⚠ `elapsed_days` is deliberately NOT touched. It is `day_offset.diff()`, a CALENDAR-day index
    difference matching Anki's scheduling semantics; "subtract a duration" is not well defined on
    a day index, and a ~10 s duration can only ever move it by crossing midnight. Flagged rather
    than changed.

    Gated on the DATASET (`review_time` present), like `clamp_negative_gaps`: published data has
    no `review_time`, so pre-rebuild runs are untouched by construction.
    """
    if "review_time" not in df.columns or "duration" not in df.columns or not len(df):
        return df
    # ★ GROUPED BY CARD, AND THAT IS WHAT MAKES IT IDENTICAL TO PR #2 -- not a deviation from it.
    # The PR writes a PLAIN frame `.shift()`, but it runs inside the BUILDER, on the frame in
    # PROTOBUF ORDER (per-card blocks), BEFORE `sort_values("review_time")`. Its shift is therefore
    # per-card by virtue of the ordering. THIS function runs on the already-SORTED parquet, where
    # the previous row is almost always a DIFFERENT card reviewed seconds earlier.
    # ⚠ MEASURED: copying the plain shift literally shortened EVERY gap bucket by a median 100%,
    # including gaps over a day -- it would have destroyed every interval in the -id dbs, silently.
    # `smoke_end_to_start.py` caught it. Same semantics, different ordering, so the faithful
    # translation is the groupby.
    answer_prev = (df["review_time"] + df["duration"]).groupby(df["card_id"]).shift()
    gap_s = ((df["review_time"] - answer_prev) / 1000.0).clip(lower=0.0)  # clamp BEFORE the cast
    out = np.floor(gap_s.to_numpy())

    # THE SENTINEL MASK IS "no known previous review FOR THIS CARD", which is NOT the same as
    # `elapsed_seconds == -1` (that marks state == 0). A card whose FIRST row in the frame is not a
    # state-0 row -- history truncated before the export window -- gets NaN from the groupby shift
    # and is missed by the state-0 test. Measured: 1 row in 3,817,339 over 40 users, affecting 1 of
    # them. Vanishingly rare per row, CERTAIN across 5,000 users, and one NaN poisons a whole user.
    # ⚠ The PR has no equivalent because its ungrouped shift never produces NaN; that is a
    # consequence of the ordering difference above, not a semantic difference.
    no_prev = np.isnan(out) | (df["elapsed_seconds"] == -1).to_numpy()
    out[no_prev] = -1.0
    assert not np.isnan(out).any(), "elapsed_end_to_start produced NaN outside the sentinel rows"
    df = df.copy()
    df["elapsed_seconds"] = out.astype("int64")
    return df


def elapsed_end_to_start_published(df):
    """END-of-previous to START-of-this, for the PUBLISHED dataset. Opt-in: RWKV_E2S_PUBLISHED=1.

    ⚠ THE FORMULA IS NOT THE SAME AS `elapsed_end_to_start`, and the difference is the whole point
    of having two functions. The two datasets put a different timestamp under the same name:

        published : the row id is the ANSWER time, and `elapsed_seconds` is answer-to-answer,
                    so                end_to_start = elapsed_seconds - duration(k)
        -id       : `review_time` is the SHOW time, so the PREVIOUS duration comes off instead.

    Both compute `show(k) - answer(k-1)`. Applying the -id formula here would subtract the wrong
    review's duration and be silently wrong -- no shape changes, no error, just a different number.

    WHY THIS QUANTITY. Decay runs from when the user last finished being shown the answer to when
    the card is next SHOWN. `duration(k)` is time spent AFTER the retrieval already happened.
    Stronger still: `duration(k)` does not exist at prediction time and it CORRELATES with the
    outcome, so the stored interval leaks a whisper of the label. This repo's own deploy contract
    already zeroes the most recent duration for exactly that reason -- and then leaves it inside
    the interval column.

    `duration` is MILLISECONDS. The -1 first-review sentinel is preserved; a corrected gap that
    would go negative is clamped to 0 BEFORE the int cast, because flooring -0.4 gives -1 and
    would silently mint a fake first review. Measured on 40 users: 0.559% of same-day rows need
    that clamp, and the existing sentinel handling covers the rest.
    """
    if not os.environ.get("RWKV_E2S_PUBLISHED") == "1":
        return df
    if "elapsed_seconds" not in df.columns or "duration" not in df.columns or not len(df):
        return df
    assert "review_time" not in df.columns, (
        "RWKV_E2S_PUBLISHED is for the PUBLISHED dataset. This frame has `review_time`, i.e. it "
        "is the -id set, where elapsed_end_to_start already applies the correction with the "
        "OTHER formula -- running both would subtract two durations."
    )
    es = df["elapsed_seconds"].to_numpy().astype("float64")
    dur_s = df["duration"].to_numpy().astype("float64") / 1000.0
    sentinel = es == -1
    out = np.floor(np.maximum(es - dur_s, 0.0))   # clamp BEFORE the cast
    out[sentinel] = -1.0
    assert not np.isnan(out).any()
    df = df.copy()
    df["elapsed_seconds"] = out.astype("int64")
    return df


def clamp_negative_gaps(df):
    """★ THE NaN LANDMINE. One line, and without it ~3.2% of users are silently destroyed.

    `build_parquet_id.py` recomputes `elapsed_seconds` from the corrected SHOW time. When a review's
    duration overlaps the next review the recomputed gap goes NEGATIVE but not -1, and
    `scale_elapsed_seconds` is `np.where(x == -1, 0, np.log(1 + 1e-5 + x))` -- so x = -26 takes the
    log branch, `log(-25.99999)` is NaN, and the NaN flows into the features, the loss, and finally
    the eval NaN guard, which skips the whole user.

    Measured on the `-id` set: 48 bad rows in 25,063,241 (0.000192%) but **10 of 313 users** hit --
    rows are a rounding error, users are not. Projected onto the real split that is ~160 of 5,000
    train and ~80 of 2,500 eval users, and it would have surfaced as `nan_users` jumping 0 -> ~80
    weeks after a multi-day rebuild, with no obvious cause.

    ⚠⚠ TWO DEPARTURES FROM THE PLAN'S ONE-LINER, BOTH FORCED BY ACTUALLY RUNNING IT.
    `FUTURE_FEATURES.md` writes the fix as `np.where(x < -1, -1, x)` on `elapsed_seconds` *and*
    `elapsed_days`. Neither half survives contact:

    1. **Not `elapsed_days`.** `is_first_review` is defined as `elapsed_days == -1`, so clamping a
       mid-card review to the sentinel would re-label it as that card's FIRST review and poison
       `cum_new_cards`, `diff_new_cards` and the label machinery. The measured landmine is on
       `elapsed_seconds` only (integer days cannot go negative from a sub-day overlap), so
       `elapsed_days` gets an ASSERT -- if it ever happens we find out loudly instead of quietly
       gaining a first review.

    2. **Clamp to 0, not to the -1 sentinel.** Clamping to -1 moves the NaN rather than removing
       it: `elapsed_seconds_cumulative` is a per-card cumsum, so a card whose history becomes
       `[-1, -1, ...]` cumulates to **-2**, and `scale_elapsed_seconds_cumulative` takes the same
       `log(negative)` branch one column over. Measured on the index case, user 486: with the -1
       clamp the raw column is clean and `scaled_elapsed_seconds_cumulative` is NaN on row 9182
       (card 1674953822938: elapsed_seconds `[-1, -1, 43164, 22550]` -> cumulative
       `[-1, -2, 43162, 65712]`). So the plan's fix would have passed a raw-column check and still
       lost the user.
       The plan's stated reason for preferring -1 was that "clamping to 0 would claim the two
       reviews were simultaneous". That argument is weaker than it looks: the overlap is bounded by
       the review's OWN duration, i.e. tens of seconds, so the true gap really is ~0 and -1 ("no
       previous review at all") is the less accurate of the two codes. Clamping to 0 is both more
       faithful and self-limiting -- with every non-sentinel value >= 0, a card's cumsum is
       `-1 + sum(nonneg) >= -1` by construction, so the cumulative column cannot go negative and
       needs no second guard. `data_processing` asserts exactly that invariant after the cumsum.
    """
    if "elapsed_days" in df.columns:
        v = df["elapsed_days"].to_numpy()
        assert (v >= -1).all(), (
            "elapsed_days < -1: clamping it would re-label a mid-card review as a first review "
            "(is_first_review is `elapsed_days == -1`). Investigate before clamping."
        )
    if "elapsed_seconds" in df.columns:
        v = df["elapsed_seconds"].to_numpy()
        bad = v < 0
        # -1 is the legitimate "no previous review" sentinel and must survive untouched.
        bad &= v != -1
        if bad.any():
            df["elapsed_seconds"] = np.where(bad, 0, v)
    return df


def _deck_depths(df_decks_raw):
    """Depth in Anki's `A::B::C` tree via parent_id. 0 = top level. Cycle-safe."""
    if not len(df_decks_raw) or "parent_id" not in df_decks_raw.columns:
        return {}
    parent = dict(zip(df_decks_raw["deck_id"], df_decks_raw["parent_id"]))
    known = set(parent)
    depth = {}

    def resolve(d, seen):
        if d in depth:
            return depth[d]
        p = parent.get(d, 0)
        if p == 0 or p not in known or p in seen:
            depth[d] = 0
            return 0
        seen.add(d)
        depth[d] = resolve(p, seen) + 1
        return depth[d]

    for d in parent:
        resolve(d, set())
    return depth


def sibling_gap_seconds(df):
    """Seconds since the END of the most recent review of a DIFFERENT card of the same note.

    ⚠ THIS IS NOT ANDREW'S LITERAL FORMULA, AND THE DIFFERENCE IS A LEAK. His example takes
    `min(|t_now - t_sib|)` over ALL siblings, which for card A1 reviewed on day 100 includes A3's
    day-110 review -- a future event. Anki knows a sibling's PAST reviews when it schedules a card
    and not its future ones, so the leaking form is not deployable, and `size` would be identical
    so no gate would catch it. Restricted to PRECEDING siblings the min collapses:

        min over PAST siblings of |t_now - t_sib|  ==  t_now - max(t_sib)

    i.e. plain recency, no `min` needed. Andrew's own example is unchanged by the restriction (its
    nearest sibling is the past one); the two forms diverge only in the leaking case.

    METHOD -- the block trick, which makes this O(n log n) rather than a per-row scan over 372 M
    rows. Sort by (note_id, review_time). Within a note, a maximal run of consecutive rows sharing
    one card is a BLOCK; adjacent blocks have different cards by construction. So for EVERY row of
    block b, the most recent different-card review is the last row of block b-1 -- one shift over
    blocks instead of a search over siblings.

    END-to-START like every other interval here (Andrew 2026-08-19): the previous sibling review's
    ANSWER time (`review_time + duration`) is the start of the gap. Clipped at 0 for the overlap
    case, and -1 (the pipeline's sentinel) where the note has no preceding sibling review at all --
    which covers single-card notes, a note's first block, and rows whose note_id is missing.

    Returns a float64 array aligned to `df`'s current row order.
    """
    import pandas as pd

    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    rt = df["review_time"].to_numpy(dtype=np.int64).astype(np.float64)
    dur = df["duration"].to_numpy(dtype=np.int64).astype(np.float64)
    nid = pd.to_numeric(df["note_id"], errors="coerce").to_numpy(dtype=np.float64)
    cid = pd.to_numeric(df["card_id"], errors="coerce").to_numpy(dtype=np.float64)

    out = np.full(n, -1.0, dtype=np.float64)
    valid = np.isfinite(nid) & np.isfinite(cid)
    if not valid.any():
        return out

    idx = np.flatnonzero(valid)
    # STABLE sort: `df` is already sorted by review_time (asserted by the caller), so a stable
    # sort on note_id alone preserves time order inside each note. A non-stable sort would break
    # the whole block argument silently.
    order = idx[np.argsort(nid[idx], kind="stable")]
    note_s = nid[order]
    card_s = cid[order]
    rt_s = rt[order]
    end_s = rt_s + dur[order]

    m = order.size
    new_note = np.empty(m, dtype=bool)
    new_note[0] = True
    new_note[1:] = note_s[1:] != note_s[:-1]
    new_block = new_note.copy()
    new_block[1:] |= card_s[1:] != card_s[:-1]
    block = np.cumsum(new_block) - 1
    nb = int(block[-1]) + 1

    # Last row of each block wins the scatter, which is exactly the row we want.
    b_last = np.zeros(nb, dtype=np.int64)
    b_last[block] = np.arange(m, dtype=np.int64)
    b_note = np.zeros(nb, dtype=np.float64)
    b_note[block] = note_s
    b_end = end_s[b_last]

    prev_end = np.full(nb, np.nan, dtype=np.float64)
    same_note = b_note[1:] == b_note[:-1]
    prev_end[1:] = np.where(same_note, b_end[:-1], np.nan)

    row_prev_end = prev_end[block]
    gap = (rt_s - row_prev_end) / 1000.0
    out[order] = np.where(np.isnan(row_prev_end), -1.0, np.maximum(gap, 0.0))
    return out


def add_id_features(df, df_cards, df_decks_raw):
    """Add every column in NEW_COLUMNS to `df`, in place. Requires `review_time` (epoch ms).

    `df` must already be merged with cards/decks and sorted by review order, and must still carry
    the RAW `card_id` / `deck_id` / `preset_id` -- call this BEFORE the ID_PLACEHOLDER fills, which
    overwrite the very timestamps these features read.
    """
    import pandas as pd

    assert "review_time" in df.columns, "review_time not found -- is DATA_PATH the -id dataset?"
    n = len(df)
    rt = df["review_time"].to_numpy(dtype=np.int64)
    assert n == 0 or np.all(np.diff(rt) >= 0), "review_time must be non-decreasing"

    cid = pd.to_numeric(df["card_id"], errors="coerce").to_numpy(dtype=np.float64)
    did = pd.to_numeric(df["deck_id"], errors="coerce").to_numpy(dtype=np.float64)
    pid = pd.to_numeric(df["preset_id"], errors="coerce").to_numpy(dtype=np.float64)

    # ---------------- time of day ----------------
    theta = (rt % int(_MS_PER_DAY)) / _MS_PER_DAY * 2.0 * np.pi
    df["tod_sin"] = np.sin(theta)
    df["tod_cos"] = np.cos(theta)

    # Running circular mean over STRICTLY PRIOR reviews (exclusive prefix sums), which is what
    # makes it causal and also what the deploy path can hold in 2 floats of per-user state.
    s_pre = np.concatenate(([0.0], np.cumsum(np.sin(theta))[:-1])) if n else np.zeros(0)
    c_pre = np.concatenate(([0.0], np.cumsum(np.cos(theta))[:-1])) if n else np.zeros(0)
    mu = np.arctan2(s_pre, c_pre)
    dev = theta - mu
    # Row 0 has no history, so the deviation is undefined. Emit (0, 0) rather than a plausible
    # angle: every defined row lies on the unit circle, so the origin is unambiguously "unknown"
    # and the net can separate it. (cos = 1 would collide with "exactly at the usual hour".)
    has_hist = np.zeros(n, dtype=bool)
    if n:
        has_hist[1:] = True
    df["tod_dev_sin"] = np.where(has_hist, np.sin(dev), 0.0)
    df["tod_dev_cos"] = np.where(has_hist, np.cos(dev), 0.0)

    # ---------------- TRUE-phase calendar ----------------
    # UTC-based; see the module docstring on timezones.
    ts = pd.to_datetime(rt, unit="ms", utc=True)
    dow = ts.dayofweek.to_numpy(dtype=np.float64)  # Monday = 0
    doy = ts.dayofyear.to_numpy(dtype=np.float64)
    df["dow_sin"] = np.sin(dow * 2.0 * np.pi / 7.0)
    df["dow_cos"] = np.cos(dow * 2.0 * np.pi / 7.0)
    df["doy_sin"] = np.sin(doy * 2.0 * np.pi / 365.25)
    df["doy_cos"] = np.cos(doy * 2.0 * np.pi / 365.25)
    df["is_weekend"] = (dow >= 5).astype(np.float64)

    # ---------------- sub-day recency and ages ----------------
    # Seconds since ANY review. The existing feature #10 is integer-DAY (built from day_offset), so
    # sub-day session structure is invisible today. Row 0 gets the -1 sentinel convention: no prior
    # review, therefore log-value 0 after standardization of the sentinel branch.
    gap = np.empty(n, dtype=np.float64)
    if n:
        gap[0] = -1.0
        # END-to-START, like `elapsed_seconds` (Andrew 2026-08-19). `np.diff(rt)` is
        # show-to-show and therefore carries the PREVIOUS review's duration inside the gap;
        # subtract it so this measures "from the end of the last review the user did, on any
        # card, to the moment this card appeared". Same defect and same fix as
        # `elapsed_end_to_start`, one level up: that one is per-card, this one is per-user.
        _du = df["duration"].to_numpy(dtype=np.int64)
        gap[1:] = np.maximum((rt[1:] - (rt[:-1] + _du[:-1])) / 1000.0, 0.0)
    df["scaled_t_since_any_review"] = np.where(
        gap == -1.0, 0.0, _std("t_since_any_review", _log_t(np.maximum(gap, 0.0)))
    )

    tenure = (rt - rt[0]) / 1000.0 if n else np.zeros(0)
    df["scaled_user_tenure"] = _std("user_tenure", _log_t(np.maximum(tenure, 0.0)))

    # Creation -> first review, a per-CARD property broadcast to the card's rows. It completes card
    # age: the existing features count from the FIRST REVIEW, so the creation->first-review span is
    # invisible. 0.2% of cards have a first review PREDATING their own creation stamp (a re-created
    # card); FUTURE_FEATURES' design correction is to clamp at 0 rather than spend a dim on the sign.
    first_rt = df.groupby("card_id")["review_time"].transform("first").to_numpy(dtype=np.float64)
    card_ts = cid >= TIMESTAMP_MIN_MS
    c2f = np.where(card_ts, (first_rt - cid) / 1000.0, np.nan)
    df["scaled_creation_to_first_review"] = np.where(
        card_ts,
        _std("creation_to_first_review", _log_t(np.maximum(np.nan_to_num(c2f), 0.0))),
        0.0,
    )

    # ---------------- deck-derived ----------------
    # ⚠ Coverage per REVIEW ROW is ~70% for deck_id, not the 99.5% measured per DECK ROW -- many
    # rows have a NaN merge or the default-deck sentinel. `deck_id_is_nan` already flags the NaN
    # part but NOT the default part, which is why `is_default_deck` exists: without it the net
    # cannot tell "no deck age" from "deck age 0".
    deck_ts = did >= TIMESTAMP_MIN_MS
    deck_age = np.where(deck_ts, np.maximum((rt - np.nan_to_num(did)) / 1000.0, 0.0), 0.0)
    df["scaled_deck_age_at_review"] = np.where(
        deck_ts, _std("deck_age_at_review", _log_t(deck_age)), 0.0
    )
    # Replaces the single `card_id - deck_id` column, which measured 57.2% negative with std 15.95
    # at n=300 -- a bimodal column whose SIGN carried most of the variance. Cards move between decks
    # constantly, so "card older than its deck" is the normal case. Deck age (always >= 0) plus this
    # flag is the same information, well-conditioned.
    df["card_predates_deck"] = np.where(
        card_ts & deck_ts, (cid < np.nan_to_num(did)).astype(np.float64), 0.0
    )
    df["is_default_deck"] = (np.nan_to_num(did, nan=-1.0) == _DEFAULT_ID).astype(np.float64)

    depths = _deck_depths(df_decks_raw)
    dep = df["deck_id"].map(lambda x: depths.get(x, 0)).to_numpy(dtype=np.float64)
    df["scaled_deck_depth"] = _std("deck_depth", dep)

    # ---------------- creation batch ----------------
    # How many of the user's cards were created near this one: an import drops hundreds of cards in
    # one second, a hand-made card is alone. ★ CLIPPED AT review_time -- see the module docstring;
    # the reference derivation counted the user's whole collection and would leak how many cards
    # they went on to create later that day.
    all_cards = np.sort(pd.to_numeric(df_cards.get("card_id", pd.Series(dtype="int64")),
                                      errors="coerce").to_numpy(dtype=np.float64))
    all_cards = all_cards[np.isfinite(all_cards) & (all_cards >= TIMESTAMP_MIN_MS)]
    for label, win_ms in (("1min", 60_000.0), ("1h", 3_600_000.0), ("1d", 86_400_000.0)):
        col = f"scaled_creation_batch_{label}"
        if all_cards.size and card_ts.any():
            lo = np.searchsorted(all_cards, cid - win_ms, side="left")
            hi = np.searchsorted(all_cards, np.minimum(cid + win_ms, rt.astype(np.float64)),
                                 side="right")
            cnt = np.maximum(hi - lo, 0).astype(np.float64)
            df[col] = np.where(card_ts, _std(f"creation_batch_{label}", _log3(cnt)), 0.0)
        else:
            df[col] = 0.0
    if all_cards.size and card_ts.any():
        lo1h = np.searchsorted(all_cards, cid - 3_600_000.0, side="left")
        pos = np.maximum(np.searchsorted(all_cards, cid, side="left") - lo1h, 0).astype(np.float64)
        df["scaled_creation_batch_pos_1h"] = np.where(
            card_ts, _std("creation_batch_pos_1h", _log3(pos)), 0.0
        )
    else:
        df["scaled_creation_batch_pos_1h"] = 0.0

    # ---------------- preset ----------------
    # Only the flag: preset AGE is defined for ~1 row in 14 and folding the rest to 0 gives a column
    # that is one constant 81% of the time. 67.4% of users never leave the default preset, so the
    # flag ("this user bothered to configure this deck") carries most of what the age could.
    df["is_default_preset"] = (np.nan_to_num(pid, nan=-1.0) == _DEFAULT_ID).astype(np.float64)

    # ---------------- the two omissions (2026-08-20) ----------------
    # Sibling recency. The note stream pools every review of a note, so it has SEEN these reviews;
    # whether its recurrence encodes the GAP to the nearest one is a separate claim and is what
    # scratchpad/features_rebuild/sibling_redundancy_screen.py measures. Kept in the rebuild either
    # way, because a column that is IN the DB can be ablated for free by zeroing it, while a column
    # that is OUT costs another full rebuild to add.
    sib = sibling_gap_seconds(df)
    df["scaled_sibling_gap"] = np.where(
        sib == -1.0, 0.0, _std("sibling_gap", _log_t(np.maximum(sib, 0.0)))
    )

    # Was the card created before this user ever reviewed anything? `rt[0]` is the user's first
    # review in their own frame, so for every row after the first this is strictly historical, and
    # for the first row it compares against that row's own timestamp -- knowable at deploy time
    # either way. Separates an imported/pre-existing collection from cards made mid-study.
    df["card_predates_first_review"] = np.where(
        card_ts, (cid < float(rt[0])).astype(np.float64), 0.0
    )

    # ---------------- REAL cycles (RWKV_REAL_CYCLES=1) ----------------
    # Same math as the pseudo cycles in prepare_batch.add_encodings, on the epoch-anchored UTC
    # day index instead of the user-relative day_offset, and with no random baseline: the phase
    # is meaningful across users, which is the whole point. Integer days, like day_offset, so the
    # 3-day cycle is stepped rather than continuous (time-of-day is its own column). `first_rt`
    # is the first row of the card in frame order -- the same anchor creation_to_first_review
    # uses, so the two features cannot disagree about when a card started.
    if real_cycles_enabled():
        day = np.floor(rt / _MS_PER_DAY)
        day_first = np.floor(first_rt / _MS_PER_DAY)
        for p in CYCLE_PERIODS:
            f = 2.0 * np.pi / p
            t = _cycle_tag(p)
            if p not in _CYCLES_WITH_REAL_REVIEW_HALF:
                df[f"cyc{t}_sin"] = np.sin(f * np.mod(day, p))
                df[f"cyc{t}_cos"] = np.cos(f * np.mod(day, p))
            df[f"cyc{t}_first_sin"] = np.sin(f * np.mod(day_first, p))
            df[f"cyc{t}_first_cos"] = np.cos(f * np.mod(day_first, p))

    for c in active_new_columns():
        assert c in df.columns, f"id_features did not emit {c}"
        v = df[c].to_numpy(dtype=np.float64)
        assert np.isfinite(v).all(), f"id_features emitted non-finite values in {c}"
    return df
