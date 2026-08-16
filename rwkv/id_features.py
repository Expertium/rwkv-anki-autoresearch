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
}


def enabled():
    return os.environ.get("RWKV_ID_FEATURES", "0") == "1"


# The model input is `card_feature_width() + ID_ENCODING_DIMS`. The ID-encoding half (the cyclic
# codes for card/note/deck/preset/user) is untouched by the rebuild, so it stays a constant; the
# card-feature half is what the flag changes. Derived rather than hardcoded per [[keep-optimizations-
# arch-agnostic]] -- `card_features_dim = 92` was written in two model files and would have been a
# silent shape mismatch the moment the rebuild landed.
ID_ENCODING_DIMS = 68
BASE_CARD_FEATURES = 24


def card_feature_width():
    return BASE_CARD_FEATURES - 1 + len(NEW_COLUMNS) if enabled() else BASE_CARD_FEATURES


def input_width():
    return ID_ENCODING_DIMS + card_feature_width()


def _log_t(x):
    """The transform `data_processing` uses for the elapsed_*/duration family."""
    return np.log(1.0 + 1e-5 + np.asarray(x, dtype=np.float64))


def _log3(x):
    """The transform `data_processing` uses for the count family (diff_new_cards etc.)."""
    return np.log(3.0 + np.asarray(x, dtype=np.float64))


def _std(name, v):
    return (v - STATISTICS_ID[f"{name}_mean"]) / STATISTICS_ID[f"{name}_std"]


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
        gap[1:] = np.maximum(np.diff(rt) / 1000.0, 0.0)
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

    for c in NEW_COLUMNS:
        assert c in df.columns, f"id_features did not emit {c}"
        v = df[c].to_numpy(dtype=np.float64)
        assert np.isfinite(v).all(), f"id_features emitted non-finite values in {c}"
    return df
