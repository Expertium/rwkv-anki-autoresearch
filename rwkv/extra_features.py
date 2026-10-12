"""RWKV_EXTRA_FEATURES (default OFF, structurally inert): input columns that need NO LMDB rebuild.

Andrew 2026-09-20: *"do a run with one new input feature: total number of cards reviewed at least once."*

The LMDB stores `card_features`, the labels, the ids, `review_ths` and the day offsets -- nothing else. A new
per-row column would normally mean a ~7 h rebuild AND a new `size` baseline, because `label_is_equalize` is baked
in at build time. This module is the exception route: a column whose value is a pure function of
`(user_id, review_th)` can be reconstructed EXACTLY as a stored row is decoded, from a small side table, and the
row already carries its `review_th`. No rebuild, no new lineage, so the arm is gated against the current champion
directly.

**`cum_new_cards`** = the number of distinct cards the user has reviewed at least once, counting the current card
when this review is its first -- `data_processing.py:364`'s `df["is_first_review"].cumsum()`, which the dataframe
stage computes, uses for `diff_new_cards`, and then throws away. It is reconstructed as

    cum_new_cards(row) = searchsorted(first_ths[user], review_th(row), side="right")

where `first_ths[user]` holds the `review_th` of that user's first reviews (`scratchpad/newcards/`).

⚠ **WHY NOT A RUNNING TOTAL IN THE LOADER** -- this is section 9's failure shape and the reason for the side
table. Training splits a user into a median of 4 chunks and eval uses 1, so a counter started at the chunk's first
row is a DIFFERENT QUANTITY in training than at eval and deploy. `review_th` is absolute, so this is not.

⚠ **QUERY AND PROBE ROWS.** A query row repeats its real row's `review_th`, and a probe row is a copy of the row
it was inserted for, so both inherit the right value with no special case. That is correct here -- the count
describes the collection, not the outcome -- and it is checked rather than assumed
(`scratchpad/newcards/smoke_extra_features.py`).

**Deploy** keeps one integer per collection and adds 1 on a card's first review (`run_as_rnn.add_same`,
`DEPLOY_FUNCTIONS.md` section 4). Anki knows it is showing a new card, so this is legal at show time.

Usage: `set RWKV_EXTRA_FEATURES=cum_new_cards`. Unset or empty => this module adds no column and every path is
byte-identical to today; an unknown name RAISES rather than being ignored.
"""
import os

import numpy as np

# name -> (the input column it appends, the side table it needs). ⚠ THE DECLARATION ORDER IS THE COLUMN
# ORDER, whatever order the flag lists them in: the input layout must be a function of the SET of features,
# not of how a runner happened to spell it, or two runners with the same features would train incompatible
# models and nothing would say so.
SPECS = {
    "cum_new_cards": ("scaled_cum_new_cards", "first_ths"),
    "review_th": ("scaled_review_th", None),            # F1: the user's total reviews so far
    "new_card_share": ("new_card_share", "first_ths"),  # F3: the share of activity spent on NEW cards
    # --- A: session structure (Andrew kept it; the 30-min cut is measured, not guessed --
    #     `scratchpad/sessions/gap_antimode.py` found no sharp antimode, so the rolling counts ride
    #     beside the cut and make the arm robust to the threshold the histogram refused to pick).
    "sess_pos": ("scaled_sess_pos", "rowtables"),       # reviews already done in this sitting
    "sess_secs": ("scaled_sess_secs", "rowtables"),     # seconds since this sitting started
    "roll10": ("scaled_roll10", "rowtables"),           # reviews in the previous 10 minutes
    "roll60": ("scaled_roll60", "rowtables"),           # reviews in the previous 60 minutes
    # --- B1: phase coherence. R alone is blind to a two-peaked user, so the second moment ships
    #     beside it; `scratchpad/phase_coherence/phase_coherence.png` is the picture of why.
    "phase_r": ("scaled_phase_r", "rowtables"),         # bias-corrected resultant length
    "phase_r2": ("scaled_phase_r2", "rowtables"),       # the second circular moment
    # --- C: study streak (consecutive UTC days with at least one review)
    "streak": ("scaled_streak", "rowtables"),
    # --- E: the running mean of GRADES, which replaces the pass rate (Andrew: "that looses signal").
    #     ⚠ THE ONE CANDIDATE THAT CAN LEAK -- it is a running mean of the label family, so an
    #     off-by-one hands the rating head its own answer. `scratchpad/rowtables/leak_check.py`.
    "grade_mean": ("scaled_grade_mean", "rowtables"),
    # --- D: SIBLING RATING (Andrew's kept item D; added 2026-09-23 after the 200-user screen in
    #     scratchpad/siblingrating/FINDINGS.md). The grade of the latest review of ANOTHER card of the
    #     same note EARLIER IN THE SAME SITTING, plus a "no such sibling" flag. Read from the
    #     precomputed int8 table `sib_flat.npy` (scratchpad/rowtables/build_sibtable.py), because the
    #     per-row card + note ids it needs would be ~8.6 GB as raw columns.
    #     ⚠ APPENDED LAST ON PURPOSE: declaration order is column order, so every layout that does not
    #     name these two is byte-identical to before.
    "sib_grade": ("scaled_sib_grade", "rowtables"),
    "sib_none": ("sib_none", "rowtables"),
    # --- The SIBLING-GAP FLAG (Andrew 2026-09-25: "fix the sibling gap"). The stored column
    #     `scaled_sibling_gap` is 0.0 when no earlier review of another card of the note exists (~97% of
    #     rows, feature audit W1), and 0.0 is also a legal scaled value, so the model cannot tell "no
    #     sibling" from "a sibling ~3 h ago". This flag is 1 exactly when the STORED value is 0.0 -- the
    #     same sentinel `clock_fix._SUB` already relies on -- so it needs no table and no rebuild, and the
    #     deploy RNN applies the same rule to the value it computes (`run_as_rnn`). Read from the row
    #     BEFORE the clock fix: existence does not depend on the clock. ⚠ APPENDED LAST, like sib_*.
    "sibgap_none": ("sibgap_none", "stored"),
}

# log1p(cum_new_cards) over the TRAIN half (users 1-5000, 357,460,180 rows), measured 2026-09-20 by
# scratchpad/newcards/measure_stats.py. Train-half only, like every other constant in STATISTICS_ID.
CUM_NEW_CARDS_LOG1P_MEAN = 8.882881
CUM_NEW_CARDS_LOG1P_STD = 1.295293
# F1 / F3, same half and the same row weighting (scratchpad/newcards/measure_stats_f1f3.py, 357,460,182
# rows). `review_th` is the stored 1-based row index, so F1 needs no side table at all and F3 reuses
# first_ths. ⚠ The row count differs by 2 from the cum_new_cards measurement, which counted through pandas
# rather than the parquet footer; at 3.6e8 rows it cannot move a mean, but the two are not the same scan.
REVIEW_TH_LOG1P_MEAN = 10.899043
REVIEW_TH_LOG1P_STD = 1.518719
NEW_CARD_SHARE_MEAN = 0.159953
NEW_CARD_SHARE_STD = 0.097551

TABLE_DIR = os.environ.get("RWKV_EXTRA_TABLE_DIR", "scratchpad/newcards")
ROW_TABLE_DIR = os.environ.get("RWKV_EXTRA_ROW_TABLE_DIR", "scratchpad/rowtables")

# The session cut. 30 minutes, and the reason is that it is NOT a discovered boundary: the gap
# histogram has no antimode (`scratchpad/sessions/gap_antimode.py`), so every threshold from 5 min
# to 6 h is defensible. 1800 is already this pipeline's "same sitting" everywhere else
# (RWKV_CLOCK_AT_PREV_ANSWER=1800), so reusing it keeps one definition rather than inventing a
# second. ⚠ Avoid anything near 60 s: that region is a scheduler artefact, not user behaviour.
SESSION_GAP = 1800
ROLL_WINDOWS = {"roll10": 600, "roll60": 3600}

# Train-half (users 1-5000) row-weighted constants, measured 2026-09-21 by
# `scratchpad/rowtables/measure_rowfeat.py` over 357,460,182 rows. ⚠ `values()` refuses to build a
# column whose constants are None, because a silently unscaled input is the kind of error no gate
# can see. The three averaging features read 357,455,182 = exactly 5,000 fewer rows: one first
# review per user, which has no history to average and is emitted as the scaled 0.0.
ROWFEAT_STATS = {
    "sess_pos": (4.163197, 1.424215),    # log1p(reviews already done this sitting)
    "sess_secs": (6.894923, 1.687621),   # log1p(seconds since the sitting started)
    "roll10": (3.283222, 1.111573),      # log1p(count in the previous 600 s)
    "roll60": (4.097617, 1.283437),      # log1p(count in the previous 3600 s)
    "phase_r": (0.440765, 0.164353),     # the bias-corrected resultant length, already in [0, 1]
    "phase_r2": (0.268032, 0.152146),    # the second circular moment
    "streak": (2.607561, 1.485916),      # log1p(consecutive UTC days)
    "grade_mean": (2.608538, 0.307149),  # the running mean rating in [1, 4]
    # sib_grade: over the train-half rows that HAVE a same-sitting sibling (the others are NaN and
    # scale to 0.0). sib_none is a 0/1 flag fed as-is: (0, 1) is the identity.
    "sib_grade": (2.718832, 0.787373),   # mean/std of the grade over the 15,304,423 train-half rows (4.28%) that have one; measure_sib.py
    "sib_none": (0.0, 1.0),
    "sibgap_none": (0.0, 1.0),
}
LOG1P_FEATURES = ("sess_pos", "sess_secs", "roll10", "roll60", "streak")
# Rows with NO history at all (the user's first review) have no defined value for the accumulator
# features. They are emitted as the SCALED value 0.0 = "the training mean", which is the same
# convention the pipeline already uses for a query row's duration, and deploy repeats it.
NO_HISTORY_ZERO = ("phase_r", "phase_r2", "grade_mean")

_tables = {}
# user_id -> {name: np.ndarray} for the user's WHOLE history. Bounded by ROWS, not by users,
# because the sizes span 30,832 (median) to 2,557,899 (largest train user) and a user count would
# mean either a useless cache or a 500 MB one. 2 M rows x 8 columns x 8 bytes = ~128 MB per fetch
# worker, and a single user larger than the budget is still admitted (then evicts everything else).
_rowcache = {}
_ROWCACHE_ROWS = 2_000_000


def active():
    """The requested feature names, in the order the columns are appended."""
    raw = os.environ.get("RWKV_EXTRA_FEATURES", "").strip()
    if not raw:
        return ()
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise RuntimeError(
            "RWKV_EXTRA_FEATURES: unknown feature(s) %s; known: %s"
            % (", ".join(unknown), ", ".join(sorted(SPECS)))
        )
    if len(set(names)) != len(names):
        raise RuntimeError("RWKV_EXTRA_FEATURES: a feature is listed twice: %s" % (names,))
    return tuple(n for n in SPECS if n in names)     # declaration order, never the flag's order


def enabled():
    return bool(active())


def row_features():
    """The active features whose values come from the per-ROW table, i.e. the ones that need the
    absolute show time. `data_processing` keys the `review_time` carry on this."""
    return tuple(n for n in active() if SPECS[n][1] == "rowtables")


def columns():
    """The input-column names this flag appends, in order."""
    return tuple(SPECS[n][0] for n in active())


def width():
    return len(active())


def _first_ths():
    """The flat side table, memory-mapped: one int32 array shared by every fetch worker.

    Kept as a flat .npy plus an index rather than the .npz the builder writes, because an .npz is
    compressed -- it cannot be mapped, so each worker would decompress and hold its own ~250 MB copy, which
    the 10 GB-free rule does not have room for. Built by `scratchpad/newcards/flatten_first_ths.py`.
    """
    t = _tables.get("first_ths")
    if t is None:
        flat = np.load(os.path.join(TABLE_DIR, "first_ths_flat.npy"), mmap_mode="r")
        idx = np.load(os.path.join(TABLE_DIR, "first_ths_index.npz"))
        t = _tables["first_ths"] = (flat, idx["users"], idx["start"], idx["count"])
    return t


def cum_new_cards(user_id, review_ths):
    """The raw count per row: how many distinct cards this user had reviewed, this one included."""
    flat, users, start, count = _first_ths()
    i = int(np.searchsorted(users, user_id))
    if i >= len(users) or users[i] != user_id:
        raise RuntimeError("no first-review table for user %s (%s)" % (user_id, TABLE_DIR))
    ths = np.asarray(flat[start[i]:start[i] + count[i]], dtype=np.int64)
    return np.searchsorted(ths, np.asarray(review_ths, dtype=np.int64), side="right")


def scale_cum_new_cards(x):
    return (np.log1p(np.asarray(x, dtype=np.float64)) - CUM_NEW_CARDS_LOG1P_MEAN) / CUM_NEW_CARDS_LOG1P_STD


def scale_review_th(x):
    return (np.log1p(np.asarray(x, dtype=np.float64)) - REVIEW_TH_LOG1P_MEAN) / REVIEW_TH_LOG1P_STD


def _rowtables():
    """The per-row side tables, memory-mapped and shared by every fetch worker."""
    t = _tables.get("rowtables")
    if t is None:
        secs = np.load(os.path.join(ROW_TABLE_DIR, "secs_flat.npy"), mmap_mode="r")
        rating = np.load(os.path.join(ROW_TABLE_DIR, "rating_flat.npy"), mmap_mode="r")
        idx = np.load(os.path.join(ROW_TABLE_DIR, "rowtables_index.npz"))
        t = _tables["rowtables"] = (secs, rating, idx["users"], idx["start"], idx["count"])
    return t


def _user_rows(user_id, upto):
    """This user's epoch seconds and ratings for rows 1..upto (1-based `review_th`)."""
    secs, rating, users, start, count = _rowtables()
    i = int(np.searchsorted(users, user_id))
    if i >= len(users) or users[i] != user_id:
        raise RuntimeError("no row table for user %s (%s)" % (user_id, ROW_TABLE_DIR))
    n = int(count[i])
    if upto > n:
        raise RuntimeError("user %s: review_th %d past the table's %d rows -- the table and the LMDB "
                           "disagree about this user" % (user_id, upto, n))
    s0 = int(start[i])
    return (np.asarray(secs[s0:s0 + upto], dtype=np.int64),
            np.asarray(rating[s0:s0 + upto], dtype=np.int64))


def _user_sib(user_id, upto):
    """This user's precomputed sibling grades for rows 1..upto, from `sib_flat.npy` (same index)."""
    t = _tables.get("sib")
    if t is None:
        t = _tables["sib"] = np.load(os.path.join(ROW_TABLE_DIR, "sib_flat.npy"), mmap_mode="r")
    _, _, users, start, count = _rowtables()
    if t.shape[0] != int(count.sum()):
        raise RuntimeError("sib_flat.npy has %d rows, the row-table index %d -- rebuild it with "
                           "scratchpad/rowtables/build_sibtable.py" % (t.shape[0], int(count.sum())))
    i = int(np.searchsorted(users, user_id))
    s0 = int(start[i])
    return np.asarray(t[s0:s0 + upto])


def _build_rowfeat(user_id, upto, names):
    """Every requested per-row column for rows 1..upto, computed from row 0 of the user.

    ⚠ FROM ROW 0, ALWAYS. This is the chunking hazard of section 9 in one line: training splits a user
    into a median of 4 chunks and eval uses 1, so a counter seeded at a chunk's first row is a
    DIFFERENT QUANTITY in training than at eval and deploy. Starting at row 0 costs one O(n) pass per
    chunk and makes the three paths compute the same thing.
    """
    secs, rating = _user_rows(user_id, upto)
    n = secs.size
    out = {}
    pos = np.arange(n, dtype=np.int64)          # how many reviews came strictly before each row

    if {"sess_pos", "sess_secs"} & set(names):
        gap = np.empty(n, dtype=np.int64)
        gap[0] = SESSION_GAP + 1                # the first review always opens a sitting
        gap[1:] = secs[1:] - secs[:-1]
        opens = gap > SESSION_GAP
        # the index and the time at which the current sitting opened
        open_idx = np.maximum.accumulate(np.where(opens, pos, -1))
        out["sess_pos"] = (pos - open_idx).astype(np.float64)
        out["sess_secs"] = (secs - secs[open_idx]).astype(np.float64)

    for nm, w in ROLL_WINDOWS.items():
        if nm in names:
            # reviews STRICTLY BEFORE this one within the window; secs is non-decreasing, so the
            # first index inside the window is a searchsorted and the count is a subtraction.
            out[nm] = (pos - np.searchsorted(secs, secs - w, side="left")).astype(np.float64)

    if {"phase_r", "phase_r2"} & set(names):
        th = 2 * np.pi * (secs % 86400) / 86400.0
        # exclusive prefix sums: the value at row i uses rows 0..i-1 only
        def _excl(v):
            c = np.cumsum(v)
            c[1:] = c[:-1]
            c[0] = 0.0
            return c
        C, S = _excl(np.cos(th)), _excl(np.sin(th))
        m = np.maximum(pos, 1)
        if "phase_r" in names:
            R = np.hypot(C, S) / m
            # Rayleigh's bias correction: a genuinely uniform user reads R ~ 1/sqrt(n) at small n, so
            # without this the feature is mostly a review-count proxy at the low end.
            unb = (m * R * R - 1.0) / np.maximum(m - 1.0, 1.0)
            out["phase_r"] = np.sqrt(np.maximum(unb, 0.0))
        if "phase_r2" in names:
            C2, S2 = _excl(np.cos(2 * th)), _excl(np.sin(2 * th))
            out["phase_r2"] = np.hypot(C2, S2) / m

    if "streak" in names:
        # Consecutive UTC days with at least one review, as it stands BEFORE this review.
        # Vectorized because a per-row Python loop is 357 M iterations on the train half and would
        # also run inside every fetch worker: collapse to distinct days, run-length them with the
        # reset-cumsum trick, then map each row back to the day of the review before it.
        day = secs // 86400
        opens_day = np.empty(n, dtype=bool)
        opens_day[0] = True
        opens_day[1:] = day[1:] != day[:-1]
        u = day[opens_day]                      # the distinct days, in order
        k = u.size
        j = np.arange(k, dtype=np.int64)
        brk = np.empty(k, dtype=bool)
        brk[0] = True
        brk[1:] = (u[1:] - u[:-1]) != 1         # a gap of 2+ days breaks the streak
        run = j - np.maximum.accumulate(np.where(brk, j, 0)) + 1
        day_idx = np.cumsum(opens_day) - 1      # which distinct day each row belongs to
        st = np.zeros(n, dtype=np.float64)
        if n > 1:
            st[1:] = run[day_idx[:-1]]          # the streak through the PREVIOUS review's day
        out["streak"] = st

    if "grade_mean" in names:
        cs = np.cumsum(rating.astype(np.float64))
        cs[1:] = cs[:-1]
        cs[0] = 0.0
        out["grade_mean"] = cs / np.maximum(pos, 1)

    if {"sib_grade", "sib_none"} & set(names):
        sib = _user_sib(user_id, upto).astype(np.float64)   # 0 = no same-sitting sibling, else 1..4
        none = sib == 0
        if "sib_grade" in names:
            g = sib.copy()
            g[none] = np.nan                    # scaled to 0.0 = the training mean of the grade
            out["sib_grade"] = g
        if "sib_none" in names:
            out["sib_none"] = none.astype(np.float64)

    for nm in NO_HISTORY_ZERO:                  # the first review has no history to average
        if nm in out:
            out[nm][0] = np.nan                 # marked here, turned into a scaled 0.0 below
    return out


def _user_length(user_id):
    _, _, users, _, count = _rowtables()
    i = int(np.searchsorted(users, user_id))
    if i >= len(users) or users[i] != user_id:
        raise RuntimeError("no row table for user %s (%s)" % (user_id, ROW_TABLE_DIR))
    return int(count[i])


def _rowfeat(user_id, upto, names):
    """The per-row columns for this user, built ONCE for the WHOLE history and cached.

    ⚠ Built for the whole user, NOT up to the chunk's last row, and the difference is 8x: a user is
    split into many chunks, so building to each chunk's end repeats the prefix over and over. Measured
    over the train half (`scratchpad/rowtables/bench_rowtables.py`): 760 s per epoch that way against
    97 s this way. Both are hidden behind the two prefetching fetch workers at a ~1.1 s step, so this
    is about not wasting a core rather than about the GPU waiting.
    """
    ent = _rowcache.get(user_id)
    if ent is not None and set(names) <= set(ent["cols"]):
        _rowcache[user_id] = _rowcache.pop(user_id)         # most recently used goes last
        return ent["cols"]
    n = _user_length(user_id)
    if upto > n:
        raise RuntimeError("user %s: review_th %d past the table's %d rows" % (user_id, upto, n))
    cols = _build_rowfeat(user_id, n, names)
    held = sum(e["rows"] for e in _rowcache.values())
    while _rowcache and held + n > _ROWCACHE_ROWS:
        held -= _rowcache.pop(next(iter(_rowcache)))["rows"]  # evict least recently used
    _rowcache[user_id] = {"rows": n, "cols": cols}
    return cols


def scale_rowfeat(name, raw):
    """z-score a per-row column; a NaN (no history) becomes 0.0 = the training mean."""
    stats = ROWFEAT_STATS.get(name)
    if stats is None:
        raise RuntimeError(
            "extra_features: %r has no measured train-half constants yet -- run "
            "scratchpad/rowtables/measure_rowfeat.py and fill ROWFEAT_STATS. An unscaled input is "
            "not something a gate can see." % name)
    m, s = stats
    x = np.asarray(raw, dtype=np.float64)
    if name in LOG1P_FEATURES:
        x = np.log1p(x)
    z = (x - m) / s
    return np.where(np.isnan(z), 0.0, z)


def scale_new_card_share(cum, th):
    """cum_new_cards / review_th. review_th >= 1 on every stored row, so the divide is safe."""
    share = np.asarray(cum, dtype=np.float64) / np.asarray(th, dtype=np.float64)
    return (share - NEW_CARD_SHARE_MEAN) / NEW_CARD_SHARE_STD


def values(user_id, review_ths, sibgap=None):
    """The appended columns for one chunk, shape (rows, width()), in `columns()` order.
    `sibgap` = the chunk's stored `scaled_sibling_gap` column, required when `sibgap_none` is active."""
    names = active()
    th = np.asarray(review_ths, dtype=np.int64)
    cum = cum_new_cards(user_id, th) if ("cum_new_cards" in names or "new_card_share" in names) else None
    rown = tuple(n for n in names if SPECS[n][1] == "rowtables")
    cols = _rowfeat(user_id, int(th.max()), rown) if rown else {}
    out = []
    for name in names:
        if name == "cum_new_cards":
            out.append(scale_cum_new_cards(cum))
        elif name == "review_th":
            out.append(scale_review_th(th))
        elif name == "new_card_share":
            out.append(scale_new_card_share(cum, th))
        elif name == "sibgap_none":
            if sibgap is None:
                raise RuntimeError("sibgap_none needs the stored scaled_sibling_gap column")
            out.append((np.asarray(sibgap, dtype=np.float64) == 0.0).astype(np.float64))
        elif name in cols:
            # a query or probe row repeats its real row's review_th, so it inherits the value, which
            # is what we want: the accumulator describes the history, not the outcome.
            out.append(scale_rowfeat(name, cols[name][th - 1]))
        else:  # unreachable: active() already rejected unknown names
            raise RuntimeError("no builder for %r" % name)
    return np.stack(out, axis=-1)
