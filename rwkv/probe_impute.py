"""RWKV_PROBE_IMPUTE_K=<k> -- R1, the imputed probe duration (PROPOSALS.md 2026-09-14, Andrew).

DEFAULT OFF (unset, empty or "0"): the callers never enter this module, so every existing run is
byte-identical.

WHY. The rectifier's +0.0019 ahead (LEDGER.md) is two costs, and the larger one, +0.0014, is that the
current review's DURATION is unknown when the four button curves are drawn: the PAVA probe rows carry
the pipeline's "no press yet" encoding, scaled_duration = 0.0 (RWKV_PROBE_DUR), and so does the imm
QUERY row (data_processing.add_queries zeroes it). 0.0 is scale_duration(~7.3 s) for every card of
every user. This module replaces that constant with a per-review estimate from the PAST:

    imputed = w * median(this card's past durations) + (1 - w) * median(the user's past durations)
    w       = n_card / (n_card + k)            (k = RWKV_PROBE_IMPUTE_K, ~3)

A card with no history takes the user median (w = 0); a user with no history keeps 0.0, as today.
Medians are of the SCALED duration (scale_duration is monotone, so this is the scaled median of the
raw durations up to bf16 storage). The same value goes to the query row of review k and to the four
probe rows of review k -- duration is spent BEFORE the press, so it cannot depend on the button.

THE LEAK RULE (Andrew 2026-09-14: "make sure the weighted mean of two medians doesn't leak information
from the future"): both medians are RUNNING medians over the REAL rows STRICTLY EARLIER in the frame
than the review being predicted. The review's own duration is exactly the quantity the probe must not
see, so it is excluded, and so is everything after it. No whole-history statistic anywhere: the value
on row r is a function of rows 0..r-1 only, so deleting rows after r cannot change it (the prefix-
invariance smoke, scratchpad/r1/smoke_probe_impute.py, checks exactly that). Skip rows (query rows,
probe rows) never enter a median: they are not reviews.

CHUNKS. Training reads a user in chunks (MAX_TRAIN_GLOBAL_LEN); the running medians restart at each
chunk boundary because the rows before it are not in the batch -- the same cold start the RNN state
itself has at a chunk boundary (CHUNKING_CORRECTION.md). Eval is one chunk for most users, deploy
has the whole history. Training therefore sees slightly MORE "no history yet" rows than deploy will.

THREE PATHS, ONE QUANTITY (AGENTS.md section 9): train / validation / eval all go through
prepare_batch.prepare(), which calls impute_before + apply_to_queries + apply_to_probes; the Python
deploy path (rwkv/run_as_rnn.py) keeps an OnlineImputer that takes the same value from the same rows
(value() before the review, push() after it); the Rust engine consumes rows its caller built, so the
Anki fork's feature builder must apply the same rule (optimization/DEPLOY_FUNCTIONS.md section 6).
"""
import bisect
import heapq
import os

import numpy as np
import torch

ENV = "RWKV_PROBE_IMPUTE_K"
# global_labels column layout (data_processing.create_sample): les, led, label_y, label_rating,
# has_label, label_is_equalize, is_query
_LBL_IS_QUERY = 6
NO_HISTORY = 0.0   # the pipeline's own "no press yet" value, kept for a user with no past review


def shrinkage() -> float:
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "0"):
        return 0.0
    k = float(raw)
    if not k > 0.0:
        raise ValueError(f"{ENV}={raw!r}: expected a positive shrinkage count, or unset")
    return k


K = shrinkage()


def enabled() -> bool:
    return K > 0.0


if enabled():
    from rwkv.data_processing import CARD_FEATURE_COLUMNS as _C

    _COL_DUR = _C.index("scaled_duration")
    print(f"[probe-impute pid {os.getpid()}] ON: k={K:g} -- query + probe rows take the shrinkage "
          f"median of PAST durations (card n/(n+k), else user; no history keeps {NO_HISTORY:g})",
          flush=True)

_COUNT = {"calls": 0, "q_rows": 0, "q_hist": 0, "p_rows": 0, "p_hist": 0}


class _RunningMedian:
    """Exact running median of a stream (two heaps); mean of the two middle values on an even count,
    the same rule as numpy.median."""

    __slots__ = ("lo", "hi")

    def __init__(self):
        self.lo = []   # max-heap (negated): the lower half, holds the extra element on an odd count
        self.hi = []   # min-heap: the upper half

    def push(self, x):
        lo, hi = self.lo, self.hi
        if not lo or x <= -lo[0]:
            heapq.heappush(lo, -x)
        else:
            heapq.heappush(hi, x)
        if len(lo) > len(hi) + 1:
            heapq.heappush(hi, -heapq.heappop(lo))
        elif len(hi) > len(lo):
            heapq.heappush(lo, -heapq.heappop(hi))

    def __len__(self):
        return len(self.lo) + len(self.hi)

    def median(self):
        lo, hi = self.lo, self.hi
        if len(lo) > len(hi):
            return -lo[0]
        return (-lo[0] + hi[0]) * 0.5


class OnlineImputer:
    """The rule, statefully: value(card) BEFORE a review is processed, push(card, scaled_duration)
    AFTER it. Used by the deploy path directly and by impute_before through one scan of a chunk."""

    __slots__ = ("k", "user", "cards")

    def __init__(self, k=None):
        self.k = float(K if k is None else k)
        self.user = _RunningMedian()
        self.cards = {}

    def value(self, card_id) -> float:
        if len(self.user) == 0:
            return NO_HISTORY
        mu = self.user.median()
        c = self.cards.get(card_id)
        if not c:
            return mu
        n = len(c)
        mc = c[n // 2] if n % 2 else (c[n // 2 - 1] + c[n // 2]) * 0.5
        w = n / (n + self.k)
        return w * mc + (1.0 - w) * mu

    def has_history(self) -> bool:
        return len(self.user) > 0

    def push(self, card_id, scaled_duration: float):
        x = float(scaled_duration)
        self.user.push(x)
        c = self.cards.get(card_id)
        if c is None:
            self.cards[card_id] = [x]
        else:
            bisect.insort(c, x)


def impute_before(data) -> np.ndarray:
    """(n,) float64: for every row of the chunk, the imputed scaled duration from the REAL rows
    strictly before it in frame order. Also returns, as the second element, the mask of rows that had
    at least one past real row (the value is not the no-history constant there)."""
    sk = data.skips.numpy().astype(bool)
    cards = data.ids["card_id"].numpy()
    dur = data.card_features[:, _COL_DUR].double().numpy()
    n = sk.shape[0]
    out = np.empty(n, dtype=np.float64)
    hist = np.zeros(n, dtype=bool)
    imp = OnlineImputer()
    for r in range(n):
        cid = int(cards[r])
        hist[r] = imp.has_history()
        out[r] = imp.value(cid)
        if not sk[r]:
            imp.push(cid, dur[r])
    return out, hist


def _write(cf: torch.Tensor, rows: np.ndarray, values: np.ndarray) -> torch.Tensor:
    out = cf.clone()
    if rows.size:
        out[torch.from_numpy(rows), _COL_DUR] = torch.from_numpy(values).to(cf.dtype)
    return out


def apply_to_queries(data, imp):
    """Query rows (skip + is_query) take their imputed duration. Real rows are never touched."""
    import dataclasses
    out, hist = imp
    lab = data.global_labels
    q = data.skips.numpy().astype(bool) & (lab[:, _LBL_IS_QUERY].float().numpy() > 0.5)
    rows = np.nonzero(q)[0]
    cf = _write(data.card_features, rows, out[rows])
    c = _COUNT
    c["calls"] += 1
    c["q_rows"] += int(rows.size); c["q_hist"] += int(hist[rows].sum())
    if c["calls"] % 500 == 1:
        print(f"[probe-impute pid {os.getpid()}] k={K:g} chunks {c['calls']} query rows {c['q_rows']} "
              f"with history {c['q_hist']} | probe rows {c['p_rows']} with history {c['p_hist']}",
              flush=True)
    return dataclasses.replace(data, card_features=cf)


def apply_to_probes(data, meta, imp):
    """The four probe rows of each target take the TARGET's imputed value, computed on the pre-probe
    chunk (imp) at the target's old row (meta.pick). Probes are skip rows, so they never entered a
    median themselves."""
    if meta is None:
        return data
    import dataclasses
    out, hist = imp
    pick = np.asarray(meta.pick, dtype=np.int64)
    pos4 = np.asarray(meta.pos4, dtype=np.int64)
    rows = pos4.reshape(-1)
    vals = np.repeat(out[pick], 4)
    cf = _write(data.card_features, rows, vals)
    _COUNT["p_rows"] += int(rows.size); _COUNT["p_hist"] += 4 * int(hist[pick].sum())
    return dataclasses.replace(data, card_features=cf)
