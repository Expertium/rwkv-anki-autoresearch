# ordcut P3 -- engagement, read from the WS checkpoints before the accuracy verdict (2026-09-05 11:52, decay running)

PREREG P3: `ord_cut_a` must move off 0 by >= 0.3 logits, else the term was inert and the verdict is
uninterpretable.

| checkpoint | a (cut, logits) | c (slope on log1p(t / 1 d)) |
|---|---|---|
| WS 1000 | +0.292 | +0.176 |
| WS 5000 | +0.649 | +0.109 |
| WS 10935 (WS end) | **+0.838** | +0.015 |

**Engaged, monotonically.** The cut settled ~0.84 logits above the Again boundary: at a success with
logit R = z, the model puts P(Good-or-Easy | success) = sigmoid(z - 0.84) -- i.e. it learned that a
Hard is what a success looks like when R is about 0.84 logits below where a Good is. The t-slope
went to ~0, so the ordinal relation is the same at every horizon >= 1 d (the sub-1-day inversion was
masked out by design). The verdict is interpretable either way: a null means the curve's logit
already carried this shape information, not that the term did nothing.

(The deployed model is unchanged by these two params; they exist only in the loss.)

## Second half (16:05, eval running): the candidate's OWN logit R separates the grades far better

Same screen instrument on the DECAYED ordcut checkpoint (cut a = 0.93), same 10 train users,
125,236 successes at t >= 1 d:

| statistic | realcyc | ordcut |
|---|---|---|
| AUC(logit R; Good vs Hard) | 0.737 | **0.851** |
| AUC(logit R; Easy vs Hard) | 0.643 | 0.774 |
| Hard share, bottom decile of logit R | 0.166 | 0.270 |
| Hard share, top decile | 0.016 | 0.001 |
| Spearman(decile, Hard share) | -1.000 | -1.000 |

PREREG P3 (second half) HOLDS with margin: the curve's logit now carries the Hard/Good distinction
it was never asked for before (+0.114 AUC). The Easy share is still not monotone (the reason the
second cut was dropped), so the one-cut design was the right reduction. What the accuracy verdict
decides is whether this extra ORDERING of R translates into a better BINARY curve at the label's t
-- the ordinal term can sharpen the ranking of successes without moving the pass/fail calibration.

