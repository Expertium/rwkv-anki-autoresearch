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
