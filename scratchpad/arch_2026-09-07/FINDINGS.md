# Architecture screens, 2026-09-07 (Andrew: "I'd be shocked if there is no low-hanging fruit")

Opened after Andrew pushed back on a phase-5 recommendation that was built on seven consecutive
TRAINING-side rejects (iters 62-68) and no architecture at all. He is right that the evidence did not
cover the model. Architecture at d=80 has had eleven experiments, four of them the interleaving 2x2;
the *expressiveness-at-fixed-params* family he opened on 2026-08-17 has **one success (interleaving,
+0.000291 ahead, the phase's largest architectural gain) and one partial null** (iter 57, which
reached only 4 of 13 channel mixers) -- against a conduct rule requiring 3-5 variants to close a
family. And the mechanism agrees: capacity adds fail 0/3 while iter 65 showed the model is
fit-limited, so what can pay is a richer or cheaper-to-fit form at the SAME parameter count.

Candidates: (1) cross-stream state fusion, (2) extra rounds via layer reuse, (3) multi-step delta on
the coarse streams, (4) a richer feature encoder, (5) a richer curve head. **This document kills (5).**

## Screen 1 -- head rank: the curve head's hidden is effectively 2.4-dimensional

`headrank_screen.py` (SVD of activations on 2 train users, 10,614 labelled rows; no solver, so
nothing to get wrong):

| activation | dim | participation-ratio rank | dims for 90% var | for 99% |
|---|---|---|---|---|
| `x` (prehead_norm, trunk output) | 80 | **10.0** | 25 | 63 |
| `x_w` (curve head hidden) | 320 | **2.4** | **3** | 7 |
| `x_p` (rating head hidden) | 320 | **10.9** | 34 | 167 |

Both heads read the same rank-10 trunk output. The rating head spreads over ~11 effective dimensions
for a 4-way output; the curve head collapses to ~2.4 for a 9-number output (w, S, d) x N=3. So across
every review the forgetting curve is drawn from a roughly 2.4-parameter family -- about what FSRS
has, in a model whose premise is a flexible learned curve. Waste, or the right inductive bias?

## Screen 2 -- is the head a SUFFICIENT STATISTIC of its own input? YES.

Take the model's own output as a feature and ask whether the trunk output adds anything on top of it.
Leave-one-user-out, standardised on train rows only, 96,968 labelled rows over 6 train users. Both
arms are equally in-sample for the model, so neither has an advantage.

| arm | by-user mean BCE |
|---|---|
| MODEL (its own ahead prediction) | 0.21189 |
| probe0 = logistic(logit p) -- recalibration only | 0.21216 |
| probe1 = + t basis | 0.21370 |
| probe2 = + t basis + the whole 80-d trunk output | 0.22805 |

Sanity PASS: probe0 reproduces the model to **0.00027**, so the fit machinery works.
**What t alone adds beyond the head's output: −0.00154. What x + t add: −0.01589, and it helps on
0 of 6 users.** Adding the trunk output does not improve on the head's answer; it only overfits.

**=> THE CURVE HEAD IS NOT THE BOTTLENECK. Its rank-2.4 hidden is the right inductive bias, not
waste: the head already extracts everything the trunk offers about the curve, and the power-law
mixture handles the horizon better than a free 4-basis logistic does (probe1 is worse than probe0).**
A richer or wider ahead head is NOT the lever, and neither is replacing the curve family. Two
candidates eliminated for ~1 h of CPU and zero GPU.

⚠ **Honest limit:** the probe is LINEAR in x, so it cannot rule out a nonlinear residual the head
misses. But the direction is not marginal -- adding 80 dimensions makes it *worse* by 0.016 -- so
there is little signal there to find.

## Two invalid versions of screen 2, recorded because the third is only trustworthy by contrast

1. **A hand-rolled Newton solver diverged** (400 steps, no line search, no convergence check) and
   returned held-out BCE of **2.0-6.2**, where a constant predictor gives ~0.30; the intercept column
   had also been standardised to all-zeros, removing the intercept. Read at face value it would have
   "shown" the head is fine -- the correct conclusion, from a broken instrument, which is exactly the
   un-diagnostic-probe failure written up for iter 67 the same day. **A screen needs a baseline whose
   value you can predict in advance; 4.4 against an expected 0.30 is what caught it.**
2. **With a real solver the numbers were sane but the comparison was confounded:** the model trained
   on all six users, the probe was leave-one-user-out. Its own sanity check (const >= t-only >= x+t)
   FAILED -- and taught that the check itself was mis-specified: `t-only` transfers WORSE than a
   constant because the t->outcome relation shifts across users. **"More features => lower held-out
   loss" is false under distribution shift**, so that ordering is not a validity test. The fix was to
   make the model's own output the baseline feature, which removes the advantage entirely.

## Where this leaves the architecture queue

The head is excluded, so a remaining lever must change what the TRUNK computes:
1. **cross-stream state fusion** -- named by the iters 41-44 topology finding as the untried
   direction ("change WHAT is computed... not when"); no stream can currently read another's WKV
   state, only the shared refined `x`. Screen: how redundant are the 5 streams' states?
2. **extra rounds via layer reuse** -- zero new parameters, strictly more computation.
3. **multi-step delta on the coarse streams** -- richer recurrence at FIXED state size (card/note
   untouched, so the state gate is safe); the delta rule is measured as massively load-bearing.
4. **a richer feature encoder** -- 109 -> 80 is a single linear, the only place the inputs mix
   before the recurrence.


## Screen 3 -- do two STRUCTURALLY DIFFERENT models err on the same reviews? YES, at r = 0.996.

The 2026-07-03 entropy-floor estimate (task #18) put the AHEAD floor at 0.2994 with the two d=128
models at 0.2992/0.2993 -- no gap -- because the estimator COLLAPSED: cross-model residual covariance
0.0950 vs each model's own Brier 0.0955. Its own caveat is the important part: that cannot separate
TRUE NOISE from a blind spot SHARED by the family. Two models of one family erring identically is
exactly what a shared blind spot looks like. So: add the most different model available and see
whether the errors decorrelate.

**Pair** (`decorrelate.py`, 4 VAL users where BOTH are out-of-sample, 409,599 reviews intersected on
`review_th`, label agreement 0.9998 so the alignment is sound):

| | params | features | Brier |
|---|---|---|---|
| A = realcyc | 563,652 | gen-5 real-timestamp, 109 dims | 0.10226 |
| B = pretrained d=128 | 2,762,884 | published, 92 dims | 0.10199 |

| statistic | value |
|---|---|
| cross-model residual covariance E[(y−pA)(y−pB)] | **0.10168** |
| ratio to A's Brier / B's Brier | **0.9943 / 0.9969** |
| **residual correlation** | **0.9957** |
| prediction correlation | 0.9686 |
| 50/50 oracle ensemble gain over the better model | **+0.00025** |

**=> The 2026-07-03 "family saturated" result EXTENDS across a 5x parameter change AND a complete
feature-layout change.** A model with 5x the parameters, trained on different inputs in a different
era, gets the same reviews wrong. Notably the feature change is included in that: the gen-5 real
clocks did not alter WHICH reviews are hard, which is consistent with realcyc scoring an exact tie
against gen4base.

**What it means for the architecture question.** Within this family and these inputs, ahead has no
headroom for width, topology, optimizer or objective work -- which is precisely the observed pattern
(capacity 0/3, topology rearrangement 1/4 with the win being interleaving's EXISTENCE, optimizer
coverage now closed 1/3, row reweighting 0/2, curve head excluded by screen 2). The measured
disagreement between our model and a 5x bigger one is worth **+0.00025** in the metric's own units,
and that is the ceiling on "capture what the other model knows".

**What it does NOT settle.** Both arms are RWKV-7 with the same 5-stream structure and the same task
framing, so a genuinely OUT-OF-FAMILY contrast is still untested. The decisive one is FSRS-6 -- a
hand-designed 3-parameter memory model, maximally different inductive bias -- whose per-review
predictions live in `srs-benchmark` (Andrew's repo; this one does not touch it). If FSRS decorrelates
materially, there IS structure our family systematically misses and the disagreement pattern points
at what to build. If it does not, the ahead residual is noise and the only remaining lever of size is
the training budget (the 2026-08-11 calibration measures 10x at ~+0.0042, i.e. 17x this ensemble gain).
⚠ Related and worth re-reading in this light: the FSRS-core hybrids (iters 60/61) were rejected at
-0.0027 ahead, but BOTH ran under the <=100k parameter cap that Andrew has since lifted ("stop making
the model smaller"). A full-size hybrid has never been tried.

## Screen 4 -- the OUT-OF-FAMILY contrast (2026-09-08). FSRS-7 adds nothing.

Screen 3 left one thing open: both its arms were RWKV-7, so it could not tell TRUE NOISE from a
blind spot shared by the family. Andrew pointed at `srs-benchmark` for the current FSRS-7
(`models/fsrs_v7.py`, the 34-parameter dual-stability model with hand-designed update rules and
per-parameter clamps) -- the most different inductive bias available without new training.

Three FSRS arms dumped per-review on the same four VAL users (`scratchpad/fsrs7/`, CPU only,
~29 min): FSRS-7 fitted per user (the leaderboard configuration), FSRS-6 fitted per user (the
within-family version-bump calibration), and FSRS-7 with `--default` parameters (the
personalization control). srs-benchmark's own `script.process` runs unmodified; the driver
chdir's into its own output dir first, so nothing is written into that repo.

**340,601 rows, label agreement 0.999859 across every pair.**

| arm | by-user BCE | resid corr vs realcyc | **best blend weight** | **gain over realcyc** |
|---|---|---|---|---|
| realcyc (563k, gen-5) | 0.28706 | -- | -- | -- |
| d128 pretrained (2.76M) | 0.28685 | 0.9955 | **0.53** | **+0.00086** |
| FSRS-7, per user | 0.30340 | 0.9760 | **0.02** | **+0.00001** |
| FSRS-6, per user | 0.33668 | 0.9468 | 0.02 | +0.00002 |
| FSRS-7, default params | 0.32041 | 0.9702 | 0.00 | +0.00000 |

Leave-one-user-out logistic stacking on the two logits is NEGATIVE for every FSRS arm
(-0.00021..-0.00028) and positive for d128 (+0.00073). Per user, FSRS-7's optimal weight is
**0.00 on three of the four users** and 0.14 on user 5003 (+0.00081); d128 takes 0.33-0.76 and
pays on all three real users.

**=> THE AHEAD RESIDUAL IS NOT A FAMILY BLIND SPOT.** The strongest out-of-family contrast
available -- a hand-designed memory model with a completely different inductive bias, fitted per
user -- contains essentially no information our model lacks. What DOES buy something is another
RWKV with 5x the parameters and ~10x the training budget (+0.00086), i.e. capacity and budget,
not architecture. That is the lever ws10 is already spending.

**★ THE METHODOLOGICAL LESSON, and it would have inverted the verdict: RESIDUAL CORRELATION
OVERSTATES DECORRELATION WHENEVER THE TWO MODELS DIFFER IN QUALITY.** FSRS-7's residual
correlation (0.9760) is visibly below the in-family 0.9955, which reads as "there is structure
here". There is not: the extra residual variance is FSRS's own error, not information, and the
fitted blend weight (0.02) says so. Screen 3's 50/50 oracle is equally misleading in the other
direction -- it makes every FSRS pair look actively harmful (-0.004..-0.011) purely because FSRS
is worse. **Price a disagreement with a FITTED blend weight; use residual correlation only
between models of comparable quality.**

⚠ **The alignment guard earned its place.** Our dumps key each row on `review_th` -- the review
the prediction is made FROM -- while FSRS keys on the review being PREDICTED; the label belongs
to `label_review_th`. Joining directly gave a label agreement of **0.7726 against a chance rate
of ~0.745**, i.e. no alignment at all, and the whole table would have been noise that looked
like a finding. The fix (`scratchpad/fsrs7/label_map.py`) rebuilds the map from the DATA alone,
so it cost minutes rather than re-running 2.7 h of RNN forward passes. The RWKV-vs-RWKV result
of screen 3 is unaffected: both arms used the same convention.
