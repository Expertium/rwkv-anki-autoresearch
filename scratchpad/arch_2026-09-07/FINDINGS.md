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
