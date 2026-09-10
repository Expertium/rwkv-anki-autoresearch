# Are the coarse streams capacity-starved, and is there a cheaper way to find out than 38 h?

Andrew, 2026-09-10, on arm 1's Q2: *"It would be nice to try to grow deck/preset/global, but 38 h
is a pretty steep price."*

He is right that 38 h is the wrong first purchase. This is the rung before it: one deploy-RNN pass
per checkpoint, CPU only, ~75 s each, zero GPU. Tool: `stream_rank.py`.

## What was measured

Effective rank of each stream's final block output, over 9,000 real rows of train users 107 / 136 /
156, on two checkpoints that differ **only** in training budget: `realcyc` (1.25 epochs) and arm 1
(12 epochs). Same architecture, same data, same 80-dim streams.

The comparison is the point. A single checkpoint's participation ratio means nothing on its own;
what matters is whether utilization RISES when the budget rises 10x. Precedent for the instrument:
`arch_2026-09-07/headrank_screen.py` measured the curve head's hidden at an effective 2.4
dimensions and killed a richer ahead head outright.

## The result

| stream | depth | PR 1.25 ep | PR 12 ep | dims for 90% | dims for 99% |
|---|---|---|---|---|---|
| card_id | 2 | 11.3 | 12.6 | 22 -> **27** | 58 -> **62** |
| note_id | 1 | 7.8 | 8.4 | 15 -> **18** | 43 -> **48** |
| deck_id | 4 | 7.4 | **4.8** | 24 -> 24 | 62 -> **64** |
| preset_id | 3 | 10.4 | 11.2 | 24 -> **29** | 61 -> **66** |
| user_id | 3 | 10.0 | 9.2 | 24 -> **28** | 61 -> **65** |

**A weak positive, and it should not be sold as more than that.**

* **The variance tail broadened on every stream, at both thresholds.** 90%-variance dims rose on
  four of five (deck flat) and 99%-variance dims rose on all five. At 12 epochs the coarse streams
  need **64-66 of their 80 dims** to reach 99%, i.e. 80-83% of the width they have. That is the
  direction "capacity is starting to bind" predicts.
* **The participation ratio disagrees**, and it is not a rounding disagreement: deck falls 7.4 ->
  4.8 and user falls 10.0 -> 9.2. PR is dominated by the leading directions, so the two statistics
  are saying that the model **concentrated its top directions while spreading its tail**.
* Zero dead dims anywhere, at either budget.

**So the screen does not kill the lever, and does not motivate 38 h of GPU either.** It says the
streams are neither idle nor obviously starved. Anyone quoting the 90/99% rise as evidence FOR
should be shown the PR column, and vice versa.

⚠ Effective rank of activations is a proxy for capacity pressure, not a measurement of it. A stream
that is information-limited rather than width-limited looks exactly like this.

## The costing, which is the part Andrew actually asked about

| option | cost | what it answers | worth it? |
|---|---|---|---|
| **A. warm-started coarse capacity, decay-only** | **~10.4 h** | does capacity help when it is added to a fully-trained model and given 2 epochs to use it? | **yes -- the cheap rung** |
| B. capacity arm at a REDUCED (2-5 ep) WS | ~23 h | nothing trustworthy -- see below | **no** |
| C. capacity arm at the full 10-ep WS | ~48-60 h | the clean answer | only if A wins |

**Option B is the obvious idea and it is unsound.** Arm 1 shows ahead saturates by ~epoch 2 for
THIS model, so a 2-5 epoch budget looks like it would sit in the saturated regime. But a LARGER
model needs MORE budget to saturate, so a reduced-budget comparison would under-train exactly the
arm that is supposed to win -- reintroducing the confound that made iters 49 / 60 / 61
uninterpretable in the first place. Do not run it.

**Option A is sound and cheap because the capacity can be added function-preservingly.** The
natural lever already exists: `RWKV_STRIP_CMIX` currently strips 9 channel mixers, six of them on
`user_id` and `preset_id`, and iter 49 tested restoring exactly those (+26,070 params, +4.7%) --
returning a null **at 1.25 epochs, the budget Q2 has just shown cannot expose a capacity limit.**
Un-stripping them on a model warm-started from ws10, with each restored mixer's output projection
**zero-initialised**, makes the model bit-identical to ws10 at step 0 (an RWKV channel mixer is
`x + cmix(x)`, so a zero output projection is the identity) and lets the 2-epoch decay train the
new capacity. Control: arm 1, exactly, same WS checkpoint and same decay length.

**The asymmetry to state before running it: a WIN is decisive, a NULL is weak.** Two epochs is much
less than twelve for brand-new parameters, so option A can only put a LOWER BOUND on what capacity
buys. That is an acceptable trade for 10.4 h against 48-60 h, provided the null is not later quoted
as "capacity does not bind" -- which is precisely the error iter 49 already caused once.
