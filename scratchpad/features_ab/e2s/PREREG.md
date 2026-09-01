# e2s — pre-registration, written before the arm ran

Recorded 2026-08-30, while featA2 (the control) was still evaluating and e2s had not started.

## The question

`featA2` and `e2s` are the same recipe on the same published dataset. The only difference is what
`elapsed_seconds` means:

| arm | `elapsed_seconds` | |
|---|---|---|
| featA2 | `answer(k) - answer(k-1)` | end-to-END — what the dataset ships, what the champion trained on |
| e2s | `show(k) - answer(k-1)` | end-to-START — **what a live Anki scheduler computes** |

Size gate verified before launch on both dbs: train 1,483,984 / test 170,384, identical.

## ★ THE STANDARD GATE DOES NOT APPLY, AND APPLYING IT WOULD INVERT THE CONCLUSION

The research gate accepts a change only if **both modes improve by ≥0.0001**. That rule assumes the
candidate is trying to be *better*. This one is trying to be *honest*: end-to-start **removes**
information, because `duration(k)` is not available at prediction time. FSRS-7 duly got slightly
worse under it (+0.000111 LogLoss at matched size).

**So a small regression here is the EXPECTED result, not a rejection.** Under the standard gate,
the arm that matches deploy would be rejected for matching deploy.

**The decision does not depend on the sign, and that is worth stating in advance so the number
cannot be read as a verdict.** Deploy computes end-to-start whatever we train on
(`vendor/jschoreels_anki/rust/rwkv.rs:322` — `now() - last_review_time`, evaluated before the user
answers). The three cases:

| outcome | what it means | what we do |
|---|---|---|
| e2s ≈ featA2 (within ±7.5e-5) | the leak buys our model nothing | switch — free correctness |
| e2s worse by X | **X = how much our reported numbers are inflated** by a quantity deploy cannot supply | switch, and restate the champion's number |
| e2s better | the train/deploy mismatch was actively costing us | switch, and it is a gain |

**This arm measures the SIZE of a correction, not whether to make it.** What it must not become is
a referendum on whether to keep training on a quantity that does not exist at deploy.

## Pre-registered predictions

1. **e2s is worse on both modes**, by an amount of the same order as FSRS-7's +0.000111 — call it
   **+0.0001 to +0.0005**. Our model reaches `duration(k)` through the same single channel FSRS
   does (the interval), because feature 7 already zeroes the most recent duration in train, eval
   and deploy alike. So there is no reason to expect a much larger effect.

2. **★ imm degrades MORE than ahead.** This is the sharp one, and it is falsifiable. `imm` predicts
   the rating of *the current review* — the very review whose duration is leaking — so the leak is
   a direct signal about the target. `ahead` predicts a future review cold, where `duration(k)`
   informs the target only through the curve's sampling point. If instead **ahead** moves more,
   my mechanism is wrong and the effect is about curve placement rather than leakage.

3. **The effect is concentrated in same-day rows.** The two definitions are numerically
   near-identical on longer intervals (duration is a median 0.001% of the gap there, and 0.00% of
   long rows move by ≥10%). Anything appearing on long rows is the **refit**, not the interval.

4. **`size` stays identical at eval**, 2,500 users, 0 mismatches. We have no `delta_t > 0` filter —
   rows are kept and only marked via `label_is_equalize` on `review_th`, from a reused
   `label_filter_db`. Already checked at the db level; the eval is the end-to-end confirmation.
   **If this fails, nothing else in the arm is interpretable** and the cause is a pipeline bug, not
   a finding.

## Analysis plan, fixed in advance

* Report deltas versus **featA2**, never versus iter 53 — featA2 is KD-off and the champion is not.
* Split the per-user delta by same-day share to test prediction 3.
* Paired per-user Wilcoxon both modes, `--intersect`, as usual.
* Report the delta as **the leak's size**, and state plainly that the champion's published number
  is optimistic by that amount as a deploy estimate.

## What would change my mind about switching

Only one thing: evidence that a live scheduler can supply an end-to-END interval. It cannot —
`duration(k)` has not happened when the prediction is made. Short of that, a worse honest number
beats a better unattainable one.
