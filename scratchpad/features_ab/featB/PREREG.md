# featB pre-registration — written 2026-09-02, while featB is mid-decay and no number exists

`featB` (23 real-timestamp features, `-id` gen-3 dbs, 114-dim input, 565,252 params) against
`featA2` (92-dim input, published `_fix` dbs, 558,212 params). Same recipe, same seed 4321, same
arch module, **KD OFF in both** (the d=128 teacher's `features2card` in_dim is 92 and cannot
forward a 114-dim batch at all).

This file is written before the arms can be compared, for the same reason the interval
pre-registration was: an analysis written after seeing the numbers can always be made to fit them.
`featB_verdict.py` is the fixed test and is not to be edited once featB reports.

## What the contrast actually contains

| component | present? | already measured elsewhere? |
|---|---|---|
| the 23 timestamp features (+7,040 params) | yes | **no — this is the thing we want** |
| end-to-**START** intervals | yes | **YES: +0.000225 ahead / +0.000400 imm** (`e2sc − fixc`) |
| the dataset itself (`-id` vs published) | yes | no — irreducible |

Held constant, and each one checked rather than assumed: Bug A fixed in both; **Bug C present in
both** (`_fix` built 08-21, `id3` built 08-24, `nan_id_fill` landed 08-26 — verified in the
artifact at ratio 0.6239 for id3); the 2026-08-19 sentinel-cumsum fix present in both (`_fix` and
`id3` are both post-08-19, which is exactly the check that featA/featA2 and iter53/fixc failed).

**Component 2 is a PENALTY featB pays inside the bundle.** End-to-start is worse by
+0.000225 / +0.000400, so featB starts roughly 0.0003 behind before the features do anything.
The features must overcome that before the headline number even reaches zero.

⚠ The subtraction is approximate: the interval cost was measured KD-**on**, at 558k params, on
published dbs. featB is KD-off at 565k on `-id`. Treat it as a first-order correction, not an
exact one.

## P1 — DIRECTION AND SIZE

**Prediction: featB beats featA2 in BOTH modes, with the ahead improvement in [0.0005, 0.004].**

Reasoning for the sign: the features add genuinely new *information* (absolute time-of-day, true
calendar phase, card/note/deck age, creation batch, seconds-resolution recency), and this trunk's
own record says it is **data-limited, not capacity-limited** at 5k — capacity adds went 0/3, but
information is the thing a data-limited model can actually use. That is a different lever, and
the 0/3 record is not evidence against it.

Reasoning for the modest size: several of the new columns are sparse (the sibling gap is defined
on only ~10–16% of rows, ceiling ~17%), and this was pre-registered at gen-2 build time as
"unlikely to move the gate" on coverage grounds.

**Falsifier:** featB worse in either mode, or better by < 0.0002 in ahead. That would say the
timestamp features carry no usable signal at this budget — the *features-only* figure is then
`observed + interval cost`, so even a null headline means the features are worth about +0.0003.

## P2 — MECHANISM: the gain should sit where fine-grained timing matters

The new columns are mostly about *when*, at resolutions the old 92-dim vector could not express.
That information is worth most at **short intervals** — time-of-day, same-day batch position,
seconds-resolution recency — and worth little across a multi-week gap.

⚠ **This cannot be read off the raw delta, and that is why it is written down now.** The interval
penalty *also* concentrates in same-day users (measured 6.6× top-vs-bottom quartile), so in this
contrast the same users both gain the most and pay the most, and the net sign is ambiguous.

**The test therefore runs on an interval-ADJUSTED per-user delta**, subtracting each user's own
measured interval cost from the `fixc`/`e2sc` pair:

    adjusted(u) = [featA2(u) − featB(u)] + [e2sc(u) − fixc(u)]

**Prediction: the adjusted gain rises with a user's same-day share** — top quartile at least 2×
the bottom quartile, and Spearman ρ > 0.05.

**Falsifier:** flat or anti-correlated. Then any gain is *not* coming from the fine-grained
timing columns, and attention should move to the age/tenure/creation-batch features instead.

## P3 — STRUCTURAL: the dataset confound is real, and bounded

The `-id` and published sets disagree on `day_offset` for ~0.001% of raw rows, but the equalize
filter amplifies that, so per-user scored counts differ for a large minority of users. The
standard size gate (identical counts) therefore **fails by construction** and must not be applied.

**Prediction: 20–40% of the 2,500 eval users differ in `size` between the arms.**

That number is not decoration — it defines a **size-matched subset** where the comparison is
genuinely apples-to-apples, and the headline is re-run there.

**Falsifier / the one that would change the conclusion:** if the full-set and size-matched
verdicts **differ in sign**, then the dataset swap is doing the work rather than the features,
and featB proves nothing about the features.

## The decision rule, fixed in advance

The research gate does **not** apply as written: gate #1 (identical `size`) fails by construction,
and the arms are cross-dataset, so this is not a champion candidate.

Adoption of the timestamp features requires **all** of:

1. featB better in **both** modes on the **size-matched subset**, p < 0.0001;
2. the same sign on the full 2,500;
3. the adjusted gain surviving the interval correction (i.e. not explained by it).

If (1) and (2) disagree, the result is **inconclusive**, not negative.

**And regardless of the outcome, the real adoption decision re-bases on generation 4**, which is
already armed behind this run: gen 3 carries Bug C (37.2% of note identity lost on cards with
missing metadata). featB's number is directionally informative; it is not the production number.
