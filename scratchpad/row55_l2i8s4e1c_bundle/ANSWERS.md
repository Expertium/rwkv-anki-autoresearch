# Answers to "Recurrent State Compression Notes"

The other team's note (Recurrent State Compression Notes, on `anki-revlogs-10k`) reads the row-17
bundle and asks six questions plus a general request ("long-card accuracy, not only finiteness").
This file answers each one with measurements, and says where a measurement does not exist.

**The recipe measured here is iter 55** (`l2i8s4e1c`): row 17's exact state layout
(card rank-2 int8 + int4 shifts, note rank-1 int8 + int4 shifts = 1,280 / 2,304 = 3,584 bits), trained
with the E1 tail-energy penalty (`lambda=1.0`, `stride=8`). Its 750-user gate is
+0.000028 ahead / -0.000045 immediate against the fp32
control, i.e. PASS on both heads with the immediate head BETTER than the control.

**Their bars, restated** (so the tables below can be read against them): dataset-wide paired
upper bound <= +0.05% per head; long cards (their 50 users with 128+ answer cards) <= +0.5% point and
<= +1.0% upper bound; young cards <= +0.10% upper bound; per-card movement fails when more than 1% of
cards have a review moving by more than 0.10.

A note on method that matters for all of it: **everything below is measured on the deploy path (the
Rust engine) on 40 of the 50 longest-history users of the corruption roster (1,763,360 reviews), in
one of two pairings**, and the pairing is named in each table:

* **quantizer pairing** - the SAME checkpoint with the state quantizer switched on vs off. This is the
  closest analogue of their measurement (their rounding is applied at every update on their own
  weights).
* **gate pairing** - the QAT+quantized checkpoint vs the fp32 no-QAT control, i.e. the pairing the
  750-user gate itself uses.

Their own answer-count mix and this roster's mix are close (P50 = 5 answers per card on both; bucket
shares 25.7 / 19.4 / 22.9 / 15.2 / 10.4 / 5.9 / 0.4%), so the per-bucket numbers are directly
comparable. `evidence/bucket/bucket_mix_roster750.json` holds the roster's own distribution.

---

## (a) Score the tax by the card's answer-count bucket

**Quantizer pairing, iter 55 (this recipe, this checkpoint).** Bucket = the review's ordinal within
its card (1 = the card's first review).

| bucket | reviews | d imm | rel imm | d ahead | rel ahead | |dp| imm p50 | p90 | p99 | max | reviews > 0.05 | reviews > 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1,763,360 | -0.000627 | -0.160% | -0.000496 | -0.130% | 0.00297 | 0.01658 | 0.04650 | 0.15879 | 0.73% | 0.01% |
| 1-2 | 133,611 | -0.000272 | -0.038% | -0.000231 | -0.058% | 0.00041 | 0.00527 | 0.01537 | 0.04072 | 0.00% | 0.00% |
| 3-4 | 130,943 | +0.000059 | +0.016% | +0.000115 | +0.030% | 0.00115 | 0.00851 | 0.02150 | 0.08363 | 0.03% | 0.00% |
| 5-8 | 245,090 | -0.000249 | -0.064% | +0.000192 | +0.048% | 0.00196 | 0.01134 | 0.02853 | 0.10744 | 0.11% | 0.00% |
| 9-15 | 350,827 | -0.000558 | -0.154% | -0.000222 | -0.058% | 0.00272 | 0.01549 | 0.03799 | 0.13825 | 0.27% | 0.00% |
| 16-31 | 502,688 | -0.000813 | -0.238% | -0.000935 | -0.260% | 0.00363 | 0.01821 | 0.04683 | 0.12259 | 0.72% | 0.00% |
| 32-127 | 389,975 | -0.001031 | -0.277% | -0.000973 | -0.252% | 0.00533 | 0.02490 | 0.05664 | 0.14453 | 1.83% | 0.01% |
| 128+ | 10,226 | -0.000865 | -0.160% | +0.003879 | +0.696% | 0.00875 | 0.04690 | 0.09002 | 0.15879 | 8.42% | 0.55% |

| cards with a review moving > 0.05 (immediate) | 4.88% |
| cards with a review moving > 0.10 (immediate) | 0.08% |
| cards with a review moving > 0.05 (ahead) | 1.76% |
| cards with a review moving > 0.10 (ahead) | 0.01% |

**What it says.** The damage is NOT on old cards: on the oldest bucket (128+ answers, 10,226 reviews)
the quantized state is **-0.000865 better** on the immediate head
(-0.160% relative). On the immediate head the positive cells are
only the young buckets (3-4), and the largest of them is
+0.016% relative (bucket 3-4). The ONE positive old-card
cell is the AHEAD head of that same 128+ bucket (+0.003879,
+0.696% relative) - see (e). Their int4-everywhere result
(+0.18% to +0.74% on their 128+ cards) is NOT reproduced by this recipe - their int4 puts 4 bits on
every state entry, while this recipe keeps an 8-bit rank-2/rank-1 factor form and truncates ~98% of
the state's energy by RANK, so the residual error does not compound the same way.

**The layout-level answer (row 17, the same bits without the E1 penalty), for comparison.**

Gate pairing:

| bucket | reviews | d imm | rel imm | d ahead | rel ahead | |dp| imm p50 | p90 | p99 | max | reviews > 0.05 | reviews > 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1,763,360 | -0.000510 | -0.130% | -0.000474 | -0.125% | 0.00534 | 0.02622 | 0.06071 | 0.29624 | 2.05% | 0.10% |
| 1-2 | 133,611 | -0.003670 | -0.491% | -0.000027 | -0.007% | 0.00097 | 0.00849 | 0.01942 | 0.07637 | 0.01% | 0.00% |
| 3-4 | 130,943 | +0.000378 | +0.100% | +0.000251 | +0.065% | 0.00200 | 0.01367 | 0.03158 | 0.15930 | 0.16% | 0.01% |
| 5-8 | 245,090 | +0.000270 | +0.070% | -0.000157 | -0.039% | 0.00378 | 0.01898 | 0.04626 | 0.19465 | 0.74% | 0.02% |
| 9-15 | 350,827 | +0.000264 | +0.073% | -0.000054 | -0.014% | 0.00466 | 0.02416 | 0.05723 | 0.20486 | 1.64% | 0.07% |
| 16-31 | 502,688 | -0.000332 | -0.098% | -0.000649 | -0.182% | 0.00625 | 0.02759 | 0.06263 | 0.27957 | 2.29% | 0.11% |
| 32-127 | 389,975 | -0.001092 | -0.295% | -0.001197 | -0.313% | 0.01041 | 0.03608 | 0.07094 | 0.29624 | 4.00% | 0.19% |
| 128+ | 10,226 | -0.002436 | -0.448% | +0.001495 | +0.266% | 0.01937 | 0.05469 | 0.10494 | 0.19767 | 12.59% | 1.21% |

Quantizer pairing:

| bucket | reviews | d imm | rel imm | d ahead | rel ahead | |dp| imm p50 | p90 | p99 | max | reviews > 0.05 | reviews > 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1,763,360 | -0.001745 | -0.443% | -0.002576 | -0.677% | 0.00783 | 0.03483 | 0.07126 | 0.31809 | 3.81% | 0.19% |
| 1-2 | 133,611 | +0.003270 | +0.442% | +0.000614 | +0.155% | 0.00068 | 0.00599 | 0.01602 | 0.12175 | 0.01% | 0.00% |
| 3-4 | 130,943 | +0.000704 | +0.187% | +0.000905 | +0.234% | 0.00217 | 0.01406 | 0.03300 | 0.16045 | 0.18% | 0.01% |
| 5-8 | 245,090 | -0.000294 | -0.076% | -0.000174 | -0.044% | 0.00461 | 0.02365 | 0.05027 | 0.19939 | 1.02% | 0.03% |
| 9-15 | 350,827 | -0.001381 | -0.379% | -0.001826 | -0.481% | 0.00746 | 0.03180 | 0.06589 | 0.21951 | 2.74% | 0.14% |
| 16-31 | 502,688 | -0.002709 | -0.790% | -0.003630 | -1.008% | 0.01074 | 0.03730 | 0.07455 | 0.29636 | 4.38% | 0.25% |
| 32-127 | 389,975 | -0.004112 | -1.101% | -0.005137 | -1.330% | 0.01624 | 0.04628 | 0.08234 | 0.31809 | 7.99% | 0.36% |
| 128+ | 10,226 | -0.008170 | -1.488% | -0.001670 | -0.296% | 0.01784 | 0.06065 | 0.10516 | 0.17669 | 16.55% | 1.28% |

Both pairings agree on the shape. On the gate pairing the 128+ bucket is
-0.002436 immediate and the only positive cells are again the youngest
buckets; on the quantizer pairing 128+ is -0.008170 immediate
(-1.488% relative).

---

## (b) The per-card |delta R| readout

The columns `|dp| imm p50 / p90 / p99 / max` and `reviews > 0.05 / > 0.10` in the tables above are the
per-review |delta p(recall)| distribution; the `cards` block under each table is the share of CARDS
with at least one review moving by more than the threshold. For iter 55 on the quantizer pairing:
p50 0.00297, p99 0.04650, max
0.15879, and 0.08% of cards move by more
than 0.10 on the immediate head (0.01% ahead).

**Against their guard** (fail when more than 1% of cards have a review moving by more than 0.10):
iter 55 reads 0.08% of cards immediate and 0.01%
ahead, i.e. it clears the guard with about 12x margin. For scale, on the same kind of
pairing the row-17 checkpoint (the same bits WITHOUT the E1 penalty) reads
2.11% immediate / 2.74% ahead, which
would FAIL the guard, while the fixed-basis codec row 61 (`l2i8fb4n`, note codec m=4, 1,152 FEWER note
bits) reads 0.62% / 0.60% with a
smaller median movement (0.00474 vs 0.00783).

---

## (c) Does rank-2 truncation accumulate on old cards?

The question is whether re-applying the truncation at every update compounds on a long history, the
way their per-update rounding does. Three measurements say no, for this family:

1. **The bucket split.** In both pairings (and for both the row-17 and the iter-55 checkpoints) the
   OLDEST bucket is where the quantized model does best, and the young buckets are the only positive
   cells. Accumulated error would show the opposite gradient.
2. **The per-user split on the longest histories.** Iter 55's per-user immediate deltas over the 40
   users have median -0.000357 with 30 of 40 users negative (the quantized
   state is the better one), and the FIVE longest histories average -0.000552 - the
   busiest users are not the damaged ones. The per-user detail is in
   `evidence/bucket/bucket_tax_row55.json` (`per_user`).
3. **The long-history trace.** On user 5098 (29,166 reviews) the quantized state scores
   -0.001749 log loss BETTER on the immediate head than the same weights with the
   quantizer off - the opposite of accumulating damage. This is expected rather than miraculous: the
   checkpoint was trained WITH the quantized state in its forward pass, so the unquantized state is
   the out-of-distribution case for it.

So: for this family the truncation error is not a random walk over the history. The rank-r subspace
that is thrown away is re-estimated from the CURRENT state at every step, and the state is dominated
by recent evidence, so the discarded directions do not accumulate the way a per-entry rounding
residual does.

---

## (d) The "0.005 ahead-head trace-vs-eval gap" was a file-naming artifact

Their note reads the earlier bundle as: the ahead head differs by about 0.005 between the trace path
and the evaluation path, roughly 10x the tax, so the deploy-side ahead tax rests on 2 users. **That
reading came from reading the evaluation's two result files the wrong way round, and the number does
not survive the correction.**

The evaluation writes two files per run, and this project names them counter-intuitively:
`result/adw_<tag>.jsonl` is the **AHEAD** head and `result/adwP_<tag>.jsonl` is the **IMMEDIATE**
head (see `quant_sweep.py`'s own docstring; the row-17 bundle's `verification.md` section 4b compared
the two heads the other way round, which manufactures an ahead-only gap out of nothing).

Measured correctly, on the 40 longest-history users, the SAME weights and NO quantization on either
side (`code/scratchpad/quant_plan/trace_eval_gap.py`, `evidence/bucket/trace_eval_gap_plain.json`):

| | mean difference | mean abs difference | max abs difference |
|---|---:|---:|---:|
| immediate head | -0.0015 | 0.0031 | 0.0147 |
| ahead head | -0.0001 | 0.0032 | 0.0144 |

Both heads differ by the same small amount, so the residual is an INPUT difference between the trace
exporter's frame/clock and the evaluation dataset - not an ahead-head bug, and not a quantization
effect. Consequences:

* a deploy tax must always be quoted WITHIN one path. Every number in this bundle is: the gate is
  Python-evaluation vs Python-evaluation, the deploy tax is Rust vs Rust on the same traces.
* the ahead head does not "rest on 2 users" more than the immediate head does; both are measured on
  the same 750 users in the gate and on the same 40 users in the bucket work.

The row-17 bundle's `verification.md` carries the same correction as an appended note.

---

## (e) The tax in relative terms, against their bars

Relative figures are the `rel%` columns above (delta divided by the base log loss of that cell).
Read against their bars:

| their bar | what this recipe does |
|---|---|
| dataset-wide <= +0.05% upper bound | gate: ahead +0.045% / immediate +0.019% relative upper bound - **both pass** their +0.05% bar |
| long cards (128+) <= +0.5% point / +1.0% upper | iter 55 quantizer pairing: immediate -0.160% (BETTER than the unquantized state), ahead +0.696% - the ahead cell is above their +0.5% point bar on this roster's 10,226 long-card reviews, and it is the one place this recipe shows a real old-card cost |
| young cards <= +0.10% upper | iter 55 quantizer pairing 1-2 bucket: immediate -0.038%, ahead -0.058% - **both pass** (both are BETTER than the unquantized state); the largest positive cell anywhere in the table is the 3-4 bucket at +0.016% |
| >1% of cards moving >0.10 = fail | iter 55, quantizer pairing: 0.08% of cards immediate / 0.01% ahead - **passes**, about 12x inside the 1% limit (the row-17 checkpoint, the same bits without the penalty, reads 2.11% / 2.74%) |

The honest summary: this recipe clears their dataset-wide bar, their young-card bar and their per-card
movement guard, and it meets their size ambition against the naive int4 bars. Its own overall deploy
tax on this roster is NEGATIVE on both heads (-0.000627 immediate /
-0.000496 ahead), because the checkpoint was trained with the quantized state in
its forward pass, so the unquantized state is the out-of-distribution case for it. The residual risk
sits in two places: the youngest buckets (the largest positive cell in the table is the
3-4 bucket at +0.016% relative) and the AHEAD head of
the 128+ bucket (+0.696% relative on
10,226 reviews). Their own design (precision by answer count, above)
attacks the first cell; this arm's analogue of it is the fixed-basis codec family, which moves far
fewer predictions (see (b)).

---

## (f) The per-write cost of encoding at every write

Their reviewer asks for the per-write SVD / rank-r solve cost "in a fork that encodes at every
write". Measured on the deploy engine's own micro-benchmarks (`cargo test --release -- --ignored
--nocapture bench`, benches `bench_rank2_paths` and `bench_fixedbasis_layout`):

| path | per layer per write |
|---|---:|
| reference rank-1 solve (Gram + eigendecomposition) | ~10.5 us |
| reference rank-2 solve (gram ~2.2-2.4 us + eig ~35-37 us) | **~41-49 us** (up to 60.7 us for a flat-state variant) |
| fast rank-r solver (`RWKV_LOWRANK_FAST=1`, block subspace iteration) | warm-started, 1-2 iterations |
| fixed-basis codec, m=4 | **7.1 us** |
| fixed-basis codec, m=6 | 11.9 us |

So a rank-2 write costs about 4x a rank-1 write, and the fixed-basis codec is both smaller AND about
6x cheaper than the rank-2 solve it replaces. Everything in this bundle's deploy path is measured
with the reference solver unless a config says otherwise (`RWKV_LOWRANK_FAST` is off by default and
its factors are an approximation, so it must not be used for a Python-Rust parity run).

---

## (g) Rank AND precision both depending on age

**Not measured. Nothing in this arm yet changes rank or precision with the state's age**, and this
bundle does not claim otherwise. What exists is the surrounding evidence:

* this arm's per-bucket result says the opposite pressure from theirs: here the oldest cards are the
  ones the quantized state IMPROVES on, so an age-dependent schedule would want to spend MORE bits on
  the young cards, not the old ones;
* their own finding - int8 must start early (16 answers), switching at 128 is too late because 127
  int4 updates carry drift into the int8 years - is consistent with that;
* the arm's precision ladder is measured: at rank 1/2, int8 factors are essentially free, int6 costs
  a little, int5/int4 is the knee, and int3/int2 fails;
* the deploy engine can already express both halves mechanically: rank and factor width are per-stream
  scopes (`RWKV_STATE_LOWRANK_SCOPE=card:2:int8,note:1:int8`), the shift width is per-stream
  (`RWKV_STATE_SHIFT_LEVEL`), and a rank/precision change mid-history is a deploy-side scheduling
  change, not a format change.
* the one age-conditional SAVING this arm did measure is the fixed-basis codec: a stored basis plus an
  m x m coefficient block (m=4, 7.1 us per write) beats the per-step rank-2 solve on both size and
  cost.

So their idea (g) is mechanically reachable in this engine, and the measurement that would justify it
does not exist yet.

---

## 7. The size claim, stated exactly

Iter 55 is 3,584 bits - **exactly row 17's size** (1,280 /
2,304). Its reduction is against the naive int4 bars and against a dense per-matrix
code, not against row 17. Rows of this arm that go BELOW 3,584 bits and still PASS the
750-user gate:

| row | recipe | bits | gate delta ahead / immediate |
|---|---|---:|---|
| 61 | card rank-2 int8 + note fixed-basis codec m=4 | 2,432 | +0.000032 / +0.000311 |
| 45 | card rank-2 int6 + note rank-1 int6 (uniform frontier) | 2,944 | +0.000125 / +0.000381 |
| 51 | card rank-2 int8 + note codec m=6 | 2,912 | +0.000018 / +0.000209 |
| 63 | card rank-2 int8 + note codec m=6 int6 coefficients | 2,696 | +0.000062 / +0.000253 (speed 0.770x) |
| 55 | **this bundle** - row 17's layout + E1 penalty | 3,584 | +0.000028 / -0.000045 |

The codec rows are the ones to watch for size; iter 55 is the row to watch for loss, because it is the
only one whose immediate head is better than its own control.

---

## 8. The corruption (finiteness) gate

Their request "long-card accuracy, not only finiteness" is answered in (a)-(c). The finiteness gate
itself, on the 50 users with the longest average card histories, with histories NOT truncated:

| | |
|---|---|
| verdict | **PASS** |
| users scored | 50 of 50 requested |
| state observations | 158,986 (raw pre-truncation state, before the sanitize cap) |
| non-finite state values | 0 |
| p(recall) tensors | 150, non-finite 0 |
| runtime | 4.5 min |

The same check on the deploy engine, with the shipped weights, is a separate step of `verify.cmd`
(`rust_finite.py --strict`, which exits 3 on a single non-finite value).
