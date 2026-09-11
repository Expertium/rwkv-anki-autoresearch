# lfd3 / lfd1 -- the LEAK-FREE decay-length pair: PRE-REGISTRATION

Written 2026-09-11 ~12:00, **before w10lf has started** (it runs behind w10clk and cmixgraft, ~09-12
evening to ~09-14 evening) and so before any leak-free number exists.

## Why it exists, and why it runs first after w10lf

The 10+2 WS:decay split was chosen on COST (6+6 = 89 h vs 10+2 = 53 h). The ~+0.0006 credited to a
longer decay came from iter 34, where the decay-ratio change also changed the total budget, so it is
"being SPENT, not disproven". A shared WS turns the question into two decay-only branches with an
exact control. The pair moved from ws10 to w10lf after phase L said the clock leak is MATERIAL
(+0.0032 imm on arm 1): on a leaked base, part of any imm gain from a longer decay can be the model
learning the leak better.

It runs FIRST after w10lf because the length it picks, L*, is the length the leak-free QAT arms use
(the QAT phase IS the decay). Choosing L* first means no QAT arm has to be re-run at a new length.

## What it is

| run | decay | from | eval | control |
|---|---|---|---|---|
| `lfd3` | 3.0 epochs = 32,805 steps | `w10lf_ws_109350` | rectified VAL half, n = 2,499 | `w10lf` (2.0 epochs) |
| `lfd1` | 1.0 epoch = 10,935 steps | `w10lf_ws_109350` | rectified VAL half, n = 2,499 | `w10lf` |

Both carry `RWKV_CLOCK_AT_PREV_ANSWER=1800` in the decay and the eval, like w10lf. Each differs from
w10lf ONLY in decay length (`mk_lf_branch.py` asserts the env equals w10lf's). Total budget varies
with decay length, so this is the endgame's operational question -- "how long should the decay be on
this WS?" -- and NOT the fixed-total-budget WS:decay de-confound, which stays Andrew's call.

## Predictions (delta = w10lf minus branch, positive = the branch is BETTER)

The only in-repo evidence on this trunk: arm 1's decay moved 10-user validation by ~-0.0001 ahead and
~-0.0012 imm, i.e. the decay pays the rating head and barely touches the curve head.

* **P1 `lfd3`:** ahead **-0.00005 .. +0.00020**; imm **+0.00005 .. +0.00060**. A longer decay helps
  imm more than ahead, if it helps at all.
* **P2 `lfd1`:** ahead **-0.00015 .. +0.00005**; imm **-0.00080 .. -0.00010**. Half the decay loses
  some of the imm the decay converts.
* **P3 shape:** the lfd3 gain is smaller than the lfd1 loss in imm (diminishing returns per epoch).
* **P4 size:** 0/2,499 against w10lf for both (same dbs, same label filter).
* **P5 engagement:** both decay logs carry the clock-fix ON banner and the chunk-count line (runner
  guards 56/57).

## The rule that picks L* (applied mechanically, from `paired_pvalue.py` on the 2,499 users)

1. **L* = 3** iff lfd3 beats w10lf by raw >= +0.0001 in BOTH modes with one-sided paired Wilcoxon
   p < 0.0001 in both (the research gate's own bar).
2. Else **L* = 1** iff lfd1 is NOT worse than w10lf beyond the noise floor in EITHER mode, i.e.
   delta >= -0.000075 in both. Reason: the QAT phase IS the decay, QAT runs at ~0.25-0.33 steps/s, and
   its tax closes by ~0.37 epochs -- so a 1-epoch decay saves ~10 h on EACH leak-free QAT arm.
3. Else **L* = 2** (w10lf's own length; nothing is re-run).

Recorded as a verdict line in this file when the pair reports. A result that makes the rule
ambiguous (for example lfd3 passes on one mode at p ~ 1e-4) goes to Andrew rather than being forced.

## Cost

lfd3 ~9.5 h decay + ~4 h eval; lfd1 ~3.2 h + ~4 h; two dry runs ~20 min. ~21 h in total.
