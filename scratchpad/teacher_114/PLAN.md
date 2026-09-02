# Can the d=128 teacher survive the move to 114-dim features? — a screen, before any run is built

## The problem the features phase inherits

`featB` ran **KD-OFF**, and not by choice: the d=128 teacher's `features2card.0.weight` is
`(512, 92)` and cannot forward a 114-dim batch at all. But KD is worth roughly **+0.0019** to this
lineage — iters 32, 35, 39 and 45 are all distillation accepts, and it is the only family with a
real hit rate (4/5 on the external-teacher sub-family).

So a gen-4 features champion trained the way featB was **forfeits ~0.0019** to buy the features'
~+0.0024 imm. That is a bad trade to make silently when it may not be necessary.

## The candidate fix, and why it is not obviously safe

The 114-dim layout is **not** the 92-dim layout plus 22 columns. `RWKV_ID_FEATURES=1` **drops**
`scaled_state` (index 22) and appends 23 new columns. So the teacher's input projection can be
re-laid-out by NAME — copy each of its 92 columns to that name's slot in the 114 layout, zero the
new ones — and it then computes exactly what it always did, **except that it no longer receives
`scaled_state`, a feature it was trained with.**

That is the whole question: **how much does the teacher lose by not seeing `scaled_state`?**

* small → re-lay-out the teacher, regenerate the dump, keep KD, keep the ~0.0019.
* large → the teacher is genuinely degraded, and KD-off (or a different teacher) is the honest
  choice.

## The measurement

Two arms, same checkpoint, same users, same published data — the data the teacher was trained
for, so nothing else moves:

| arm | env | meaning |
|---|---|---|
| A | `RWKV_ARCH_MODULE=…architecture_old_d128.py` | the teacher as it has always run |
| B | A + `RWKV_ZERO_FEATURES=22` | the same teacher with `scaled_state` removed |

`B − A` is exactly the cost of the re-lay-out, measured directly rather than argued.

**`RWKV_ARCH_MODULE`, not the file swap.** `run_base5k_eval.cmd` copies `architecture_old_d128.py`
over `rwkv/architecture.py` and restores it afterwards; a crash mid-run leaves the working tree on
the wrong architecture. The env override exists specifically to replace that footgun, and gen 4 is
building in the same tree.

**Zeroing an input column is exactly equivalent to deleting the feature**, because `y = Wx + b` is
linear in `x` — the same identity `model.rs` relies on for `RWKV_ZERO_FEATURES` at load.

## Pre-registered

I expect **a small loss, under 0.002 on imm**. `scaled_state` is Anki's own card-state enum, and
Andrew's rebuild directive drops it precisely because the timestamp features are expected to
subsume what it carries. If instead it costs more than ~0.004, the teacher is materially crippled
and the re-lay-out should not be used.

⚠ This bounds the teacher's degradation, **not** the KD gain that survives it. A teacher that is
0.002 worse does not necessarily distil 0.002 worse — KD here pays through target-variance
reduction, which is not linear in teacher quality. A small result licenses building the arm; it
does not predict its size.

300 users (5001–5300), which resolves a 0.002 effect comfortably while leaving the GPU free for
gen 4's successor.
