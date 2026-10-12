# Pre-registration: `cmixgraft` -- does coarse-stream capacity bind at the real budget?

Written 2026-09-10, before any GPU time and before any number exists.

## Why it exists

Arm 1's Q2: at 12 epochs our 563k model is **0.002954 BEHIND** the old 2.76M d=128 model on ahead,
while **beating it by 0.001520 on imm**. Every capacity reject in the record (iters 49, 60, 61,
"capacity-at-5k 0/3") was measured at ~1.25 epochs -- a budget arm 1 has now shown cannot expose a
capacity limit. So capacity is **unresolved at the real budget**, not closed.

Andrew, 2026-09-10: *"It would be nice to try to grow deck/preset/global, but 38 h is a pretty
steep price"*, and separately *"I'd rather not increase card/note state size"*. Both constraints are
satisfied here: gate #3 has always allowed deck/preset/global to grow freely because those streams
are per-deck and per-user, not per-card, so **the 9 B/card and 27 B/note deploy contract is
untouched**; and the graft costs a decay, not a WS phase.

## The lever

`RWKV_STRIP_CMIX` drops from
`user_id:0,1,2 preset_id:0,1,2 deck_id:1,2 card_id:1` to `deck_id:1,2 card_id:1` -- restoring the
**six** user/preset channel mixers. **563,652 -> 641,862 params (+78,210, +13.9%).**

⚠ This is iter 49's lever **extended**, not repeated: iter 49 restored the two LAYER-0 mixers
(+26,070, +4.7%). Six is a bigger dose, chosen because a 2-epoch decay is a weak trainer of new
parameters and a larger dose gives the test more power.

## Why it costs 10.4 h instead of 38-60

The graft is **function-preserving at init**. A channel mixer returns `in_BTC + dropout(W_v(...))`
and `RWKV7ChannelMixer.__init__` already zeroes `W_v`, so a fresh mixer is the identity. A stripped
mixer is not absent from the checkpoint either -- `rwkv_model.py:578-581` builds it from a
`d_model=1` dummy config precisely to keep checkpoint interchange working -- so grafting is a SHAPE
replacement of 30 tensors, not a missing-key fill.

**Verified by execution, both halves** (`smoke_graft.py`, two separate processes because the arch
flags are read at import):

* graft vs arm 1 over 300 deploy predictions x 2 heads: **max |diff| = 0.000e+00**;
* a PERTURBED control (one restored `W_v` set to 1e-3) differs at **1.132e-06**, so the restored
  mixers really are executed and the identity above is not vacuous.

Artifact: `scratchpad/ws10/w10g_ws_109350.pth`, built by `expand_ckpt.py`, which refuses unless
exactly six mixers change shape and all six `W_v` are exactly zero.

## ⚠ The OPTIMIZER state had to be grafted too (found 2026-09-10 12:40, before any GPU)

A decay branch loads the WS checkpoint's optimizer state as well as its weights, and
`expand_ckpt.py` wrote only the weights -- so the decay would have died at load
(`train_rwkv.py:709` loads `<ws>_<step>_optim.pth` with no existence check). ws10's optimizer file
could not simply be copied either: `get_optimizer` groups by `len(param.squeeze().shape) >= 2`, a
stripped mixer's `W_k`/`W_v` are `(1, 1)` dummies in the AdamW `other` group, and the restored
`(80, 80)` matrices belong in the Muon `channel_mixer` group. The group sizes differ, so
`load_state_dict` refuses, and a positional copy could have attached moments to the wrong tensors.
`expand_optim.py` rebuilds both optimizers with the REAL `get_optimizer` (env parsed from each
runner file; they differ ONLY in `RWKV_STRIP_CMIX`) and maps ws10's state BY NAME: **391 params
keep their moments bit-for-bit, the 30 grown tensors start fresh, and exactly the 12 restored
`W_k`/`W_v` move AdamW -> Muon** -- the treatment every surviving channel mixer already gets, so it
is part of "having a mixer", not a second variable. Verified in a third process: the state loads,
every carried entry equals ws10's by name, and one synthetic step gives the grown tensors
correctly sized state. Output `scratchpad/ws10/w10g_ws_optim_109350.pth`, which
`write_decay_setup.py` copies to the resume name.

## ⚠ The second variable, priced rather than hidden

`Block.forward` wraps the mixer's output in `self.dropout` (`dropout_layer`), while the STRIPPED
branch returns `x_BTC` with no dropout. `DROPOUT_LAYER = 0.01 * 0.5 = 0.005` here, so restoring a
mixer also switches on **p=0.005 layer dropout on six of thirteen layer-steps**. The graft is
identity in EVAL mode exactly (which is what the smoke measures) and identity-plus-0.005-dropout in
TRAIN mode.

Not "fixed", because every available fix is a bigger variable than the bug: zeroing `dropout_layer`
for those blocks is a code change no other run has, and moving `RWKV_DROPOUT_SCALE` changes the
whole model. **The direction is favourable**: extra dropout is extra regularisation, and arm 1's own
gap screen says this model is FIT-limited at this budget, so it should if anything HURT the
treatment arm.

## Predictions, recorded before the run

* **P1 (the question).** Capacity binds at the real budget => the graft improves **ahead** by
  >= +0.0003 vs arm 1. That is 3x the accept bar, chosen because a 2-epoch decay can only realise
  part of what a full-budget capacity change would.
* **P2.** imm moves less than ahead, and may not move at all. Arm 1 showed imm is already ahead of
  the 2.76M model, so the capacity deficit is an ahead-side phenomenon.
* **P3 (engagement, to be measured BEFORE the number).** The six restored `W_v` must be non-zero at
  the end of the decay, with a Frobenius norm at least 10% of a neighbouring trained `W_v`. If they
  are still ~zero the decay never used the capacity and the run is uninterpretable, not a null.
* **P4 (falsifier).** ahead within +/-0.0001 of arm 1 WITH P3 satisfied => two epochs of coarse
  capacity buys nothing, and the full-budget version (option C, 48-60 h) needs a better argument
  than this one before it is worth Andrew's GPU.

## The asymmetry, stated before the result rather than after

**A WIN IS DECISIVE; A NULL IS WEAK.** Two epochs is much less than twelve for brand-new
parameters, so this run bounds capacity's value from BELOW. A null must never be quoted as
"capacity does not bind" -- that is precisely the error iter 49 already caused once, and the reason
this question is open at all.

## Gate

Both-modes rule vs arm 1 (a trunk capacity change can move either mode). Control is arm 1 exactly:
same WS checkpoint, same 21,870-step decay, same env but for `RWKV_STRIP_CMIX` and the checkpoint
prefix. Not a champion candidate -- the gen-5 lineage's champion is decided on the research gate,
and this is an endgame branch.

## PREVIEW at step 8,000 of 21,870 (2026-09-12 12:50) -- NOT the verdict

10-user validation (row-weighted, 4 dp) against arm 1 at the same steps, from the two decay logs:

| step | cmixgraft ahead / imm | arm 1 ahead / imm | delta |
|---|---|---|---|
| 50 | 0.3220 / 0.2968 | 0.3221 / 0.2968 | -0.0001 / +0.0000 |
| 2000 | 0.3222 / 0.2968 | 0.3222 / 0.2968 | +0.0000 / +0.0000 |
| 4000 | 0.3204 / 0.2969 | 0.3208 / 0.2968 | -0.0004 / +0.0001 |
| 6000 | 0.3216 / 0.2967 | 0.3210 / 0.2966 | +0.0006 / +0.0001 |
| 8000 | 0.3218 / 0.2960 | 0.3219 / 0.2960 | -0.0001 / +0.0000 |

The deltas oscillate around zero with no trend: no visible capacity effect so far. This instrument
predicted arm 1's gate deltas within 0.0004 (2026-09-10), so an effect of the size the PREREG hopes
for on ahead (~0.001+) would likely show by the decay's end. The gate on 2,499 users decides; a null
stays a WEAK null (two epochs for new parameters).

## PREVIEW at the END of the decay (step 21,870, 2026-09-12 18:05) -- NOT the verdict

Late 10-user validation deltas vs arm 1 (cmixgraft minus arm 1; positive = cmixgraft WORSE):
step 10,000 +0.0002 / -0.0002, 12,000 +0.0004 / +0.0000, 14,000 +0.0007 / -0.0001, 16,000 +0.0001 /
+0.0000, 18,000 +0.0003 / +0.0000, 20,000 +0.0003 / -0.0001, **21,870 +0.0003 / -0.0001** (ahead / imm).
From step 10,000 on the ahead delta is positive at every point: the early read is **ahead ~0.0003
WORSE, imm equal**, i.e. a null or a small loss, not the win P1 hoped for. Read it with the PREREG's
two caveats: the restored mixers also switch on p=0.005 layer dropout on six blocks (which should hurt
the treatment), and 2 epochs is short training for new parameters -- so a null here is WEAK.

## VERDICT (2026-09-12 21:05, `verdict_cmixgraft.py`, n = 2,499, size 0/2,499) -- A WEAK NULL

| mode | arm 1 | cmixgraft | gain | one-sided Wilcoxon (candidate better) |
|---|---|---|---|---|
| ahead | 0.297577 | 0.297492 | **+0.000085** | p = 0.948 (47.2% of users better, median gain -0.000022) |
| imm | 0.262066 | 0.262105 | **-0.000039** | p = 1.0 (39.0% better) |

* **P3 HOLDS (read first):** all six restored `W_v` are engaged, ||W_v||_F at 19.8-31.2% of the
  neighbouring trained median (21.11) against a 10% bar. The run is interpretable.
* **P1 FAILS:** ahead +0.000085 against the >= +0.0003 bar. **P2 holds** (|imm| < |ahead|).
* **P4 FIRES:** ahead within +/-0.0001 with P3 satisfied -- two epochs of coarse-stream capacity buy
  nothing measurable.
* **Where the +0.000085 sits (size quartiles, ahead):** Q1 +0.000156 (50.7% better), Q2 +0.000136,
  Q3 +0.000068, **Q4 -0.000017 (45.1% better)**. So the mean is carried by the smallest, noisiest users,
  and the users with the most coarse-stream history -- where user/preset capacity should matter most --
  gain nothing. imm is slightly worse in every quartile, most often for the largest users (Q4: 31.7%
  better), which fits the favourable-direction caveat above: the restored mixers also switch on p=0.005
  layer dropout on six layer-steps.
* **Read it with the asymmetry stated before the run: a WEAK null.** It bounds user/preset channel-mixer
  capacity from below at a 2-epoch decay; it does not say "capacity does not bind". It also does NOT
  test the card stream, which is where the per-user gap to the d=128 model grows (with reviews per
  card); the card grafts `lfcmf2` / `lflora16` test that, on w10lf's WS.
* Consequence for option C (the 48-60 h full-budget capacity run): per P4 it needs a better argument
  than this run before it is worth the GPU.
