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
