"""Greedy coordinate-descent HP tuner for the MAX=65536 era -- REBUILT 2026-07-30.

WHY IT WAS REBUILT: the previous version targeted the d=32 H=2/K=16 arch at MAX=110000, ran
QUANT-AWARE throughout, used a 2-epoch WS budget and evaluated on users 101-200. Every one of
those is now wrong. The trunk is the d=80 A18 lineage (iter 31/32), training is PLAIN (QAT has
been parked since iter 14), WS is FIXED at 1 epoch (Andrew 2026-07-09) and the gate metric is
the RECTIFIED logloss (Andrew 2026-07-27).

WHAT IT IS FOR: the 2026-07-30 speedup phase adopted MAX_TRAIN_GLOBAL_LEN = 65536, which is
1.61x faster but drops the group count 22,346 -> 10,935 -- i.e. HALF the optimizer steps per
epoch at unchanged LR. That cost -0.000264 ahead / -0.000307 imm vs iter 31 rectified (both
modes the same direction = a real systematic loss, not seed noise). Andrew accepted the speed
and directed this tuner to recover the accuracy. Hence the lever order below is NOT the old one:
the batch doubled, so the LEARNING RATES come first.

RECIPE PER TRIAL (mirrors scratchpad/maxval/run_maxval.cmd, the validated MAX=65536 run):
  40-step sanity (VRAM + proves the speed flags actually engaged)
  -> WS 1 epoch, train users 1-5000, train_db_5k_h1, MAX=65536, NUM_FETCH_PROCESSES=2
  -> write_decay_setup -> cosine decay (WS x decay_ratio epochs)
  -> write_eval_toml -> RECTIFIED eval (RWKV_EVAL_PAVA=1) on the tune-eval subset 5001-6000
  -> self-record to the journal.
Cost ~4.3 h/trial (WS 2h37 + decay ~40 min + eval ~1 h; the subset is 51.0M of the VAL half's
128.8M reviews).

THE BASELINE IS FREE: the `maxval` run IS the default config, and its rectified jsonls already
cover 5001-7500. Restricted to 5001-6000 it scores ahead 0.299250 / imm 0.266335, seeded into
the journal as the baseline row -- so trial 1 is a real probe, not an anchor re-run.

TUNE-EVAL SUBSET: 5001-6000 (1000 users), the post-champ5k_t1 remedy -- the old 200-user subset
could not resolve sub-0.001 effects and inverted at n=5000. Sanity check on this subset: it
ranks maxval vs iter 31 the same way the full VAL half does (subset +0.000113/+0.000309 vs full
+0.000264/+0.000306), so it is a usable proxy. Any sub-0.001 winner STILL needs confirming on
the full VAL half (5001-7500) before it becomes the recipe.

VALIDATION-BASED EARLY PRUNING is on, against optimization/tuner65k_vprune_ref.json (built from
maxval's own val trajectory + its 5001-6000 finals -- a matched reference on this exact trunk
and batch size). The trainer aborts (exit 42) iff BOTH modes' val loss exceed the reference by
>= 0.004 ahead AND 0.006 imm at 2 consecutive checkpoints. min_step = max(1000, 2 x the trial's
warmup) so a long-warmup trial is not killed for being slow by construction. This is the
sign-correct rule for regularization levers (the train-loss rule is not -- see the
decay_ratio_0p1 false-kill audit). It matters most for the LR grid, where a 2.8x LR probe can
diverge and would otherwise burn 4.3 h.

Objective minimized = ahead + imm (rectified, by-user mean on 5001-6000).
CLI: next / record <name> / record-pruned <name> / record-baseline <ahead> <imm> / status / loop.
"""
import json
import os
import subprocess
import sys

ROOT = "C:/Users/Andrew/rwkv-anki-autoresearch"
JOURNAL = f"{ROOT}/optimization/tuner_5k_log.jsonl"
TRIAL_DIR = f"{ROOT}/scratchpad/tuner65k"
VPRUNE_REF = f"{ROOT}/optimization/tuner65k_vprune_ref.json"

# Verified from the maxval run: its WS final checkpoint is mvws_10935.pth, i.e. 1 epoch of
# train_db_5k_h1 (users 1-5000) at MAX=65536 = 10,935 groups. (The old optimization/groups_5k.json
# says 6554 -- that is MAX=110000 and is stale; it is not read any more.)
GROUPS_PER_EPOCH = 10935
WS_EPOCHS = 1                      # FIXED (Andrew 2026-07-09, the champ5k_b1 budget A/B)
TRAIN_DB = "train_db_5k_h1"
USTART, UEND = 1, 5000
EVAL_USTART, EVAL_UEND = 5001, 6000
NUM_FETCH = 2                      # adopted 2026-07-30: halves the 3.8 GB/h RAM climb, fetch is
                                   # 2.3 ms of a ~1,450 ms step so it costs nothing
MAX_LEN = 65536                    # the adopted batch dim
VALIDATE_EVERY = 1000              # MUST match the vprune ref's cadence (pairing is by exact step)

# The iter-31/A18 trunk. Every run on this lineage sets all of these; a missing flag silently
# trains a different model, so they live in one string used verbatim by every trial.
TRUNK_ENV = (
    "set RWKV_DETERMINISTIC=1\n"
    "set RWKV_AUGMENT_SEED=1234\n"
    "set RWKV_EMPTY_CACHE_EVERY=1\n"
    "set RWKV_EMPTY_CACHE_WINDOW=0\n"
    "set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4.py\n"
    "set RWKV_GRU_HEAD=3\n"
    "set RWKV_PAVA_LAMBDA=0.1\n"
    "set RWKV_PROBE_DENSITY=0.08\n"
    "set RWKV_PROBE_DUR=0.0\n"
    "set RWKV_MUON=1\n"
    # ⚠ RWKV_MUON_MOMENTUM deliberately NOT set here. It lived in this trunk block until
    # 2026-08-03, while the per-trial block ALSO set it -- so every trial emitted it twice.
    # That was benign only because cmd.exe takes the LAST `set` and the per-trial block comes
    # after (verified: the momentum=0.9 trial logged "momentum=0.9"). But it is exactly the trap
    # that silently flattens a coordinate: reorder the template so the trunk block lands second
    # and every momentum trial runs at 0.95, producing a clean null with no error anywhere.
    # The per-trial line always emits, defaulting to 0.95, so removing this is byte-identical.
    "set RWKV_NO_AHEAD_RESIDUAL=1\n"
    "set RWKV_STRIP_L0_VLORA=1\n"
    "set RWKV_ZERO_FEATURES=22\n"
    "set RWKV_STATE_CLAMP_TAU=300\n"
    "set RWKV_STATE_CLAMP_WINDOW=32768\n"
    "set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
    "deck_id:1,deck_id:2,card_id:1\n"
)
# The speed stack adopted 2026-07-30 (1.68x, validated accuracy-neutral). Defaults are OFF in
# code, so these must be set explicitly. Cleared before eval, exactly as run_maxval.cmd does.
SPEED_ENV = ("set RWKV_MUON_BATCHED=1\n"
             "set RWKV_NO_JIT=1\n"
             "set RWKV_QAT_COMPILE=1\n")

BASE_PEAK_LR = 1e-3      # the AdamW group's base LR (57,412 params)
BASE_MUON_LR = 0.02      # the Muon groups' base LR (500,800 matrix params -- the bulk of the model)

# (param, grid). ORDER IS THE COORDINATE ORDER and it is deliberate: MAX doubled the batch, so
# the learning rates are the levers with a mechanistic reason to have moved. Everything after
# them is the usual robustness sweep.
SPACE = [
    # 200 steps was 0.9% of a 22,346-step epoch and is now 1.8% of a 10,935-step one. Bigger
    # batches usually want proportionally MORE warmup, not less; upstream used 20,000.
    # RESOLVED 2026-07-31: 400 wins, interior optimum (ahead is an inverted U), no extension needed.
    ("warmup_steps",  [200, 400, 800]),
    # Muon carries ~90% of the parameters, so re-balance its share against AdamW's at fixed overall
    # scale. muon_lr = BASE_MUON_LR * lr_mult * muon_lr_mult.
    # ★ 0.5 was the phase's big win (+0.000601/+0.000371 incremental, first trial to clear the bar
    # in both modes), and it sits on the LOW EDGE -- hence 0.25, added 2026-07-31 on Andrew's call
    # to probe it now rather than after the remaining coordinates.
    # ⚠ 2.0 was DROPPED, not silently skipped. It had been generated and ran ~25 min before being
    # stopped. Two independent results predict it is worse: the lr_mult coordinate is a 4-point
    # MONOTONIC dose-response in which more LR is always worse, and on this very lever 0.5 beat 1.0
    # by a wide margin. Spending 4.2 h to confirm that was worse value than 0.25 and lr_mult 0.7.
    # Add it back here if the shape is ever in doubt (warmup turned out non-monotonic, so shapes
    # are not always safe to assume).
    # ★★ 0.125 added 2026-08-01 -- SECOND consecutive edge win, and the two modes disagree about
    # whether it is exhausted. ahead is decelerating (1.0->0.5 gave +0.000601, 0.5->0.25 only
    # +0.000181) but imm is NOT (+0.000371 then +0.000411). imm was the mode still short of the
    # bar, so the lever that is still paying on it is worth one more probe.
    # ⚠ RESEARCH FLAG, not a tuning detail: RWKV_MUON_LR=0.02 was tuned at MAX=32768 and iter 29
    # accepted Muon on the strength of its imm gain. If 0.125 (muon_lr 0.0025, 8x below default)
    # keeps winning, the honest question becomes whether Muon still earns its place at THIS batch
    # size -- an iter-29-level question, not a coordinate. Raise it with Andrew; do not answer it
    # by letting the grid slide toward zero.
    ("muon_lr_mult",  [1.0, 0.5, 0.25, 0.125]),
    # Robustness levers. wd kept winning grid edges in the d=32 era (0.1, then 0.2), but that was
    # a different arch and 4x the params -- start from the champion 0.01 and probe upward.
    ("weight_decay",  [0.01, 0.05, 0.1]),
    ("clip",          [0.25, 0.5]),
    # ★ CEILING RAISED 2026-08-01 by Andrew ("extend it to 1/1.4") after 0.4 won both modes
    # (+0.000203/+0.000185) while sitting on the OLD ceiling. That old 0.4 = 1/2.5 was the top of
    # the sanctioned range in the 5k methodology (Andrew 2026-07-01), NOT an arbitrary grid choice
    # -- which is why it was escalated rather than extended the way muon_lr_mult was.
    # 0.7143 (= 1/1.4, rounded; the 1e-5 difference is ~0.1 step out of ~7,810) is probed BEFORE
    # 0.55 deliberately: if the new ceiling wins, the direction holds all the way and 0.55 only
    # fills in the curve; if it loses, an interior optimum exists and 0.55 is exactly the right
    # bisection. More information per trial either way.
    # ⚠ These trials cost MORE than the standard 4.2 h -- decay is WS x ratio, so 0.7143 means
    # ~7,810 decay steps (~1h44m) vs 0.25's 2,733 (~38 min). Budget ~5.1 h and ~4.8 h.
    # ★★ CEILING RAISED AGAIN 2026-08-02 by Andrew ("raise it to 1") -- 0.7143 won at the previous
    # ceiling and the lever shows NO sign of exhausting. All four measured points are strictly
    # monotonic in BOTH modes:
    #     0.25  0.298117/0.265373   0.40  0.297914/0.265188
    #     0.55  0.297648/0.265035   0.7143 0.297478/0.264877
    # ⚠ WHAT THIS LEVER ACTUALLY BUYS IS TOTAL TRAINING (WS 1 ep + decay = ratio ep), so 0.25 ->
    # 1.0 is 1.25 -> 2.0 epochs = 1.6x. That makes it partly a cheap early read on the ENDGAME's
    # "10x the epoch budget" hypothesis, and mild independent evidence the model is undertrained --
    # consistent with the +0.0037/+0.0043 gap to upstream's ~12 epochs.
    # Only 1.0 is added, not 0.85: four points on a straight line do not need more interior
    # resolution, they need to know where the line ENDS.
    # ⚠ COST: decay = WS x ratio, so ratio 1.0 is ~10,935 decay steps (~2h25m, as long as WS
    # itself) -> a ~5.8 h trial rather than 4.2 h.
    ("decay_ratio",   [0.25, 0.4, 0.7143, 0.55, 1.0]),
    # ★ lr_mult MOVED TO LAST and its grid REPLACED, 2026-07-31.
    # It originally ran FIRST with [1.0, 1.41, 2.0, 2.8] -- upward only, because the design
    # anchored on "the batch doubled, so the LR should rise". That heuristic was wrong in both
    # directions: raising the LR hurt monotonically across all four points, and the phase's biggest
    # win came from LOWERING an LR (muon_lr_mult 0.5). The original grid could not have found that,
    # since it contained no downward probe at all.
    # So the live question is the one it never asked: at the tuned config, does lowering the JOINT
    # LR help? Hence [1.0, 0.7], evaluated LAST so it sees the winning warmup/muon/wd/clip/decay.
    # The upward points are NOT re-run: that result is settled, monotonic over 4 points, and
    # recorded in optimization/TRAINING_SPEED.md. Their journal rows are untouched.
    # ⚠ Reordering is SAFE for the replay: coordinate descent evaluates each coordinate at the
    # incumbent of the ones BEFORE it, and every recorded row already has lr_mult=1.0 (the
    # default), so warmup/muon rows still match their configs exactly. Verified before relaunch.
    ("lr_mult",       [1.0, 0.7]),
    # ★ ADDED 2026-08-02 (Andrew: "let's add momentum and beta2 then").
    # Muon's momentum -- the direct companion of the lever that carried this round. Its LR was 8x
    # too high FOR THIS BATCH SIZE; momentum enters the effective step the same way (~lr/(1-m))
    # and governs the same 500,800 params, so the same batch change plausibly mis-set it too.
    # Bracketed in (1-m) = [0.2, 0.1, 0.05, 0.025], NOT geometrically in m: a quantity pinned near
    # 1 cannot be bracketed by ratios (bracket(0.95) would propose 1.9).
    ("muon_momentum", [0.95, 0.9, 0.8, 0.975]),
    # AdamW beta2. Bounded upside -- it governs only the 57,412 AdamW params, which is exactly why
    # lr_mult (moving both families) was a wash -- but 0.999 is a ~1000-step second-moment horizon,
    # ~9% of a 10,935-step epoch, so the estimate barely converges inside a 1-epoch budget.
    ("adamw_beta2",   [0.999, 0.98, 0.95]),
    # ★★ ADDED 2026-08-03, after Andrew: "We're looking for algorithmic improvements of any kind,
    # there is no reason to stop (other than exhausting this particular lever)."
    #
    # wd_head_mult -- weight decay for the SRS-HEAD group alone, as a multiple of `weight_decay`.
    # ⚠ THE FLAT `weight_decay` RESULT IS THE ARGUMENT *FOR* THIS, NOT AGAINST IT. Until today all
    # THREE wd groups (matrix decay / channel-mixer / head) read the SAME env var, so that
    # coordinate could only move them together. A flat response to a shared knob has two readings:
    # wd genuinely does not matter, or the groups want OPPOSITE things and cancelled. This model
    # has already produced exactly the second case once -- peak_lr looked tuned while governing
    # only 10% of the weights, and splitting Muon's LR off was worth +0.00183, the round's biggest
    # win. Same structure, so the same prior applies. Split wired 2026-08-03
    # (RWKV_WEIGHT_DECAY_HEAD / _CMIX, each defaulting to RWKV_WEIGHT_DECAY == byte-identical).
    # The head is probed first because it is the most functionally distinct group: a curve mixture
    # plus a rating head, not part of the recurrent trunk.
    ("wd_head_mult",  [1.0, 0.2, 5.0]),
    # dropout_scale -- ⚠ RE-PROPOSED ON PURPOSE. NEXT_ROUND_CANDIDATES below says "not worth
    # adding", on the grounds that wd (10x) and clip (2x) were both flat. That argument stands on
    # its own, but two things now outweigh it:
    #  1. ★ THE LEVER WAS A NO-OP ON THIS TRUNK UNTIL TODAY. architecture_d80_lora4.py HARDCODED
    #     the three rates when it was forked from rwkv/architecture.py, so RWKV_DROPOUT_SCALE was
    #     silently ignored. Adding this coordinate before the fix would have produced two
    #     BYTE-IDENTICAL trials and "confirmed" the family-is-flat hypothesis with zero
    #     information -- the trap is what makes it worth naming, not the lever.
    #  2. The budget changed underneath the flat results: decay_ratio 1.0 took total training from
    #     1.25 to 2.0 epochs, and wd/clip were measured at 0.25. Regularization is more live at
    #     1.6x the training.
    # Conduct rule 5 also applies: wd + clip = 2 in-family rejects = "deprioritized", not closed.
    # This is the third distinct variant, and it is a different MECHANISM (stochastic, not
    # shrinkage), so it is not redundant with the first two.
    ("dropout_scale", [1.0, 0.5, 2.0]),
]
DEFAULTS = {"lr_mult": 1.0, "warmup_steps": 200, "muon_lr_mult": 1.0,
            "weight_decay": 0.01, "clip": 0.25, "decay_ratio": 0.25,
            "muon_momentum": 0.95, "adamw_beta2": 0.999,
            "wd_head_mult": 1.0, "dropout_scale": 1.0}
PARAMS = [p for p, _ in SPACE]

# =============================================================================
# POLICY (added 2026-08-02, after the first full run). Everything here exists
# because the 2026-07-30..08-02 grid needed a HUMAN to intervene, four times, for
# things a rule can decide. Each entry names the incident it prevents.
# =============================================================================
SPACE_OVERRIDE = f"{ROOT}/optimization/tuner65k_space.json"

# Cross-seed spread on an IDENTICAL recipe, per mode (the seed-pair doctrine). The objective is
# ahead+imm, so its noise is ~2x this. Used to flag coordinate winners that are really coin flips.
SEED_NOISE = 0.0004

POLICY = {
    # --- auto edge-extension ------------------------------------------------
    # INCIDENT: muon_lr_mult won at 0.5 (the low edge), then 0.25, then 0.125 -- three manual
    # stop-edit-verify-relaunch cycles. decay_ratio did the same at 0.4 -> 0.7143 -> 1.0. Each
    # cycle threw away ~20 min of an in-flight trial and leaked orphaned fetch workers.
    "auto_extend": True,
    # Extend only while the edge win is bigger than objective noise -- otherwise the grid would
    # crawl outward on coin flips.
    # CALIBRATED, not guessed: replayed against all 11 extend/stop decisions this grid actually
    # made. 0.0008 agreed on 7, 0.0005 on 8, 0.0002 on 8; 0.0003 agrees on 11 of 11 (with the
    # bound-clamp below). Re-run the replay in the docstring of tools/replay if this changes.
    "extend_min_gain": 0.0003,
    # ...and only while the gain is not collapsing. INCIDENT: I stopped muon_lr_mult at 0.125 by
    # eyeballing "imm increments went +0.000411 -> +0.000084"; decay_ratio's per-unit slope halved
    # twice before I called it. This is that judgement, written down: require the newest increment
    # to be at least this fraction of the previous one.
    "extend_saturate_frac": 0.45,
    "extend_max_per_param": 4,
    # ⚠ BOUNDS ARE A HUMAN DECISION AND THE TUNER NEVER CROSSES THEM. This is what makes
    # auto-extension safe: decay_ratio's 0.4 ceiling was Andrew's documented methodology limit
    # (1/2.5), not a grid choice of mine, which is why raising it was escalated rather than
    # automated -- twice. Encoded here, the tuner extends freely INSIDE the sanctioned range and
    # stops at the edge with a message asking for a decision, instead of either crawling past a
    # policy limit or stalling on one that was never really a limit.
    "bounds": {
        "decay_ratio":  [0.1, 1.0],      # 1.0 = Andrew 2026-08-02; was 1/2.5 then 1/1.4
        "muon_lr_mult": [0.03, 4.0],
        "lr_mult":      [0.5, 3.0],
        "warmup_steps": [100, 3200],
        "weight_decay": [0.0, 0.5],
        "clip":         [0.05, 2.0],
        # momentum is bounded in m-space; the extender's geometric step is meaningless this close
        # to 1, so the bounds do the real work of keeping it sane.
        "muon_momentum": [0.5, 0.99],
        "adamw_beta2":   [0.9, 0.9995],
        # Multipliers on `weight_decay` / the base dropout rates. 0 is a legitimate endpoint for
        # wd_head_mult (no decay on the heads at all) but NOT for dropout_scale, where the extender
        # would then geometrically crawl toward 0 forever without ever reaching a decisive answer.
        "wd_head_mult":  [0.0, 20.0],
        "dropout_scale": [0.1, 5.0],
    },
    # --- housekeeping between trials ---------------------------------------
    # INCIDENT: killing a trial to change the grid orphaned its fetch workers; 12 had accumulated
    # across four relaunches, still holding LMDB mappings, on a box that had already hit 0.3 GB
    # free once that day.
    "reap_orphans": True,
    # --- honest cost reporting ---------------------------------------------
    # INCIDENT: the ETA said "4.2 h/trial" while decay_ratio=1.0 trials actually took 5.8 h,
    # because decay steps scale with the ratio. Nobody was misled for long, but the queue estimate
    # was quietly wrong for a day.
    "steps_per_sec": 1.253,              # measured on this trunk at MAX=65536
    "eval_hours": 1.0,                   # ~1000 users, rectified
    "sanity_hours": 0.05,
}


# =============================================================================
# CANDIDATES FOR THE NEXT TUNING ROUND (2026-08-02). NOT active -- adding them to
# SPACE now would make the loop demand ~7 more trials, and the plan is seed-pair
# then iter 35. Ranked by expected value, with the reasoning, so the next round
# starts from an argument instead of a blank page.
# =============================================================================
NEXT_ROUND_CANDIDATES = [
    # ---- 1. HIGHEST EV: Muon's momentum. Already wired (RWKV_MUON_MOMENTUM, default 0.95),
    # never tuned. It is the direct companion of the lever that carried this entire round.
    # Muon's LR turned out 8x too high FOR THIS BATCH SIZE, and momentum enters the effective
    # step size the same way lr does (~lr/(1-m)) while governing the same 500,800 params. If one
    # was mis-set by the batch change, the prior that the other is too is strong.
    # ⚠ PARAMETERIZE IN (1-m), NOT m: geometric bracketing of a quantity pinned near 1 is
    # meaningless (bracket(0.95) would propose 1.9). 1-m = 0.05 -> [0.0125, 0.025, 0.05, 0.1, 0.2]
    # i.e. momentum [0.9875, 0.975, 0.95, 0.9, 0.8].
    ("muon_momentum", [0.8, 0.9, 0.95, 0.975], "env RWKV_MUON_MOMENTUM; bracket in (1-m)"),

    # ---- 2. AdamW beta2. Wired (RWKV_ADAMW_BETA2), hardcoded 0.999 since forever, never tuned
    # on this trunk. Upside is bounded -- it governs only the 57,412 AdamW params, ~10% of the
    # model, which is exactly why lr_mult (which moved both families) was a wash. But 0.999 is a
    # ~1000-step second-moment horizon, ~9% of a 10,935-step epoch, which is slow for a 1-epoch
    # budget: the estimate barely converges before training ends.
    ("adamw_beta2", [0.95, 0.98, 0.999], "env RWKV_ADAMW_BETA2"),
]
# ---- NOT worth adding, and why (so they are not re-proposed):
#   * ~~dropout_scale -- weight_decay was flat across 10x and clip across 2x. Two independent nulls
#     in the regularization family is decent evidence the family is flat on this trunk.~~
#     ★ PROMOTED TO SPACE 2026-08-03. The reasoning above is still sound as far as it goes, but the
#     lever was a NO-OP on this trunk until that date (architecture_d80_lora4.py hardcoded the
#     rates), so adding it earlier would have run two byte-identical trials and "confirmed" the
#     flat-family read with no information at all. Plus decay_ratio=1.0 moved total training
#     1.25 -> 2.0 epochs, which is 1.6x more than where wd/clip were measured flat.
#   * ns_steps (Muon's Newton-Schulz iterations, muon.py default 5) -- a compute/accuracy
#     tradeoff, not an accuracy lever; 5 is the standard value.
#   * cb_lr_mult -- meaningless in plain training; it tuned codebook groups that no longer exist.
# ---- TWO THAT ARE METHODOLOGY DECISIONS, NOT COORDINATES (raise with Andrew, do not just add):
#   * WS_EPOCHS / where the budget is spent. decay_ratio 1.0 won, so total training is now 2.0
#     epochs, and the split between stable and decay has never been tested at that budget --
#     WS 1 + decay 1.0 vs WS 1.5 + decay 0.5 at matched cost. ⚠ AND THE CONSTRAINT PINNING WS=1
#     IS WEAKER THAN IT LOOKS: it rests on the champ5k_b1 A/B ("2nd epoch adds nothing"), which
#     CLAUDE.md itself flags was run with augmentation OFF -- i.e. with BYTE-IDENTICAL epochs,
#     the one configuration in which extra epochs CANNOT help. That null does not license WS=1
#     at a 2-epoch budget.
#   * Augmentation ON. Same argument from the other side: at 2.0 epochs the second pass is a
#     byte-identical replay, and augmentation is precisely the regularizer that regime wants.
#     ⚠ The cost is real though -- augmentation off is what gives ~zero run-to-run variance, and
#     this tuner's ability to resolve 0.0003 effects depends on it.


def bracket(x, steps=2, ratio=2.0, additive=False):
    """Grid that brackets the incumbent on BOTH sides. USE THIS when adding a lever.

    ★ THE MOST EXPENSIVE LESSON OF THE 2026-07-30 RUN. The `lr_mult` grid was [1.0, 1.41, 2.0,
    2.8] -- upward ONLY, because the design anchored on "the batch doubled, so the LR should
    rise". Four full trials (~17 h) went into a hypothesis that was not merely wrong but
    BACKWARDS: every upward point was worse, monotonically, and the phase's single biggest win
    turned out to be an 8x REDUCTION in Muon's LR. That win was reachable only by accident,
    because `muon_lr_mult` happened to contain 0.5 -- the one downward probe anywhere in the
    design. A grid built with this helper would have found it in trial 2 instead of trial 7.

    Guard against direction priors generally: if you believe a lever should move one way, that
    belief is exactly the thing the grid should be able to falsify.
    """
    if additive:
        return sorted({x} | {x + k * ratio for k in range(1, steps + 1)}
                      | {x - k * ratio for k in range(1, steps + 1) if x - k * ratio > 0})
    out = {float(x)}
    for k in range(1, steps + 1):
        out.add(round(x * ratio ** k, 6))
        out.add(round(x / ratio ** k, 6))
    return sorted(out)


def load_space():
    """SPACE/DEFAULTS, with an optional JSON override re-read on EVERY call.

    INCIDENT this fixes: changing the grid meant killing the loop AND the in-flight trial, editing
    this file, re-verifying the journal replay, and relaunching -- four times, each costing ~20 min
    of training plus a batch of orphaned workers. The loop now re-reads the space between trials,
    so a grid edit lands at the next trial boundary with nothing killed and nothing lost.
    Absent or unreadable file -> the in-code SPACE, so this cannot break a run.
    """
    if not os.path.exists(SPACE_OVERRIDE):
        return [(p, list(g)) for p, g in SPACE], dict(DEFAULTS)
    try:
        with open(SPACE_OVERRIDE) as fh:
            d = json.load(fh)
        space = [(e["param"], list(e["grid"])) for e in d["space"]]
        defaults = dict(d.get("defaults") or DEFAULTS)
        if not space:
            raise ValueError("empty space")
        # MERGE, don't blindly replace. Without this the JSON silently SHADOWS the in-code SPACE,
        # so adding a coordinate in code would do nothing and the tuner would quietly skip a lever
        # someone thought they had added -- a failure with no error message, which is the worst
        # kind. Params present in code but missing from the JSON are appended (in code order);
        # grids for params the JSON already has win, since those carry auto-extensions.
        have = {p for p, _ in space}
        for p, g in SPACE:
            if p not in have:
                space.append((p, list(g)))
                print(f"[space] '{p}' is in SPACE but not in the override -- appending it "
                      f"(the JSON was written before this coordinate existed)", flush=True)
        for p in DEFAULTS:
            defaults.setdefault(p, DEFAULTS[p])
        return space, defaults
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[space] override unreadable ({e!r}) -- falling back to the in-code SPACE",
              flush=True)
        return [(p, list(g)) for p, g in SPACE], dict(DEFAULTS)


def save_space(space, defaults, note=""):
    tmp = SPACE_OVERRIDE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"note": note or "written by hp_tuner_5k.py; edited between trials is SAFE "
                                  "-- the loop re-reads this file before each trial",
                   "defaults": defaults,
                   "space": [{"param": p, "grid": list(g)} for p, g in space]}, fh, indent=2)
    os.replace(tmp, SPACE_OVERRIDE)


def est_trial_hours(cfg, space=None):
    """Wall-clock estimate for ONE trial from its config. decay steps scale with decay_ratio, so
    a ratio-1.0 trial is ~40% dearer than a ratio-0.25 one -- the flat '4.2 h' the old status line
    printed was wrong for half this grid."""
    sps = POLICY["steps_per_sec"]
    ws = ws_steps()
    dec = int(ws * float(cfg.get("decay_ratio", DEFAULTS["decay_ratio"])))
    return (ws + dec) / sps / 3600.0 + POLICY["eval_hours"] + POLICY["sanity_hours"]


def reap_orphans():
    """Kill python fetch workers whose parent is gone. Returns the count.

    These accumulate whenever a trial is stopped (grid change, crash, manual abort) and they hold
    LMDB mappings that re-balloon, on a box whose RAM headroom is already the binding constraint.
    Deliberately narrow: only python.exe running multiprocessing's spawn_main, and only when the
    recorded parent pid no longer exists -- so it cannot touch Andrew's srs-benchmark run, the
    Reddit bot, the Telegram bridge, or a live trial."""
    if not POLICY["reap_orphans"] or os.name != "nt":
        return 0
    ps = ("$live=@{}; Get-CimInstance Win32_Process | ForEach-Object { $live[[int]$_.ProcessId]=1 };"
          "$n=0; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {"
          " $_.CommandLine -like '*spawn_main*' -and -not $live.ContainsKey([int]$_.ParentProcessId)"
          "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $n++ };"
          "Write-Output $n")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90)
        return int((out.stdout or "0").strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def peak_lr(cfg):
    return BASE_PEAK_LR * float(cfg["lr_mult"])


def muon_lr(cfg):
    return BASE_MUON_LR * float(cfg["lr_mult"]) * float(cfg["muon_lr_mult"])


def canon(cfg):
    # .get with the DEFAULT for every later-added lever: journal rows written before a lever
    # existed ran with the (env-unset ==) default value, so they canon onto the same point.
    return tuple(round(float(cfg.get(p, DEFAULTS[p])), 8) for p in PARAMS)


# THE BAR: "recover what MAX=65536 cost" == reach iter 31's numbers ON THIS SAME 1000-user
# subset. Computed 2026-07-30 by restricting result/RWKV{,-P}-iter31_algo_rect.jsonl to
# 5001-6000. Note the two modes are NOT equidistant from the baseline (0.299250/0.266335):
# MAX=65536 hurt imm ~2.7x more than ahead here, so a trial that recovers ahead alone has
# not done the job.
BAR = {"ahead": 0.299137, "imm": 0.266026}


def obj(rec):
    return rec["ahead"] + rec["imm"]


def load_journal():
    recs = []
    if os.path.exists(JOURNAL):
        for line in open(JOURNAL):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def find(recs, cfg):
    k = canon(cfg)
    for r in recs:
        if canon(r["config"]) == k:
            return r
    return None


def compute(recs, space=None, defaults=None):
    """Replay coordinate descent. Returns ('need', cfg, param) or ('done', best_cfg)."""
    space = space if space is not None else [(p, list(g)) for p, g in SPACE]
    defaults = defaults if defaults is not None else dict(DEFAULTS)
    best = dict(defaults)
    if find(recs, defaults) is None:
        return ("need", dict(defaults), "baseline")
    for param, grid in space:
        results = {}
        for v in grid:
            cfg = dict(best)
            cfg[param] = v
            r = find(recs, cfg)
            if r is None:
                return ("need", cfg, param)
            results[v] = obj(r)
        best[param] = min(results, key=lambda v: results[v])
    return ("done", best)


def coordinate_report(recs, space, defaults):
    """Per-coordinate: winner, its gain over the DEFAULT, and whether adopting it is justified.

    WHY vs the default and NOT vs the runner-up: the runner-up of a monotonic lever is its
    adjacent grid point, which is close BY CONSTRUCTION -- muon_lr_mult 0.125 beats 0.25 by only
    +0.000266, yet beats the default 1.0 by +0.00187. Scoring against the runner-up made every
    coordinate in this grid look like a coin flip, which is exactly backwards for the two levers
    that carried the whole result. The decision the tuner actually makes is "change this
    hyperparameter or keep it", so the honest test is winner vs DEFAULT.

    WHY flag it at all: this grid adopted warmup 400 over the default 200 on +0.000265, well
    inside the ~0.0008 objective noise floor (2 x the per-mode cross-seed spread). That went into
    the recipe looking like a finding. A tuned recipe should say which of its values are real and
    which are coin flips it happened to land on."""
    best = dict(defaults)
    out = []
    for param, grid in space:
        vals = []
        for v in grid:
            cfg = dict(best)
            cfg[param] = v
            r = find(recs, cfg)
            if r is not None:
                vals.append((obj(r), v))
        if not vals:
            out.append((param, None, None, None, None))
            continue
        vals.sort()
        win_o, win_v = vals[0]
        runner = (vals[1][0] - win_o) if len(vals) > 1 else None
        dflt = defaults.get(param)
        dflt_o = next((o for o, v in vals if v == dflt), None)
        gain = (dflt_o - win_o) if dflt_o is not None else None
        best[param] = win_v
        if gain is None:
            verdict = "default not measured"
        elif win_v == dflt:
            verdict = "UNCHANGED (default held)"
        elif gain >= 2 * SEED_NOISE:
            verdict = "REAL"
        else:
            verdict = "COIN FLIP (adopted inside noise)"
        out.append((param, win_v, gain, runner, verdict))
    return out, best


def _next_edge_value(grid, at_low):
    """Next point beyond an edge, by the grid's own geometric step. Falls back to a factor of 2."""
    s = sorted(grid)
    if len(s) < 2:
        return None
    if at_low:
        a, b = s[0], s[1]
        ratio = (a / b) if b else 0.5
        nxt = a * (ratio if 0 < ratio < 1 else 0.5)
    else:
        a, b = s[-1], s[-2]
        ratio = (a / b) if b else 2.0
        nxt = a * (ratio if ratio > 1 else 2.0)
    return round(nxt, 6)


def maybe_extend(recs, space, defaults, verbose=True):
    """If a COMPLETE coordinate was won at a grid edge and is still paying, append the next point.

    Returns a new space if it extended, else None. This is the rule that replaces four manual
    stop-edit-verify-relaunch cycles. It refuses to extend when:
      * the edge win is inside noise               -> the grid would crawl on coin flips
      * the gain is collapsing                     -> the lever is saturating (the judgement I made
                                                      by eye for muon_lr_mult and decay_ratio)
      * the next point would cross a declared bound -> a HUMAN decision; it says so and stops
      * the coordinate has already been extended max_extend times
    """
    if not POLICY["auto_extend"]:
        return None
    best = dict(defaults)
    for idx, (param, grid) in enumerate(space):
        measured = []
        for v in grid:
            cfg = dict(best)
            cfg[param] = v
            r = find(recs, cfg)
            if r is None:
                return None          # coordinate incomplete -- nothing to decide yet
            measured.append((v, obj(r)))
        measured.sort(key=lambda t: t[0])
        win_v = min(measured, key=lambda t: t[1])[0]
        best[param] = win_v
        if win_v not in (measured[0][0], measured[-1][0]) or len(measured) < 2:
            continue                 # interior optimum -> this coordinate is settled
        at_low = win_v == measured[0][0]
        ordered = measured if at_low else measured[::-1]      # winner first, walking inward
        gain = ordered[1][1] - ordered[0][1]                  # win vs its neighbour (positive=better)
        if gain < POLICY["extend_min_gain"]:
            if verbose:
                print(f"[extend] {param}: edge win {win_v:g} but gain {gain:+.6f} is inside noise "
                      f"({POLICY['extend_min_gain']:.6f}) -- not extending", flush=True)
            continue
        if len(ordered) >= 3:
            prev_gain = ordered[2][1] - ordered[1][1]
            if prev_gain > 0 and gain < POLICY["extend_saturate_frac"] * prev_gain:
                if verbose:
                    print(f"[extend] {param}: SATURATING (last gain {gain:+.6f} < "
                          f"{POLICY['extend_saturate_frac']:g} x previous {prev_gain:+.6f}) "
                          f"-- lever spent, not extending", flush=True)
                continue
        base_n = len(dict(SPACE).get(param, []))
        if len(grid) - base_n >= POLICY["extend_max_per_param"]:
            if verbose:
                print(f"[extend] {param}: already extended {len(grid)-base_n}x "
                      f"(max {POLICY['extend_max_per_param']}) -- stopping", flush=True)
            continue
        nxt = _next_edge_value(grid, at_low)
        if nxt is None:
            continue
        lo, hi = POLICY["bounds"].get(param, (float("-inf"), float("inf")))
        # CLAMP to the bound rather than refuse. The geometric step routinely overshoots a
        # sanctioned limit while the limit itself is still an untried, useful point: at
        # decay_ratio [0.25, 0.4, 0.7143] the step wants 1.276 but the bound is 1.0, and 1.0 was
        # exactly what Andrew asked for. Refusing there was the rule's only disagreement with the
        # human across all 11 decisions this grid made.
        nxt = min(max(round(nxt, 6), lo), hi)
        if nxt in grid or any(abs(nxt - g) < 1e-9 for g in grid):
            if verbose:
                print(f"[extend] ** {param} won at {win_v:g}, which is the edge of its sanctioned "
                      f"range [{lo:g}, {hi:g}], and the lever is STILL PAYING ({gain:+.6f}). "
                      f"This is a HUMAN decision -- widen POLICY['bounds'][{param!r}] if the "
                      f"range should change. **", flush=True)
            continue
        new_space = [(p, list(g)) for p, g in space]
        new_space[idx] = (param, list(grid) + [nxt])
        if verbose:
            print(f"[extend] {param}: won at the {'low' if at_low else 'high'} edge {win_v:g} "
                  f"with gain {gain:+.6f} -> appending {nxt:g}", flush=True)
        return new_space
    return None


def trial_name(param, cfg):
    if param == "baseline":
        return "t65_baseline"
    v = cfg[param]
    vs = f"{v:g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"t65_{param}_{vs}"


def ws_steps():
    return WS_EPOCHS * GROUPS_PER_EPOCH


def write_trial_files(name, param, cfg):
    folder = f"{TRIAL_DIR}/{name}"
    os.makedirs(folder, exist_ok=True)
    ws_ts = ws_steps()
    decay_ep = WS_EPOCHS * float(cfg["decay_ratio"])
    plr, mlr = peak_lr(cfg), muon_lr(cfg)
    pval_str = f"{cfg[param]:g}" if param in cfg else "baseline"

    # Stale-result hygiene, done HERE rather than in the .cmd. The .cmd retries a failed eval
    # WITHOUT deleting (eval_sharded skips users it already banked, which is what makes the
    # giant-user OOM recoverable -- the 2026-07-30 big-eval ops rule). Deleting at generation
    # time keeps that property while still guaranteeing a regenerated trial starts clean.
    for f in (f"{ROOT}/result/RWKV-{name}.jsonl", f"{ROOT}/result/RWKV-P-{name}.jsonl",
              f"{ROOT}/result/RWKV-{name}-s0.jsonl", f"{ROOT}/result/RWKV-P-{name}-s0.jsonl",
              f"{folder}/{name}_ws_trace.jsonl", f"{folder}/{name}_ws_trace.jsonl.val.jsonl",
              f"{folder}/{name}_ws_trace.jsonl.pruned.json"):
        if os.path.exists(f):
            os.remove(f)

    # vprune min_step: never kill a trial before 2x its own warmup (a long-warmup trial is worse
    # early BY CONSTRUCTION), and never before the documented floor of 1000.
    vprune_min = max(1000, 2 * int(cfg["warmup_steps"]))

    ws_toml = f"""# HP tuner (MAX=65536 era) trial {name}: param={param} -> {pval_str}
# Full config: {json.dumps(cfg)}  ->  PEAK_LR {plr:g}, RWKV_MUON_LR {mlr:g}
# Recipe = scratchpad/maxval/run_maxval.cmd (the validated MAX=65536 run) with the HPs swapped.
TRAIN_USERS_START = {USTART}
TRAIN_USERS_END = {UEND}
VALIDATE_USERS_START = 5001
VALIDATE_USERS_END = 5010

TRAIN_DATASET_LMDB_PATH = "{TRAIN_DB}"
TRAIN_DATASET_LMDB_SIZE = 400_000_000_000
VALIDATE_DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
VALIDATE_DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000

NUM_FETCH_PROCESSES = {NUM_FETCH}
MAX_TRAIN_GLOBAL_LEN = {MAX_LEN}

TRAIN_MODE = "WS"
STEP_OFFSET = 1
WARMUP_STEPS = {int(cfg["warmup_steps"])}
EPOCHS = {WS_EPOCHS}
VALIDATE_EVERY = {VALIDATE_EVERY}
PEAK_LR = {plr:g}

LOAD_MODEL = false
SAVE_MODEL_FOLDER = "scratchpad/tuner65k/{name}"
SAVE_MODEL_PREFIX = "{name}ws"
DEVICE = "cuda"
DTYPE = "bfloat16"

USE_WANDB = false
WANDB_PROJECT_NAME = "rwkv"
WANDB_RESUME = false
WANDB_RESUME_ID = ""
"""
    with open(f"{folder}/{name}_ws.toml", "w") as f:
        f.write(ws_toml)

    with open(f"{folder}/{name}.json", "w") as f:
        json.dump({"name": name, "param": param, "config": cfg, "ws_steps": ws_ts,
                   "peak_lr": plr, "muon_lr": mlr}, f)

    cmd = f"""@echo off
REM Auto-generated by optimization/hp_tuner_5k.py -- do NOT edit while running (cmd.exe re-reads
REM a running .cmd at a saved byte offset).
cd /d C:\\Users\\Andrew\\rwkv-anki-autoresearch
set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\tuner65k\\{name}
set LOG=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\tuner65k\\{name}.log
set STAMP=%RANDOM%%RANDOM%

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\\Users\\Andrew\\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
{TRUNK_ENV}REM ---- this trial's HPs ----
set RWKV_MUON_LR={mlr:g}
set RWKV_MUON_MOMENTUM={cfg.get("muon_momentum", 0.95):g}
set RWKV_ADAMW_BETA2={cfg.get("adamw_beta2", 0.999):g}
set RWKV_WEIGHT_DECAY={cfg["weight_decay"]:g}
set RWKV_WEIGHT_DECAY_HEAD={cfg["weight_decay"] * cfg.get("wd_head_mult", 1.0):g}
set RWKV_DROPOUT_SCALE={cfg.get("dropout_scale", 1.0):g}
set RWKV_CLIP={cfg["clip"]:g}
REM ---- the adopted speed stack (cleared again before eval) ----
{SPEED_ENV}
echo ===== TRIAL {name} (param={param}={pval_str}) cfg={json.dumps(cfg)} peak_lr={plr:g} muon_lr={mlr:g} START %DATE% %TIME% ===== > "%LOG%"

echo === SANITY 40 steps (VRAM + speed-flag proof) %TIME% === >> "%LOG%"
set RWKV_MAX_STEPS=40
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner65k/{name}/{name}_ws.toml > "%DIR%\\sanity_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_SANITYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 13
)
set RWKV_MAX_STEPS=
REM An env typo that silently disables a speed flag would cost ~2 extra hours PER TRIAL, so prove
REM both engaged before committing to the run.
findstr /C:"BATCHED Newton-Schulz" "%DIR%\\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_MUONNOTBATCHED %DATE% %TIME% >> "%LOG%"
  exit /b 16
)
findstr /C:"[compile] torch.compile" "%DIR%\\sanity_%STAMP%.log" >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOCOMPILE %DATE% %TIME% >> "%LOG%"
  exit /b 17
)
echo SANITY OK %TIME% >> "%LOG%"

set RWKV_STEP_TRACE=scratchpad/tuner65k/{name}/{name}_ws_trace.jsonl
set RWKV_VPRUNE_REF=optimization/tuner65k_vprune_ref.json
set RWKV_VPRUNE_MIN_STEP={vprune_min}
echo === WS {WS_EPOCHS} epoch ({USTART}-{UEND}, MAX={MAX_LEN} -^> ~{ws_ts} steps) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner65k/{name}/{name}_ws.toml > "%DIR%\\ws_%STAMP%.log" 2>&1
if %ERRORLEVEL%==42 (
  echo === VAL-PRUNED - recording estimated logloss %TIME% === >> "%LOG%"
  .venv\\Scripts\\python.exe optimization/hp_tuner_5k.py record-pruned {name} >> "%LOG%" 2>&1
  echo DONE_EXIT_PRUNED %DATE% %TIME% >> "%LOG%"
  exit /b 0
)
REM A crashed WS must NOT cascade into decay/eval -- write_decay_setup takes the LATEST ckpt, so
REM a half-trained one would be silently decayed and evaluated as if it were the real trial.
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 4
)
echo WS OK %TIME% >> "%LOG%"
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=

echo === DECAY SETUP ({decay_ep:g} ep, ratio {cfg["decay_ratio"]:g}) %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_decay_setup.py scratchpad/tuner65k/{name} {name}ws {name}d scratchpad/tuner65k/{name}/{name}_decay.toml {TRAIN_DB} {USTART} {UEND} {decay_ep:g} {plr:g} {MAX_LEN} > "%DIR%\\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL %DATE% %TIME% >> "%LOG%"
  exit /b 5
)
echo === DECAY %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u -m rwkv.train_rwkv --config scratchpad/tuner65k/{name}/{name}_decay.toml > "%DIR%\\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 6
)
echo DECAY OK %TIME% >> "%LOG%"

echo === WRITE EVAL TOML %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe scratchpad/write_eval_toml.py scratchpad/tuner65k/{name} {name}d scratchpad/tuner65k/{name}/{name}_eval.toml RWKV-{name} RWKV-P-{name} {EVAL_USTART} {EVAL_UEND} > "%DIR%\\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)
REM Eval runs WITHOUT the training speed flags, exactly as run_maxval.cmd does.
set RWKV_MUON_BATCHED=
set RWKV_QAT_COMPILE=
set RWKV_NO_JIT=
set RWKV_EVAL_PAVA=1
REM Users 5002/5905/5995 (266k-367k reviews) OOM the 12 GB card iff the desktop is holding
REM several GB of VRAM. eval_sharded SKIPS users already banked, so a retry costs only the
REM remainder -- hence three attempts and NO del between them (the 2026-07-30 ops rule).
echo === EVAL {EVAL_USTART}-{EVAL_UEND} RECTIFIED, attempt 1 %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/{name}/{name}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\\Scripts\\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/{name}/{name}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo EVAL attempt 2 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\\Scripts\\python.exe -u optimization/eval_sharded.py --config scratchpad/tuner65k/{name}/{name}_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\\eval3_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 8
)
echo EVAL OK %TIME% >> "%LOG%"

echo === RECORD {name} %TIME% === >> "%LOG%"
.venv\\Scripts\\python.exe optimization/hp_tuner_5k.py record {name} >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
"""
    with open(f"{folder}/{name}.cmd", "w") as f:
        f.write(cmd)
    return ws_ts


def by_user_mean(path):
    tot, n = 0.0, 0
    for line in open(path):
        r = json.loads(line)
        tot += r["metrics"]["LogLoss"]
        n += 1
    return tot / n, n


def cmd_next():
    recs = load_journal()
    out = compute(recs)
    if out[0] == "done":
        best = out[1]
        print("DONE")
        print("BEST CONFIG:", json.dumps(best))
        r = find(recs, best)
        if r:
            print(f"  ahead {r['ahead']:.6f}  imm {r['imm']:.6f}  (objective {obj(r):.6f})")
        return
    _, cfg, param = out
    name = trial_name(param, cfg)
    ts = write_trial_files(name, param, cfg)
    pv = "(baseline HPs)" if param == "baseline" else f"{cfg[param]:g}"
    print(f"NEXT {name}")
    print(f"  param={param}  value={pv}  full={json.dumps(cfg)}")
    print(f"  peak_lr={peak_lr(cfg):g}  muon_lr={muon_lr(cfg):g}  ws_steps={ts}")
    print(f"  cmd=scratchpad/tuner65k/{name}/{name}.cmd")


def cmd_record(name):
    side = json.load(open(f"{TRIAL_DIR}/{name}/{name}.json"))
    ahead, na = by_user_mean(f"{ROOT}/result/RWKV-{name}.jsonl")
    imm, ni = by_user_mean(f"{ROOT}/result/RWKV-P-{name}.jsonl")
    rec = {"name": name, "param": side["param"], "config": side["config"],
           "ahead": round(ahead, 6), "imm": round(imm, 6), "users": na,
           "peak_lr": side["peak_lr"], "muon_lr": side["muon_lr"]}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED {name}: ahead {ahead:.6f} imm {imm:.6f} (users {na}/{ni}) obj {ahead+imm:.6f}")


def fit_vprune_alpha(recs):
    """Calibrate the val-delta -> final-tune-eval-delta shrinkage slope (per mode), from trials
    that COMPLETED honestly and carry a val sidecar: x = (trial val - reference val) at each WS
    checkpoint >= 1000, y = (trial tune-eval - baseline tune-eval). Early val gaps compress as
    training converges AND val (review-pooled, 10 users) is a different scale from the recorded
    metric (by-user mean, 1000 users) -- a single through-origin slope absorbs both. Caveat: pairs
    within a trial share one y (effective n = #trials, not #pairs), and completed trials only
    populate small |x|, so kill-scale (>=0.004) estimates are linear extrapolation -- hence the
    clamp. Returns (alpha_ahead, alpha_imm, n_pairs, n_trials); alpha=1.0 fallback."""
    if not os.path.exists(VPRUNE_REF):
        return 1.0, 1.0, 0, 0
    champ = json.load(open(VPRUNE_REF))
    if "val_step" not in champ:
        return 1.0, 1.0, 0, 0
    cvals = {int(s): (a, i) for s, a, i in zip(champ["val_step"], champ["val_ahead"], champ["val_imm"])}
    base = next((r for r in recs if r["param"] == "baseline"), None)
    if base is None:
        return 1.0, 1.0, 0, 0
    xa, ya, xi, yi, n_trials = [], [], [], [], 0
    for r in recs:
        if r.get("pruned") or r["param"] == "baseline":
            continue
        sidecar = f"{TRIAL_DIR}/{r['name']}/{r['name']}_ws_trace.jsonl.val.jsonl"
        side_path = f"{TRIAL_DIR}/{r['name']}/{r['name']}.json"
        if not os.path.exists(sidecar) or not os.path.exists(side_path):
            continue
        # re-probes at a new base REUSE the trial name (and overwrite the dir): only pair a
        # sidecar with the journal row whose config matches the dir's current side json
        side = json.load(open(side_path))
        if canon(side["config"]) != canon(r["config"]):
            continue
        pts = 0
        for line in open(sidecar):
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            s = int(v["step"])
            if s < 1000 or s not in cvals:
                continue
            xa.append(v["val_ahead"] - cvals[s][0]); ya.append(r["ahead"] - base["ahead"])
            xi.append(v["val_imm"] - cvals[s][1]);   yi.append(r["imm"] - base["imm"])
            pts += 1
        n_trials += 1 if pts else 0

    def slope(x, y):
        sxx = sum(v * v for v in x)
        if n_trials < 3 or sxx < 1e-10:
            return 1.0
        a = sum(u * v for u, v in zip(x, y)) / sxx
        if a <= 0:  # noise/anti-correlation = no information -> naive slope, not a fake floor
            return 1.0
        return min(max(a, 0.25), 1.5)
    return slope(xa, ya), slope(xi, yi), len(xa), n_trials


def cmd_record_pruned(name):
    """Record a pruned trial from its .pruned.json marker: the journal gets the ESTIMATED logloss
    flagged "pruned": true, so descent proceeds (an abysmal trial never wins a coordinate anyway).
    Estimate = baseline_tune_eval + alpha * mean(strike-window val deltas): the window mean cuts
    single-checkpoint noise, the fitted alpha corrects early-gap compression and the val ->
    tune-eval scale, and anchoring on the BASELINE JOURNAL ROW keeps pruned rows on the same
    1000-user scale as honest rows."""
    side = json.load(open(f"{TRIAL_DIR}/{name}/{name}.json"))
    marker = json.load(open(f"{TRIAL_DIR}/{name}/{name}_ws_trace.jsonl.pruned.json"))
    rec = {"name": name, "param": side["param"], "config": side["config"],
           "ahead": round(float(marker["estimated_ahead"]), 6),
           "imm": round(float(marker["estimated_imm"]), 6),
           "pruned": True, "pruned_at_step": int(marker["pruned_at_step"]),
           "rule": marker.get("rule", "val"),
           "peak_lr": side["peak_lr"], "muon_lr": side["muon_lr"]}
    detail = ""
    if "val_delta_ahead" in marker:
        rec["val_delta_ahead"] = marker["val_delta_ahead"]
        rec["val_delta_imm"] = marker["val_delta_imm"]
        detail = f"(val d_a {marker['val_delta_ahead']:+.4f}, d_i {marker['val_delta_imm']:+.4f})"
        recs = load_journal()
        base = next((r for r in recs if r["param"] == "baseline"), None)
        if base is not None:
            window = marker.get("window") or [[marker["pruned_at_step"],
                                               marker["val_delta_ahead"], marker["val_delta_imm"]]]
            mda = sum(w[1] for w in window) / len(window)
            mdi = sum(w[2] for w in window) / len(window)
            a_a, a_i, n_pairs, n_trials = fit_vprune_alpha(recs)
            rec["ahead"] = round(base["ahead"] + a_a * mda, 6)
            rec["imm"] = round(base["imm"] + a_i * mdi, 6)
            rec["est_alpha"] = [round(a_a, 4), round(a_i, 4)]
            rec["est_window_mean"] = [round(mda, 6), round(mdi, 6)]
            rec["est_naive"] = [round(float(marker["estimated_ahead"]), 6),
                                round(float(marker["estimated_imm"]), 6)]
            detail += (f" est = baseline + alpha*window_mean, alpha ({a_a:.2f},{a_i:.2f}) "
                       f"from {n_trials} trials/{n_pairs} pairs")
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED-PRUNED {name} @ step {rec['pruned_at_step']}: est ahead {rec['ahead']:.6f} "
          f"est imm {rec['imm']:.6f} {detail}")


def cmd_record_baseline(ahead, imm):
    rec = {"name": "t65_baseline", "param": "baseline", "config": dict(DEFAULTS),
           "ahead": round(float(ahead), 6), "imm": round(float(imm), 6), "users": 1000,
           "peak_lr": BASE_PEAK_LR, "muon_lr": BASE_MUON_LR,
           "source": "scratchpad/maxval (the validated MAX=65536 run) restricted to 5001-6000"}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RECORDED baseline: ahead {ahead} imm {imm}")


def cmd_loop():
    """Self-driving coordinate descent: run every remaining trial (its WS->decay->eval->record
    .cmd) until DONE. Resumable -- replays the journal on restart, so a teardown continues from
    the next trial. Launch this DETACHED (survives Esc). Each trial .cmd self-records.

    Between trials it now also: re-reads the grid (so an edit needs no kill), auto-extends a
    coordinate won at a still-paying edge, and reaps orphaned fetch workers. All three were manual
    in the 2026-07-30..08-02 run."""
    while True:
        space, defaults = load_space()           # hot reload -- edits land with nothing killed
        recs = load_journal()

        ext = maybe_extend(recs, space, defaults)
        if ext is not None:
            save_space(ext, defaults, note="auto-extended by the edge rule; see POLICY")
            continue                             # re-plan against the extended grid

        out = compute(recs, space, defaults)
        if out[0] == "done":
            best = out[1]
            print("TUNER DONE. best:", json.dumps(best), flush=True)
            r = find(recs, best)
            base = next((x for x in recs if x["param"] == "baseline"), None)
            if r and base:
                print(f"  best vs baseline: ahead {base['ahead']-r['ahead']:+.6f}  "
                      f"imm {base['imm']-r['imm']:+.6f}  (obj {obj(base)-obj(r):+.6f})", flush=True)
            rep, _ = coordinate_report(recs, space, defaults)
            for param, win, gain, runner, verdict in rep:
                if win is None:
                    continue
                gs = f"{gain:+.6f}" if gain is not None else "n/a"
                print(f"  {param:14} -> {win:<9g} vs default {defaults.get(param, 0):<9g} "
                      f"gain {gs:>10}  {verdict}", flush=True)
            print("  NOTE: a sub-0.001 winner still needs confirming on the full VAL half "
                  "(5001-7500) -- run scratchpad/tuner65k/run_confirm.cmd.", flush=True)
            return

        _, cfg, param = out
        name = trial_name(param, cfg)
        write_trial_files(name, param, cfg)
        n = reap_orphans()
        if n:
            print(f"[reap] killed {n} orphaned fetch worker(s) before starting", flush=True)
        print(f"\n===== TRIAL {name}  param={param}  cfg={json.dumps(cfg)} "
              f"peak_lr={peak_lr(cfg):g} muon_lr={muon_lr(cfg):g} "
              f"est {est_trial_hours(cfg):.1f} h =====", flush=True)
        cmd_path = f"{TRIAL_DIR}/{name}/{name}.cmd".replace("/", "\\")
        rc = subprocess.call(["cmd", "/c", cmd_path])
        if find(load_journal(), cfg) is None:
            print(f"ABORT: {name} did not record (rc={rc}). "
                  f"Check scratchpad/tuner65k/{name}.log. Stopping.", flush=True)
            return


def cmd_status():
    recs = load_journal()
    print(f"{'name':28} {'param':14} {'peak_lr':>9} {'muon_lr':>9} "
          f"{'ahead':>9} {'imm':>9} {'obj':>9}  note")
    base = next((x for x in recs if x["param"] == "baseline"), None)
    for r in recs:
        note = f"PRUNED@{r['pruned_at_step']} (estimated)" if r.get("pruned") else ""
        if base is not None and r is not base:
            # positive = better than baseline; "BAR" marks a trial that cleared it in BOTH modes
            da, di = base["ahead"] - r["ahead"], base["imm"] - r["imm"]
            cleared = r["ahead"] <= BAR["ahead"] and r["imm"] <= BAR["imm"]
            note = (f"d {da:+.6f}/{di:+.6f}" + ("  ** CLEARS THE BAR **" if cleared else "")
                    + (("  " + note) if note else ""))
        print(f"{r['name']:28} {r['param']:14} {r.get('peak_lr', 0):9.2e} "
              f"{r.get('muon_lr', 0):9.2e} {r['ahead']:9.6f} {r['imm']:9.6f} {obj(r):9.6f}  {note}")
    print(f"{'--- BAR (iter 31 on 5001-6000)':28} {'':14} {'':>9} {'':>9} "
          f"{BAR['ahead']:9.6f} {BAR['imm']:9.6f} {BAR['ahead']+BAR['imm']:9.6f}  "
          f"reach BOTH to have recovered what MAX=65536 cost")
    space, defaults = load_space()
    out = compute(recs, space, defaults)
    if out[0] == "done":
        best = out[1]
        print("\nCOORDINATE DESCENT COMPLETE. best:", json.dumps(best))
        r = find(recs, best)
        base = next((x for x in recs if x["param"] == "baseline"), None)
        if r and base:
            print(f"  vs baseline: ahead {base['ahead']-r['ahead']:+.6f}  "
                  f"imm {base['imm']-r['imm']:+.6f}")
        # Which coordinate choices are real and which are coin flips (see coordinate_report).
        rep, _ = coordinate_report(recs, space, defaults)
        print(f"\n  {'coordinate':14} {'winner':>9} {'default':>9} {'gain vs default':>16}"
              f"  verdict")
        for param, win, gain, runner, verdict in rep:
            if win is None:
                continue
            gs = f"{gain:+.6f}" if gain is not None else "n/a"
            print(f"  {param:14} {win:>9g} {defaults.get(param, 0):>9g} {gs:>16}  {verdict}")
        print("\n  NOTE: a sub-0.001 winner still needs confirming on the full VAL half "
              "(5001-7500) before it becomes the recipe.")
    else:
        _, cfg, param = out
        if param == "baseline":
            print("\nNEXT: seed the baseline row -- `record-baseline 0.299250 0.266335` "
                  "(the maxval run restricted to 5001-6000); no GPU run needed.")
        else:
            print(f"\nNEXT: probe param={param} value={cfg[param]:g}  ({trial_name(param, cfg)})"
                  f"  peak_lr={peak_lr(cfg):g} muon_lr={muon_lr(cfg):g}"
                  f"  est {est_trial_hours(cfg):.1f} h")
    # Cost is per-config, not a flat constant: decay steps scale with decay_ratio, so the old flat
    # "4.2 h x N" understated this grid by ~25% on its longest trials.
    todo, hours = 0, 0.0
    probe = dict(defaults)
    for param, grid in space:
        for v in grid:
            cfg = dict(probe)
            cfg[param] = v
            if find(recs, cfg) is None:
                todo += 1
                hours += est_trial_hours(cfg)
        got = [(obj(find(recs, {**probe, param: v})), v) for v in grid
               if find(recs, {**probe, param: v}) is not None]
        if got:
            probe[param] = min(got)[1]
    print(f"\n(grid: {sum(len(g) for _, g in space)} points, {todo} unrun "
          f"=> ~{hours:.0f} h at {POLICY['steps_per_sec']:g} steps/s)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "next":
        cmd_next()
    elif cmd == "record":
        cmd_record(sys.argv[2])
    elif cmd == "record-pruned":
        cmd_record_pruned(sys.argv[2])
    elif cmd == "record-baseline":
        cmd_record_baseline(sys.argv[2], sys.argv[3])
    elif cmd == "status":
        cmd_status()
    elif cmd == "loop":
        cmd_loop()
    else:
        print("unknown command:", cmd)
