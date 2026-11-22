"""Sweep card/note state-quantization candidates with the canonical sharded evaluator.

One candidate = one eval of ONE checkpoint with quantization env vars set. PTQ: the recurrent state
is round-tripped to the deploy format at EVERY step of the forward pass, which is the same
simulation the Rust deploy engine performs when it persists a compressed state. Everything else is
held fixed (checkpoint, roster, eval toml, shard settings), so candidates are comparable to each
other and to the unquantized reference by a paired per-user comparison.

Recorded per candidate:
  * mean LogLoss per head (ahead = result/adw_<tag>.jsonl, imm = result/adwP_<tag>.jsonl)
  * paired delta vs a reference tag, its one-sided 95% upper bound, and a paired Wilcoxon p
    (H1: quantized is WORSE). PASS = upper bound <= +0.0005 in BOTH heads (constraint 3).
  * state bits per stream vs the bar (constraint 2)
  * eval wall-clock and users/s vs the reference (constraint 6)

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/quant_sweep.py --model <pth> \
      --ref qbl_incumbent_a04 --candidates nj,cn2,c2,n2,l1i8,l1i4 [--limit 3] [--shards 2]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "adaptive_wd"))
import adaptive_train as ad  # noqa: E402

VAL_DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
LABEL_DB = "F:/rwkv_lmdb/label_filter_db_id_e2s"
RUN_DIR = ROOT / "scratchpad" / "quant_plan"
DEFAULT_ROSTER = RUN_DIR / "roster750_seed20261119.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# Trunk geometry, read off the checkpoint by probe_arch.py: N_HEADS=1, HEAD_DIM=K=32, so the
# recurrent state per layer is one 1x32x32 WKV matrix plus two token-shift vectors of length 32.
LAYERS = {"card": 1, "note": 3}
STATE_FLOATS_PER_LAYER = 32 * 32 + 2 * 32  # 1088
# Constraint 2: "int4" = plain fp32 -> int4 of the whole state; the bar is half of that.
INT4_BITS = {s: LAYERS[s] * STATE_FLOATS_PER_LAYER * 4 for s in LAYERS}
BAR_BITS = {s: INT4_BITS[s] // 2 for s in LAYERS}
# int6/int5 are NEW here (2026-11-22). They have always existed in the Rust deploy engine
# (`parse_level`, model.rs:2104) and in rust_grid.QBITS, but not in the training path, so no QAT
# run could train for them. Additive only: no existing recipe names them. Motive is axis 1 -- the
# factor-precision rungs BELOW int8 at row 17's exact layout, e.g. card:2:int6 + note:1:int6 =
# 1,024/1,920 bits, which is 640 bits under row 17's 1,280/2,304.
_QMAX = {"int8": 127.0, "int6": 31.0, "int5": 15.0, "int4": 7.0, "int3": 3.0, "int2": 1.0,
         "fp32": float("inf")}
_QBITS = {"int8": 8, "int6": 6, "int5": 5, "int4": 4, "int3": 3, "int2": 2, "fp32": 32}

# name -> (env overrides, comment, bits override). Every candidate also gets RWKV_NO_JIT=1.
CANDIDATES = {
    "nj": ({}, "unquantized control in the no-JIT path (the speed reference)", None),
    "c2": ({"RWKV_QAT_SCOPE": "card:int2", "RWKV_QAT_SHIFT_SCOPE": "card:int2"},
           "card only: WKV + shifts at int2", None),
    "n2": ({"RWKV_QAT_SCOPE": "note:int2", "RWKV_QAT_SHIFT_SCOPE": "note:int2"},
           "note only: WKV + shifts at int2", None),
    "cn2": ({"RWKV_QAT_SCOPE": "card:int2,note:int2", "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int2"},
            "the simplest scheme at the bar: both streams WKV + shifts at int2", None),
    # Card rank-r. `cn8` (no truncation at all, 8 bits) has a ~zero tax (+0.000033/+0.000022 vs
    # the fp32 control), so 8-bit precision of the full state is free and the rank-1 tax is
    # TRUNCATION. The per-stream attribution says the card owns it (card-only +0.001142 imm vs
    # note-only +0.000258), and the card's 2,176-bit bar has room for more ranks. The note stays
    # rank-1 int8 in all four. Added 2026-11-21.
    # The note pool is 2/3 of the total bits (3,072 of 4,096 at row 4) and the note's OWN rank-1
    # int8 tax is +0.000373/+0.000258 - inside the ceiling by itself. So the note is where the
    # bits can be cut. `n4` isolates the note's int4 tax on row 4's exact card config; the last two
    # combine the likely card fix (more rank) with the note cut. Added 2026-11-21.
    "nr4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",
            "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int4"},
           "note rank-1 int4 factors + int4 shifts, card at row 4's exact config (1,024/1,536 bits)", None),
    "nr4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int4",
              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int2"},
             "note rank-1 int4 factors + int2 shifts (1,024/1,152 bits)", None),
    "l2i8s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-2 int8 + int4 shifts, note rank-1 int4 + int4 (1,280/1,536 bits)", None),
    "l3i8s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-3 int8 + int4 shifts, note rank-1 int4 + int4 (1,792/1,536 bits)", None),
    # Equal-bits (rank, precision) probes. Rank is what buys the immediate head and 8-bit precision
    # of the full state is free, so at a FIXED bit count the question is rank vs precision. The note
    # is held at rank-1 int8 + int4 shifts (2,304 bits) except in `n2i4s4`, which makes the same
    # comparison on the note side. Added 2026-11-21.
    "l2i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int4,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "card rank-2 int4 + int4 shifts, 768 card bits (vs row 9 rank-1 int8)", None),
    "l4i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:4:int4,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "card rank-4 int4 + int4 shifts, 1,280 card bits (vs row 17 rank-2 int8)", None),
    "l7i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:7:int4,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "card rank-7 int4 + int4 shifts, 2,048 card bits (vs row 16 rank-3 int8)", None),
    "n2i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int4",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "note rank-2 int4 + int4 shifts, 2,304 note bits (vs row 17 note rank-1 int8)", None),
    # The follow-ups of that equal-bits question on the two streams the ladder has NOT cut yet:
    # the CARD's rank at int4 (l3i4s4) and the note's SHIFT precision (int3 as a half-step between
    # the working int4 and the harmful int2 of rows 19-20). `n1i8s3` and `n2i4s3` are the same
    # 2,112 note bits with rank-1 int8 vs rank-2 int4 factors, so they answer "rank or precision?"
    # on the note the way rows 9/23 answered it on the card. Added 2026-11-21 23:20.
    "l3i4s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int4,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "card rank-3 int4 + int4 shifts, note rank-1 int8 + int4 (1,024/2,304 = 3,328)", None),
    "l2i8s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int4"},
               "card rank-2 int8 + int2 shifts, note rank-1 int8 + int4 (1,152/2,304 = 3,456)", None),
    "n1i8s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
               "note rank-1 int8 + int3 shifts, card rank-2 int8 + int4 (1,280/2,112 = 3,392)", None),
    "n2i4s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int4",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
               "note rank-2 int4 + int3 shifts, same 2,112 note bits as n1i8s3 (1,280/2,112)", None),
    # Chain 13: the two axes crossed. Chain 11 cuts the CARD (rank-3 int4 reaches 1,024 bits, rank-2
    # int4 768) and the shrink chain cuts the NOTE (rank-1 int3 factors 1,344, int3 shifts 2,112), but
    # each is measured with the other stream held at row 17's level. These three points put the cheap
    # card UNDER the cheap note, which is the only way the total reaches 2,368-3,136 instead of 3,584.
    # They are informative whatever chain 11 returns: a failure says the two cuts do not compose, which
    # is the answer the frontier needs. Added 2026-11-21 23:35.
    "l3i4s4n8s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int4,note:1:int8",
                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
                    "card rank-3 int4 + int4 shifts, note rank-1 int8 + int3 shifts (1,024/2,112 = 3,136)", None),
    "l3i4s4n3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int4,note:1:int3",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-3 int4 + int4 shifts, note rank-1 int3 factors + int4 shifts (1,024/1,344 = 2,368)", None),
    "l2i4s4n8s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int4,note:1:int8",
                    "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
                   "card rank-2 int4 + int4 shifts, note rank-1 int8 + int3 shifts (768/2,112 = 2,880)", None),
    # Aggressive shrink combos: push the TOTAL below row 15's 4,608 bits.
    "l3i4s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int4,note:1:int4",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-3 int4 + int4 shifts, note rank-1 int4 + int4 (1,024/1,536)", None),
    "l2i8s4n3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int3",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-2 int8 + int4 shifts, note rank-1 int3 + int4 (1,280/1,344)", None),
    "l2i8s2n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int4"},
                 "card rank-2 int8 + int2 shifts, note rank-1 int4 + int4 (1,152/1,536)", None),
    "l3i8s4n4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",
                    "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int2"},
                   "card rank-3 int8 + int4 shifts, note rank-1 int4 + int2 (1,792/1,152)", None),

    # Andrew's 2026-11-22 12:01 direction: put RANK 2 on the NOTE at int8 factors. The note is the big
    # pool (2/3 of a passing row's bits) and it is the one stream whose truncation tax nobody has cut
    # with precision - rows 19-22, 26 and the shrink combos all moved the note DOWN in bit width and
    # failed, while the card's ladder shows rank is what buys a truncated stream. `n2i4s4` (note rank 2
    # int4 at the SAME 2,304 bits as row 17) failed by a hair on the imm head (ub95 +0.000543), so the
    # untried point is rank 2 at int8: it SPENDS note bits (2,304 -> 3,840) instead of keeping them.
    # The note bar is 6,528 bits, so rank 2 (3,840) and even rank 3 (5,376) fit with room to spare.
    # `l2i8n2i8` is the direct A/B against row 17: same card, only the note's rank changes.
    "l2i8n2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "card rank-2 int8 + int4 shifts, note rank-2 int8 + int4 (1,280/3,840 = 5,120)", None),
    "l3i8n2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:2:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "row 18's card (rank-3 int8) under the note rank-2 int8 (1,792/3,840 = 5,632)", None),
    "l2i8n2i8s8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:2:int8",
                    "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int8"},
                   "note rank-2 int8 with int8 shifts: is the note's SHIFT precision free at rank 2? (1,280/4,608)", None),
    "l2i8n3i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:3:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "note rank-3 int8: the rank ladder on the note, 5,376 of the 6,528 note bits", None),
    "l2i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},
             "card rank-2 int8 factors + int8 shifts (1,536/3,072 bits)", None),
    "l3i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int8",
              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},
             "card rank-3 int8 factors + int8 shifts (2,048/3,072 bits)", None),
    "l2i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "card rank-2 int8 factors + int4 shifts (1,280/2,304 bits)", None),
    # ---- E1: the tail-energy penalty (RWKV_QAT_TAIL_PENALTY), zero state bits, zero deploy cost --
    # The gate's tax is RANK TRUNCATION (`cn8`, full-matrix int8, is +0.000033/+0.000022), and the
    # measured tail of row 17's own states (corruption/tail_energy_l2i8s4.json: 5,388 raw states on
    # the eval path, users 7213/7044/6960) is: the card at rank 2 keeps 0.99381 of the state energy
    # (tail 0.62%), the note at rank 1 keeps 0.96943 (tail 3.06%). The deploy path stores a rank-r
    # matrix, so a state whose energy is EXACTLY rank-r loses nothing to the truncation; the tail is
    # precisely the information the deploy path throws away, and the QAT LOSS CANNOT SEE IT, because
    # the straight-through estimator passes gradient through the truncation as if it were identity.
    # The penalty adds `lambda * mean over streams of the tail fraction` to the training loss. It is
    # a RATIO, so it is scale-invariant: shrinking the whole state does not reduce it, only a
    # steeper spectrum does. Lambda is bracketed by the measured tail (mean ~0.0245 over the four
    # quantized layers): 1.0 is ~2.4% of the ~1.02 training loss, 10.0 is ~25%.
    # RWKV_QAT_FUSED=0 is REQUIRED (the fused rank-1 kernel truncates inside the kernel), and it is
    # also the documented bit-parity twin of the fused kernels, so `l2i8s4pf` separates "fused off"
    # from "penalty on" - without it a small E1 delta could be misattributed. Added 2026-11-22.
    "l2i8s4pf": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                  "RWKV_QAT_FUSED": "0"},
                 "row 17's recipe on the Python per-step QAT loop, NO penalty - the E1 control (1,280/2,304)", None),
    "l2i8s4e1a": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0"},
                  "row 17 + tail-energy penalty lambda=1.0 (~2.4% of the loss) (1,280/2,304)", None),
    "l2i8s4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},
                  "row 17 + tail-energy penalty lambda=10.0 (~25% of the loss) (1,280/2,304)", None),
    # ---- E1 applied to the two NEAR-MISS shrink layouts (chain 16, gated on chain 15 result) ----
    # Row 21's layout (`l2i8s4n4`: card rank-2 int8 + int4 shifts, note rank-1 int4 + int4, 2,816
    # bits) failed the log-loss gate by 0.00014 on imm (+0.000328/+0.000638). Row 22's layout
    # (`l3i8s4n4`: card rank-3 int8 + the same int4 note, 3,328 bits) failed by only 0.000025
    # (+0.000297/+0.000525). Row 17 is 3,584 bits, so these save 768 and 256 bits.
    # Both failures are the note's rank-1 int4 tax, and the note's rank-1 int8 tax alone is
    # +0.000258 imm, so a penalty that removes part of the note's truncation tax could flip them.
    "l2i8s4n4e1a": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",
                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0"},
                    "row 21's layout + tail penalty lambda=1.0 (1,280/1,536 = 2,816)", None),
    "l2i8s4n4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",
                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},
                    "row 21's layout + tail penalty lambda=10.0 (1,280/1,536 = 2,816)", None),
    "l3i8s4n4e1a": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",
                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0"},
                    "row 22's layout + tail penalty lambda=1.0 (1,792/1,536 = 3,328)", None),
    "l3i8s4n4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",
                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},
                    "row 22's layout + tail penalty lambda=10.0 (1,792/1,536 = 3,328)", None),
    # THE int6/int5 FACTOR RUNGS (2026-11-22) -- axis 1. Row 17's EXACT layout (card rank 2, note
    # rank 1, int4 shifts), with only the FACTOR PRECISION changed, so each point isolates it.
    # Every earlier rung below row 17 used int4 factors; int6/int5 were never runnable at all
    # because _QMAX lacked the keys (patched above). Bits:
    #   l2i6s4   card 2x2x32x6 + 64x4 = 1,024  note 3x(2x1x32x6 + 64x4) = 1,920  = 2,944
    #   l2i5s4   card 2x2x32x5 + 64x4 =   896  note 3x(2x1x32x5 + 64x4) = 1,728  = 2,624
    #   l2i4s4n4 card 2x2x32x4 + 64x4 =   768  note 3x(2x1x32x4 + 64x4) = 1,536  = 2,304
    # A 2-user Rust PTQ probe on the row-17 checkpoint said int6 costs +0.000003 ahead / +0.000002
    # imm vs int8 and int5 +0.000017/-0.000073 (rust_grid_int6_probe.json) -- i.e. no visible
    # penalty for 640/960 fewer bits. That probe CANNOT settle the gate: calibrated against rows 21
    # and 23 it reproduces the note-int4 cost at only 54% and gets the card-int4 sign WRONG. So
    # these are full 750-user QAT points, not PTQ.
    "l2i6s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "row 17 layout at int6 factors (1,024/1,920 = 2,944 bits)", None),
    "l2i5s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int5,note:1:int5",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "row 17 layout at int5 factors (896/1,728 = 2,624 bits)", None),
    "l2i4s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int4,note:1:int4",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
                 "row 17 layout at int4 factors, the bottom rung (768/1,536 = 2,304 bits)", None),
    # THE NOTE-ONLY FIXED-BASIS CANDIDATES (2026-11-22, after the leave-one-user-out measurement).
    # WHY THE NOTE ONLY: at rank 2 the CARD is reproduced EXACTLY for 1,280 bits, and the best fixed
    # basis on the card needs m=16 (2,304 bits) to reach only 0.9596 held out - the card is the wrong
    # place for a codec. The NOTE's shipped rank-1 captures 0.904-0.951 and passes the gate, and a
    # fixed basis reaches 0.876-0.921 held out at m=8 for the SAME 768 bits/layer (0.82-0.87 at m=6,
    # 544 bits/layer). So these hold the card at rank 2 and switch only the note (RWKV_FIXEDBASIS_SCOPE):
    #   l2i8fb8n  card 1,280 + note 3x768 = 3,584  (row 17's EXACT size -> a pure codec A/B)
    #   l2i8fb6n  card 1,280 + note 3x544 = 2,912  (672 below row 17 -> the size claim)
    "l2i8fb8n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:8:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                  "RWKV_FIXEDBASIS_SCOPE": "note",
                  "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
                 "card rank 2 (row 17's card) + NOTE fixed basis m=8 int8 (1,280/2,304 = 3,584 bits)",
                 (1280, 2304)),
    "l2i8fb6n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:6:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                  "RWKV_FIXEDBASIS_SCOPE": "note",
                  "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
                 "card rank 2 (row 17's card) + NOTE fixed basis m=6 int8 (1,280/1,632 = 2,912 bits)",
                 (1280, 1632)),
    "fb4i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:4:int8,note:4:int8",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
               "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
              "FIXED-BASIS state coding (approach 1): m=4 modes per side, int8 coefficient block (384/1152 = 1536 bits)",
              (384, 1152)),
    "fb6i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:6:int8,note:6:int8",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
               "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
              "FIXED-BASIS state coding (approach 1): m=6 modes per side, int8 coefficient block (544/1632 = 2176 bits)",
              (544, 1632)),
    "fb8i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:8:int8,note:8:int8",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
               "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
              "FIXED-BASIS state coding (approach 1): m=8 modes per side, int8 coefficient block (768/2304 = 3072 bits)",
              (768, 2304)),
    "fb12i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:12:int8,note:12:int8",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
               "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
              "FIXED-BASIS state coding (approach 1): m=12 modes per side, int8 coefficient block (1408/4224 = 5632 bits)",
              (1408, 4224)),
    "fb8i4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:8:int4,note:8:int4",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
               "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
              "FIXED-BASIS state coding (approach 1): m=8 modes per side, int4 coefficient block (512/1536 = 2048 bits)",
              (512, 1536)),
    "fb16i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:16:int8,note:16:int8",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
               "RWKV_FIXEDBASIS_PATH": "C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\quant_plan\\corruption\\fixedbasis_l2i8s4all.npz"},
              "FIXED-BASIS state coding (approach 1): m=16 modes per side, int8 coefficient block (2304/6912 = 9216 bits)",
              (2304, 6912)),

    # MIXED precision (2026-11-22 20:20): the ladder above coarsens BOTH streams at once, so a result cannot say
    # WHERE the int6 tax comes from. An 11-user deploy screen says it is mostly the NOTE's: card int6 + note
    # int8 (`l2i6c`) costs +0.000115 imm [+0.000018,+0.000218] and +0.000052 ahead [-0.000007,+0.000109], while
    # card int8 + note int6 (`l2i6n`) costs +0.000307 imm [+0.000124,+0.000558] and +0.000268 ahead
    # [+0.000079,+0.000601] (paired vs l2i8s4, rust_wide_screen_it36int6split.json). So the CARD's int6 is
    # nearly free at PTQ and is the one worth a 750-user QAT point.
    "l2i6c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
              "row 17 layout with the CARD at int6 and the note left at int8 (1,024/2,304 = 3,328 bits)",
              None),
    "l2i6n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int6",
               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
              "row 17 layout with the NOTE at int6 and the card left at int8 (1,280/1,920 = 3,200 bits)",
              None),
    "l2i6s4n1i8s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
                     "card rank-2 int6 + int4 shifts and note rank-1 int8 + int3 shifts "
                     "(1,024/2,112 = 3,136 = 256 BELOW row 35, the smallest passing row)",
                     (1024, 2112)),
    "l2i6s4n1i8s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int2"},
                     "card rank-2 int6 + int4 shifts and note rank-1 int8 + int2 shifts "
                     "(1,024/1,920 = 2,944 = 448 BELOW row 35) - gated second point of chain 21",
                     (1024, 1920)),
    "l2i6s4n1i6s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
                     "card rank-2 int6 + int4 shifts and note rank-1 int6 + int3 shifts "
                     "(1,024/1,728 = 2,752 = 640 BELOW row 35, and the only sub-3,136 rung whose "
                     "7-user deploy screen sits inside the +0.0005 ceiling on BOTH heads: "
                     "+0.000343/+0.000435)",
                     (1024, 1728)),
    "l3i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "card rank-3 int8 factors + int4 shifts (1,792/2,304 bits)", None),
    "l1i8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int8",
              "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},
             "rank-1 factors at int8, shifts int8", None),
    # Andrew's pick (2026-11-19 22:03) for the QAT redo: card keeps the full matrix at int2 (exactly
    # at the 2176-bit bar, and card int2 went nearly free under QAT in qat_log #39), note gets the
    # rank-1 int8 structure (3072 bits, 2x under its 6528-bit bar).
    "c2_nl1i8": ({"RWKV_QAT_SCOPE": "card:int2",
                  "RWKV_QAT_LOWRANK_SCOPE": "note:1:int8",
                  "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int8"},
                 "card full-matrix int2 + note rank-1 int8 (Andrew's choice for the QAT redo)", None),
    "l1i4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",
              "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
             "rank-1 factors at int4, shifts int4 (mirrors the verified Rust config)", None),
    "l1i3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int3,note:1:int3",
              "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3"},
             "rank-1 factors at int3, shifts int3 (384/1152 bits)", None),
    "l1i2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int2,note:1:int2",
              "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int2"},
             "rank-1 factors at int2, shifts int2 (256/768 bits)", None),
    "l1i4s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",
                "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int2"},
               "rank-1 factors at int4, shifts int2 (smaller card state)", None),
    # Which side of the rank-1 split carries the tax? These two have the SAME state size
    # (768 card / 2,304 note bits) with the bits moved from one side to the other, so the pair
    # is an attribution test that costs no state. Added 2026-11-20 for the axis-1 curve.
    # Attribution for row 4 at its exact allocation: which stream owns which head? Card-only and
    # note-only rank-1 int8, the other stream left in fp32. Added 2026-11-21.
    "l1i8c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8", "RWKV_QAT_SHIFT_SCOPE": "card:int8"},
              "attribution: card rank-1 int8 only, note untouched", None),
    "l1i8n": ({"RWKV_QAT_LOWRANK_SCOPE": "note:1:int8", "RWKV_QAT_SHIFT_SCOPE": "note:int8"},
              "attribution: note rank-1 int8 only, card untouched", None),
    "l1i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int8,note:1:int8",
                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},
               "rank-1 int8 factors + int4 shifts (768/2304 bits)", None),
    "l1i4s8": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",
                "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},
               "rank-1 int4 factors + int8 shifts (768/2304 bits)", None),
    "l1i4c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4", "RWKV_QAT_SHIFT_SCOPE": "card:int4"},
              "attribution: card rank-1 int4 only, note untouched", None),
    "l1i4n": ({"RWKV_QAT_LOWRANK_SCOPE": "note:1:int4", "RWKV_QAT_SHIFT_SCOPE": "note:int4"},
              "attribution: note rank-1 int4 only, card untouched", None),
    # DIAGNOSTIC, far over the bar: no rank truncation at all, 8 bits per state element. It tells
    # us whether the residual immediate-head tax is rank-1 truncation or factor precision.
    "cn8": ({"RWKV_QAT_SCOPE": "card:int8,note:int8", "RWKV_QAT_SHIFT_SCOPE": "card:int8,note:int8"},
            "DIAGNOSTIC, 4x over the bar: full-matrix int8 both streams (no rank truncation)", None),
    "cn3": ({"RWKV_QAT_SCOPE": "card:int3,note:int3", "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3"},
            "DIAGNOSTIC, over the bar: full-matrix int3 both streams", None),
    "c3": ({"RWKV_QAT_SCOPE": "card:int3", "RWKV_QAT_SHIFT_SCOPE": "card:int3"},
           "DIAGNOSTIC, over the bar: card int3", None),
    "c4": ({"RWKV_QAT_SCOPE": "card:int4", "RWKV_QAT_SHIFT_SCOPE": "card:int4"},
           "DIAGNOSTIC, over the bar: card int4", None),
    "n3": ({"RWKV_QAT_SCOPE": "note:int3", "RWKV_QAT_SHIFT_SCOPE": "note:int3"},
           "DIAGNOSTIC, over the bar: note int3", None),
    "n4": ({"RWKV_QAT_SCOPE": "note:int4", "RWKV_QAT_SHIFT_SCOPE": "note:int4"},
           "DIAGNOSTIC, over the bar: note int4", None),
    "q72u": ({"RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",
              "RWKV_QAT_PQ": "reference/pq_cb_wkv_q72u.txt",
              "RWKV_QAT_SHIFT_PQ": "reference/pq_cb_shift_q72u.txt",
              "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3",
              "RWKV_QAT_NORM_BITS": "1", "RWKV_QAT_FUSED": "1"},
             "the project's locked deploy recipe (72 bits/layer, learned catalogs)", (72, 216)),
}


def base_env():
    """The eval environment of the 92-input lineage. Must start from os.environ.copy()."""
    env = os.environ.copy()
    env.update({
        "RWKV_ID_FEATURES": "0",
        "RWKV_REAL_CYCLES": "0",
        "RWKV_FINAL_LAYOUT": "0",
        "RWKV_E2S_PUBLISHED": "0",
        "RWKV_VRAM_CAP_GB": "11",
    })
    return env


def bits_for(env, bits_override=None):
    """State bits per stream for a candidate, WKV plus token shifts, excluding the per-step scale.

    full matrix: layers * (1024 + 64) * qbits
    rank r int-N: layers * (2*r*K*qbits + 64*shift_bits)
    """
    if bits_override is not None:
        return {"card": bits_override[0], "note": bits_override[1]}
    shift_bits = {}
    for entry in env.get("RWKV_QAT_SHIFT_SCOPE", "").split(","):
        if ":" in entry:
            name, lvl = entry.split(":")
            shift_bits[name] = _QBITS[lvl]
    lr = {}
    for entry in env.get("RWKV_QAT_LOWRANK_SCOPE", "").split(","):
        parts = entry.split(":")
        if len(parts) >= 3:
            lr[parts[0]] = (int(parts[1]), _QBITS.get(parts[2], 32))
    fq = {}
    for entry in env.get("RWKV_QAT_SCOPE", "").split(","):
        parts = entry.split(":")
        if len(parts) == 2:
            fq[parts[0]] = _QBITS[parts[1]]
    out = {}
    for s, n in LAYERS.items():
        sb = shift_bits.get(s, 32)
        if s in lr:
            r, qb = lr[s]
            out[s] = n * (2 * r * 32 * qb + 64 * sb)
        elif s in fq:
            out[s] = n * (1024 * fq[s] + 64 * sb)
        else:
            out[s] = n * (1024 + 64) * 32
    return out


def load_pair(tag):
    """{user: (ahead_logloss, imm_logloss)} from the merged canonical result files."""
    out = {}
    for idx, name in ((0, f"adw_{tag}.jsonl"), (1, f"adwP_{tag}.jsonl")):
        path = ROOT / "result" / name
        if not path.exists():
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out.setdefault(int(rec["user"]), [None, None])[idx] = float(rec["metrics"]["LogLoss"])
    return {u: tuple(v) for u, v in out.items() if v[0] is not None and v[1] is not None}


def paired(ref, cur, mode_idx):
    users = sorted(set(ref) & set(cur))
    if len(users) < 5:
        return None
    d = np.array([cur[u][mode_idx] - ref[u][mode_idx] for u in users])
    delta = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    try:
        p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    mean_cur = float(np.mean([cur[u][mode_idx] for u in users]))
    mean_ref = float(np.mean([ref[u][mode_idx] for u in users]))
    return {"n": len(users), "delta": delta, "ub95": delta + 1.645 * se, "p_worse": p,
            "mean_cur": mean_cur, "mean_ref": mean_ref}


def run_candidate(model, name, users, shards, solo, tag):
    env_extra, _comment, bits_override = CANDIDATES[name]
    env = base_env()
    env.update({k: str(v) for k, v in env_extra.items()})
    env["RWKV_NO_JIT"] = "1"
    cfg = RUN_DIR / f"eval_{tag}.toml"
    ad.eval_config(ROOT / "rwkv" / "get_result_config_base5k.toml", cfg, model, tag, users,
                   VAL_DB, LABEL_DB)
    users_path = RUN_DIR / f"users_{tag}.json"
    users_path.write_text(json.dumps(list(users)), encoding="utf-8")
    log = RUN_DIR / f"eval_{tag}.log"
    cmd = [str(PY), str(ROOT / "optimization/eval_sharded.py"), "--config", str(cfg),
           "--users-file", str(users_path), "--shards", str(shards),
           "--solo-threshold", str(solo), "--fetch-per-shard", "1", "--threads-per-shard", "7"]
    t0 = time.monotonic()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} shards={shards} solo={solo} env={env_extra} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    secs = time.monotonic() - t0
    pair = load_pair(tag)
    return {"name": name, "tag": tag, "env": env_extra, "comment": _comment,
            "bits": bits_for(env_extra, bits_override), "rc": rc, "secs": secs,
            "users": len(pair) if pair else 0, "pair": pair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", default="nj,cn2")
    ap.add_argument("--ref", default="", help="tag of the reference eval (unquantized, same roster)")
    ap.add_argument("--users-file", default=str(DEFAULT_ROSTER))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--solo", type=int, default=1000000)
    ap.add_argument("--stamp", default=time.strftime("%m%d_%H%M"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"checkpoint not found: {model}")
    doc = json.loads(Path(args.users_file).read_text(encoding="utf-8"))
    users = list(doc["users"] if isinstance(doc, dict) else doc)
    if args.limit:
        users = users[:args.limit]
    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    for n in names:
        if n not in CANDIDATES:
            raise SystemExit(f"unknown candidate {n}; known: {sorted(CANDIDATES)}")

    ref_pair = load_pair(args.ref) if args.ref else None
    if args.ref and ref_pair is None:
        print(f"WARNING: reference tag '{args.ref}' has no merged result files yet; "
              "deltas will be skipped")
    print(f"model={model}\nroster={args.users_file} users={len(users)} shards={args.shards} "
          f"solo={args.solo} ref={args.ref or '-'}")
    print(f"bar bits: card<={BAR_BITS['card']} note<={BAR_BITS['note']} "
          f"(int4 card={INT4_BITS['card']} note={INT4_BITS['note']})")

    rows = []
    for name in names:
        tag = f"q{name}_{args.stamp}"
        print(f"\n--- {name}: {CANDIDATES[name][1]}  tag={tag}")
        r = run_candidate(model, name, users, args.shards, args.solo, tag)
        bits = r["bits"]
        over = [s for s in LAYERS if bits[s] > BAR_BITS[s]]
        r["bits_ok"] = not over
        if r["pair"] is None:
            print(f"    NO RESULT (rc={r['rc']}, {r['secs']/60:.1f} min)")
        else:
            msg = (f"    bits card={bits['card']} note={bits['note']}"
                   f"  {'OK (<= bar)' if not over else 'OVER BAR: ' + ','.join(over)}"
                   f"  n={r['users']}/{len(users)}  wall={r['secs']/60:.1f} min"
                   f"  {r['users']/max(r['secs'],1e-9)*60:.1f} users/min")
            print(msg)
            if ref_pair:
                for idx, mode in ((0, "ahead"), (1, "imm")):
                    st = paired(ref_pair, r["pair"], idx)
                    if st is None:
                        continue
                    r.setdefault("gate", {})[mode] = st
                    verdict = "PASS" if st["ub95"] <= 0.0005 else "fail"
                    print(f"    {mode:5s} cur={st['mean_cur']:.6f} ref={st['mean_ref']:.6f} "
                          f"delta={st['delta']:+.6f} ub95={st['ub95']:+.6f} "
                          f"p_worse={st['p_worse']:.4g} -> {verdict} vs +0.0005")
        rows.append(r)

    if ref_pair:
        ref_secs = [r["secs"] for r in rows if r["name"] == "nj"]
        if ref_secs:
            print(f"\nspeed (wall-clock, same roster/shards): nj={ref_secs[0]/60:.1f} min")
            for r in rows:
                if r["name"] != "nj":
                    print(f"  {r['name']:5s} {r['secs']/60:.1f} min  "
                          f"ratio={ref_secs[0]/max(r['secs'], 1e-9):.3f} "
                          f"(constraint 6 needs >= 0.75)")
    out = Path(args.out) if args.out else RUN_DIR / f"quant_sweep_{args.stamp}.json"
    slim = [{k: v for k, v in r.items() if k != "pair"} for r in rows]
    out.write_text(json.dumps({"model": str(model), "ref": args.ref, "users": len(users),
                               "shards": args.shards, "bar_bits": BAR_BITS, "rows": slim},
                              indent=2), encoding="utf-8")
    print(f"\nsummary written to {out}")


if __name__ == "__main__":
    main()
