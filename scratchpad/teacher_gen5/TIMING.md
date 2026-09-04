# d=128 teacher retrain -- cost measurement (2026-09-04 18:07-18:47)

Model: `scratchpad/architecture_old_d128.py` on the gen-5 SSD dbs, 109-dim layout, the modern recipe
minus the two d80-specific strip flags (GRU head N=3, PAVA 0.2, interleave, Muon incl. LoRA, KD off).
**2,578,838 params** (the upstream teacher had 2,762,884 on the 92-dim layout).

| MAX_TRAIN_GLOBAL_LEN | fits? | steady steps/s | VRAM | note |
|---|---|---|---|---|
| 32768 | yes | **0.83** (window after warm-up; first-500 window 0.30) | 9.0-9.4 GB | measured WHILE the CPU screen pass ran (this step is dispatch-bound), so a LOWER bound -- A0 did 0.95 on a quiet machine at the same MAX |
| 65536 | **no** | 0.056 (47 steps in 14 min) | 11.9 GB pinned, 46 W | the WDDM-paging signature; killed at 18:46 rather than spend 5 h confirming it |

So the teacher trains at MAX 32768 = **22,346 optimizer steps per epoch**, i.e. **6.5-7.5 h per epoch**
(0.95-0.83 steps/s), plus the same rate for the decay epochs, plus a ~4 h eval.

## What a USEFUL teacher costs, and why the cheap one is worthless

A teacher pays only if it beats the student. The student (realcyc, KD off) is 0.298083 / 0.263592 on
the gen-5 VAL set. A d=128 model at our 1.25-epoch recipe scored **0.2983 / 0.2679** on the published
set (A0) -- i.e. no better than the student on ahead and worse on imm -- and the teacher-114 screen
showed a teacher worse than its student cannot pay. The record's budget calibration says a 3x budget
step is worth ~+0.002, and upstream's 12-epoch d=128 model sits at 0.294612 / 0.263561 on the VAL half
(published lineage): ~0.0035 ahead better than the student, about equal on imm. KD's +0.0019 over
iters 32/35/39/45 came from a teacher of roughly that margin.

| budget (WS + 2-epoch decay) | steps | GPU time | expected teacher vs student |
|---|---|---|---|
| 1.25 ep (A0's recipe) | 28k | ~10 h + eval | WORSE -- do not run |
| 4 + 2 ep | 134k | ~45-50 h | ~+0.002 ahead: marginal teacher |
| 10 + 2 ep (upstream-class) | 268k | **~90-100 h (~4 days)** + eval | ~+0.0035 ahead, ~0 imm: the teacher that made KD pay |

**The honest framing for Andrew:** the useful teacher IS a 10x-budget d=128 run, i.e. the same class of
cost as the endgame's own plain 10x arm. Two alternatives to weigh against it: (a) born-again KD from
the frozen realcyc checkpoint (ranked-queue #3; 2 h dump + one normal run; the record's iter-54 tie
says a same-size separate-pass teacher delivers the decay-phase KD gain in full, but it has never been
measured against NO teacher directly); (b) skip KD on this lineage and put the 4 days into the endgame's
10x run itself, on the argument that KD's value is variance reduction at a LOW budget and may shrink at
12.5 epochs. Not a decision I can make: it sets the phase-4 budget.

Re-measure the 32768 rate on a quiet GPU before quoting it in a plan (30 min: `run_t128_timing.cmd`
with the 65536 phase removed).
