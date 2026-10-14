# THE LEDGER -- every change relative to the old d=128 model

> Andrew 2026-09-13: *"record all changes relative to the old d=128 model. They're not 100% additive, but they're
> additive enough for the table to give a good idea of what changed."* 2026-09-14: *"keep this table somewhere
> at hand, so that you can always pull it up for me and/or edit."* **This file is that place.** Pull it up on
> request; edit it whenever a pending row lands (say which row moved and why in the commit).

Each row is the MEASURED difference between two consecutive runs of the record (positive = worse LogLoss,
negative = better), so the rows telescope: they sum to the bottom line exactly. They are not strictly
additive as CAUSES (an effect measured on one base can differ on another), but each was measured as a
single-variable pair where the record has one, and bundles are named. Yardstick = the old d=128 model,
unquantized, unrectified, on the VAL half (0.294612 / 0.263561). Rounded to 4 dp; the raw pairs are in the
rows referenced.

| # | change | ahead | imm | measured as |
|---|---|---|---|---|
| 1 | Our training recipe at the SAME d=128 arch (1.25 epochs, augmentation off, MAX 32768, peak LR 1e-3, warmup 200) | +0.0037 | +0.0043 | A0 vs the upstream weights |

| 2 | Smaller model: 2.76 M -> 558 k params (4.95x), the width ladder | +0.0010 | +0.0005 | A18 vs A0 |
| 3 | PAVA + GRU head (N=3) + Muon | -0.0004 | -0.0008 | iter 31 vs A18 |
| 4 | Distillation from the d=128 teacher | -0.0006 | -0.0004 | iter 32 vs iter 31 (removed again in row 8) |
| 5 | Forgetting-curve rectifier = the deploy contract (current duration zeroed + PAVA pooling), a METRIC change the d=128 yardstick does not pay | +0.0019 | +0.0001 | iter 32 rectified vs unrectified |
| 6 | HP tuning at MAX=65536 (Muon LR /8, decay ratio 1.0, dropout x0.5; 1.68x faster) | -0.0013 | -0.0010 | iter 34 vs iter 32 rect |
| 7 | Six accepted algorithmic iterations: seed pair, PAVA lambda 0.2, KD alpha 0.9, interleaving, KD through the decay, Muon on LoRA | -0.0014 | -0.0010 | iter 53 vs iter 34 |
| 8 | Distillation REMOVED (Andrew: skip KD) + Bug A fix + rebuilt dbs, one bundle | +0.0007 | +0.0004 | featA2 vs iter 53 |
| 9 | End-to-start intervals (what a live scheduler can compute) | +0.0004 | +0.0005 | e2sc vs iter 53, applied inside row 10's bundle |
| 10 | New input features: the 23 real-timestamp columns (net of row 9) | -0.0007 | -0.0029 | featB vs featA2 = rows 9 + 10 (-0.0003 / -0.0024 as a bundle) |
| 11 | Gen 4: Bug C fix + the scored set re-selected with `delta_t > 0` on e2s gaps (drops ~0.19% of rows that are 1.46x easier, so this is mostly the set, not the model) | +0.0002 | +0.0003 | gen4base vs featB (new lineage) |
| 12 | Real calendar cycles replace the pseudo cycles (+ row 11 dropped) | -0.0000 | +0.0000 | realcyc vs gen4base (tie) |
| 13 | 10x epoch budget (10 WS + 2 decay epochs) | -0.0005 | -0.0015 | w10plain vs realcyc |
| 14 | Clock-leak fix (query rows see the previous answer's time), WS + decay + eval; also the final 103-input layout (~0) | +0.0002 | +0.0021 | w10lf vs w10plain (w10clk, decay-only, gave the same +0.0002 / +0.0021: a leak-free WS recovers nothing more) |
| 14b | Decay length 2 -> 1 epoch (L* = 1 by the pre-registered rule; 3 epochs was +0.0002 WORSE on ahead) | -0.0003 | +0.0000 | lfd1 vs w10lf (ahead +0.000278 p 7e-8; imm -0.000049, inside the noise floor) |
| | **Bottom line at d=80 (lfd1 = w10lf's WS + a 1-epoch decay, plain, leak-free, the d=80 base) vs the old d=128 model** | **+0.0029** | **+0.0007** | 0.297477 / 0.264243 (w10lf: 0.297755 / 0.264195) |
| 14c | Smaller model again: d=80 -> d=64, 561,732 -> 332,922 params (1.69x), 10+1 epochs; adopted by Andrew 2026-09-22/23 | +0.0000 | +0.0004 | 104 `s64hc10` vs lfd1 (+0.000028 / +0.000402, n=2492) |
| 14d | Sibling rating (2 read-time inputs) + Muon on the 26 scale matrices | -0.0000 | -0.0001 | 110 `s64hsm10` vs s64hc10 (-0.000011 / +0.000105, imm p 7e-11), 2026-09-25 |
| | **Bottom line today (s64hsm10 = d=64 + row 14d, plain, leak-free, the incumbent) vs the old d=128 model** | **+0.0029** | **+0.0010** | 0.297534 / 0.264587 |
| 15 | QAT for the deploy state (card/note quantized), LEAK-FREE at L* = 1, **on d=80** | +0.0022 | +0.0035 | lfq1 vs lfd1 (2026-09-18; the leaked base's tax was +0.0025 / +0.0045 -- ~a fifth of its imm part was leak precision); QAT-KD REJECTED (85 `lfqkd1`: -0.0001 / +0.0000); b13 arm cancelled in favour of residual VQ |
| 15b | ... at +100% of state bits (3 residual VQ stages of b10; card 185 -> 385 b, note 105 -> 205 b) | +0.0016 | +0.0026 | **99 `lfq1rvq3b10v2` vs lfd1 (2026-09-22)** -- the tax falls ~25% in both modes, so bits DO buy it back. ⚠ ROW 95's -0.0008 / -0.0012 IS WITHDRAWN: that eval ran a 3-stage-trained model through a 1-stage deploy quantizer on the box. Still short of Andrew's < 0.0010 target in both modes. **+200% (5 stages, 107 `lfq1rvq5b10`): +0.0013 / +0.0021** -- still short; the imm tax looks like a floor that WKV index bits cannot remove. d=80 only; not shipped |
| | **Deployed at d=80 (MEASURED: lfq1, quantized) vs the old d=128 model** | **+0.0050** | **+0.0042** | 0.299630 / 0.267711 |
| 16 | QAT on the d=64 trunk (the same recipe as row 15: b10 WKV, m2b12 shift, 1-bit norms, learned catalogs) | +0.0028 | +0.0040 | 108 `s64hq1` vs s64hc10 (1.29x / 1.15x the d=80 tax). `s64hqnr` (111, the WKV norm range refitted on the plain model) made it WORSE: tax +0.0059 / +0.0079. `s64hqr4b` (112, 4 residual stages, card 403 b / note 214 b) cuts it to **+0.0018 / +0.0028** |
| | **Deployed today (MEASURED: s64hqr4b = d=64, 4 stages, quantized) vs the old d=128 model** | **+0.0048** | **+0.0039** | 0.299369 / 0.267496 |

**⚠ ROWS 1, 2 AND 13 INTERACT, AND THE TABLE CANNOT SPLIT THEM (Andrew 2026-09-14: "reducing model size
only had a mild effect ... yet scaling up epochs to 10+2 didn't buy that much. Something's off").** Row 1 was
measured at d=128 with 1.25 epochs; row 2 (size) was measured at 1.25 epochs, where a 558k model is not
capacity-limited; row 13 (10x epochs) was measured at 558k. If the small model saturates before 10 epochs,
then at the FULL budget the size cost is not +0.0010 but most of the ~0.003 ahead that never came back, and
row 1 is mostly epochs after all. The record supports that reading: the per-user check of 2026-09-12 (d=128
beats w10clk on ahead for 78% of users, a gap that GROWS with reviews per card) and "capacity binds at the
real budget" (AGENTS.md). The alternatives inside row 1 -- augmentation off, peak LR 1e-3 vs 7e-4, warmup 200
vs 20,000, MAX 32768 vs 66000, no KD -- are not separated either. **The single experiment that settles it is
d=128 (the A0 architecture) at 10+2 epochs with our recipe** (~4 GPU days at 0.83 steps/s, MAX 32768;
`scratchpad/teacher_gen5/TIMING.md`): if it reproduces the upstream 0.2946 / 0.2636, the offender is SIZE at
the real budget and row 1 collapses into row 13; if it does not, the rest is augmentation / LR / warmup. The
cheaper partial test is already queued: the card-stream grafts (lfcmf2 / lflora16) on w10lf's WS.
**THE ARITHMETIC (Andrew 2026-09-14, "something definitely doesn't add up"):** row 1 charges the recipe
+0.0037 / +0.0043; HP tuning (row 6) and the 10x budget (row 13) recover only -0.0013 -0.0005 = -0.0018 ahead
and -0.0010 -0.0015 = -0.0025 imm. So **+0.0019 ahead / +0.0018 imm of the recipe's cost is still unrecovered
at 10x epochs**, on top of the size row. The 2026-08-11 budget calibration had projected +0.0042 at 10x; the
10x run delivered 12% of that on ahead. Suspects, none separated: (i) the small model saturates (size, see
above); (ii) **augmentation is OFF, so all 10 epochs are byte-identical replays** of the same random ID codes
and cycle phases (`optimization/DATASETS.md`; upstream's 12 epochs re-drew them every batch) -- the model
can spend capacity memorizing fixed codes it never sees at eval; (iii) the WS hyper-parameters (peak LR 1e-3,
Muon 2.5e-3, warmup 200, MAX 65536) were tuned at 1.25 epochs and never re-tuned for 10 (upstream: 7e-4,
warmup 20,000, MAX 66000). Discriminating runs, cheapest first: the plain budget curve lfb2 / lfb5 (queued;
flat from 5 to 10 epochs = epochs stopped paying), augmentation ON for one 10-epoch WS at d=80 (~38 h; Andrew
vetoed "that particular kind" on 2026-08-16 for run-to-run variance and KD reasons -- KD is gone and the
endgame is a single run, so the veto is worth re-asking), and d=128 at 10+2 (~4 days, settles size).

Three things the sum hides. (a) Rows 1 and 13 are the same lever in opposite directions: the 1.25-epoch
recipe cost +0.0037 / +0.0043 at d=128, and going to 10x epochs at 558 k bought back only -0.0005 /
-0.0015 -- so ~0.003 of ahead is where "capacity at the real budget" lives (the card-graft arms test it).
(b) Row 5 is a change of what is MEASURED, not of the model: the old model would pay ~+0.0036 on the same
rectified metric (A18 unrectified vs rectified). (c) The gen-4/5 rows (11-14) are scored on a re-selected
set of reviews with n = 2,499 (user 6701 excluded), against a yardstick scored on the published set; row 11
is that basis shift. Pending rows: the leak-free QAT arms and the b13 (the decay-length pair landed as row 14b; the card grafts lfcmf2 / lflora16 both came back NULL on 2026-09-16/17, so no row: the shipped base is unchanged)
bits (row 15), phase-5 tuning of the QAT decay. (w10lf landed 2026-09-14 and completed row 14.)


## Non-telescoping diagnostic (2026-10-14)

`d128e10` tested the d=128 architecture at 10+2 epochs. On the `-id5` VAL half, 2,279 users scored: ahead 0.296280 / RMSE(bins) 0.057933; imm 0.261156 / RMSE(bins) 0.034402. There were 220 OOM-skipped users; 6701 was excluded by design. Do not add this as a telescoping delta: against published `base5k`, 1,955 of the 2,279 overlapping users have unequal scored-review counts because the label-filter databases differ. See `GPU_RUN_BOARD.md` for the full caveat and next-work order.



## Training speed diagnostic (2026-10-14)

Five-step CPU/CUDA profile of the `s64hc10` WS path, at steps 150-154, with `RWKV_EMPTY_CACHE_EVERY=1` and `RWKV_VRAM_CAP_GB=11`: GPU kernel time 840.75 ms/step (81.64% elementwise/reduce/copy/optimizer, 14.82% WKV recurrence, 3.44% GEMM). The CPU-side profiler reported 311.6 `cudaFree` calls/step at 190 ms, 412.4 `cudaStreamSynchronize` calls at 122 ms, and 16,818 CUDA launches at 120 ms. A representative 9.6 MB pageable batch transfer took 3.78 ms synchronous; separate pageable copies took 4.04 ms. This does not justify copy-stream staging. Keep `RWKV_EMPTY_CACHE_EVERY=1`; investigate the `cudaFree` profile separately before considering allocator changes. Full trace: `scratchpad/dispatch/profile_s64/profile.log`.
