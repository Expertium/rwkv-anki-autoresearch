# GPU training run board

Updated 2026-10-12. This is the current source for active and planned GPU training runs.
`research_5k.md` and `research_log.jsonl` remain the record of completed results.

## Goal

Match the old d=128 model's ahead and imm LogLoss on the VAL users, 5001-7500, while keeping
quantized card and note state near 1,000 times smaller than the unquantized d=128 reference.
The reference is 0.294612 ahead and 0.263561 imm. The current d=64 quantized model, `s64hqr4b`,
scores 0.299369 ahead and 0.267496 imm. It uses 403 card-state bits and 214 note-state bits.
The card state is about 1,036 times smaller than the old fp32 card state.

## Active and queued runs

| Order | Run | State | Change | Control / decision |
|---:|---|---|---|---|
| 1 | `d128e10` | **Queued to resume** | The 10-epoch WS ended at step 223,460. The 2-epoch decay reached step 2,000 of 44,692. The resume config now loads step 2,000 weights and optimizer state, continues the LR schedule, and skips completed groups. | Compare with `w10lf`; this is a scaling diagnostic, not a deploy candidate. It gates the next WS-tuning trials. The original WS had 225 failed validation checks under the 11 GiB cap, so do not use its missing validation points as evidence. |
| 2 | `wtmuon0p5` | Queued behind `d128e10` | Keep AdamW LR fixed. Set the Muon LR to 0.5 times its current value during the 10-epoch WS, then run its 1-epoch decay. | Compare with `s64hc10`. This tests the Muon-to-AdamW LR ratio. |
| 3 | `wtmuon2` | Queued behind `wtmuon0p5` | Keep AdamW LR fixed. Set the Muon LR to 2 times its current value during the 10-epoch WS, then run its 1-epoch decay. | Compare with `s64hc10`. Together, runs 2-3 bracket the current ratio. |
| 4 | `wtwarmup2000` | Queued behind both Muon trials | Increase WS warmup from 400 to 2,000 steps. This trial needs only the WS and its validation endpoint. | Compare with `s64hc10`. The chain skips it if its guards fail. |
| 5 | QAT weight-decay sweep | **Planned; runner not generated** | On the accepted 4-stage WKV QAT recipe, change QAT-decay weight decay from 0.01 to 0.0 and 0.03. Keep the WS, quantizer, catalogs, and all other settings fixed. | Control: `116 s64hcq4`. Use the same 1,000-user screen as Phase 5. Evaluate quantized. Confirm any screen winner on all VAL users. |

The chain files are `scratchpad/d128e10/wait_d128.log` and
`scratchpad/wstune/chain_ws4.cmd`. The rented GPU was destroyed on 2026-09-29.
These runs must use Andrew's local RTX 4070.

## Next work after the queued runs

1. **(Recommended) Reduce QAT tax without spending more state bits.** Finish the QAT weight-decay sweep. If it does not help, review the quantization components against the current 403/214-bit limit. Do not repeat the closed WKV-stage, norm-range, norm-bit, or catalog/decay-LR trials.
2. Improve the full-precision model with training or architecture changes that fit the same state budget. Gate each change on both ahead and imm LogLoss.
3. Improve the training pipeline. Keep training, evaluation, and inference inputs aligned. Re-run the full deploy checks after any input or objective change.

## Current constraints

- The 403-bit card and 214-bit note states meet the approximate 1,000x target. Do not spend more bits without Andrew's approval.
- Every QAT result must use the learned catalogs and quantized evaluation path.
- Compare results on the same VAL users and the same equalized reviews. Record users skipped by the VRAM cap.
- The RTX 4070 cap is 11 GiB. A capped training run must pass `scratchpad/count_skips.py`.
- Keep at least 10 GB of system RAM available. Do not overlap a giant-user eval with another memory-heavy task.

## Utilization note

At 2026-10-12 15:33, Windows GPU Engine counters attributed about 25% of the 3D engine to
Agent Canvas GPU process PID 4616 and 8-12% to `dwm.exe` PID 1516. The Canvas process is Chromium's
GPU process. It showed no CUDA or compute-engine use. The main Canvas renderer had high cumulative
CPU time, but low current CPU use. Minimizing the window did not reliably lower the reported 3D
activity. A later `nvidia-smi` sample showed only 1% GPU use, 34 W, and 1,737 MiB. The cause is not
confirmed. Do not call the 25% draw normal. Compare Windows 3D-engine counters with NVIDIA utilization
and power over the same interval.
