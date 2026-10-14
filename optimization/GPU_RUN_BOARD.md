# GPU training run board

Updated 2026-10-14. This is the current source for active and planned GPU training runs.
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
| 1 | `d128e10` | **Evaluation complete** | The d=128, 10+2 run completed with 2,279 scored users from the 2,499-user roster. Ahead LogLoss is 0.296280 / RMSE(bins) 0.057933. Imm LogLoss is 0.261156 / RMSE(bins) 0.034402. The eval log ends `EVAL_OK` and `DONE_EXIT_0`; the shard log and merged outputs are present. | 220 users were OOM-skipped; user 6701 was excluded by design. The saved OOM list is the union for this paired run. Against the published `base5k` result file, 1,955 of the overlapping 2,279 users have unequal stored review counts because the base uses the published label filter while this run uses `label_filter_db_id_e2s`. A user-paired LogLoss delta is not valid on that mixed review set. Do not treat these means as same-set accuracy or a clean capacity result. This run is a scaling diagnostic, not a deploy candidate. The WS had 225 failed validations under the 11 GiB cap; do not use its missing validation points as evidence. |
| 2 | Architecture-agnostic speedup | **Diagnostic profile complete; no transfer win** | Five-step `s64hc10` profile at steps 150-154; 11 GiB allocator cap, `RWKV_EMPTY_CACHE_EVERY=1`, empty-cache window 0. GPU kernel time: 840.75 ms/step. CPU view: 311.6 `cudaFree` calls/step cost 190 ms; 412.4 synchronizations cost 122 ms; 16,818 CUDA launches cost 120 ms. | A 9.6 MB pageable batch copy took 3.78 ms synchronous and 4.04 ms split. No case for copy-stream staging. Preserve `RWKV_EMPTY_CACHE_EVERY=1`. Smoke passed: 437 WS steps, final checkpoint and optimizer saved, 0 skipped batches, 0 failed validations, full Muon LR echoed; 0.941 steps/s, 9.756 GB peak allocator reserve. `scratchpad/wtmuon0p5smoke/`. |
| 3 | `wtmuon0p5` | **Running WS** | Muon LR 0.00125 (0.5x), AdamW unchanged; 10-epoch WS, followed by the existing 1-epoch decay and eval. | Detached runner PID 21896; started 2026-10-14 12:58 local. `RWKV_VRAM_CAP_GB=11`, `RWKV_EMPTY_CACHE_EVERY=1`. Smoke runner passed preflight and completed all trainer guards. |
| 4 | `wtmuon2` | Queued behind `wtmuon0p5` | Keep AdamW LR fixed. Set the Muon LR to 2 times its current value during the 10-epoch WS, then run its 1-epoch decay. | Compare with `s64hc10`. Together, runs 3-4 bracket the current ratio. |
| 5 | `wtwarmup2000` | Queued behind both Muon trials | Increase WS warmup from 400 to 2,000 steps. This trial needs only the WS and its validation endpoint. | Compare with `s64hc10`. The chain skips it if its guards fail. |
| 6 | QAT weight-decay sweep | **Planned; runner not generated** | On the accepted 4-stage WKV QAT recipe, change QAT-decay weight decay from 0.01 to 0.0 and 0.03. Keep the WS, quantizer, catalogs, and all other settings fixed. | Control: `116 s64hcq4`. Use the same 1,000-user screen as Phase 5. Evaluate quantized. Confirm any screen winner on all VAL users. |

The chain files are `scratchpad/d128e10/wait_d128.log` and
`scratchpad/wstune/chain_ws4.cmd`. The rented GPU was destroyed on 2026-09-29.
These runs must use Andrew's local RTX 4070.

## Monitoring

The hourly context heartbeat and five-minute GPU cold/low-util watch are enabled. Both passed live checks on 2026-10-14. The heartbeat sent no message while this conversation was active. The GPU watch reported the active `wtmuon0p5` run and skipped the destroyed rented box. The watch only reports an idle GPU when queued work remains; it does not restart runs.

## Next work

1. Finish the active WS tuning queue: `wtmuon0p5`, then `wtmuon2`, then `wtwarmup2000`. Check each trial's final artifact, skip count, validation failures, and paired LogLoss before recording its verdict. Keep `RWKV_EMPTY_CACHE_EVERY=1` and the 11 GiB cap.
2. After tuning is complete, ask Sol whether to increase the training budget. Do not launch a 100+10 run before that review. The current epoch comparisons show gains for the first move to 10 WS epochs, but do not measure a tenfold extension from today's model.
3. Continue work on QAT tax only if it remains in the planned queue; do not spend state bits above the updated limits without further review. Do not repeat closed WKV-stage, norm-range, norm-bit, or catalog/decay-LR trials.

## Current constraints

- State budgets: card <= 450 bits and note <= 400 bits. Andrew raised the note-state budget to 400 bits on 2026-10-14. Current deploy state: 403 card bits, 214 note bits.
- Every QAT result must use the learned catalogs and quantized evaluation path.
- Compare results on the same VAL users and same equalized reviews. Record users skipped by the VRAM cap.
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
