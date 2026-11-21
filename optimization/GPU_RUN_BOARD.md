# GPU training run board

Updated 2026-10-23. This is the current source for active and planned GPU training runs.
`research_5k.md` and `research_log.jsonl` remain the record of completed results.

Status checked 2026-11-04. Run 5 is complete. Queue item 6 is complete: WD=0.0 and WD=0.03 full decays and standard screen evaluations completed cleanly; both arms passed isolated 0.04-epoch smoke, quantized probe, and 2-user production-evaluator gates. Each treatment scored 983/1,000 screen users and OOM-skipped the same 17; control 116 scored 998 and omitted users 5626 and 5859. On the shared 983-user roster, WD=0.0 gains were +0.000031/+0.000029 (p=0.431/0.142); WD=0.03 gains were +0.000018/+0.000055 (p=0.310/0.000695). Both treatments fail the raw +0.0001 and p<1e-4 gates, so no full-VAL confirmation is due. Each decay completed 10,935 steps with zero skipped batches and failed validations. Host RAM was 33.2 GiB free and the RTX 4070 idle at the latest check. The adaptive-WD experiment is not ready to train: the previous adaptive LMDBs store 69 card features (109 total model inputs with ID encodings), and the second full-run attempt failed at baseline load before training. Preserve that log. Fresh sequential CPU rebuild is complete with exit code 0 for both splits: `F:/rwkv_lmdb/train_db_5k_h1_adaptive92` and `F:/rwkv_lmdb/test_db_5k_adaptive92`, from `../anki-revlogs-10k-id` with `F:/rwkv_lmdb/label_filter_db_id_e2s`; optional ID/real-cycle features were off. Sampled width audit passed for both databases: 24 stored card columns. Both full LMDB audits now pass. Train: all 5,000 users present, 4,417 multi-batch users, every feature tensor is 24-wide, zero bad tensors. VAL: all 2,500 users present, the known four multi-batch users (5626, 6630, 6701, 6810), every tensor 24-wide, zero bad tensors. The audit outputs are preserved in `scratchpad/adaptive_wd/audit_train_db_92.log` and `audit_val_db_92.stdout.log`; the earlier full VAL audit was stopped and is superseded by the successful rerun. A fresh RAM-guard telemetry entry was recorded at 08:48 on Nov 5, and available RAM was above 34 GiB at smoke launch. The isolated 92-input `EPOCHS=0.04` smoke in `scratchpad/adaptive_wd/smoke_92_20261105/` completed with `ADAPTIVE_SMOKE_OK` and exit 0. Training reached step 1,734, saved a model+optimizer checkpoint pair, and the production baseline and trial evaluation both scored user 5002 in ahead and immediate modes with no OOM/NaN skips. The state file recorded the trial and incumbent; the smoke trial was accepted. This is plumbing evidence only, not an accuracy result. RAM stayed above 35 GiB free and the GPU was idle after the smoke. The RAM-guard process remains alive; its latest log entry is from smoke execution and `scratchpad/ram_guard.py` is absent from this checkout, so it cannot be restarted from source here. Adaptive-WD full run is active in `scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/` on the 92-input model and rebuilt -id/e2s databases. Attempts 1–4 each completed 43,354 steps and passed the strict dual-mode screen on the same deterministic 750-user roster. All four were accepted at WD=0.01. Attempt 4 scored ahead 0.304392 and immediate 0.272142; its model and optimizer pair are saved at step 173,416.

The old 30-attempt controller stopped before attempt 5 with `FileExistsError` because temporary stop markers occupied the next-attempt paths. The saved state was intact. The markers were verified and removed, and the controller resumed from saved state with Andrew's limits: at most 7 rejected epochs and at most 10 accepted epochs. The new controller and resume-schedule tests pass.

As of 2026-11-15 14:00 Moscow time the adaptive-WD search is finished and its final QAT stage is blocked on a recipe decision. Attempt 11 at WD 0.64 was rejected at 0.311561/0.279959, taking the rejected count to 7 (4 accepted, 11 attempted). The incumbent is still attempt 4 at WD 0.01 (0.304392/0.272142), so the adaptive WD schedule bought nothing over a fixed WD=0.01 in this 1-epoch-per-attempt regime, and larger WD is monotonically harmful in both modes. The controller then crashed in its final QAT stage: the QAT child died with FileNotFoundError because `final_qat/validation_users.json` was never written, and a second bug passed the 750-user screen roster where the trainer expects the 2-user internal roster. Both are fixed and a real 86-step smoke of the fixed path completes. The QAT stage still cannot be launched as specified: `qat_env.json` mostly sets flags no code reads, so the epoch would run at full precision and its full VAL would not be a deploy number. The GPU is idle, free RAM is about 34 GiB, no training or eval process is running, and `scratchpad/GPU_COLD.txt` is absent. Rows 8 and 9 wait on the same QAT recipe so that all three arms share one final stage. On 2026-11-15 Andrew pivoted away from adaptive weight decay to the TA3 arm (row 10) and authorized its four-epoch run. The adaptive result stands as a screen-only negative result: adaptive WD switching found nothing better than its fixed WD=0.01 incumbent, and its final QAT stage stays blocked (recipe decision) and unrun. That first chain attempt died at epoch-1 step 4,930 on 2026-11-15 (see row 10) and the GPU was idle ~12 h afterwards; the split was fixed, a fresh 0.04-epoch smoke passed, and the chain restarted 2026-11-16 08:37.

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
| 3 | `wtmuon0p5` | **Evaluation complete; rejected** | 10+1 run with Muon LR x0.5 and AdamW unchanged. WS and decay completed cleanly. Normal full-VAL eval completed at 11:41 Moscow time on 2026-10-16 after resume from the F: drive disconnect. Candidate own means: ahead 0.297417 / imm 0.264327. Paired against 104 `s64hc10` on 2,454 users with identical review counts: delta -0.000053 ahead (p_better 0.265) / -0.000000 imm (p_better 0.657). | EITHER-MODE gate fails: no significant improvement, no significant harm. 2,454 scored users, 45 OOM skips; size unchanged on all paired users. Muon LR x1 remains the best measured setting. The 11 GiB cap and `RWKV_EMPTY_CACHE_EVERY=1` remained set. |
| 4 | `wtmuon2` | **Halted; host-RAM investigation** | WS reached checkpoint 4,000/10,935 after resuming from 2,000; runner hit its 40-attempt guard at 17:19 on 2026-10-23 after RAM exhaustion. | Checkpoint pair at 4,000 is preserved. Resume only after a memory-safe worker/prefetch plan is validated. Do not run decay/evaluation from this intermediate checkpoint. |
| 5 | `wtwarmup2000` | **WS + decay complete; standard full-VAL evaluation complete** | The original two-fetch-worker WS stopped near step 6,076 during host-RAM pressure. Andrew authorized a one-worker resume from the preserved step-6,000 model/optimizer pair. The resumed WS finished global step 109,350 and saved its model+optimizer pair. Its remaining warmup was correctly 1 step; the original runner falsely expected 2,000 and exited 72 after WS, so it was not rerun. A separate one-fetch-worker decay from WS-final then completed all 10,935 steps and saved the complete model+optimizer pair at step 10,935. | WS evidence: `scratchpad/wtwarmup2000/ws_2880716635.log`; resume audit: `endpoint_audit.json` (last-five paired 10-user deltas vs `s64hc10`: about -0.000680 ahead / +0.000260 imm). Decay: `decay_3159526217.log`; runner `wtwarmup2000_resume1w_de_20261025.log` ends `DECAY_OK`, `EVAL_OK`, `DONE_EXIT_0`. Decay had zero skipped batches and zero failed validations. Final pair: `wtwarmup2000_d_10935.pth` + `wtwarmup2000_d_optim_10935.pth`; WS-final pair: `wtwarmup2000_ws_109350.pth` + `wtwarmup2000_ws_optim_109350.pth`. Canonical `optimization/eval_sharded.py` full VAL completed 2,454/2,499 users, 45 CUDA OOM skips, 0 NaN skips; mean by-user LogLoss 0.297460 ahead / 0.264357 imm. Outputs: `result/RWKV[-P]-wtwarmup2000.jsonl`, OOM list `result/RWKV-wtwarmup2000.oomskip.jsonl`; shard log `scratchpad/wtwarmup2000/eval_shard_3159526217.log`. This is an evaluation record, not yet a paired acceptance verdict. RAM-guard telemetry was fresh at the resumed decay launch; available RAM remained above the 10-GiB floor in observed checks. The GPU watcher process is alive, but no idle-transition event was in its history at the most recent audit. |
| 6 | QAT weight-decay sweep | **Complete; both treatments rejected at screen** | Both arms passed separate 0.04-epoch smoke runs (437 steps; 0 skipped batches and 0 failed validations), learned-catalog probes, and 2-user standard sharded-eval checks. Each full decay from `s64hc10_ws_109350` completed 10,935 steps, saved model+optimizer and learned catalogs, passed its quantized probe, and had zero skipped batches/failed validations. Standard production screen evaluations scored 983/1,000 users per treatment, with 17 OOM skips (list in the final column). Against control 116 on the common 983-user roster, WD=0.0 gains were +0.000031/+0.000029 (p=0.431/0.142); WD=0.03 gains were +0.000018/+0.000055 (p=0.310/0.000695). Both fail raw +0.0001 and p<1e-4; neither is eligible for full-VAL confirmation. The WD=0.03 smoke/probe/2-user eval were repeated 2026-10-28 after its full run had completed. | Control 116 `s64hcq4` scored 998 and omitted 5626/5859. Both treatment arms omitted 5126, 5162, 5164, 5209, 5333, 5414, 5456, 5507, 5626, 5641, 5715, 5731, 5823, 5835, 5859, 5900, 5991. Shared roster n=983. The fourth optimizer group is fixed `encode_params` decay 0.01 in `rwkv/train_rwkv.py`; each treatment changed the other three groups. Paired details: `scratchpad/phase5/compare_s64_wd.py`; logs: `scratchpad/p5wd0s64/`, `scratchpad/p5wd0p03s64/`. |
| 7 | Adaptive weight-decay training (92-input) | **Search complete; final QAT stage blocked** | Start: QAT46 step-248 model+optimizer. Fixed 750-user screen from VAL 5001–7500; accept an epoch only if both ahead and immediate mean LogLoss improve. Attempts 1–4 (WD 0.01) accepted: 0.308803/0.275131, 0.307044/0.274275, 0.305847/0.272732, 0.304392/0.272142 (incumbent, step 173,416). Attempts 5–11 all rejected, and every larger WD is worse in BOTH modes: WD 0.01 0.305330/0.272290, 0.02 0.305640/0.272493, 0.04 0.305999/0.273207, 0.08 0.307400/0.274062, 0.16 0.308387/0.275737, 0.32 0.309645/0.276878, 0.64 0.311561/0.279959. The 7-rejected limit ended the search: 11 attempted / 4 accepted / 7 rejected, so adaptive WD switching found nothing better than the WD=0.01 incumbent. | Run dir `scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/`; state `adaptive_state.json`; controller log `resume_controller_20261109.log`. Baseline on the fixed roster 0.346251/0.314766; every screen scored 750/750 users, 0 OOM, 0 NaN. The final QAT stage crashed twice on runner bugs, both fixed 2026-11-15 (missing `final_qat/validation_users.json`; the QAT `make_train_config` call passed the 750-user screen roster instead of the 2-user internal roster, tripping the trainer's VAL-range assertion). An 86-step real-path smoke in `scratchpad/adaptive_wd/qat_smoke_20261115/` now completes (`MODEL SAVED.`, validation 0.3256/0.3038, model+optimizer pairs). **BLOCKED on a recipe decision:** `scratchpad/adaptive_wd/qat_env.json` sets `RWKV_QAT_NORM`, `RWKV_QAT_LORA` and `RWKV_QAT_CLOCK_FIX`, which no code reads, and `RWKV_QAT_PQ=1`/`RWKV_QAT_SHIFT_PQ=1`, whose values the loader treats as codebook FILE PATHS; the flags that actually gate quantization (`RWKV_QAT_LOWRANK_SCOPE`, `RWKV_QAT_SHIFT_SCOPE`, `RWKV_QAT_NORM_BITS`) and the learnable-catalog flags are absent. The smoke's validation matched the plain attempts, so no quantization was active. The canonical full VAL also runs in the plain env, against the recorded rule that a QAT run must be evaluated with quantization on. Needs Andrew's decision before launch. |
| 8 | Fixed-WD low control | **Queued second, after adaptive final evaluation** | Fixed WD=0.01 for 10 accepted training epochs, then one separate final decay/QAT epoch at WD=0.01. Match the adaptive run's start checkpoint (QAT46 step-248 model+optimizer), 92-input/-id/e2s data and features, optimizer, per-epoch decay LR schedule, worker/resource limits, fixed 750-user screen, final-QAT settings, and canonical full-VAL evaluation. No screen-based accept/reject or WD switching. | Run after row 7 completes its adaptive attempts, final decay/QAT and canonical full VAL. Run the isolated 0.04-epoch smoke and preflight first. Preserve artifacts and compare the same users. This control uses exactly 10 training epochs; report adaptive's actual attempt count separately. See `scratchpad/adaptive_wd/NOTES.md`. |
| 9 | Fixed-WD high control | **Queued third, after row 8** | Fixed WD=0.64 (the maximum adaptive-grid value) for 17 training epochs, then one separate final decay/QAT epoch at WD=0.01 (18 epochs total). Match rows 7–8 in start checkpoint (QAT46 step-248 model+optimizer), 92-input/-id/e2s data and features, optimizer, LR/schedule, worker/resource limits, fixed screen roster, final-QAT settings, and canonical full-VAL evaluation. No accept/reject or WD switching. This asks whether a static high-WD run uses the adaptive run's maximum allowed training-attempt budget more effectively. | Run after rows 7 and 8. Run the isolated 0.04-epoch smoke and preflight first. Preserve artifacts; report actual training attempts and elapsed compute for each arm, since the adaptive run may stop before all 17 attempts. See `scratchpad/adaptive_wd/NOTES.md`. |

The chain files are `scratchpad/d128e10/wait_d128.log` and
`scratchpad/wstune/chain_ws4.cmd`. The rented GPU was destroyed on 2026-09-29.
These runs must use Andrew's local RTX 4070.

## Monitoring

The hourly context heartbeat and five-minute GPU watch are enabled. On 2026-10-16, the watch missed an idle GPU because it treated a stale `eval_sharded.py` parent as proof of active GPU work after the normal evaluation had completed. Andrew's screenshots were at 12:59 and 16:04 Moscow time (UTC+3); the normal evaluation ended at 11:41 Moscow time. OpenHands showed the screenshot time as 16:25, but Windows confirmed Moscow time (UTC+3); use Windows event times as local time. The watch keeps the process, log, and queue checks and now uses a 20-minute rolling GPU rule: idle when the 90th-percentile utilization is below 30% and the 90th-percentile VRAM use is below 2,000 MiB. Prefix checks on one 20-minute training, evaluation, and idle capture classified all 12 five-, ten-, fifteen-, and twenty-minute windows correctly. It keeps its one-minute sample when the scheduled five-minute task runs late by up to eight minutes; otherwise it restarts the history. A smoke test and one live invocation passed. The live check reported 42% utilization p90 and 933 MiB VRAM p90, so it classified the GPU as active. The watch skips the destroyed rented box and never restarts runs.

## Fast evaluation implementation (2026-10-23)

- `optimization/fast_evaluate.py` is an experimental accelerator for `rwkv.get_result`. It preloads complete LMDB sections per user, then runs one model forward per original section to preserve reference seed/RNG, reduction order, and whole-user OOM/NaN semantics. It uses the current model and scorer plus equalized label-filter tensors; RAW mode is unsupported. Optional result JSONL writing requires exact parity for the full requested roster and no skipped users.
- Fast smoke for users 5005-5014: 10/10 completed, no skips; all five users present in the older `wtmuon0p5` output matched exactly, but that output lacks users 5010-5014. A separate two-section user 5002 test passed exact parity: `result/fast-eval-multisection-5002.json` / `scratchpad/fast_eval_multisection.log`. Cross-user packing was removed after a metric mismatch; see `scratchpad/fast_eval_crossuser_parity10.log`.
- A fresh standard `rwkv.get_result` reference run for users 5005-5014 exists as `RWKV-fast-ref10-20261022` / `RWKV-P-fast-ref10-20261022`. The fast run matched exact metrics and sizes for all 10 users in both modes, with no skips. Timing: 66.28 s total, 60.10 s forward, 3.01 s preparation wall, peak allocated 4.43 GiB / reserved 5.76 GiB. Run: `result/fast-eval-reference-compare-retry3.json`; log: `scratchpad/fast_eval_reference_compare_retry3.log`. The standard reference took about 9m30s by timestamps but its Python process returned native-teardown error `0xC0000409` after producing all 10 outputs; its wrapper marker is not an independent success proof.
- Cross-user section packing was removed after an exactness mismatch. Per-user smoke (5005-5014) and user 5002's two-section case passed exact parity; user 5626 OOM-skipped as a whole without output. Full VAL completed: 2,457/2,499 users, 42 OOM skips, no NaN/incomplete skips. All 2,453 shared users matched exact metrics and sizes; 4 fast-only users were in the standard run's shard OOM list. Standard completed user 6197 but fast OOM-skipped it. Runtime 18,996 s (5h16m), forward 16,984 s, preparation 1,490 s, peak allocated/reserved 10.65/11.00 GiB; standard log reports 6,169 s (1h43m), so this implementation is about 3.08x slower. Report/log: `result/fast-eval-fullval-20261023.json`, `scratchpad/fast_eval_fullval_20261023/fast_eval_fullval.log`. A 1-user JSONL-output integration test and overwrite negative test passed using the fresh standard reference: `result/fast-eval-integration-5005.json` and `result/RWKV[-P]-fast-integration-5005-retry1.jsonl`; log `scratchpad/fast_eval_fast-integration-5005-retry1.log`. Correctness and output gates pass; speed gate fails. Do not use this implementation for upcoming evaluations. It is not wired into `eval_sharded.py` or standard scripts; standard evaluation remains the queue path.

## Next work

1. Fast-eval gate is closed for this implementation. Full-VAL metric/size parity passed on 2,453 shared users; 42 users OOM-skipped in fast mode, and user 6197 succeeded only in standard evaluation. The 1-user JSONL-output and overwrite-protection test passed. But fast evaluation took 5h16m vs 1h43m standard (3.08x slower), so it fails the speed goal. Keep `optimization/fast_evaluate.py` experimental. Standard `eval_sharded.py` remains the production/queue path while any new speed design gets built and parity-tested.
`wtmuon2` WS resumed from checkpoint 2,000 and saved through step 4,000; its runner later exhausted 40 attempts and stopped with `DONE_EXIT_29` after RAM pressure. Preserve its checkpoint pair; do not evaluate it. `wtwarmup2000`, launched at Andrew's instruction, was stopped around step 6,076; preserve its complete model+optimizer pair at step 6,000. During the unassisted two-worker full runs, each fetch worker showed ~19-20 GiB working set but ~2.6-3.1 GiB private bytes; after stopping the trees, available RAM recovered from <1 GiB to ~42.5 GiB. This points to worker memory/cache use associated with LMDB access, but does not establish one sole cause. LMDB `map_size` is a virtual mapping limit, not eager physical allocation. The separate 1-worker smoke below shows substantially lower peak memory under short-run load.

An isolated one-worker 0.05-epoch GPU smoke completed: 546 steps, saved model+optimizer, 0 skipped batches, 0 failed validations. A separate 30-minute telemetry process captured 95 samples over ~18 minutes (training lasted ~16.5 minutes; monitor continued after the run). Available RAM ranged 28.83-41.87 GiB (mean 34.86 GiB), never below 10 GiB. The single fetch worker peaked at 9.82 GiB RSS / 9.67 GiB USS; train parent peak RSS was 2.47 GiB. The periodic RAM guard also trimmed the two workers during this one-worker-config smoke; therefore these measurements validate one worker plus the guard, not one worker unassisted. The earlier unassisted two-worker full runs had minima below 1 GiB. Thus one worker passed this short/smoke-sized safety check, not a full-run guarantee. The 2-worker full runs remain stopped; do not resume them automatically. For any approved full resume, set `NUM_FETCH_PROCESSES=1`, keep the RAM guard enabled, enforce >=10 GiB available, and monitor. If one worker becomes a throughput bottleneck, evaluate readahead/other mitigations separately rather than increasing worker count without measured headroom.

The Windows RAM guard (`scratchpad/ram_guard.py`, task `ClaudeRamGuard`) is enabled and running. It caps fetch-worker working sets; this can reduce hot-page residency and hurt I/O throughput. During the one-worker smoke it initialized a 6-GiB cap, then raised it to 15.6 GiB while available RAM stayed >=28.8 GiB; the worker peaked at 9.82 GiB RSS. Thus the guard was running but did not bind the observed worker peak. It is still not a full-run guarantee. The GPU idle notifier is enabled; sampler PIDs 16408/50852 were alive after restart, with a 19:57:50 `watch_started` record. It logs errors and idle/activity transitions, not periodic samples. No training/eval workload is active; diagnostic artifacts are isolated under `scratchpad/ramprobe1w_20261023/`.
3. The QAT weight-decay sweep is complete; both WD arms failed the paired screen gate. Results are in `research_5k.md` rows 126-127. No full-VAL confirmation is warranted. Next: assess the separately authorized 3,000/3,000-bit QAT experiment against kernel capacity, engine/exporter support, and projected run time before making a runner. Keep existing deployment budgets and contract unchanged until the authorized experiment passes its checks.

## Current constraints

- Current deployed state remains 403 card bits / 214 note bits. The queued QAT recipe uses caps of card <= 450 bits and note <= 400 bits. Andrew approved a separate post-queue experiment with 3,000 card-state bits and 3,000 note-state bits; this is not a deploy-contract change.
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

| 10 | TA3 (Training-Time Testing) arm | **ABORTED 2026-11-19 mid epoch 4 by Andrew's decision; recorded as a screen-only NEGATIVE result (research_log #128, research_5k row 128)** | `rwkv/ta3.py` + trainer hook, env-gated (`RWKV_TA3=1`, inert when unset). Per step: A/Q loss-mask split (data stream unchanged), normalised downhill trial step on A (eta 0.3, scope = 28 channel-mixer weight matrices), Q's gradient at the perturbed weights, exact weight restore, and a per-group weighted merge (ahead-masked group and immediate group carry their own half/full row-count ratios; each group is back-propagated on a mode-masked pass because the model reports its term averages detached). Probe on 30 real batches: downhill on A lowers Q's loss, beating equal-norm random 27/30 (p 1e-7..2.6e-6). Smoke `EPOCHS=0.04`: 1,734 steps, 0 tracebacks, 0 skips, artifacts written. | Gate run `scratchpad/ta3_gradcheck_20261115/`: weights restored bit-exactly (0.0 drift) and optimizer state untouched (0.0) PASS; the eta=0 gradient differs from the plain gradient by 2.1-12.7% BUT the identical plain pass repeated differs by 0.7-10.9%, so the eta=0 difference is at the path's own reproducibility floor (bf16, no QAT, dropout off) and the eta=0.3 effect (18.7-51.1%) is above it. Value-level check: the objective equals the two groups to 1e-7, so the decomposition is right. Restated as a statistical gate it passes; an exact gate needs fp32 (scripted model rejects it today) or a nondeterminism investigation. Four-epoch arm mirrors the plain chain (offsets 1 / 43,355 / 86,709 / 130,063). STARTED 2026-11-15: the corrected 0.04-epoch smoke `scratchpad/ta3_smoke_fixed_20261115/` (built from the REAL start config `attempt_01_epoch_01_wd_0.01` = QAT46 step-248) and a config pre-flight (`scratchpad/ta3/ta3_train.py --dry-run`, which reproduces offsets 1 / 43,355 / 86,709 / 130,063 and the load chain rwkv_iter46_248 -> ta3_43354 -> ta3_86708 -> ta3_130062) both passed. The resumable chain controller is `scratchpad/ta3/ta3_train.py` (launcher `scratchpad/ta3/run_ta3_chain.cmd`, chain log `scratchpad/ta3/chain.log`, run dir `scratchpad/ta3_fullrun_20261115/`). Nothing differs from the plain chain except `RWKV_TA3=1` (eta 0.3, scope cmix, frac 0.5). Pre-registered endpoints and hazards: `scratchpad/ta3/NOTES.md`. Primary endpoint = the 4-epoch TA3 checkpoint vs the plain 4-epoch checkpoint (0.304392/0.272142) on the same fixed 750-user roster, gate = both modes improve >= 0.0001 with p < 1e-4; per-epoch screens are diagnostic only, with no accept/reject and no rollback. The per-step `[ta3]` log line prints the A-phase ahead-masked pass, so its `imm` field is `0.000` by construction; progress is tracked through the trainer's own 2-user validation, not that line. Corrected 0.04-epoch smoke result (2026-11-15 16:53-17:34, `scratchpad/ta3_smoke_fixed_20261115/`): 1,734 TA3 steps, 0 tracebacks, 0 skipped batches, 0 aborts, 0 non-finite steps, 5 internal validations (0.3650/0.3386 -> 0.3385/0.3137 on the 2-user roster), model+optimizer pairs at steps 50/500/1000/1500/1734, and 0.729 steps/s (~44 steps/min) versus ~40 steps/min for the plain attempts, i.e. the extra three passes cost almost nothing in wall time because the run is fetch-bound. Epoch 1 started 17:36 and its step-50 validation (0.3650/0.3386) reproduces the smoke's exactly, so the two paths start identically. Expected: ~17-18 h per epoch, about 3 days for the chain. **INCIDENT 2026-11-15/16.** Epoch 1 aborted at step 4,930 (19:55, last pair `ta3_4500.pth`) with `RuntimeError: TA3: mode 'ahead' has no labelled rows in this phase`, and the GPU then sat idle ~12 h because nothing was watching. Root cause: the A/Q split used ONE cut at `round(0.5*T)` on the raw time axis, and some real batches put a whole loss group on one side of it. Fail-closed, so nothing was corrupted. FIXED in `rwkv/ta3.py`: each loss group is now split by RANK over its own labelled rows (`_rank_split`/`_mode_split`); a group with fewer than two labelled rows lands whole in phase A, where the merge gives it weight 1 and its Q term weight 0 -- exactly its full-batch gradient -- and `training_step` runs only the (phase, group) passes that have rows. A first revision that cut each group at its own MIDPOINT as a time threshold was rejected by the test: with more than one chunk per batch several rows share a time position and a threshold collapses whole tied groups. Evidence: `scratchpad/ta3/test_split.py` (1,200 random layouts, all invariants) and a read-only scan of EVERY training chunk (`scratchpad/ta3/scan_batch_layouts.py`, `scratchpad/ta3/layout_scan_lean.log`): over 5,000 users / 46,062 chunks the OLD rule left a phase empty for 55 chunks (plus 4 in immediate mode, 0.13%; short tail chunks where all ahead rows precede all imm rows), 43 chunks have a group with fewer than two labelled rows, and the NEW rule has ZERO failures. Smoke #2 after the fix (`scratchpad/ta3_smoke_fix2_20261116/`): 1,734 steps, 0 tracebacks/skips/aborts/non-finite, 5 internal validations 0.3668/0.3406 -> 0.3378/0.3139, final pair `ta3smoke2_1734.pth` + optim. The chain restarted 2026-11-16 08:37 in `scratchpad/ta3_fullrun_20261116/` (chain log `scratchpad/ta3/chain_20261116.log`); the aborted `scratchpad/ta3_fullrun_20261115/` is preserved and NOT resumed. `scratchpad/ta3/ta3_train.py` also retries a crashed epoch from scratch (`--retries 2`, logs `train_retryN.log`, failures recorded in `ta3_state.json`) so a crash no longer idles the GPU until a human notices. **Progress 2026-11-19 14:45:** epochs 1-3 completed cleanly (43,354 steps each, 88 validations each, zero tracebacks, zero skipped batches, zero failed validations, a final model+optimizer pair per epoch) and the chain moved to epoch 4 without a retry. Their DIAGNOSTIC screens on the fixed 750-user roster are worse than the EPOCH-MATCHED plain screens: e01 0.309108/0.275626 vs plain 0.308803/0.275131 (paired ahead delta -0.000305, p=0.9995; imm -0.000494, p=1.0); e02 0.309924/0.273467 vs 0.307044/0.274275 (ahead -0.002880, p=1.0; imm +0.000808, p=5.0e-29); e03 0.306259/0.273344 vs 0.305847/0.272732 (ahead -0.000412, p=0.9994; imm -0.000613, p=1.0). Deltas are plain minus TA3, so negative = TA3 worse. All three epochs fail the dual-mode gate and the ahead mode is worse in every one, so the interim trend is negative; the pre-registered PRIMARY endpoint is still the epoch-4 checkpoint (`epoch_04/ta3_173416.pth`) versus the plain epoch-4 checkpoint on the same roster. Epoch 4 started 2026-11-19 10:33:49 and is at step 138,044 (7,982 of 43,354) at ~31-32 steps/min, slower than the 44 steps/min the 0.04-epoch smoke showed; epochs 2 and 3 each took ~23.5 h of training, so epoch-4 training ends about 2026-11-20 09:00 and its screen verdict about 10:00. **ABORTED 2026-11-19 15:39 by Andrew's decision** (the chain process tree was killed; the pre-registered 4-epoch endpoint was NOT reached, so this arm stands as screen-level evidence only). Recorded as a NEGATIVE result: `optimization/research_log.jsonl` entry #128 and `optimization/research_5k.md` row 128. The TA3 direction is CLOSED here; the next direction is quantization (card + note state at <= half the int4 bit count, both heads within +0.0005 of the current best). Epoch-4 partial artifacts (`epoch_04/` to step 138,000) and the whole `scratchpad/ta3_fullrun_20261116/` tree are preserved, not resumed. `ta3_state.json` holds epochs 1-3 with their screens; the RAM-guard heartbeat is fresh at 20.8 GB free (above the 10 GiB floor); the GPU is active at ~49% / 3.2 GB. |
| 11 | **Card/note state quantization (Andrew's new direction, 2026-11-19)** | **PTQ phase complete and negative; QAT decay epoch ACTIVE since 2026-11-19 21:25 Moscow** | Quantize the card and note recurrent WKV states only (deck/preset/global untouched) to at most half of plain fp32->int4, with both heads within +0.0005 of the current best and at most 25% slower on the Rust deploy engine. PTQ sweep on the seeded random 750-user VAL roster (seed 20261119) against the unquantized baseline 0.300609/0.272009: rank-1 int4 factors+shifts 512/1,536 bits +0.000943/+0.001397 FAIL; full-matrix int2 exactly AT the 2,176/6,528 bit bar +0.009785/+0.011579 FAIL; rank-1 int8 1,024/3,072 bits +0.000794/+0.001132 FAIL (best untrained scheme). No untrained scheme passes, so the arm moved to the QAT option of Andrew's constraint 4. | QAT: ONE decay epoch from `adaptive_173416.pth` (the same checkpoint as the arm's baseline) with `RWKV_QAT_LOWRANK_SCOPE=card:1:int8,note:1:int8` + `RWKV_QAT_SHIFT_SCOPE=card:int8,note:int8`; runner `scratchpad/quant_plan/qat_decay.py --candidate l1i8 --epochs 1 --tag l1i8e1` (detached, launcher log `scratchpad/quant_plan/qat_l1i8e1_launcher.log`, trainer log `scratchpad/quant_plan/qat_l1i8e1/train.log`). The required 0.04-epoch smoke `qat_l1i8smoke` passed first: step 1,734, `MODEL SAVED.`, 0 skipped batches, and the `[QAT-LOWRANK] card_id=rank1/fq127.0, note_id=rank1/fq127.0` plus `[QAT-SHIFT]` banners proved quantization ACTIVE (the adaptive-WD QAT stage's broken `qat_env.json` is bypassed). 43,354 steps expected, ~12 h at the observed ~1.04 steps/s. Next: evaluate the saved checkpoint on the same 750-user roster with the same quantizer (and measure the Rust speed) and compare the paired deltas against `qbl_incumbent_a04`. Results table: `scratchpad/quant_plan/RESULTS.md`; notes: `scratchpad/quant_plan/NOTES.md`. One GPU workload at a time; the deployed budget stays 403 card / 214 note bits until a candidate passes. |

Row 11 amendment (2026-11-19 22:05, Moscow). The first full QAT decay epoch (`qat_l1i8e1`, launched
21:35 from `adaptive_173416.pth`) was **STOPPED** at 21:45, because Andrew's anchor rule requires the
QAT redo to start from the checkpoint BEFORE the run's one decay epoch, and `adaptive_173416` is the
checkpoint AFTER it. The partial folder `scratchpad/quant_plan/qat_l1i8e1/` is kept but must not be
resumed or scored. The validated 0.04-epoch smoke `qat_l1i8smoke` still stands as the code-path check
(step 1,734, `MODEL SAVED.`, 0 skipped batches, quant banners active). Anchor resolution: the adaptive
attempts turned out to be one-epoch segments of a single flat-LR 4-epoch schedule
(`attempt_04/train.log`: `LR schedule continues at step 130062 of 173416 (factor 1.000000)`,
`LR decay shape = constant`), so the only clean pre/post-decay pairs in this checkout are the old-regime
runs `iter36` (WS-final step 124 -> step 558) and `iter44`/`iter45` (8/16-epoch decay-QAT from the same
WS-final). `scratchpad/quant_plan/score_anchors.py` is scoring `iter45_496`, `iter46_248`, `iter44_248`
and `iter36_558` unquantized on the 750-user roster against the current 0.300609/0.272009 baseline; the
anchor and then every result row will be re-based on the winner. `qat_decay.py` now takes
`--start-folder/--start-name/--start-optim/--peak-lr/--dry-run` so the epoch can be redone from any
checkpoint. Details: `scratchpad/quant_plan/NOTES.md` ("Anchor investigation") and `RESULTS.md`.

Row 11 amendment 2 (2026-11-19 22:35, Moscow). **The iter36 pre/post pair was mapped the other way
round in amendment 1.** The decay phase RESTARTS its step counter, so `rwkv_iter36_558.pth` is the
WS-final (**pre**-decay, the QAT start) and `rwkv_iter36_124.pth` is the post-decay champion (**the
baseline**). Evidence: qat_log #40 describes the lineage as "558 WS + 124 decay"; every decay run is
named after its own step count (iter39/43 = 124, iter44 = 248, iter45 = 496, iter46 = 248); the
opt_iter36 mtimes come in two phases (200-558 written 15:41-15:45, then 50/100/124 written 15:46-15:47)
and the two phases have different optimizer file sizes. So the decay is a **124-step** phase, not
"2 epochs": every step count in this lineage is a multiple of 31, i.e. the old data config had 31
groups/epoch and iter36's decay was 4 of them. The old users-1-100 training dbs (`train_db`,
`train_db_sc8k`) are GONE -- no lmdb survives anywhere on C: or F: except the 5k ones -- so the redo
trains on `train_db_5k_h1_adaptive92` restricted to users 1-100 at MAX_TRAIN_GLOBAL_LEN=66000
(`probe_groups.py`: 160 groups/epoch, 0 chunks dropped, 9.66M rows). Also corrected: iter43/44/45
were NOT QAT runs (HISTORY.md: the restored arch_iter36 was missing the `[QAT]` parser), so their
unquantized scores are valid and they remain legitimate anchor candidates.

Chain launched 22:23 (`scratchpad/quant_plan/it36qat_chain.py`, log `it36qat_chain.log`): it waits
for the anchor baseline eval of `it36_124`, then runs a 0.04-epoch smoke -> the `--no-quant` fp32
control (124 steps) -> the QAT run (candidate `c2_nl1i8` = card full-matrix int2 + note rank-1 int8,
124 steps, 2,176/3,072 bits against the 2,176/6,528 bar) -> an unquantized eval of both new
checkpoints against tag `qnj_it36_124` -> the remaining anchor diagnostics (`it36_558`, `it45_496`,
`it46_248`, `it44_248`). One GPU workload at a time; the chain stops the eval chain itself before
training starts.


Row 11 amendment 3 (2026-11-20 07:47, Moscow). **Anchor redo of iter36 is trained and measured; the
3,000/3,000-bit experiment is VETOED.** Andrew's master spec for this arm caps the card state at half
of naive fp32->int4 (card <= 2,176 bits, note <= 6,528 bits) and he rejected the higher bit budget:
"at that bit budget codebooks become too big + inference speed drops". The "Current constraints" line
below that lists a 3,000/3,000-bit post-queue experiment is therefore superseded -- do NOT launch it.
Anchor = iter36: `rwkv_iter36_558.pth` (WS-final, pre-decay) -> 124-step decay -> `rwkv_iter36_124.pth`
(post-decay, the baseline). All redo runs used `--epochs 2 --max-steps 124 --train-users 1 100
--max-train-len 66000` (5k db restricted to users 1-100: 160 groups/epoch, 0 skipped chunks) from
`rwkv_iter36_558.pth`: smoke `qat_it36qatsmoke` (0.04 ep, rc=0, all three quant banners), fp32 control
`qat_it36ctl` (`--no-quant`, 124 steps, 10.2 min), QAT `qat_it36qat` (candidate `c2_nl1i8` = card
full-matrix int2 + note rank-1 int8, 2,176/3,072 bits, 124 steps, 3.8 min). Unquantized on the seeded
750-user roster, paired per user against `qnj_it36_124` (0.342813 ahead / 0.314272 imm): `it36_558`
0.351102/0.322708 (+0.008289/+0.008436, worse, so the decay does help), `it36ctl` 0.333802/0.306211
(-0.009011/-0.008061), `it36qat` 0.336100/0.307813 (-0.006713/-0.006459). So the 5k-db substitution
gives a BETTER decay than iter36's own, and QAT costs +0.002298 ahead / +0.001602 imm against its
budget-matched fp32 twin. Old-regime diagnostics on the same roster are all 0.036-0.042 behind the
adaptive incumbent (0.300609/0.272009): `it45_496` 0.336628/0.306384, `it46_248` 0.342243/0.313402,
`it44_248` 0.340167/0.311412 -- this lineage is a MECHANISM BED, not a deploy candidate. Running now:
`it36qat_eval2.py` (quantized deploy evals of `qat_124`/`ctl_124`), then `it36ptq_chain.py` (re-base
of the PTQ rows on `rwkv_iter36_124.pth`), then `it36qatl1i8_chain.py` (second QAT screening slot:
candidate `l1i8` = rank-1 int8 on both streams, 1,024/3,072 bits, plus its unquantized twin, its
quantized deploy eval and the constraint-6 speed run on the idle machine). The malformed duplicated
copy of amendment 2 was removed (23 duplicate lines). Results table: `RESULTS.md`.


Row 11 amendment 4 (2026-11-20 09:25, Moscow). **Anchor results: the QAT redeploy is compliant, every
untrained scheme fails.** Everything is on the iter36 anchor, the seeded 750-user roster, paired per
user (`scratchpad/quant_plan/compare_tags.py`), with results in `scratchpad/quant_plan/RESULTS.md`.
Baseline row 0 = `rwkv_iter36_124.pth` unquantized 0.342813 ahead / 0.314272 imm. The other rows:
* control (row 5) = the same 124-step decay replayed in fp32 from `rwkv_iter36_558.pth` on the 5k db
  restricted to users 1-100 (160 groups/epoch, 0 skipped): 0.333802/0.306211, i.e. `-0.009011/-0.008061`.
  The replay's data substitution alone is worth 0.009 -- it is BETTER than iter36's own decay.
* PTQ on the anchor (rows 1-4, `it36ptq`, `quant_sweep_it36ptq.json`): cn2 (2,176/6,528 bits)
  +0.004108/+0.005544; l1i4 (512/1,536) +0.001719/+0.001419; l1i8 (1,024/3,072) +0.001884/+0.001044;
  c2_nl1i8 (2,176/3,072) +0.002085/+0.002054. All FAIL the +0.0005 ceiling by 2.1x-11x. Note that the
  rank-1 family again beats the mixed/full-matrix one at the same base.
* QAT + quantized deploy (row 8, `qc2_nl1i8_it36qatq`) = **0.335038/0.307253 = -0.007776/-0.007019 vs
  the anchor** -> PASS constraint 3 as written, 15x inside the bar, at 2,176/3,072 bits (constraint 2
  satisfied: card exactly at the bar, note half of it).
* QAT with the naive at-the-bar scheme (row 9, `qcn2_it36qatq`) = 0.337559/0.311924: better than the
  anchor but +0.001459/+0.004111 against its own fp32 twin -> QAT does NOT rescue full-matrix int2.
* Tax decomposition vs the budget-matched fp32 control (row 5): PTQ COST +0.001424/+0.002676; MODEL
  DRIFT (QAT trained, evaluated in fp32) +0.002298/+0.001602; PRECISION DEGRADATION (QAT+quant vs its
  own fp32 twin) -0.001063/-0.000560; FULL TAX (QAT+quant vs row 5) **+0.001236/+0.001042**
  (ub95 +0.001375/+0.001205); RECOVERED BY QAT +0.002487/+0.003236. QAT mainly fixes the IMMEDIATE head.
**TWO READINGS OF CONSTRAINT 3, both true:** vs the anchor champion the deploy row is 0.0078/0.0070
better (PASS with a 15x margin); vs its own budget-matched fp32 twin it is +0.0012/+0.0010 worse
(2.4x/2.1x OVER the +0.0005 ceiling). Andrew's decision is needed on which reference the gate uses.
No deploy-contract change is proposed, and the deployed 403 card bits / 214 note bits stay untouched.
Running/queued, one GPU job at a time: `it36qatl1i8` (row 10, candidate `l1i8` at 124 steps, plus its
fp32 twin and the constraint-6 speed run on the idle machine), then `it36qat496` (candidate `c2_nl1i8`
at iter45's 496-step decay length plus its own 496-step fp32 twin -- a scaling diagnostic of the
residual tax), then `it36qatl1i4` (candidate `l1i4` at 124 steps: 512/1,536 bits, four times under the
bar, in case QAT makes the cheaper allocation pass the micro-bar).


Row 11 amendment 5 (2026-11-20 10:15, Moscow). **Best recipe so far: QAT with rank-1 int8 on BOTH
streams, and constraint 6 passes.** Row 10 of `RESULTS.md` (`ql1i8_it36qatl1i8q`): QAT 124 steps from
`rwkv_iter36_558.pth` with candidate `l1i8` (card 1,024 bits / note 3,072 bits, both rank-1 int8, 2x
under the card bar and half the note bar), evaluated WITH quantization = **0.334305 ahead / 0.307221
imm** = -0.008508/-0.007052 vs the iter36 anchor champion (ub95 -0.007324/-0.006205). That is the best
row so far: it beats row 8 (`c2_nl1i8`, 2,176/3,072 bits) by 0.000733 ahead / 0.000032 imm while
spending HALF the card bits, and its QAT fp32 twin (0.335448/0.307029) needed less adaptation drift
(+0.001647/+0.000817 vs the control) than the int2-card candidate (+0.002298/+0.001602). On the idle
machine (`rust_speed_it36qatl1i8.json`, 2 reps, users 107/136/156) both recommended configs PASS
constraint 6: `l1i8` 2,586 rev/s = 0.903x / 0.806x states/s / median per-user 0.898, `c2_nl1i8` 2,586 =
0.903x / 0.804x / 0.894, against plain 2,863 rev/s. Quantizer mismatch matters: evaluating the
`l1i8`-trained checkpoint with the `c2_nl1i8` quantizer gives 0.335534/0.308766, i.e. +0.001229/+0.001545
worse than the matched quantizer, so a deploy must use the training-time quantizer.
Queue, one GPU job at a time: `it36qat496` is RUNNING (496-step `c2_nl1i8` QAT + its own 496-step fp32
twin, then fp32 and quantized evals of both: a diagnostic of whether the residual full tax falls with
QAT length), then `it36qatl1i4` (candidate `l1i4` = rank-1 int4, 512/1,536 bits - four times under the
bar - in case QAT makes the cheaper allocation pass the micro-bar). The residual to close is the
immediate head: for row 10 the full tax is +0.000503 ahead (at the ceiling) but +0.001009 imm.
No deploy-contract change is proposed; the deployed 403 card bits / 214 note bits stay untouched.

Row 11 amendment 6 (2026-11-20 22:50, Moscow). **The gate is redefined: the anchor is the run that is
identical EXCEPT for quantization, and the 496-step control shows the decay length dominates.**
Andrew's ruling (22:22): every quantized point is judged against the *same* run with quantization OFF,
not against the iter36 champion. For the 124-step decay that twin is `qnj_it36ctl` = 0.333802 ahead /
0.306211 imm (750-user seeded roster), and the previously quoted "-0.0085/-0.0071 vs the champion" is
WITHDRAWN: the champion `rwkv_iter36_124.pth` differs in data and step budget, so that number was
never a quantization result. `scratchpad/quant_plan/RESULTS.md` was rebuilt around row 0 = the control
(rows 1-4 = the gate-comparable PTQ/QAT points, then the fp32-twin table, the tax decomposition, and
Appendix A holding the PTQ-on-champion numbers explicitly marked non-comparable); `NOTES.md` carries
the same ruling plus the resume hazards.

New reference point, row 0b: the SAME 124-step recipe run for 496 steps with no quantization
(`qnj_it36ctl496`) scores **0.321659 ahead / 0.294614 imm** - 0.012143/0.011598 BETTER than row 0
(ub95 +0.012911/+0.012336, p ~1e-108) and 0.021154/0.019659 better than the iter36 champion. So at
this scale the decay length moves the loss about 25x more than the entire quantization tax does
(0.0121/0.0116 vs 0.0005/0.0010), and any 496-step candidate must be compared against this twin.

REBUILD, NOT RESUME: the 10:28 machine outage killed `it36qat496_chain` while the 496-step `c2_nl1i8`
QAT was around global step 250, leaving `scratchpad/quant_plan/qat_it36qat496/qat_50.pth` with an
EMPTY train.log. That checkpoint must NOT be scored and that folder must NOT be resumed. Its clean
control (`qat_it36ctl496/ctl_496.pth`, finished 10:21:28) survived and is the row 0b above.

RUNNING, one GPU job at a time: `it36qatsmall_chain` (launched 2026-11-20 22:25, log
`scratchpad/quant_plan/it36qatsmall_chain.log`). Job 0 scored the 496-step control (done 22:45:52,
rc=0, 20.4 min). Then the bit ladder, each point = 0.04-epoch smoke + a 124-step QAT decay from
`rwkv_iter36_558.pth` + one quantized eval against `qnj_it36ctl`, for the reductions that matter most
to Andrew's first axis: `l1i4` (512/1,536 bits), `l1i4s2` (384/1,152), `l1i2` (256/768); and finally
`l1i8` at 496 steps (1,024/3,072 bits) against row 0b. Every point skips itself if its merged result
already exists, so a restart is safe.

Deliverables are on GitHub: commit `69121550` ("Document the card/note state-quantization arm, gated
on the identical unquantized twin") - 421 files, the arm's runners, RESULTS.md/NOTES.md, the board and
research-log updates, and new .gitignore rules for the regenerable per-user prediction dumps. The
2,158 locally deleted stale scratchpad paths were deliberately NOT committed. The deployed budget
stays 403 card bits / 214 note bits; no deploy-contract change is proposed.

Row 11 amendment 7 (2026-11-21 00:00, Moscow). **The axis-1 bit ladder is running and the curve is
steep.** Two more points are scored against the identical 124-step fp32 control (`qnj_it36ctl`,
0.333802/0.306211):

  * row 5, `l1i4` (rank-1 int4 factors + int4 shifts, card 512 bits / note 1,536 bits) =
    0.334852/0.307894 = **+0.001050 ahead / +0.001682 imm** (ub95 +0.001164/+0.001841, p 2e-63/3.1e-75)
    - 2.1x/3.4x over the +0.0005 ceiling, and roughly double row 4's tax for half the bits.
  * row 6, `l1i4s2` (int4 factors + int2 shifts, 384/1,152 bits) = 0.335247/0.308491 =
    **+0.001445 ahead / +0.002280 imm** (ub95 +0.001592/+0.002464, p 3.0e-66/1.9e-88) - 2.9x/4.6x over.

So the token-shift vectors look like the expensive side of the rank-1 split (int4 -> int2 shifts cost
+0.000395/+0.000598 on their own). Both streams are still far under their constraint-2 bars (card 512
or 384 vs 2,176; note 1,536 or 1,152 vs 6,528), so axis 1 is a trade-off curve, not a hard limit.

Still running, one GPU job at a time: `it36qatsmall_chain` finished the `l1i2` smoke, started the
`l1i2` 124-step QAT at 23:44:46, and will then run `l1i8` at 496 steps against the row 0b twin
(`qnj_it36ctl496`). Its last two points are ROWS 7 and 8 of `RESULTS.md`.

QUEUED, second screening chain: `scratchpad/quant_plan/it36qatsmall2_chain.py`, launched detached
2026-11-20 23:47. It waits for `it36qatsmall_chain` to log its end marker with zero matching processes
(it does not count itself, and it takes `scratchpad/quant_plan/it36qatsmall2_chain.lock` so a double
launch cannot happen; delete that lock by hand after an outage). Its points, in order:
  * `l1i4` at 496 steps (tag `it36l1i4496`) vs `qnj_it36ctl496` - can the longer decay absorb row 5's
    +0.001050/+0.001682 at 512/1,536 bits?
  * `l1i8s4` and `l1i4s8` at 124 steps (tags `it36small8s4`, `it36small4s8`) - the new attribution pair
    in `quant_sweep.CANDIDATES`: both are 768 card / 2,304 note bits, with the bits moved between the
    rank-1 factors and the shifts, so the pair separates "factor error" from "shift error".
  * `l1i4s2` at 496 steps, then `l1i2` at 496 steps (the last one only if the 124-step `l1i2` deltas
    are within +0.0025, because a longer decay will not rescue a catastrophic allocation).
Nothing here changes the deploy contract: the deployed 403 card bits / 214 note bits stay untouched.

Row 11 amendment 8 (2026-11-21 01:00, Moscow). **The bit ladder is closed, and the longer-decay route to
compliance is REFUTED.** Rows 7 and 8 of `RESULTS.md` are scored against the identical unquantized runs:

  * row 7, `l1i2` (rank-1 int2 factors + int2 shifts, 256/768 bits) = 0.337406/0.311143 =
    +0.003604 ahead / +0.004932 imm vs the 124-step control = 7.2x/9.9x over the ceiling.
  * row 8, `l1i8` at 496 steps (row 4's exact bits) = 0.322791/0.296287 = **+0.001132/+0.001674** vs its
    own 496-step fp32 twin (row 0b). The tax MORE than doubles versus the 124-step pair
    (+0.000503/+0.001009) even though the absolute quantized model is 0.0115 better than row 4: the fp32
    twin keeps improving faster than a fixed-bit quantizer can follow. Compliance must therefore be won
    at the champion's own decay length, and more training is not a lever.

The axis-1 curve is now complete and convex (card/note bits -> d ahead / d imm vs the 124-step control):
1,024/3,072 -> +0.000503/+0.001009; 512/1,536 -> +0.001050/+0.001682; 384/1,152 -> +0.001445/+0.002280;
256/768 -> +0.003604/+0.004932. Row 4 remains the best point on both heads and is 2.1x/4.3x under the
constraint-2 bars, so the bars are not the binding constraint - the +0.0005 loss ceiling is, and the
immediate head is the blocker.

Run state: `it36qatsmall_chain` ENDED at 00:51:55 (all four points scored, rows 5-8).
`it36qatsmall2_chain` took the GPU at 00:51:57 and is running `l1i4` at 496 steps (the same longer-decay
question at the cheapest allocation); it will then run `l1i8s4` and `l1i4s8` (the 768/2,304-bit
attribution pair) and `l1i4s2` at 496 steps, and its own guard skips `l1i2`@496 because the 124-step
`l1i2` point is +0.003604 (over its +0.0025 threshold). `it36qatsmall3_chain` is queued behind it
(lock-guarded, detached 00:55): the per-stream attribution pair `l1i8c` (card rank-1 int8 only) and
`l1i8n` (note rank-1 int8 only) on row 4's own checkpoint, which splits row 4's ahead/imm tax into a
card part and a note part. Both are attribution rows only - they leave one stream in fp32, so they do
not meet constraint 2 and are never candidates.

OPEN DECISION FOR ANDREW (no run is waiting on it): row 4's allocation spends 1,024 of the allowed
2,176 bits per layer, and inside the rank-1 fast path the only ways to spend the headroom are finer
levels - int8 factors + int16 shifts (1,536 bits/layer), int16 factors + int8 shifts (2,048), or
unquantized factors + int2 shifts (2,176, exactly at the bar). Each needs a small machinery change in
BOTH the QAT path (`rwkv/architecture.py` `_QMAX`) and the Rust engine's level handling plus the bit
accounting, so it is a research-direction choice rather than a screening point. The alternative is to
accept row 4 as the best available recipe (ahead at the ceiling, imm 2x over) or to change the state
STRUCTURE (per-row scales, a diagonal term) rather than the level set.

The deployed 403 card bits / 214 note bits stay untouched; no deploy-contract change is proposed.

Row 11 amendment 8b (2026-11-21 01:00, Moscow). **The third chain was restarted with one more point, so
that the machinery question is answered by data first.** `it36qatsmall3_chain` (relaunched 00:59:20,
lock re-taken) now runs, in order: `l1i8c` and `l1i8n` (the per-stream attribution pair on row 4's own
checkpoint), then `cn8` = full-matrix int8 on both streams with its own 124-step QAT decay. `cn8` is
8,704 card bits / 26,112 note bits, four times over the bars, so it is a DIAGNOSTIC and never a
candidate - but it has no rank truncation at all, so its tax separates the two candidate mechanisms
behind row 4's remaining immediate-head error: the rank-1 structure, or the 8-bit factor precision.
That decides whether spending the rank-1 headroom on a finer level (int16 factors, int16 shifts, or
unquantized factors) can help at all, which is the open decision recorded in amendment 8. `cn8` needs
its own QAT run rather than a re-eval of row 4's checkpoint, because evaluating a checkpoint with a
quantizer it was not trained for costs about +0.0012/+0.0015 and would swamp the diagnostic.

Row 11 amendment 9 (2026-11-21 01:10, Moscow). **Andrew relaxed the QAT step budget, and chain 4
(`it36qatsmall4`) is queued to exploit it.** The 124-step QAT length was the champion's own decay
length, so every earlier QAT point copied it; Andrew now allows any length: "instead of using 124
steps, you can use more steps. It will make results less apples-to-apples comparable, but if it
ultimately ends up turning a failing recipe into a winning one, it's fine". Chain 4 (launched detached
01:05:50, lock `it36qatsmall4_chain.lock`, waits behind chain 3) runs: (1) the 992-step fp32 twin
`it36ctl992` -> tag `qnj_it36ctl992`, scored against `qnj_it36_124` for context only; (2) `l1i8` at 992
steps against that twin (RESULTS row 11); (3) `l1i4` at 992 steps only if the 496-step `l1i4` ahead tax
beat the 124-step one (row 12). Every point keeps a same-length fp32 twin: longer training helps fp32
too (row 0b is 0.0121/0.0116 better than row 0), so without its own twin a longer point cannot separate
more training from quantization. 992 = 8x the champion's 124 and 2x the last measured point, so it is
the last sensible length on this curve. The tax rises from 124 to 496 steps (+0.000503/+0.001009 ->
+0.001132/+0.001674); if it keeps rising at 992, this route is closed for good and only bit allocation
is left. Chain 3 (the `l1i8c`/`l1i8n` attribution pair plus the `cn8` mechanism diagnostic) still runs
before it. GPU at amendment time: chain 2's `l1i4`@496 QAT, 6.3 GB / 71%; free RAM 41.8 GB.

Row 11 amendment 10 (2026-11-21 08:00, Moscow). **Four chains finished overnight and the mechanism is
settled: the rank-1 tax is RANK TRUNCATION, not precision - and row 9 is a 25%-smaller state at the same
tax.** `it36qatsmall2` ended 03:08:54, `it36qatsmall3` 04:33:14, `it36qatsmall4` 06:12:50, all rc=0.
Key results: (1) `cn8`, the full-matrix int8 diagnostic with NO rank truncation (8,704/26,112 bits, 4x
over the bars, never a candidate) scores +0.000033/+0.000022 against row 0, so an 8-bit state is
lossless when it is not truncated and a finer level cannot help - the fix must add RANK. (2) The
per-stream attribution on row 4's own checkpoint splits it: card-only +0.001282/+0.001142, note-only
+0.000373/+0.000258, so the note is inside the ceiling by itself and the card owns the immediate head.
(3) Row 9 `l1i8s4` (int8 factors + int4 shifts, 768/2,304 bits) = +0.000516/+0.001030, i.e. row 4's tax
for 25% fewer bits: the shifts can drop int8 -> int4 for free and the bits belong in the factors. (4) The
decay-length axis is CLOSED - `l1i8`'s tax is +0.000503/+0.001009 at 124, +0.001132/+0.001674 at 496 and
+0.000874/+0.001675 at 992 - although the 992-step quantized run is 0.0162/0.0159 BETTER than row 4 in
absolute terms (its fp32 twin is 0.317213/0.289632), so absolute and relative readings disagree. Also:
`it36qatsmall2`'s own guard correctly skipped `l1i2`@496 and chain 4's skipped `l1i4`@992. RUNNING:
`it36rank` (launched 07:45) measures the card rank ladder - rows 15-18, card rank-2/3 int8 with int8 or
int4 shifts, 1,280-2,048 card bits, note unchanged. QUEUED behind it: `it36notecut` (note rank-1 int4
pool cut, plus the gated `l*i*s4n4` combinations only if the best card-rank imm delta comes in under
+0.00070). BLOCKER: rank >= 2 has no fused path in the deploy engine (measured 0.629x rev/s, 0.448x
states/s), so any rank >= 2 recipe needs a fused rank-r readout in the engine before it can be deployed
- that is a deploy-path change and needs Andrew's approval. GPU at amendment time: `it36rank`'s `l2i8`
QAT decay, chain `it36notecut` waiting.
