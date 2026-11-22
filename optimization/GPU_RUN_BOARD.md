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
+0.00070). BLOCKER (SUPERSEDED 2026-11-22 14:15): rank >= 2 had no fused path in the deploy engine (measured
0.629x rev/s, 0.448x states/s), so a rank >= 2 recipe needed a fused rank-r readout before it could be
deployed. That path now EXISTS: `RWKV_LOWRANK_FAST=1` (E2) replaces the KxK Gram + symmetric
eigensolve with a warm-started block subspace iteration + Rayleigh-Ritz, and it takes the rank-2 NOTE
from 0.618x to 0.803x rev/s - see the E2 amendment at the end of this file. GPU at amendment time: `it36rank`'s `l2i8`
QAT decay, chain `it36notecut` waiting.

Row 11 amendment 11 (2026-11-21 08:40, Moscow). **ROW 15 PASSES ALL SIX CONSTRAINTS - the card/note state
quantization arm has its first passing recipe.** `l2i8` = QAT card RANK-2 int8 factors + int8 shifts, note
rank-1 int8: card 1,536 bits (192 B, 2.83x smaller than naive int4, bar 2,176), note 3,072 bits (384 B,
2.13x smaller, bar 6,528). Loss on the 750-user roster = 0.333848/0.306518, i.e. +0.000046 ahead (ub95
+0.000112) and +0.000307 imm (ub95 +0.000369) against the identical unquantized 124-step decay (row 0) -
both heads inside the +0.0005 ceiling including the upper bounds. Deploy engine speed 2,323.6 rev/s =
0.812x (states/s 0.660x, median per-user 0.803x), above the 0.75x floor, and that measurement was about
3% pessimistic because a QAT decay was training (`l1i8` read 0.875x there against 0.903x idle). The
mechanism is confirmed: rank-1 imm +0.001009 -> rank-2 +0.000307. The earlier "rank >= 2 cannot be
deployed" conclusion came from `l2i4`, which put rank 2 on the NOTE as well (4 layers) and measured
0.629x; card-only rank-2 is 0.812x and needs NO engine change, because the engine already parses
`card:2:int8`. (2026-11-22 14:15: the NOTE side is solved too - `RWKV_LOWRANK_FAST=1` measures
`l2i8n2i8` at 0.803x rev/s and `l3i8n2i8` at 0.791x. See the E2 amendment at the end of this file.) Artifacts: `rust_artifacts/it36l2i8_124.safetensors`, `rust_speed_rank2card.json`, and the
3-user deploy grid `rust_grid_it36l2i8.json`. OWED: an idle-machine speed re-measurement, and Andrew's
decision on whether this recipe replaces the deployed 403/214-bit one. STILL RUNNING: rows 16-18 (card
rank-3, int4 shifts) and `it36notecut` behind them. GPU at amendment time: `it36rank`'s `l3i8` QAT decay.

Row 11 amendment 12 (2026-11-21 09:25, Moscow). **Row 16 (card rank-3 int8) is the lowest-tax row, and the
card rank ladder shows the knee.** `l3i8` = card rank-3 int8 + int8 shifts, note rank-1 int8
(2,048/3,072 bits) scores 0.333808/0.306388 = **+0.000006 ahead / +0.000176 imm** against row 0 (ub95
+0.000070/+0.000234). Immediate-head tax by card rank: rank-1 +0.001009, rank-2 +0.000307, rank-3
+0.000176 - so rank-3 is the axis-2 choice (near-zero tax, and it frees ~+0.0003 of the immediate-head
budget for a smaller note pool) and rank-2 (row 15) is the axis-1 choice with 512 fewer card bits.
Andrew asked for more gains on 2026-11-21 09:12 with the priority order 1) smaller state, 2) lower log
loss, 3) speed, and to queue lots of runs, so three new chains were launched at 09:15 behind the two
already running: `it36rankprec` (equal-bits rank-vs-precision probes - card rank-2 int4 at 768 bits vs
row 9's rank-1 int8, rank-4 int4 at 1,280 vs row 17's rank-2 int8, rank-7 int4 at 2,048 vs row 16's
rank-3 int8, note rank-2 int4 at 2,304 vs row 17's note rank-1 int8), `it36longdecay` (rank-2 at 496 and
992 steps against their own same-length fp32 twins, the absolute-loss axis), and `it36shrink` (four
combinations under row 15's 4,608 bits: 2,560 / 2,624 / 2,688 / 2,944). Eight new candidates were added
to `quant_sweep.CANDIDATES` (39 total). Queue depth ~13 points, roughly 8 h of GPU work. Andrew also
noted the deployed recipe is no longer his to change - another developer owns it and is running int4
cards + int8 notes - so this arm is an exploration for him, and the deployed 403/214-bit recipe is not
ours to touch.

Row 11 amendment 13 (2026-11-21 13:05, Moscow). **GPU IDLE SINCE 12:25: the three 09:15 chains are in a
circular wait and need Andrew.** `it36notecut` ended normally at 12:25 after scoring `l3i8s4n4`. Since
then `it36rankprec`, `it36longdecay` and `it36shrink` have all been parked in `wait_for_prev`: each one's
`BUSY_PAT` lists the OTHER two chain scripts, so each sees the other two's two processes and logs
`matching processes=4` forever. `it36rankprec` is the only one whose predecessor has ended (it36notecut),
so it would start immediately if it did not count its two successors as busy. No chain can break the tie
by itself, and the waiter gives up after `WAIT_MAX_MIN=900`, i.e. at about 2026-11-22 00:15. The deploy
measurement owed by section 3 of `autorun/QUEUE.md` cannot run while these chains are waiting, so the
0.812x/0.800x numbers are still the pessimistic same-time measurements. The hourly supervisor is not
authorized to kill or restart a chain whose log is fresh, so it recorded RESULTS.md row 22 and stopped.
Fix options for Andrew (needs a decision, because it means editing chain files): (a) drop the other
chain script names from each `BUSY_PAT` and keep only real workload processes
(`quant_sweep.py|rwkv.train_rwkv|rust_speed.py|rust_grid.py`), which makes the end-marker check the only
sequencer, or (b) launch the three chains one after another instead of at once. Results landed this hour:
row 22 `l3i8s4n4` (card rank-3 int8 + int4 shifts, note rank-1 int4 + int4 shifts, 1,792/1,536 bits) =
+0.000297 ahead / +0.000525 imm, FAIL by 0.000025 on the immediate head - an int4 note blocks the imm
head on the rank-1 (row 19), rank-2 (row 21) and rank-3 (row 22) card alike, so 2,304 note bits stay the
cheapest passing note.

Row 11 amendment 14 (2026-11-21 22:55, Moscow). **The three-chain deadlock is fixed and the queue is
moving again. The corruption/finiteness check is queued behind it.** Fix (`_fix_chain_busypat2.py`,
22:20): `BUSY_PAT` in `it36rankprec`, `it36longdecay` and `it36shrink` listed each other's chain
SCRIPT, so each saw its two siblings' waiting processes forever (`matching processes=4`) and none
could take the GPU; all three now match real GPU work only
(`qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|rwkv-infer`).
The waiters were killed, the three stale locks deleted, and the chains relaunched 22:21:59-22:22:00.
`it36rankprec` ran the `l2i4s4` 0.04-epoch smoke, its 124-step QAT decay (6.1 min, rc=0) and started
the quantized eval at 22:29:18; `it36longdecay` and `it36shrink` wait behind it correctly. Cost of the
deadlock: the GPU was idle 12:25-22:23, about 10 h.

New chain `it36corrupt` (launched 22:50, waits for `it36shrink`): the corruption check Andrew asked for
at 23:52 on 2026-11-20 - score the 50 users with the longest average review histories, with their
histories NOT truncated, and fail if even ONE card/note state value or even ONE p(recall) is
non-finite. The eval's own guards cannot answer it: `_get_loss` returns None only on NaN (never on
+inf) and the QAT state path deliberately caps non-finite states before quantizing
(`_sanitize_state`, `rwkv_ops.py:313`), which is exactly what hides a blow-up. The probe
(`scratchpad/quant_plan/corruption/finite_probe.py`) therefore monkey-patches the eval path - no repo
source is edited - and watches the RAW state, the shift vectors and every p_curve/p_imm/p_imm_all at
`RWKV_QAT_FUSED=0` (the documented bit-parity Python twin of the fused kernels), one process, no
sharding, on `corruption/corrupt50.json` (50 users, 4,609,289 section rows, longest single-card
history 1,409; the 62 users whose full section exceeds the 200,000-row VRAM cap are recorded in that
file's `excluded_too_big` and backfilled - Andrew: "skip users that cannot fit into VRAM"). Points, in
value order: `l2i8s4` (row 17), `l2i8` (15), `l3i8` (16), `l3i8s4` (18), then the quantization-OFF
control (`qat_it36ctl/ctl_124.pth`) for attribution. CPU smoke (user 7325, full history, `l2i8`): 828
state calls / 189,923,328 state values, maxabs 0.345, 0 non-finite, 3 p(recall) tensors clean -> PASS.
Two guards were added with it: `status_report.py`'s WORKLOAD pattern now includes
`probe_get_result|check_corruption` (the check runs `get_result` under its own entry point, so without
this the report would have called the GPU idle while it scored), and the chain logs a heartbeat every
5 minutes while a point runs (a 50-user point can take an hour, and a silent `run()` would have looked
like a dead chain to the supervisor).

Row 11 amendment 14 (2026-11-21 22:55, Moscow). **The three-chain deadlock is fixed and the queue is
moving again. The corruption/finiteness check is queued behind it.** Fix (`_fix_chain_busypat2.py`,
22:20): `BUSY_PAT` in `it36rankprec`, `it36longdecay` and `it36shrink` listed each other's chain
SCRIPT, so each saw its two siblings' waiting processes forever (`matching processes=4`) and none
could take the GPU; all three now match real GPU work only
(`qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|rwkv-infer`).
The waiters were killed, the three stale locks deleted, and the chains relaunched 22:21:59-22:22:00.
`it36rankprec` ran the `l2i4s4` 0.04-epoch smoke, its 124-step QAT decay (6.1 min, rc=0) and started
the quantized eval at 22:29:18; `it36longdecay` and `it36shrink` wait behind it correctly. Cost of the
deadlock: the GPU was idle 12:25-22:23, about 10 h.

New chain `it36corrupt` (launched 22:50, waits for `it36shrink`): the corruption check Andrew asked for
at 23:52 on 2026-11-20 - score the 50 users with the longest average review histories, with their
histories NOT truncated, and fail if even ONE card/note state value or even ONE p(recall) is
non-finite. The eval's own guards cannot answer it: `_get_loss` returns None only on NaN (never on
+inf) and the QAT state path deliberately caps non-finite states before quantizing
(`_sanitize_state`, `rwkv_ops.py:313`), which is exactly what hides a blow-up. The probe
(`scratchpad/quant_plan/corruption/finite_probe.py`) therefore monkey-patches the eval path - no repo
source is edited - and watches the RAW state, the shift vectors and every p_curve/p_imm/p_imm_all at
`RWKV_QAT_FUSED=0` (the documented bit-parity Python twin of the fused kernels), one process, no
sharding, on `corruption/corrupt50.json` (50 users, 4,609,289 section rows, longest single-card
history 1,409; the 62 users whose full section exceeds the 200,000-row VRAM cap are recorded in that
file's `excluded_too_big` and backfilled - Andrew: "skip users that cannot fit into VRAM"). Points, in
value order: `l2i8s4` (row 17), `l2i8` (15), `l3i8` (16), `l3i8s4` (18), then the quantization-OFF
control (`qat_it36ctl/ctl_124.pth`) for attribution. CPU smoke (user 7325, full history, `l2i8`): 828
state calls / 189,923,328 state values, maxabs 0.345, 0 non-finite, 3 p(recall) tensors clean -> PASS.
Two guards were added with it: `status_report.py`'s WORKLOAD pattern now includes
`probe_get_result|check_corruption` (the check runs `get_result` under its own entry point, so without
this the report would have called the GPU idle while it scored), and the chain logs a heartbeat every
5 minutes while a point runs (a 50-user point can take an hour, and a silent `run()` would have looked
like a dead chain to the supervisor).

Row 11 amendment 15 (2026-11-21 23:45, Moscow). **The night queue is closed; the corruption column fills itself.**

- **Row 23 recorded (`l2i4s4`, FAIL).** Card rank-2 **int4** + int4 shifts with row 9's note: +0.000266 ahead / +0.000580 imm, at the SAME 768/2,304 bits as row 9 (rank-1 int8). At equal bits rank-2 int4 beats rank-1 int8 by about +0.00025/+0.00045, so at a fixed budget RANK beats factor precision on the card; the imm head is still 1.16x over the ceiling.
- **New column `corruption`** (Andrew's gate) in `scratchpad/quant_plan/RESULTS.md`: `pending` for row 0 and rows 15-18 (queued in `it36corrupt`), `-` where the row fails the log-loss gate and so is not a deploy candidate.
- **Chain 11 `it36equalbits`** (launched 23:01, waits for `it36corrupt`): `l3i4s4` 1,024/2,304, `l2i8s2` 1,152/2,304, `n1i8s3` and `n2i4s3` 1,280/2,112. Its BUSY_PAT carries `probe_get_result|check_corruption`, the trap section 2 of the supervisor's `QUEUE.md` warns about.
- **Chain 13 `it36combo`** (launched 23:08, waits for `it36equalbits`): the cheap card under the cheap note - `l3i4s4n8s3` 3,136, `l3i4s4n3` 2,368, `l2i4s4n8s3` 2,880 total bits. All inside the bars (card 768-1,024 of 2,176; note 1,344-2,112 of 6,528). Same note as chain 11: a card rank >= 2 needs no engine work on the card stream.
- **Chain 12 `it36corrupt2`** (launched 23:08, waits for `it36combo`): the corruption verdict for every later row, gate-guarded (skips a row whose own sweep fails the log-loss gate, and skips when the verdict is missing - never a pass). `corruption/check_corruption.py` CONFIGS gained the eight later tags.
- **`corruption/record_corruption.py`** is the only writer of the corruption cells, and `QUEUE.md` section 4 step 4 orders the hourly supervisor to run it before it pushes RESULTS.md. Verified in a dry run plus one real write (then reverted: RESULTS.md is byte-identical to the pushed state).
- Order: `it36rankprec` (running, `l4i4s4` eval since 23:02) -> `it36longdecay` -> `it36shrink` -> `it36corrupt` -> `it36equalbits` -> `it36combo` -> `it36corrupt2`. Host RAM 43 GB free, RAM guard heartbeating, one process per chain, no stale locks.
Row 11 amendment 16 (2026-11-22 11:35, Moscow). **The 02:33-11:15 stall was an F: disk I/O error, and the corruption check now owns the GPU alone.**

- **Root cause of the 8.7 h gap - hardware, not code.** `it36shrink`'s `l3i4s4n4` QAT decay started 02:32:04 and its `train.log` stopped at 02:33:24. `corruption/_diag_stall.py` then measured `rwkv.train_rwkv` (pid 29396) and its fetch worker (26952) at **0.000 s user and 0.000 s system CPU over a 6 s window** while the process was `running` with 50 threads and rss 2.23 GB; `corruption/_diag_disk.py` found the System-log events `[disk] 153` ("The IO operation at logical block address 0x131488190 for Disk 1 (PDO name: \Device\00000060) was retried", 02:33:45) and `[disk] 51` ("An error was detected on device \DR1 during a paging operation", 02:33:46). `Get-Partition` confirms F: = Disk 1 (2.49 TB free). So an LMDB read on F: wedged the training loop while every guard stayed green.
- **Fix 1 - the watchdog.** `it36ladder_lib.run` now logs a heartbeat, tracks log growth AND the total CPU of the child's whole process tree, and if neither changes for `STALL_MIN=20` minutes it kills the tree with `taskkill /F /T` and retries once (`STALL_RC=-9`, `STALL_RETRIES=1`, `POLL_S=30`). `it36corrupt_chain.py` and `it36corrupt2_chain.py` delegate to it.
- **Fix 2 - the report can see a wedge.** `status_report.py` prints `CPU over 5 s: <pid>=<seconds> ... -> CPU alive (n/m busy)`, with `WEDGE? live workload processes burned no CPU` for a stuck tree, `gone` for a process that exited during the sample, and "all sampled processes exited during the sample" when a chain changed point between the two samples. A healthy 4-process tree reads about 14 s of CPU per 5 s of wall time (~2.8 cores).
- **Andrew's stop instruction (11:21).** "Let's stop further runs for now and ONLY do corruption checks for (as of right now) passed runs." The queue was reordered so `it36corrupt` runs FIRST (it no longer waits for `it36shrink`), and at 11:24 the waiters `it36shrink`, `it36equalbits`, `it36combo` and `it36corrupt2` were killed with their lock files deleted. `QUEUE.md` sections 2 and 5 now mark them PAUSED and forbid a restart: their logs have no end marker, so the supervisor's restart rule would otherwise fire as soon as the GPU is idle and their logs are 45 min old.
- **Rows that pass as of now: 15, 16, 17 and 18 only.** All five overnight sweeps FAIL: 24 `l4i4s4` +0.000218/+0.000463, 25 `l7i4s4` +0.000209/+0.000466, 26 `n2i4s4` +0.000180/+0.000492, 27 `l2i8496` +0.000715/+0.000793, 28 `l2i8992` +0.000554/+0.000829.
- **Corruption point 1 (`l2i8s4`, row 17) PASSES** at 11:25:24: rc=0, 50 of 50 users, **158,986 raw state observations and 150 p(recall) tensors, 0 non-finite**, 0 NaN skips, 4.8 min on the `corrupt50` roster with histories NOT truncated, on the Python QAT path (`RWKV_QAT_FUSED=0`) where the raw state is visible before `_sanitize_state` caps it. Verdict in `corruption/corruption_corrupt_l2i8s4.json`; the RESULTS.md cell is written by `corruption/record_corruption.py`. Points left: `l2i8` (row 15), `l3i8` (16), `l3i8s4` (18), and the quantization-OFF control `ctl_fp32`.
Row 11 amendment 17 (2026-11-22 11:45, Moscow). **The corruption check is DONE: all five points PASS.**

`it36corrupt` logged `=== it36corrupt chain end ===` at 11:41:38.

| point | RESULTS row | state calls | state non-finite | p(recall) tensors | p non-finite | users | min | mean LogLoss ahead / imm |
|---|---|---|---|---|---|---|---|---|
| `l2i8` | 15 | 158,986 | 0 | 150 | 0 | 50/50 | 4.7 | 0.382082 / 0.366948 |
| `l3i8` | 16 | 158,986 | 0 | 150 | 0 | 50/50 | 4.8 | 0.382168 / 0.366710 |
| `l2i8s4` | 17 | 158,986 | 0 | 150 | 0 | 50/50 | 4.8 | 0.382094 / 0.366979 |
| `l3i8s4` | 18 | 158,986 | 0 | 150 | 0 | 50/50 | 4.7 | 0.382198 / 0.366806 |
| `ctl_fp32` | 0 | 0 (no quantizer) | - | 150 | 0 | 50/50 | 1.8 | 0.382646 / 0.366875 |

- Every verdict is `corruption/corruption_corrupt_<tag>.json`; the RESULTS.md cells were written by
  `corruption/record_corruption.py`. No cell is `pending` any more.
- The state count is the same for all four quantized rows because the roster, the layer count and the
  sequence plan are the same: rank and bit width change the representation, not the number of state
  values, so the four rows are directly comparable.
- On this roster quantization is not worse than fp32 on the ahead head (0.38208-0.38220 vs 0.38265) and
  about equal on the immediate head. A 50-user roster is not the 750-user gate roster, so this is the
  finiteness gate plus a sanity band, not an acceptance result.
- The chain ran with `RWKV_QAT_FUSED=0` (the Python twin of the fused CUDA kernels) because the raw state
  is only visible there, and with `probe_get_result.py` monkey-patching the eval path, so no repo source
  was edited for the measurement.
- **Next: the GPU is idle and no chain is queued** (every other chain is PAUSED by Andrew's 11:21
  instruction), so the only authorized work left is the owed idle deploy measurement of `QUEUE.md`
  section 3 (`rust_speed.py --tag idle_l2i8`), which fixes the ~3% pessimistic 0.812x by measuring the
  rank-2 card path on an idle machine.

## Amendment 13 (2026-11-22 14:15): the E2 fast rank-r solver unblocks a rank>=2 NOTE

`RWKV_LOWRANK_FAST=1` replaces the rank>=2 truncation (KxK Gram matrix + full `nalgebra::SymmetricEigen`,
about 35 us of a 43 us compress call on a 32x32 state) with a warm-started block subspace iteration on
fixed-size arrays plus Rayleigh-Ritz on the (R+1)x(R+1) projection. The tracked block is the R wanted
directions plus one extra direction, refreshed from the strongest column of A each call, so a new
dominant direction can still enter. `RWKV_LOWRANK_FAST_ITERS` (default 2) sets the iteration count, and
the basis travels with the per-entity state (`FastLayerState::basis`).

Measured A/B, same session, `it36l2i8n2i8_124.safetensors`, users 107/136/156, 2 reps:

| config | reference solver | fast solver | note |
|---|---:|---:|---|
| plain | 2,915.1 rev/s (1.000x) | 3,025.1 rev/s (1.000x) | baseline |
| `l2i8` (rank-2 card) | 2,357.8 (0.809x) | 2,544.9 (0.841x) | the card path improves too |
| `l2i8s4` | 2,357.8 (0.809x) | 2,544.9 (0.841x) | |
| `l2i8n2i8` (rank-2 NOTE) | 1,801.5 (0.618x) | **2,429.2 (0.803x)** | was the blocker |
| `l3i8n2i8` (rank-3 NOTE) | 1,781.4 (0.611x) | **2,393.0 (0.791x)** | |

Why it is safe to deploy: `RWKV_LOWRANK_FAST_CHECK=1` runs both solvers, records the largest relative
Frobenius difference, and still writes the REFERENCE factors, so a checked run is bit-identical to the
no-flag run (verified on 3 configs). Of 64,132 rank>=2 calls on `l2i8n2i8`, 95.6% are under 1e-3
relative, 57 are above 1e-1, none above 1.0, and 3.2% fall back to the reference because the state is
degenerate (cheap there). Scored on the same traces the fast solver costs imm +0.000051 / ahead
+0.000000 (`rust_score.json`) - the same order as the quantization tax. The finiteness check with the flag
ON passes (`rust_finite_fastrank.json`: 0 non-finite of 65,671,168 state values).

Artifacts: `rust_fastrank_check.py`, `rust_fastrank_check.json`, `rust_speed_fastrank_{on,off}.json`,
`rust_finite_fastrank.json`, `rust_artifacts/it36l2i8n2i8_124.safetensors`. The flag is recorded in
`rust_grid.CONFIGS` for the three rank>=2-note configs and does NOT change `bits_for`; leave it UNSET for
any Python<->Rust parity run, because the fast factors are an approximation, not a bit-identical
truncation. RESULTS.md row 29 (`l2i8n2i8`) is now a loss-gate AND speed pass, so it is a deploy candidate
once `it36corrupt2` clears it.

## Amendment 14 (2026-11-22 15:35): row 31 is loss-only, E1 is queued and waiting, the iter-17 bundle is zipped and re-verified

**RESULTS row 31 (`l2i8n2i8s8`, card rank-2 int8 + int4 shifts, note rank-2 int8 + int8 shifts,
1,280/4,608 = 5,888 bits) is LOSS-ONLY.** ahead 0.333683 (delta -0.0001185, ub95 -8.7842e-05,
p 0.99999999995) and imm 0.306295 (delta +0.0000836, ub95 +0.0001077, p 7.1967e-11) both PASS the
log-loss gate, but the DEPLOY ENGINE cannot express the recipe: `RWKV_STATE_SHIFT_LEVEL` takes ONE bare
level, and a per-stream string such as `card:int4,note:int8` is silently dropped by `parse_level`
(`rust/rwkv-infer/src/model.rs` matches only `intN`), after which the shifts fall back to the WKV FACTOR
level. Verified by three engine runs on user 107: `card:int4,note:int8` gave predictions IDENTICAL to
`int8` (max abs diff 0.0) and 0.087 away from `int4`. 768 extra note bits over row 29 buy 0.000019 of
imm, so int8 note shifts are not free either.

**The 20-minute wedge monitor worked.** The `l2i8n3i8` 0.04-epoch smoke wedged at 14:57:17: all four
workload processes sat at 0.00 s CPU for 20 min, GPU 1,701 MiB at 2%, and the System log had no disk
events in the preceding 2 h. At 15:17:18 `it36ladder_lib` killed the tree (`rc=-9`) and the retry passed
in 1.6 min, so the decay started 15:18:52. Second wedge of the day (the first is the 02:33 F: I/O error)
and the first one the monitor caught by itself. Four processes from the 14:51 `l2i8n2i8s8` eval are STILL
alive at 0 CPU and 4-55 MB (pids 22560, 33820, 34852, 33020): the shard parents returned rc=0 while their
children lingered. They did not block the retry, so they are left alone as debris.

**Chain 15 `it36e1` (E1) is launched and WAITING behind `it36corrupt2`** (lock taken 15:22:07;
`WAIT_MAX_MIN` raised to 2,400 min inside its own script, because the queue ahead is 11 sweep points plus
a 16-point corruption chain and the shared 900-min default could expire). E1 is the advisor's
first-ranked lever: `lambda * mean over streams of the truncated tail energy` added to the TRAINING loss -
zero state bits, zero deploy cost. Measured tails of row 17's own states
(`corruption/tail_energy_l2i8s4.json`, 5,388 raw states, users 7213/7044/6960): the card at rank 2 keeps
0.99381 of the state energy (tail 0.62%), the note at rank 1 keeps 0.96943 (tail 3.06%). Three points, all
scored against `qnj_it36ctl` exactly like row 17: `l2i8s4pf` (the fused-off control, which separates
"fused off" from "penalty on"), `l2i8s4e1a` (lambda 1.0 = ~2.4% of the ~1.02 loss) and `l2i8s4e1b`
(lambda 10.0 = ~25%). E1 must run on the Python per-step QAT loop (`RWKV_QAT_FUSED=0`, the bit-parity
twin of the fused kernels, because the fused rank-1 kernel truncates inside the kernel), which is slower
per step, so the three points cost about 4 h. Decision rule recorded in the chain docstring: if neither
lambda moves the tax, E1 is closed; if `e1b` moves it without wrecking the loss, lambda is refined.

**The iter-17 bundle is zipped and re-verified from the zip alone.** `scratchpad/row17_l2i8s4_bundle.zip`
(12,610,091 bytes, 180 entries) and the folder hold 177 files / 44.04 MB; `MANIFEST.md` verifies 177/177
SHA-256 against the folder with no extra files. `verify.cmd` was extracted into the repo root and run: the
`cargo build --release` finished in 1m18s, the deploy tax printed +0.000335 / +0.000316 (identical to every
earlier measurement), the strict finiteness check passed on 43,474,944 state values, 4a/4b printed
`PARITY: PASS` to at most 8.5e-07 per review on 3,465/3,865/12,915/23,455-review histories, 4c printed the
expected FAIL (the trace's own Python reference is UNQUANTIZED), and the speed step measured 0.804x rev/s
(median per-user 0.801x). Two defects in `verify.cmd` were found by that run and fixed: it named user 156,
whose trace the bundle does not ship (the current label DB puts review 1142 of that user in the equalize
set and the exporter aborts on it), and step 4c - which fails by design - was wired as fatal.

**The Rust finiteness check is being pointed at the real 50-user roster.** `corruption/check_corruption.py`
already covers this in Python (row 17: 50/50 users, 158,986 state calls, 0 non-finite, 285.8 s), but the
DEPLOY engine's own check had only ever run on the 3-user parity trace. `corruption/export_corrupt50_traces.py`
is exporting the 50 longest-history users (5 shards x 10 users; 10 of 50 done at 15:30) so that
`rust_finite.py --strict` can run on the same users the Python gate used, at the same untruncated history
lengths.

## Amendment 15 (2026-11-22 17:00): row 32 closes the note's rank ladder; `int6`/`int5` become trainable and chain 17 is queued

**RESULTS row 32 is recorded: `l2i8n3i8`, chain 14's last point.** Card rank-2 int8 + int4 shifts
(row 17's card), note **rank-3 int8** + int4 shifts, 1,280/5,376 = 6,656 bits: ahead 0.333817
(+0.000015, ub95 +0.000033, p_worse 0.215), imm 0.306334 (+0.000122, ub95 +0.000142, p_worse 2.0e-30),
**PASS on both heads**. It CLOSES the note's rank ladder: against row 29 (`l2i8n2i8`, the same card)
rank 3 is WORSE on BOTH heads (-0.000115 ahead / +0.000103 imm at rank 2) while spending 1,536 MORE
note bits, and against row 30 (rank 2 on a rank-3 card) it is worse still. So the note's residual tax
is NOT rank-limited - extra note rank cannot buy it back - which is precisely the premise the E1
tail-energy penalty rests on, and keeps chains 15/16 as the right next move. Speed n/a: a rank-3 note
needs `RWKV_LOWRANK_FAST=1` (that pair has not been timed; 0.791x was the rank-2 note on a rank-3 card,
row 30).

**`int6` and `int5` are now TRAINABLE. The factor-precision rungs below int8 had never been tested at
all.** Every rung below row 17 so far used int4 factors (rows 19-26). `int6`/`int5` were not runnable:
`rwkv/architecture.py:151` `_QMAX` carried only int8/int4/int3/int2/fp32, so `card:2:int6` raised
`KeyError` in `_apply_qat_scopes`, even though the Rust deploy engine has ALWAYS spoken both levels
(`parse_level`, `rust/rwkv-infer/src/model.rs:2104-2105`, and `rust_grid.QBITS` - whose docstring has
recorded the gap for a while) and both quantization paths are generic in qmax (`rwkv7_cuda.cu:468-474`
clamps to `+-qmax`; `rwkv_ops._fake_quant_factor` uses `amax/qmax`). The change is therefore three
ADDITIVE edits and no kernel work: `_QMAX` in the trainer, `_QMAX`/`_QBITS` in `quant_sweep` (bits
accounting), and three candidates. No existing recipe names the new levels, so nothing that already ran
changes behaviour. Chain 17 `it36int6` is queued as a waiter behind `it36e1shrink`. All three points sit
at row 17's EXACT layout (card rank 2, note rank 1, int4 shifts), so the ONLY change is factor precision:

| candidate | card/note bits | total | vs row 17 | note |
| --- | --- | --- | --- | --- |
| `l2i6s4` | 1,024/1,920 | 2,944 | 640 fewer | least aggressive cut first |
| `l2i5s4` | 896/1,728 | 2,624 | 960 fewer | also 192 below row 21's 2,816 |
| `l2i4s4n4` | 768/1,536 | 2,304 | 1,280 fewer | the only BOTH-int4 combination (row 21 was note-int4 only, row 23 card-int4 only) |

**A 2-user Rust PTQ probe screens the rungs but cannot settle the gate.** On the row-17 checkpoint,
`rust_grid_int6_probe.json`: `l2i6s4` +0.000003 ahead / +0.000002 imm against `l2i8s4` for 640 fewer
bits; `l2i5s4` +0.000017 / -0.000073 for 960 fewer. That is no visible penalty, but the probe is
calibrated against two layouts whose 750-user QAT numbers already exist, and it FAILS that calibration:
it reproduces the note int8->int4 cost at only 54% (+0.000167 vs the known +0.000311) and gets the card
int8->int4 SIGN wrong (-0.000334 vs the known +0.000253). So every chain-17 point is a full 750-user
QAT run with the standard paired test, exactly like rows 15-31. The probe is only a screen: it shows
the rungs are not catastrophic, nothing more.

**The int6 path was pre-flighted in the REAL trainer on CPU, while the GPU was busy.** With
`DEVICE=cpu` the `[QAT-LOWRANK]` banner reported `card_id=rank2/fq31.0, note_id=rank1/fq31.0` and the
first step trained finite (0.837127 / ahead 0.3954 / imm 0.442), so the level reaches the layer configs
and the trainer runs on it. Chain 17's own 0.04-epoch GPU smoke covers the rest.

**Operational lesson (cost me two jobs and idled the GPU): a CPU-only job that matches a chain's
BUSY_PAT strands the queue.** `it36ladder_lib.running()` matches a process's COMMAND LINE against the
chain's pattern, so `rust_grid.py` (and its `rwkv-infer.exe` children) and even `python -m rwkv.train_rwkv
--config ...qat_*cpufake\train.toml` on `DEVICE=cpu` both count as "busy". A waiting chain returns only
when `ended AND busy == 0`, so a long CPU job keeps the GPU idle for its whole duration: at 16:42
`it36shrink` had `ended=True` and could not start while `rust_grid.py` was measuring. Both jobs were
killed; the queue took the GPU at 16:46:55 (`it36equalbits`, candidate `l3i4s4`). **Rule: run such jobs
only inside a point's own window** - a point is roughly smoke 1-3 min + decay 10 min + eval 56 min - so
the job must finish well before the point does. The wide-roster fast-solver grid (the 22 landed
`trace_corrupt50` users, `plain/l2i8s4/l2i8s4f/l2i8s4f1`) was launched for that purpose and will be
re-run that way.

## Amendment 16 (2026-11-22 17:50): the fast solver is free on long histories; factor-precision cost scales with history; VLEVEL is dropped

**Axis 3 - the shipped row-17 weights get 16% faster for free.** On the three LONGEST corrupt50
histories (users 7250 / 5052 / 5939, 28-29k reviews each, 80-84 MB traces) `l2i8s4`, `l2i8s4f`
(`RWKV_LOWRANK_FAST=1`) and `l2i8s4f1` (plus `RWKV_LOWRANK_FAST_ITERS=1`) agree to **<=3e-6 on both
heads** (tax -0.003711/-0.003620, -0.003714/-0.003623, -0.003714/-0.003621). Together with the speed
measurements (0.800x -> 0.844-0.889x -> 0.927x rev/s, median per-user 0.921x) this makes ITERS=1 the
deploy setting for row 17: same weights, same bits, same log loss, 16% faster, no retraining. The flag
must stay OFF for any Python<->Rust parity run, because the fast factors are an approximation rather
than a bit-identical truncation. Evidence: `rust_grid_fast_wide3.json`, `rust_speed_fastrow17.json`,
`rust_speed_fastiters.json`.

**Axis 1 - the cost of cutting factor precision GROWS with history length, so the 2-user screen
understated it ~100x.** Per-user paired imm deltas against `l2i8s4` (`rust_grid_long_paired.json`):

| config | 7250 | 5052 | 5939 | mean | std |
| --- | --- | --- | --- | --- | --- |
| `l2i6s4` (1,024/1,920 = 2,944 bits) | +0.000067 | +0.000226 | +0.000742 | +0.000345 | 0.00035 |
| `l2i5s4` (896/1,728 = 2,624 bits) | +0.000629 | +0.001147 | +0.002877 | +0.001551 | 0.00118 |
| `l2i8s4v4` (V int4, 2,944 bits) | +0.000653 | +0.000057 | +0.002334 | +0.001015 | 0.00118 |
| `l2i8s4v3` (V int3, 2,784 bits) | +0.001639 | -0.006228 | +0.003570 | -0.000340 | 0.00519 |
| `l2i8s4v2` (V int2, 2,624 bits) | +0.003583 | -0.006194 | +0.005652 | +0.001013 | 0.00633 |

Both exponent rungs are the SAME SIGN on every user and grow with length (the longest user, 5939, pays
the most). **Expectation now on record before chain 17 runs:** `l2i6s4` is a coin flip - row 17's own
imm is +0.000327 with ub95 +0.000390 against a +0.0005 ceiling, so an added ~+0.0001-0.0003 leaves
little room - and `l2i5s4` plus the bottom rung `l2i4s4n4` are EXPECTED TO FAIL on imm. They still run,
because the gate is the 750-user roster and a roster average sits below the long-user figure but above
zero, and the measured answer is the deliverable. If int6 lands, it is a 640-bit (18%) cut.

**Axis 1 negative - `RWKV_LOWRANK_VLEVEL` is dropped.** At the SAME 640-bit saving, coarsening V while U
stays int8 costs +0.001015 (std 0.00118) against int6's +0.000345 (std 0.00035): three times larger and
three times noisier, so the bits are better spent symmetrically. Worse, the VLEVEL spread is ~15x wider
than the symmetric rungs' and V int3's negative mean is driven by ONE user (5052 reads -0.006228 while
the other two read +0.001639/+0.003570), which means coarsening V does not add a smooth error - it
destabilizes the state trajectory. No trainer support and no chain are added for it; the engine knob
stays available for future work.

**Method note.** The per-user paired difference is what makes these readable: the grid JSONs store only
per-config means, and on these long traces every quantized config reads BETTER than `plain` by ~0.0036
(the trace-path offset already documented), so absolute taxes cannot rank configs. Differencing the same
users pairwise cancels that offset and leaves the std above, which is what separates
`int6` (sign-consistent, std 0.00035) from VLEVEL (sign-flipping, std 0.005-0.006).

## Amendment 17 (2026-11-22 17:55): error feedback is implemented, was never measured, and the 3-user screen cannot settle it

**`RWKV_LOWRANK_EF` (error feedback / noise shaping) has no record anywhere in this lineage.** The
deploy engine implements it and documents it as "carry a per-card quant-error buffer, add it back to
the fresh state BEFORE re-compressing each step, then update e = compensated - reconstructed. Cancels
the compounding DC bias of the low-rank factor quantization", with budget knobs `RWKV_EF_ERANK` (rank
of the carried buffer), `RWKV_EF_ELEVEL` (its factor width) and `RWKV_EF_PQ`. Searching RESULTS.md,
NOTES.md, this board and every other .md/.json under `optimization/` and `scratchpad/quant_plan` for
LOWRANK_EF / error-feedback / noise-shaping / EF_RANK / EF_LEVEL / EF_PQ returns NOTHING, while the
same search finds the neighbouring HADAMARD and per-column findings - so it is an unmeasured lever,
not a re-run. It is well motivated by today's other result: the cost of cutting factor precision
GROWS with history length, which is what a compounding bias looks like, and EF is the implemented
mechanism for cancelling one.

**First screen (row-17 deploy weights, the 3 longest corrupt50 users, rank-1 int4 error buffer):**

| config | bits card/note | total | imm | vs row 17 | per-user imm deltas (7250/5052/5939) |
| --- | --- | --- | --- | --- | --- |
| `l2i8s4` (row 17) | 1,280/2,304 | 3,584 | 0.324755 | - | - |
| `l2i8s4ef` | 1,536/3,072 | 4,608 | 0.322530 | -0.002225 | +0.000187 / -0.005988 / -0.000874 |
| `l2i4s4n4ef` | 1,024/2,304 | 3,328 | 0.323445 | -0.001310 | +0.000774 / -0.005933 / +0.001229 |

On paper `l2i4s4n4ef` is attractive: BOTH streams at int4 plus a cheap error buffer, 3,328 bits, i.e.
256 BELOW row 17 while reading 0.0013 better on imm. **It is not established, and the reason matters
for every future deploy-side probe:** the 3-user mean is dominated by user 5052, whose imm is
0.533-0.545 against 0.170 (7250) and 0.265 (5939). 5052 moves by about -0.006 under several UNRELATED
configs (VLEVEL int3 -0.006228, both EF configs -0.00599), i.e. on that one hard trace coarse state
reads up to 0.012 better than fp32, while the other two users barely move and EF is slightly negative
on them. With three users, one such user is a third of the mean. So the EF delta is a one-user effect,
not a demonstrated improvement.

**Blocker to resolve before EF can become a recipe: the 750-user gate cannot score it.**
`RWKV_LOWRANK_EF` exists only in the Rust deploy engine; the Python eval path
(`rwkv_ops.fake_lowrank_state`, which `quant_sweep.py` uses) has no EF, and the formal gate is an
eval-path measurement (fp32 twin versus QAT+quantized on the 750-user roster). So EF is measurable
only on the Rust trace path, which is a ranking instrument and not the gate. Making EF a candidate
therefore needs a decision: either implement EF in the Python eval path, or extend the accepted gate
to a Rust-path measurement. **That is Andrew's call, not a change to make quietly** - it touches how
results are accepted.

**Next step that needs no decision:** widen the screen rather than conclude from three users. The
`trace_corrupt50` export (26/50 complete at 17:45) gives up to 50 long users for free, and the
exporter is CPU-only (`device=torch.device("cpu")`) and matches none of the chains' BUSY_PAT, so a
GENUINE wide screen can be built without touching the GPU queue. If EF holds up across tens of users,
the eval-path question above becomes worth Andrew's time; if it does not, the idea is closed for the
cost of a few CPU hours.

## Amendment 18 (2026-11-22 18:30): the error-feedback screen closes - EF is real but cannot shrink the state

The EF question is now answered on **21 of the longest corrupt50 histories** (per-user paired deltas
against `l2i8s4` within each batch, 2,000 bootstrap resamples). Three points, in order of what they
settle:

| config | bits card/note | total | ahead delta [95%] | imm delta [95%] | better on |
| --- | --- | --- | --- | --- | --- |
| `l2i8s4ef` (row 17 + EF) | 1,536/3,072 | 4,608 | -0.001167 [-0.002081, -0.000339] | -0.002453 [-0.004826, -0.000539] | 10/15 ahead, 8/15 imm |
| `l2i4s4n4ef` (both int4 + EF) | 1,024/2,304 | 3,328 | +0.000049 [-0.000737, +0.000875] | -0.000376 [-0.001852, +0.000911] | 10/21 ahead, 6/21 imm |
| `l2i4s4n4ef2` (both int4 + EF, buffer int2) | 896/1,920 | 2,816 | +0.003439 [+0.000587, +0.007513] | +0.002696 [+0.001449, +0.004236] | 1/6 ahead, 0/6 imm |

**(1) EF genuinely works.** At row 17's own config it improves BOTH heads with intervals that exclude
zero, by about -0.0012 ahead and -0.0025 imm. The error buffer is real state, so that costs +1,024 bits
(card +256, the three note layers +768). This is axis 2, and axis 2 is already saturated - rows 29, 30
and 31 sit at or below their own fp32 twins - so the trade is not one the project needs.

**(2) EF cannot shrink the state, which was the whole point of testing it.** The best it does is
`l2i4s4n4ef`: 3,328 bits, i.e. 256 below row 17, reading STATISTICALLY EQUAL to row 17 on both heads
(+0.000049 ahead, -0.000376 imm, both intervals spanning zero, 10/21 users better on ahead and 6/21 on
imm). So int4 factors plus a rank-1 int4 error buffer is a genuine -7% state at equal accuracy, but not
more than that, and it is 7x smaller a cut than the 2,304-bit bottom rung that nobody expects to pass.
Note the direction of the earlier 3-user read: it showed -0.001310 imm and looked like a 256-bit win,
which 21 users cut to -0.000376 and an interval over zero. That is exactly the trap the wide screen
exists to catch.

**(3) A coarser error buffer does not work.** `l2i4s4n4ef2` puts the buffer at int2, which is where the
bits would come from (2,816 bits, 768 below row 17 and exactly row 21's size - the layout that failed by
0.00014). It is WORSE than row 17 on both heads by +0.0034 / +0.0027 with intervals well clear of zero,
and better on 1/6 and 0/6 users. The buffer is the mechanism, so coarsening it removes the benefit. The
engine's `RWKV_EF_PQ` (~96 b) is the only remaining way to cheapen it, and by this result the budget for
the buffer is the whole effect, so it is not worth a chain.

**Disposition.** EF is closed as an axis-1 lever. It stays documented as an axis-2 option that the
current gate cannot score anyway (`RWKV_LOWRANK_EF` exists only in the Rust engine, and the gate is an
eval-path measurement), so it needs no decision from Andrew now - the question only becomes live if a
future recipe wants the -0.0025 imm at more bits.

**Instrument kept.** `corruption/wide_screen.py` aggregates any number of `rust_grid` batch tags into
per-user paired deltas with sign counts and bootstrap intervals, and it is what turned a
three-user-looking win into a measured non-result. Deploy-side probes should go through it rather than
quoting a 2-3 user mean.

## Amendment 19 (2026-11-22 19:35): the corruption column closes for rows 29-32 - the four note-rank candidates all pass on the 50 longest histories

Rows 29-32 passed the loss gate but had no corruption verdict, so none of them was deployable. The
verdicts are now in, from `corruption/check_corruption.py` on the **50 longest-mean-review users** with
their histories **NOT truncated** (`corruption/corrupt50.json`, 2,349,827 reviews / 4,609,289 section
rows, 62 too-big users excluded and backfilled):

| row | config | card/note bits | total | state calls | p(recall) tensors | non-finite | time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | `l2i8n2i8` | 1,280/3,840 | 5,120 | 158,986 | 150 | **0** | 5.2 min |
| 30 | `l3i8n2i8` | 1,792/3,840 | 5,632 | 158,986 | 150 | **0** | 5.3 min |
| 31 | `l2i8n2i8s8` | 1,280/4,608 | 5,888 | 158,986 | 150 | **0** | 5 min |
| 32 | `l2i8n3i8` | 1,280/5,376 | 6,656 | 158,986 | 150 | **0** | 5 min |

All four: **PASS**, 50/50 users scored, no NaN-guard skips, zero non-finite state values, zero
non-finite p(recall). The observation count is IDENTICAL to row 17's (158,986 state calls, 150
p(recall) tensors), which is the control that the probe really was exercised: a PASS with fewer
observations would be INCOMPLETE, not a pass. So the note's rank ladder is clean at the history lengths
that matter, not just on the short parity traces.

**The scheduling problem this exposed, and the tool that fixes it.** Every `it36*_chain.py` waiter
decides "the GPU is free" by counting matching processes (`it36ladder_lib.running`) and a chain hands
off from one point to the next with NO idle gap, so a job that is not part of a chain cannot wait for a
free GPU without racing a handoff - the race is exactly what stranded the queue on 2026-11-22 16:42
(a `rust_grid` run matched BUSY_PAT and `it36shrink` sat for 10+ min with the GPU at 1%). Meanwhile
Andrew's instruction is that a corruption check must NOT be left pending once a row passes.
`corruption/gpu_hold.py` resolves it: a CPU-only, self-expiring process whose command line carries a
BUSY_PAT token (`--tag rust_grid.py`), so it holds the queue "busy" for a bounded window while the
checks run back to back inside it, and it cannot create a second GPU workload because it only sleeps.
The four checks above ran in one such window (19:10-19:33), started the moment the active chain logged
its end marker. The hold ALWAYS exits after `--secs`, so a crash cannot wedge the queue.

**Still open on the Rust side.** The Python/eval-path gate is now clean for rows 15-18 and 29-32. The
deploy engine has its OWN non-finiteness guard (`RWKV_FINITE_CHECK`, `--strict`), and it has still only
seen the 3-user parity trace. `corruption/export_corrupt50_traces.py` is producing the 50 long-history
traces for that check (`trace_corrupt50/`, 39/50 complete at 19:35); `corruption/corrupt50_finite_run.py`
runs it over every complete trace once the export finishes.

## Amendment 20 (2026-11-22 19:45): int6/int5 priced on 21 long histories - chain 17's expectation is on record

Chain 17 (`it36int6`) was queued on a 2-user short-trace PTQ probe that read int6 as free. Its own
calibration test had already invalidated that probe (it reproduced a known note tax at 54% and got a
known card tax's sign wrong), and Amendment 18's method lesson said to route deploy-side probes through
`corruption/wide_screen.py`. That has now been done, on the longest histories available:

| config | bits card/note | total | ahead delta [95%] | imm delta [95%] | better on |
| --- | --- | --- | --- | --- | --- |
| `l2i6s4` | 1,024/1,920 | 2,944 | +0.000442 [+0.000232, +0.000701] | +0.000726 [+0.000420, +0.001100] | 2/21 (both heads) |
| `l2i5s4` | 896/1,728 | 2,624 | +0.001458 [+0.000826, +0.002251] | +0.002189 [+0.001389, +0.003065] | 1/21 (both heads) |

21 users, per-user PAIRED deltas against `l2i8s4` inside each batch, 2,000 bootstrap resamples
(`rust_wide_screen_fast_wide3_int6long_it36int6wide_b4.json`). The sign is consistent on 19-20 of 21
users, so the means are not carried by an outlier. The 3-user read of the same two rungs
(+0.000345/+0.001551 imm) understated int6 by 2.1x.

**What it means for chain 17.** PTQ is not QAT, and QAT recovers roughly half of a PTQ tax on this
sample - but row 17's own QAT imm tax is +0.000327 with ub95 +0.000390 against the +0.0005 ceiling, so
`l2i6s4` would have to win back almost the whole +0.0007 PTQ gap to pass. The chain still runs, because
a 750-user roster is cheaper than a wrong assumption and the measured answer is the deliverable, but the
expectation is now on record: **int6 is at best a coin flip, `l2i5s4` and `l2i4s4n4` are expected to
fail.** The 2,304-bit bottom rung was not measured: `l2i4s4n4` is a `quant_sweep` candidate name that
does not exist in `rust_grid`'s CONFIGS, so that batch ended in a KeyError after the four configs above
were complete; with int5 at +0.0022 imm it was not re-run.

## Amendment 21 (2026-11-22 20:10): the deploy engine's own finiteness guard clears 40 long histories

Andrew's blowup gate was clean in PYTHON first (rows 15-18 and 29-32, `check_corruption.py` on the
50 longest-mean-history users). The DEPLOY engine has its own guard (`RWKV_FINITE_CHECK` with
`--strict`, which exits 3 on any non-finite value) and it had only ever seen the 3-user parity trace,
so the implementation that actually ships had never been exercised at the history lengths the gate is
about. `corruption/export_corrupt50_traces.py` reached the driver's minimum of 40 complete traces, and
`corruption/corrupt50_finite_run.py --tag corrupt50` ran over all of them inside the active chain's
point window (CPU-only engine, so no GPU contention).

| config | compressions | state values | shift-vector values | p(recall) values | non-finite |
| --- | --- | --- | --- | --- | --- |
| `plain` | 0 (nothing quantized) | 0 | 0 | 3,459,629 | **0** |
| `l2i8s4` (row 17) | 7,053,440 | 7,222,722,560 | 451,420,160 | 3,459,629 | **0** |

`rc=0` for both configs over all 40 users: zero bad values in every per-user record, on the RAW and
the QUANTIZED side of every state compression and on every shift vector, plus zero planted failures
(the guard's own self-test). Largest |state| 0.553, largest |shift vector|
5.380 - i.e. the states are small and stay small; nothing is near an overflow.

On histories of up to 95,597 reviews this is about 45,000x the Python gate's 158,986 state calls on the
same roster, so the two implementations agree where it matters. Note: the engine's `[finite] TOTAL`
line reports zeros because its cumulative counter does not aggregate across users - the per-user
records are the numbers above. The last 10 traces were still exporting; re-run with
`--tag corrupt50_rest` when they land.


## Amendment 22 (2026-11-22 20:35): the int6 tax splits - the CARD-only int6 rung is nearly free, the NOTE's is not

The 21-user int6 price (Amendment 20: `l2i6s4` +0.000442 ahead / +0.000726 imm) is now split by quantizing one stream at a time on 11 long-history users, as per-user PAIRED deltas against `l2i8s4` (`rust_wide_screen_it36int6split.json`; absolutes `rust_grid_it36int6split.json`):

| config | card bits | note bits | total | d ahead (95% CI) | d imm (95% CI) |
|---|---:|---:|---:|---|---|
| `l2i6c` card int6, note int8 | 1,024 | 2,304 | 3,328 | +0.000052 [-0.000007, +0.000109] better 2/11 | +0.000115 [+0.000018, +0.000218] better 2/11 |
| `l2i6n` card int8, note int6 | 1,280 | 1,920 | 3,200 | +0.000268 [+0.000079, +0.000601] better 0/11 | +0.000307 [+0.000124, +0.000558] better 1/11 |

`l2i6c` is 256 bits BELOW row 17 and its ahead interval contains zero; `l2i6n` is 128 bits below the card rung and worse on both heads. The card rung is therefore the axis-1 point worth a 750-user QAT run: chain 18 (`it36int6card`, points `l2i6c` then `l2i6n`, 124 steps vs `qnj_it36ctl`) is queued behind chain 17 (`it36int6`). Both configs were added to `rust_grid.CONFIGS` and `quant_sweep.CANDIDATES`; the deploy engine's `parse_level` already accepts int6, so `card:2:int6,note:1:int8` is expressible and this rung is DEPLOYABLE, unlike the per-stream-shift rows (31/34/37).


## Amendment 23 (2026-11-22 21:40): approach 1 (fixed basis) measured - the basis does not transfer across users - and the per-stream shift level makes row 35 deployable

Approach 1 of the six state-size mechanisms stores a FIXED codec basis instead of re-describing one every step: with P, Q fixed K x m matrices the payload is the m x m coefficient block C = P^T W Q, so a layer costs `m*m*qbits + 64*shift_bits` bits and the deploy work is two small matmuls. It is a model-side codec object (no discrete entries, no nearest-neighbour lookup, no per-user vectors) - not a codebook. Fitted on 1,347 raw pre-truncation states per (stream, layer) from 3 users of row 17's checkpoint (`corruption/fit_fixedbasis.py`, `state_subspace_l2i8s4all.npz`, every=1) and scored leave-one-user-out:

| key | n | in-sample m=4 / m=8 / m=16 | LOO m=4 / m=8 / m=16 |
|---|---:|---|---|
| card L0 | 1347 | 0.9938 / 0.9997 / 1.0000 | 0.7181 / 0.9195 / 0.9596 |
| note L0 | 1347 | 0.9935 / 0.9995 / 1.0000 | 0.8559 / 0.9205 / 0.9604 |
| note L1 | 1347 | 0.9856 / 0.9995 / 1.0000 | 0.7510 / 0.8939 / 0.9659 |
| note L2 | 1347 | 0.9848 / 0.9993 / 1.0000 | 0.7816 / 0.8757 / 0.9647 |
| note ALL | 4041 | 0.8636 / 0.9679 / 0.9992 | 0.5671 / 0.7616 / 0.9220 |

The in-sample column is an overfit artifact; the LOO column is the deploy-relevant one. It says (a) the CARD is the wrong place for a codec - its shipped rank-2 truncation is exact (0.9998) for 1,280 bits while the best fixed basis needs m=16 (2,304 bits) for 0.9596 held out - and (b) the NOTE is worth measuring, because its shipped rank-1 keeps only 0.904-0.951 and still passes. Two candidates hold the card at rank 2 and switch ONLY the note: `l2i8fb8n` (1,280/2,304 = 3,584 bits, row 17's exact size) and `l2i8fb6n` (1,280/1,632 = 2,912, 672 below). Both are implemented in the Python path (`fake_fixedbasis_state`, `RWKV_FIXEDBASIS_SCOPE`) and the Rust engine (`fixedbasis_roundtrip` in `model.rs` + the `fast.rs` branch, `RWKV_STATE_FIXEDBASIS_SCOPE` + `RWKV_FIXEDBASIS_PATH`, basis text `corruption/fixedbasis_l2i8s4all.txt`), registered in `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`, and the engine was smoked on real traces (`rust_grid_it36fbsmoke.json`: plain 0.369532/0.363273, `l2i8s4` 0.368217/0.361383, `fb8i8` 0.369105/0.362425, `l2i8fb8n` 0.368729/0.363213). Chain 19 `it36fixedbasis` trains both points (124 steps vs `qnj_it36ctl`) and then runs its own corruption phase for whichever passes the gate.

Also this amendment: `RWKV_STATE_SHIFT_LEVEL` is now PER-STREAM in the deploy engine (`CompressCfg::shift_level: HashMap<usize, f64>`; a bare level still fills all five streams, a `stream:level` list sets only the named ones). That closes the gap behind the loss-only rows 31, 34, 35 and 37: row 35 (`n1i8s3`, 3,392 bits, the smallest passing row) measures 2155.5 rev/s = 0.789x rev/s (median per-user 0.791) on users 7213/7250 against plain 2732.5, above the 0.75x floor, so it is now DEPLOYABLE (`rust_speed_underload_n1i8s3.json`).


## Amendment 24 (2026-11-22 22:20): row 41 misses the imm ceiling by 0.000016 at 3,136 bits; the held-out fixed-basis screen rejects the codec note; 10 traces cannot be exported

**(1) The smallest total yet is 0.000016 short.** `l3i4s4n8s3` (card rank-3 int4 + int4 shifts = 1,024 bits, note rank-1 int8 + int3 shifts = 2,112) reads 0.334034/0.306727 = +0.000232/+0.000516 against `qnj_it36ctl` (ub95 +0.000301/+0.000589), i.e. ahead at 0.46x of the ceiling and the immediate head 1.03x over it. The two cuts compose SUBLINEARLY - +0.000219/+0.000509 additive vs the measured +0.000232/+0.000516 - so 3,136 bits is a real frontier point and the question is only which 1,024-bit card to put under it. The int6 rank-2 card gets the same 1,024 bits for +0.000052/+0.000115 on 11 long histories (`l2i6c`, Amendment 22) where rank-3 int4 costs +0.000208/+0.000472 (row 33). Chain 21 (`it36int6combo`, point `l2i6s4n1i8s3`) tests that swap at the same 3,136 bits; it is registered in `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`, launched behind chain 19, and pre-priced on the 7 long histories by `rust_grid_it36int6shift.json`.

**(2) The fixed basis is a codec that costs 4-6x the ceiling on the note.** The basis (`corruption/fixedbasis_l2i8s4all.npz`) was fitted on users 7213/7044/6960, so the fair test is the 7 OTHER long histories. Running row 17's deploy weights and pairing per user against `l2i8s4` (`rust_wide_screen_it36fbwide_b1.json`, 2,000 bootstrap resamples):

| config | card/note bits | total | d ahead (95% CI) | d imm (95% CI) |
|---|---:|---:|---|---|
| `l2i8fb8n` codec note m=8 | 1,280/2,304 | 3,584 | +0.002035 [+0.001001, +0.003044] better 1/7 | +0.002756 [+0.001430, +0.004158] better 0/7 |
| `l2i8fb6n` codec note m=6 | 1,280/1,632 | 2,912 | +0.001455 [+0.000604, +0.002400] better 1/7 | +0.002154 [+0.001058, +0.003456] better 0/7 |

Better on one user out of 7 on ahead and none on imm - a consistent sign, not an outlier. For scale, the note's int6 factor cut costs +0.000268/+0.000307 and the card's +0.000052/+0.000115, so the codec pays 7-9x a factor-width cut for 672 bits. This is PTQ against rank-r-trained weights, so QAT is the one regime the screen cannot see: chain 19 keeps both points (m=8 = the QAT-recovery A/B, m=6 = the 2,912-bit size claim), with the recorded expectation that the codec FAILS unless QAT recovers most of the gap.

**(3) The deploy-twin corruption coverage is 40 of 50, for a documented reason.** Ten traces fail inside `export_rnn_trace.export_user` with `AssertionError: <review_th> not found in pred_dict` (`rwkv/get_result.py:166`): the frame comes from `../anki-revlogs-10k` while the equalize set comes from the eval's label DB `F:/rwkv_lmdb/label_filter_db_id_e2s`, and for those users that DB holds a review_th beyond the frame's length. They include 5486 (#4 by the roster's metric), 6734 (#19), 6188 (#22), so the gap is not confined to the short histories. The GATE is unaffected - the Python probe covers all 50 through the eval path - and closing the deploy twin means either intersecting the equalize set with the trace's own review_th (a patch to the CERTIFIED exporter) or sourcing the frame from the same DB; it is recorded here as a decision, not applied.


## Amendment 25 (2026-11-22 22:50): the note's int3 FACTOR rung blocks 2,368 bits on any card; the 3,136-bit int6-card swap is pre-priced as cheap; chain 21 gains a gated second rung

**(1) 2,368 bits is closed.** `l3i4s4n3` (card rank-3 int4 + int4 shifts = 1,024, note rank-1 int3 factors + int4 shifts = 1,344) reads 0.334657/0.307554 = +0.000856/+0.001343 (ub95 +0.000963/+0.001478), i.e. 1.71x/2.69x over the ceiling. Against row 41 (same card, note at int8 factors + int3 shifts) the 768-bit note factor cut costs about +0.00062/+0.00083, the same order as row 38's +0.00053/+0.00070 for the same cut on row 17's card - so the note's FACTOR width blocks that size whatever card sits under it, and the frontier below 3,136 bits has to come from the note's SHIFT bits, not its factors.

**(2) The 3,136-bit swap is pre-priced as cheap.** Row 41 reached 3,136 bits with rank-3 int4 card factors and missed the immediate-head ceiling by 0.000016. The int6 rank-2 card buys the same 1,024 bits for +0.000052/+0.000115 (Amendment 22), so `l2i6s4n1i8s3` was priced on the 7 held-out long histories as a per-user PAIRED delta against `l2i8s4` (`rust_wide_screen_it36fbwide_b1_it36int6shift.json`, 2,000 bootstrap resamples): +0.000164 [+0.000071, +0.000274] better 1/7 ahead and +0.000205 [-0.000099, +0.000498] better 1/7 imm. The imm interval contains zero and the ahead interval only just excludes it, so stacking the int6 card ON the note's int3 shift rung does not compound - on this screen it is cheaper on both heads than row 41's card. Chain 21 measures it at 750 users.

**(3) Chain 21 now carries a gated second rung.** `l2i6s4n1i8s2` (the same card with the note's int2 shifts: 1,024/1,920 = 2,944 bits, 448 below row 35) runs ONLY if the 3,136-bit point passes its own gate, because the note's int2 shift rung cost +0.000408 imm in row 20 and is therefore a coin flip at best. Both candidates are registered in `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`; the chain was restarted at 22:31 so the running process holds the gated list.


## Amendment 26 (2026-11-22 23:20): 2,880 bits is closed, `it36combo` is exhausted, and the CPU screen reproduces to 1e-5

**(1) 2,880 bits is closed.** `l2i4s4n8s3` (card rank-2 int4 + int4 shifts = 768, note rank-1 int8 + int3 shifts = 2,112) reads 0.334083/0.306823 = +0.000281/+0.000611 (ub95 +0.000352/+0.000689). The ahead head passes at 0.70x of the ceiling; the immediate head is 1.22x over. It was `it36combo`'s last point.

**(2) The int4-card ladder is closed as a whole, and the CARD is what closes it.** Row 23 (the same card, note int4 shifts, 3,072 bits) is 1.16x over on the immediate head; row 43 is 1.22x over. The note's int3 shift rung therefore costs only +0.000015/+0.000031 for 192 note bits, while the step from row 17's int8 card to an int4 card costs the imm head about +0.00039. So the reduction below 3,136 bits has to come from the card's factor width (int6 -> int5) or from the note's int2 shift bit, not from the note's factors or its rank-1 int8 precision.

**(3) The 7-user CPU screen is reproducible to 1e-5.** `rust_grid.py --tag it36wide5` re-ran the row-17 deploy weights over the same 7 held-out long histories: `plain` 0.411068/0.398831 (earlier 0.411063/0.398798), `l2i8s4` 0.407618/0.395728 (earlier 0.407610/0.395718), `l2i6s4n1i8s3` 0.407791/0.395947 = +0.000173/+0.000219 (earlier +0.000164/+0.000205). Two new rungs are registered for it in `rust_grid.CONFIGS`: `l2i5s4n1i8s3` (3,008) and `l2i5s4n1i8s2` (2,816).

**(4) Chain state.** `it36combo` ended at 23:17:03 with its list exhausted; `it36corrupt2` skipped the three gate-failing points and started the 50-user check for `n1i8s3` (row 35) at 23:21:17. The remaining waiters are `it36e1shrink`, `it36fixedbasis`, `it36e1`, `it36int6card`, `it36int6combo` and `it36int6`.


## Amendment 27 (2026-11-22 23:26): row 35 PASSES the corruption gate - the corruption column is CLOSED and 3,392 bits is fully deployable

`it36corrupt2` scored `n1i8s3` (card rank-2 int8 + int4 shifts, note rank-1 int8 + int3 shifts, 1,280/2,112 = 3,392 bits) on the 50-user long-history roster: 50/50 users, 78,911 raw card/note state observations before `_sanitize_state`, 0 NaN, 0 +inf, 0 -inf, max |state| 0.5899 (after quantization 0.5902), 582 shift-vector passes at max |x| 5.5312, 150 p(recall) tensors with 0 non-finite, 4.7 min on CUDA. **PASS** (`corruption_corrupt_n1i8s3.json`).

This closes the corruption column: rows 15, 16, 17, 18, 29, 30, 31, 32 and 35 all PASS, every other row is a gate FAIL that is never checked, and no cell says `pending` any more. Row 35 is therefore the smallest recipe that satisfies ALL SIX constraints - the loss ceiling on both heads, both bit bars, the 0.75x speed floor (0.789x) and finiteness at 3,392 bits - and it is the number to beat.

`it36corrupt2` then logged `=== it36corrupt2 chain end ===` at 23:26:01: its first three points are gate Fails (rows 23, 33, 34), rows 39-43 fail too, and its last four are `it36noterank2` rows already recorded. `it36e1` took the GPU at 23:29:46 with `l2i8s4pf`, its fused-off QAT control, followed by the two lambda brackets `l2i8s4e1a`/`l2i8s4e1b`.


## Amendment 28 (2026-11-23 00:05): the sub-3,136 corner is priced - int5 is worse than int6, the note's int2 shift rung is a dead end, 2,752 bits is the only survivor

One CPU-only screen (`rust_grid.py --tag it36wide5`, row-17 deploy weights, the 7 held-out long histories, paired vs `l2i8s4`, 2,000 bootstrap resamples; `rust_wide_screen_it36wide5.json`) priced five rungs at once:

| config | card/note bits | ahead | imm |
|---|---|---|---|
| `l2i6s4n1i8s3` (chain 21) | 1,024/2,112 | +0.000174 [+0.000083, +0.000287] better 0/7 | +0.000219 [-0.000069, +0.000509] better 1/7 |
| `l2i6s4n1i6s3` (chain 22) | 1,024/1,728 | +0.000343 [+0.000229, +0.000492] better 0/7 | +0.000435 [+0.000188, +0.000722] better 1/7 |
| `l2i5s4n1i8s3` | 896/2,112 | +0.000404 [+0.000058, +0.000830] better 1/7 | +0.000803 [+0.000082, +0.001753] better 1/7 |
| `l2i6s4n1i6s2` | 1,024/1,536 | +0.000104 [-0.000576, +0.000755] better 3/7 | +0.001641 [+0.000406, +0.003225] better 1/7 |
| `l2i5s4n1i8s2` | 896/1,920 | +0.000162 [-0.000816, +0.001090] better 2/7 | +0.002081 [+0.000707, +0.003632] better 1/7 |

**(1) The int5 card is worse than int6.** At the same 2,112-bit note the int5 card costs +0.000404/+0.000803 against int6's +0.000174/+0.000219: about +0.000230/+0.000580 for the last 128 card bits, roughly 3x the price of any other card bit measured so far. The 2-user PTQ probe that called int5 free (`rust_grid_int6_probe.json`) does not hold on long histories. The 3,008 rung is therefore not worth a GPU run.

**(2) The note's int2 SHIFT rung is a dead end for the immediate head.** At int6 factors it costs +0.001641 against +0.000435 (+0.0012), and at int8 factors +0.002081 against +0.000803 (+0.0013), for 192 fewer note bits each time. Chain 21's gated 2,944-bit second point depends on that rung, so it is a long shot by construction.

**(3) 2,752 bits is the only survivor, and it is marginal.** `l2i6s4n1i6s3` = card rank-2 int6 and the note at rank-1 int6 with int3 shifts reads +0.000343 [+0.000229, +0.000492] ahead and +0.000435 [+0.000188, +0.000722] imm. The point estimate is inside the +0.0005 ceiling on both heads and 640 bits below row 35, but the imm upper bound is above the ceiling, while chain 21's 3,136 point reads +0.000219 [-0.000069, +0.000509] and sits AT it. So chain 21 runs first and chain 22 (`it36int6note`, created 23:58:57, waiting behind chain 21) carries the size claim. Both are PTQ prices against weights trained for the rank-1-int8 note, so neither can see what QAT recovers - which is what the 750-user runs answer.


## Amendment 29 (2026-11-23 00:20): the E1 control PASSES and the Python QAT loop matches the fused kernels; the E1 penalty itself OOMs at the standard batch, so E1 is untested

**(1) The E1 control is clean.** `l2i8s4pf` - row 17's exact layout (card rank-2 int8 + int4 shifts, note rank-1 int8 + int4 shifts, 1,280/2,304 = 3,584 bits) trained on the Python per-step QAT loop (`RWKV_QAT_FUSED=0`) with NO penalty - reads 0.333852/0.306537 = +0.000050/+0.000325 (ub95 +0.000117/+0.000388) against `qnj_it36ctl`. Row 17 (fused kernels) is +0.000052/+0.000327, so the two paths agree to about 2e-6 and `RWKV_QAT_FUSED=0` is bit-equivalent to the fused kernels at the gate's resolution. This is the control the E1 design needs: the penalty must run on the Python loop, so its delta can now be attributed to the penalty rather than to the fused/Python difference. 37.4 min for the 750-user sweep.

**(2) The penalty OOMs, so E1 has no verdict.** `it36e1` then ran `l2i8s4e1a` (lambda=1.0) and `l2i8s4e1b` (lambda=10.0); both raised `torch.OutOfMemoryError` at 10.34-10.44 GiB of the 11.00 GiB allocator cap inside `RWKV7_WKV`'s Python loop, after 14 s and 27 s and BEFORE the first training step. The chain skipped both points and logged its end marker at 00:19:35; `it36e1shrink` read `it36l2i8s4e1a`/`it36l2i8s4e1b` as `null`, found no chain-15 penalty sweep, and ran nothing. The mechanism is the penalty's own cost: `accum_tail_penalty` calls a DIFFERENTIABLE `torch.linalg.svdvals` on the state at every timestep of every rank-truncated layer and keeps each step's SVD graph alive until backward, which adds roughly 3x the state-graph memory per step over the control. The control fits the cap; the penalty does not. E1 is therefore UNTESTED rather than refuted, and the cheap fix is to subsample the steps the penalty sees (or take the tail off the rank-r reconstruction already computed in `fake_lowrank_state` as a Frobenius residual) - not to shrink the batch or the bit budget.

**(3) Chain state.** `it36e1` (chain 15) and `it36e1shrink` (chain 16) both ENDED at 00:19:35-00:19:38; `it36int6` (chain 17) took the GPU at 00:19:59 with `l2i6s4`, whose smoke passed in 1.2 min, and its 124-step decay is running. The waiters behind it are `it36int6card`, `it36fixedbasis`, `it36int6combo` (chain 21) and `it36int6note` (chain 22).


## Amendment 30 (2026-11-23 01:05): row 45 (`l2i6s4`) is the smallest passing row at 2,944 bits, and the E1 penalty is fixed and re-queued

**(1) A new axis-1 frontier.** `it36int6` point 1 - row 17's layout with BOTH factor widths cut int8 -> int6 (card rank-2 int6 + int4 shifts = 1,024 bits, note rank-1 int6 + int4 shifts = 1,920, 2,944 bits total) - reads 0.333927/0.306593 = +0.000125/+0.000381 against `qnj_it36ctl` (ub95 +0.000189/+0.000446), a PASS on both heads. That is 448 bits below row 35 (3,392, the previous smallest passing row) and 640 below row 17, for about +0.000073 ahead / +0.000054 imm over row 17 (difference of deltas, not a paired test). The imm head sits at 0.89x of the +0.0005 ceiling, so the next 128-bit cut must come from the note's FACTOR width: the note's int2 shift rung is closed (row 20) and 2,368 was shown unreachable on the int4 card (row 42). Its finiteness check ran in the first free GPU window, held by `corruption/gpu_hold.py` so chain 18 could not take the GPU first.

**(2) E1 is testable again.** The OOM that killed chain 15's two penalty points was memory, not mathematics: `accum_tail_penalty` ran a DIFFERENTIABLE `svdvals` once per TIMESTEP per rank-truncated layer and every one of those graphs stayed alive until backward, so the cost scaled with the training window. `rwkv/model/rwkv_ops.py` now samples ONE step in `RWKV_QAT_TAIL_STRIDE` (default 8) PER STREAM TAG, an unbiased estimate of the mean tail at 1/stride of the graph memory. Verified twice before queueing: `scratchpad/_test_e1_stride.py` (stride 4 -> 2 of 8 calls, 8 -> 1 of 8, 1 -> 8 of 8, counter per tag) and the real trainer on CPU (`corruption/e1_cpu_smoke.py` with `E1SMOKE_STRIDE=8`, logging `[tail-penalty] lambda=1.0 stride=8 card=0.002546(n=7) note=0.011098(n=19)`). The lever is still inert when `RWKV_QAT_TAIL_PENALTY` is unset, so nothing running is affected.

**(3) Chains.** `it36int6` (17) is running its second point (`l2i5s4`, 2,624 bits) with `l2i4s4n4` after it; the waiters are `it36int6card`, `it36fixedbasis`, `it36int6combo` (21), `it36int6note` (22) and the new `it36e1b` (23: E1 lambda 1.0/3.0/10.0 at stride 8, self-diagnosing because a point whose smoke OOMs is skipped).


## Amendment 31 (2026-11-23 01:20): the note's int5 factor rung costs 6x more per bit - the sub-2,752 corner is closed, so the next gain must come from lower TAX

A second CPU-only deploy screen (`rust_grid.py --tag it36wide6`, row-17 deploy weights, the same 7 held-out long histories, paired against `l2i8s4`, 2,000-resample bootstrap, seed 20261119) priced the note's FACTOR width below int6, at the int6 card:

    config            card/note bits   d ahead                  d imm
    l2i6s4n1i6s3      1,024/1,728      +0.000343 [+0.000229,+0.000492]  +0.000435 [+0.000188,+0.000722]   (chain 22's point, unchanged)
    l2i6s4n1i5s3      1,024/1,536      +0.000948 [+0.000528,+0.001399]  +0.001182 [+0.000663,+0.001756]
    l2i6s4n1i4s3      1,024/1,344      +0.002327 [+0.000675,+0.004053]  +0.002925 [+0.001377,+0.004729]

The note's factor ladder has a KNEE at int6. The int8 -> int6 cut costs +0.000170/+0.000216 for 384 bits (`it36wide5`), but int6 -> int5 costs +0.000605/+0.000747 for 192 bits - about 6x more per bit - and int4 is 4x further out again. Note also that the int5 rung was bought at exactly the size of the closed int2-shift rung (2,560 bits): it is BETTER on the immediate head than that rung (+0.001182 against +0.001642) but 9x worse on ahead (+0.000948 against +0.000104), and both miss the imm ceiling any way round - so neither way of buying 2,560 bits works. Every rung below 2,752 is now measured and closed: 2,752 is the floor of this (rank, factor width, shift width) design space, and the only route further down is removing deploy TAX at a fixed size - which is what the E1 tail-energy penalty tests in chains 23 and 24.

Screen artifacts: `rust_grid_it36wide6.json` (absolutes), `rust_wide_screen_it36wide6.json` (paired deltas + bootstrap), `rust_grid_it36wide6_stdout.log`. The screen ran beside chain 17 (CPU only, no GPU work) and was killed before the chain-17 hand-off, as every grid must be.


## Amendment 32 (2026-11-23 02:00): chain 17 closes the int6 factor ladder - 2,944 PASSES, 2,624 and 2,304 FAIL; the next knob is SHIFT width

`it36int6` ran three points on row 17's layout, cutting the factor widths from int8 -> 6 and then below:

    row  config      card/note bits   d ahead   d imm    verdict
    45   l2i6s4      1,024/1,920      +0.000125  +0.000381  PASS (ub95 +0.000189/+0.000446)
    46   l2i5s4        896/1,728      +0.000285  +0.000566  FAIL (imm 1.13x)
    47   l2i4s4n4      768/1,536      +0.000588  +0.001015  FAIL (1.34x/2.25x)

So int6 is a knee on both streams: the 640 bits from int8 take +0.000073/+0.000054, the next 320 bits (card int6 -> int5) take +0.000160/+0.000184, and int4 breaks both heads. This matches the CPU deploy screen run beside the chain (`it36wide6`: the note's int6 -> int5 step costs +0.000605/+0.000747 for 192 bits), so two independent methods agree on the knee. QAT does recover part of the PTQ price, which is why these rungs need GPU runs rather than screens alone.

Consequence: 2,944 bits (row 45) is the axis-1 frontier, and the one measured-cheap knob left is SHIFT width (row 35: the note's int4 -> int3 step cost +0.000011/+0.000037 for 192 bits). A CPU screen (`rust_grid.py --tag it36wide7`) is pricing the card's shift ladder (`l2i6s3n1i6s4` = 2,880 with only the card's shifts cut, `l2i6s3n1i6s3` = 2,688 with both streams' shifts at int3) before any further GPU time goes into it. Chain 18 (`it36int6card`) started as soon as the row-45 finiteness check released the GPU.
