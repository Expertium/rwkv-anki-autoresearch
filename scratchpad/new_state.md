## Optimization state (research phase, 100/100 workbench)

> Numeric record = `optimization/log.md` + `baseline_log.jsonl` + `research_log.md`. Verbose narrative,
> superseded plans, and the full pre-2026-06-30 snapshot = `optimization/HISTORY.md`. This section keeps
> ONLY the current champion, deploy config, acceptance gate, compact lesson bank, live state, and ops.

### Workbench + baselines
- **Workbench:** train users 1-100 / eval 101-200 (`--short --secs`), `train_db_sc8k` (8192-review chunks),
  MAX_TRAIN_GLOBAL_LEN=66000, WS(+decay), **augmentation OFF** (RWKV_AUGMENT_SEED=1234) + RWKV_DETERMINISTIC=1
  -> run-to-run variance ~0 (so the 0.0003 gate is usable). Eval `python -m rwkv.get_result` (CUDA, JIT-on ->
  REQUIRES the `@torch.jit.ignore` fix on `quant_aware_rwkv7`). ~16 min train + ~4 min eval per experiment.
- **Baseline-to-beat (accuracy TARGET, fp32, NOT deployable):** the d=128 2.76M model trained on 1-100, eval
  101-200 = **ahead 0.320295 / imm 0.281913** (eval via arch-swap `scratchpad/architecture_old_d128.py`).
- **Iteration-0 reference:** d=128, 2,762,884 params, ahead 0.374046 / imm 0.319475 (historical floor).

### CHAMPION = WS-15 + 4-epoch cosine decay  (d=32, 192,800 params)
- arch `[1,4,3,3,3]` (card,deck,note,preset,user), d=32 / K=32 / H=1. ckpt
  `scratchpad/tuner/decay15/decay15_640.pth`; weights `reference/champ_decay15.safetensors`.
- **fp32: ahead 0.314807 / imm 0.280200** -- BEATS the d=128 baseline on BOTH modes (ahead +0.0055, imm
  +0.0017) at 14x fewer params, pure training. Recipe = HP-tuned {peak_lr 1e-3, warmup 200, wd 0.01, clip
  0.25, epochs 15} + 4-epoch decay. (The d=32 arch was never capacity-limited, just badly undertuned.)
- **DEPLOYED (the OFFICIAL comparison number = quant + low-rank, via the Rust engine) [[champion-logloss-deployed]]:**
  BOTH card AND note state = rank-2 low-rank with int4-quantized factors + int4-quantized token shifts.
  ~ahead 0.3140 / imm 0.2806 (still beats d=128 on both). State = card 96 B + note 288 B -> BOTH hard targets
  met (card <=0.15 KB; note >=2x smaller). PTQ penalty is tiny (note int2 +0.0020 imm; no QAT needed -- the
  low-rank card + well-trained decay states quantize cleanly). Rust flags:
  `RWKV_STATE_LOWRANK_SCOPE="card:2:int4,note:2:int4" RWKV_QUANT_SHIFTS=1`. int2 factors DEFERRED (per-column
  scaling rescues to +0.014 imm but not free) -- see [[deploy-known-issues]].

### ACCEPTANCE GATE (research phase) -- accept iff ALL hold (record binary accepted/rejected per iter):
1. "size" (equalized review count, 101-200) IDENTICAL to champion (data-integrity; any change = pipeline bug).
2. params <= **225,000**.   3. card AND note per-entity state UNCHANGED (deck/preset/global MAY grow freely).
4. ahead improves by >= **0.0003** vs the CURRENT champion.   5. imm improves by >= **0.0003**.
=> accept ONLY changes that improve BOTH modes by >=0.0003 (a monotonic champion). [[research-acceptance-gate]]
Two HARD INVARIANTS (never change): hierarchy card->note->deck->preset->global; same preprocessed 92-dim
inputs / existing LMDBs (no new/changed inputs).

### LESSON BANK -- do NOT re-run these (full numbers in log.md / HISTORY.md)
- KEPT: SRS heads 128->64 * card->deck rebalance (compensation deck>preset>user, NOT note) * card 2->1 layer
  * 4-epoch decay * **HP tuning (peak_lr 7e-4->1e-3, clip 0.5->0.25, epochs->15) = the big win; the model was
  undertuned** * scoped state-quant card int4 + note int8 ~free * QAT makes card int2 + note int4 ~free
  (warm-started) * **LOW-RANK rank-2 int4 card/note WKV state BEATS int2 -- smaller AND more accurate (rank-2
  keeps ~98.7% energy)** * shifts must be quantized for honest deploy size (RWKV_QUANT_SHIFTS).
- FAILED/REJECTED: FC/head-width 4->2 (imm +0.053) * note 3->2 layer-cut (iter38, +0.0018) * all-streams
  blanket state-quant (long-recurrence user/global sink it) * note int4 PTQ (>2x budget) * weight PTQ int8/int4
  (no speed win) * QAT from scratch (iter40, +0.0118 -- MUST warm-start) * naive low-rank QAT (iter46; STE
  can't guide a structural rank change -> low-rank stays PTQ, int-quant stays QAT) * capacity adds at 100
  users: num_curves/points 64->128, channel_mixer 1.0->1.5, WS 18 epochs, 8-epoch decay -- ALL reject =>
  **the d=32 model is DATA-limited at 100 users, not capacity-limited; training levers are the wins.**
- DATA-DROP bug (FIXED): `get_groups` silently skips any batch with size>MAX_TRAIN_GLOBAL_LEN. At the old
  MAX=20000 the early loop trained on ~5% of the data; MAX=66000 = full coverage (worth ~0.013 imm -- larger
  than the entire early iter0->iter36 loop). Iter-to-iter rankings stayed valid (same subset) but absolute
  quality was on a biased slice.
- GPU-training speedups (arch-agnostic, non-gating): `torch._foreach_*` for copy_downcast_/grad-transfer +
  skip grad_norm/log_model when wandb off + JIT restored via `@torch.jit.ignore` on `quant_aware_rwkv7` (the
  QAT-lowrank `torch.linalg.svd` had SILENTLY broken TorchScript -> would crash plain WS/eval) = ~1.38x over
  the no-JIT body. `torch.compile` is Windows-blocked (no Triton). Gate parallelism (run_qat_eval.sh NPROC)
  made the Rust gate ~8x faster.
- BLOCKED/DEFERRED: **K<32** (head dim) is hardwired ONLY in the per-row warp reduction of `rwkv7_cuda.cu`
  (one 32-element state row = one 32-lane warp; `__shfl_down_sync offset=16..1`) -- a LOCALIZED, moderate
  kernel change (K-aware sub-warp / shared-mem reduction), NOT a rewrite. It is the path to a 2x smaller AND
  faster model (H=2/K=16: WKV state 1024->512 floats/layer, ~half the matmul) = task #3. CUDA graphs (variable
  shapes, ~1.1-1.3x only). Stateful-BPTT carry SHELVED (smaller chunks don't speed training; the verified
  stateful WKV kernel is done + UNCOMMITTED in rwkv_ops.py / rwkv7_cuda.cu) [[stateful-bptt-shelved]].

### SPEED -- where GPU training time actually goes (RE-DIAGNOSED 2026-06-30) [[gpu-training-speed-levers]]
- **Fetching is already HIDDEN -- NOT a lever.** `data_fetcher.get()` waits ~2.5-3 s on the FIRST batch then
  ~3-7 ms/step (7 workers + FETCH_AHEAD=5 fully hide prep+IPC); the input `.to(device)` H2D is ~0 ms on the
  critical path (~21 MB batch). Async-pinned prefetch / mp.Queue swap / vectorizing prepare() buy ~nothing.
  (This CORRECTS the earlier "fetch overlap 1.5-1.85x" claim, which was wrong about the mechanism.)
- **Cheap win = `RWKV_EMPTY_CACHE_EVERY`** (env added; default 1 = byte-identical). The per-step
  `torch.cuda.empty_cache()` (first 1000 steps, an OOM-fragmentation guard) costs +~150 ms/step -> ~1.2x for
  short (960-2400-step) runs. Numerics-neutral (allocator only); validate via `scratchpad/run_ectest.cmd`
  (steps/s + no-OOM, model is tiny ~6/12 GB).
- **Real lever = the WKV-kernel compute floor (fwd 140 + bwd 403 = ~543 ms/step, ~80% of the step).** Only a
  smaller model / K<32 kernel / bigger batch moves it => task #3 (2x smaller) is ALSO the main speed win.
  Param breakdown (192,800): 5 RWKV streams 75.5% (deck 4L 21.6%, note/preset/user 3L 16.2% each, card 1L
  5.4%), SRS heads 16.0%, input FC 8.4%; ~10.4k params per d=32 layer.

### LIVE STATE (2026-06-30)
- **RUNNING (detached; monitor via OS truth -- watchers die on teardown):** `build_1500` = building
  `train_db_sc8k_1500` (users 1000-2499, ~56 GB, sc8k 8192-chunk) for the "varied data, few epochs"
  experiment. RESUMABLE via `scratchpad/run_build_1500.cmd` (skips `_done`); monitor `scratchpad/build_1500.log`
  (tqdm + DONE_EXIT_).
- **QUEUED (after build frees the CPU):** (1) `scratchpad/run_ectest.cmd` -- validate the empty_cache speedup.
  (2) `scratchpad/run_train_1500.cmd` -- 1 epoch WS on 1000-2499 (compute-matched to the champion) -> eval
  101-200 -> score vs champion (data VARIETY vs REPETITION; keep the better recipe). (3) EMA experiment
  `scratchpad/run_exp_ema.cmd` (WS-15 + EMA 0.999, eval averaged weights).
- **ACTIVE AGENDA (Andrew, 2026-06-30):** improve fetching [DONE -- not a lever; empty_cache banked] ->
  100u*15ep vs 1500u*1ep experiment, keep the better -> **reduce model size ~2x (H=2/K=16, the localized
  kernel change above)**. META-GOAL: make training on 5000 users practical = speedups + smaller model +
  (stretch) the K<32 kernel rewrite. The arch search is DATA-limited at 100 users -> training levers + the
  size/kernel work are the frontier, not capacity adds.
- **UNCOMMITTED code (commit-when-asked):** `rust/rwkv-infer` (low-rank sort-fix + per-column low-rank);
  `train_rwkv.py` (EMA + HP env overrides + augmentation seed + RWKV_EMPTY_CACHE_EVERY); `data_processing.py`
  (label_filter-optional); `architecture.py` (RWKV_NUM_CURVES/POINTS / CHANNEL_MIXER_FACTOR / LORA env
  overrides); `rwkv_ops.py` + `rwkv7_cuda.cu`/`.cpp` (stateful WKV kernel -- verified, currently unused).
- Tuner `optimization/hp_tuner.py` (greedy coordinate descent, resumable from `tuner_log.jsonl`) is ready for
  the next tune -- run SPARINGLY (only after a big arch change or several accumulated small changes). Arch
  env-overrides default to the champion. Literature-review queue: `optimization/LIT_REVIEW.md`.

### Ops
- **Compaction (ONLY sanctioned way):** run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
  then yield idle and STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) +
  focus-bearing flag (stale/empty = purged). Never hand-create `pending_compact.txt`. The injector is 24/7
  (ClaudeLoopController every 3 min; acts only on a stale heartbeat) and may inject EXACTLY `/compact <focus>`
  or a short `Continue` -- nothing else.
- **ESC-PROOF detached launches:** Esc / session teardown tree-kills Claude's Bash/PowerShell bg jobs INCLUDING
  training. Launch each training as a self-contained `.cmd` via `scratchpad/detach.ps1` (WMI Win32_Process ->
  parented to WmiPrvSE, survives); log to a STABLE repo path (`scratchpad/*.log`, NOT the rotating session
  temp); end the .cmd with `echo DONE_EXIT_%ERRORLEVEL%`. MONITOR via OS truth (poll the log / Get-Process /
  ckpt mtime) -- detached runs give NO tool-completion event. A Bash watcher gives notifications but is itself
  Esc-killable (re-arm it each turn; the training survives). Beat the heartbeat each working turn
  (`claude-automation/beat.ps1`). **Do NOT kill the FSRS benchmark PIDs (the ~80000s-CPU python procs).**
- **DATA FACT:** anki-revlogs-10k has NO absolute timestamp / review-id (anonymized; raw `revlogs` parquet =
  card_id, day_offset [integer DAY counter], rating, state, duration, elapsed_days, elapsed_seconds). Time-of-
  day is UNRECOVERABLE -> a time-of-day input feature is impossible here. elapsed_seconds (time-since-last) is
  already an input.
- Quant papers: `scratchpad/{rwkvquant,rwkvedge}.txt` (poppler installed; the Read tool handles PDFs). Use the
  CURRENT session's scratchpad dir for transient logs (it rotates on teardown -- check task-output paths).
