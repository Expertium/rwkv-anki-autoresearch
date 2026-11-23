# Project instructions
## User identity
- Andrew is the user; GitHub username: `Expertium`.

## User and writing
- Andrew knows Python/PyTorch. Explain Rust, CUDA, build tools, and git internals when needed. Recommend a choice; do not ask him to explain model internals.
- Write chat replies in concise Simplified Technical English. Keep caveats that affect correctness.
- Treat hand-edited changes as intentional. Do not revert them without a clear reason.
- Do not edit sibling repositories or read-only datasets. Use sibling source only to check behavior.

## Work order
- **GPU first:** before unrelated work, check whether a safe, authorized GPU run can start. Start it first. If a run ends or fails, resume or launch the next safe run before unrelated work.
- Keep the GPU busy whenever safe work is available. Do not bypass run order, resource limits, or required approval. Do not overlap GPU jobs when a gate forbids it.
- Continue independent CPU work while the GPU is busy. Do not stop only because the GPU is busy.
- Andrew authorized queued experiments without repeated permission requests. A new multi-day endgame run, a deploy-contract or state-budget change, destructive data work, or a genuine research-direction choice still needs his decision.
- Check logs, artifacts, skipped batches, and active processes. A successful exit code alone does not prove a run succeeded.

## Research and data safety
- Keep training, evaluation, and CPU/Rust inference aligned. For input, output, or objective changes, check all three paths and run relevant parity tests.
- Gate experiments with the current lineage and acceptance rules. Record each result in the research log and summary. Update `optimization/GPU_RUN_BOARD.md` when run state changes.
- Do not change the deploy contract or state budget without Andrew's approval. Preserve the fixed hierarchy and input requirements stated by the current research plan.
- Protect run artifacts and read-only data. Do not delete databases, checkpoints, logs, or queued-run files without authorization. Do not copy secrets into public or tracked locations.
- Use `.venv/Scripts/python.exe` in PowerShell. Use `scratchpad/detach.ps1` for long runs so terminal resets do not stop them. Never edit a running `.cmd` file.
- Before starting a new runner, inspect it, run `scratchpad/preflight_runner.py`, and use `scratchpad/mk_dryrun.py` when the runner or phase shape is new. Check GPU and RAM constraints first. (`preflight_runner.py`, `mk_dryrun.py` and `count_skips.py` are absent from this checkout as of 2026-11-15; `scratchpad/detach.ps1` was recreated. Substitute `scratchpad/adaptive_wd/preflight_adaptive.py`, a runner `--dry-run` config build, and the required 0.04-epoch smoke.)
- Keep at least 10 GB of system RAM available. Current local training cap is 11 GiB; capped runs need the skipped-batch check. Do not run another GPU workload beside an active gated experiment.

## Git and external services
- Do not commit, push, or open a pull request unless Andrew asks. Never push large data or checkpoint files.
- For external posts on Andrew's behalf, disclose that an AI agent wrote the content. Never call the agent Claude unless Andrew asks.
- **Uncommitted worktree deletion set (first seen 2026-11-23 02:00):** ~2,158 tracked files under
  `scratchpad/` (old arms: `ab_*`, `arch_*`, `baseline_gru|lstm/`, `bench/`, `bughunt/`, `budgetcal/`,
  `workload/`, `ws10/`, ...) are deleted in the worktree but still in `HEAD`. No file under
  `quant_plan/`, `corruption/`, `rwkv/`, `rust/` or `optimization/` is affected, so the science is
  intact. Do NOT commit those deletions - `git add -A` is unsafe in this checkout - and do not restore
  them either (the deletion looks deliberate). Stage explicit paths on every commit.

## Current references (check these before acting)
- `PROJECT_MAP.md` - folders and key files. Update it when the tracked project structure changes.
- `optimization/GPU_RUN_BOARD.md` - active runs and queue. Read this first before scheduling experiments.
- `scratchpad/adaptive_wd/NOTES.md` - adaptive-WD lineage, gates, and resume hazards; use the run board for live status.
- `scratchpad/ta3/NOTES.md` - TA3 (Training-Time Testing) arm: pre-registered endpoints, gate history, epoch plan, monitoring, resume hazards.
- `scratchpad/quant_plan/RESULTS.md` + `scratchpad/quant_plan/NOTES.md` - card/note state-quantization arm (anchor `iter36`): results table with bit/speed columns, the QAT-vs-PTQ tax decomposition, live status and the queued screening chains. **Gate (Andrew, 2026-11-20): the anchor is the run that is identical EXCEPT for quantization**, so deltas are quoted against `qnj_it36ctl` (124-step decay) or `qnj_it36ctl496` (496-step decay), never against the iter36 champion. **Step budget (Andrew, 2026-11-21): a QAT decay may use MORE than the champion's 124 steps** ("it will make results less apples-to-apples comparable, but if it ultimately ends up turning a failing recipe into a winning one, it's fine"); each length still needs its own fp32 twin at the SAME length, because longer training helps fp32 too. Screening chains take a lock file (`scratchpad/quant_plan/it36qatsmall*_chain.lock`); delete a stale lock by hand after an outage, and do not add a chain's own script to its busy pattern (the venv `python.exe` is a shim, so one chain shows two processes). **Mechanism (2026-11-21): `cn8` (full-matrix int8, no rank truncation, 4x over the bars, diagnostic only) scores +0.000033/+0.000022, so an 8-bit state is lossless when it is not truncated - the rank-1 tax is RANK TRUNCATION, not precision, and a finer level cannot fix it.** The per-stream attribution on row 4's checkpoint gives card-only +0.001282/+0.001142 vs note-only +0.000373/+0.000258, so the CARD owns the immediate-head tax (and the note alone is inside the ceiling). **Corrected 2026-11-21 23:00: rank >= 2 IS deployable on the CARD alone.** The 0.629x rev/s / 0.448x states/s figure was rank-2 on BOTH streams (`l2i4`, which also puts rank 2 on the note's 3 layers). Rank 2 on the CARD only - 1 layer, the cheap stream - measures **2,323.6 rev/s = 0.812x (states/s 0.660x, median per-user 0.803x)**, above the 0.75x floor, and the deploy engine already parses `card:2:int8` (rows 15-18 need no engine work). A rank >= 2 NOTE is a different matter: the engine PARSES and RUNS it, but it FAILS the floor - 1,743.0 rev/s = 0.600x at rank 2 and 1,686.8 = 0.581x at rank 3 (`rust_speed_probe_note2.json`, user 107, 1 rep, while a decay trained; `l2i8s4` read 0.783x in that same probe, so the idle figure is about 0.62x) - because `compress_wkv_state` uses the rank-1 power iteration for rank 1 and a full KxK Gram eigendecomposition for rank >= 2. So a rank >= 2 NOTE needed a FAST rank-r engine path, not a parser change - and since 2026-11-22 it HAS one: `RWKV_LOWRANK_FAST=1` swaps the Gram eigensolve for a warm-started block subspace iteration + Rayleigh-Ritz and measures **0.803x rev/s for `l2i8n2i8`** (rank-2 note) and 0.791x for `l3i8n2i8`, against 0.618x/0.611x for the reference solver in the same session, at a deploy log-loss cost of imm +0.000051 / ahead +0.000000 (`rust_fastrank_check.json`, `rust_speed_fastrank_{on,off}.json`). The knobs are `RWKV_LOWRANK_FAST` (off by default), `RWKV_LOWRANK_FAST_CHECK` (run BOTH solvers, record the largest relative Frobenius difference, still WRITE the reference factors, so a checked run is bit-identical to the reference) and `RWKV_LOWRANK_FAST_ITERS` (default 2). The flag must stay OFF for any Python<->Rust parity run: the fast factors are an approximation, not a bit-identical truncation. That measurement ran while a QAT decay was training, and the idle-machine re-measurement (`rust_speed_idle_l2i8.json`, 2026-11-22 11:47) reads **0.809x** for the rank-2 card path (`l1i8` 0.902x, plain 2,915.1 rev/s), so the under-load numbers are accurate to about 1%. Row 9 `l1i8s4` (768/2,304 bits) is the axis-1 best: row 4's tax for 25% fewer bits, because the shift vectors can drop int8 -> int4 for free.
- `optimization/PROPOSALS.md` - ranked research queue and proposal rules.
- `scratchpad/quant_plan/autorun/` - **the hourly supervisor** (Windows Task Scheduler task `RWKV-Quant-Autorun`, every hour): `quant_hourly.yaml` is the goose recipe, `QUEUE.md` is its contract, `status_report.py` is the deterministic state dump it reads first, `run_autorun.cmd` + `autorun_lock.py` are the wrapper and the single-instance lock, `AUTORUN_LOG.md` holds one block per run, `RECORDED.txt` lists the sweep stamps already written into RESULTS.md. It may only (a) restart a LIVE chain whose log has stalled for >45 min while the GPU is idle, (b) run the one-time idle deploy measurement `rust_speed.py --tag idle_l2i8`, and (c) record landed results in RESULTS.md/NOTES.md. It never starts a new research direction, never writes to git, and never runs a second GPU workload. **Coordination rule: before editing `RESULTS.md`, `NOTES.md` or `optimization/GPU_RUN_BOARD.md`, run `.venv\Scripts\python.exe scratchpad\quant_plan\autorun\autorun_lock.py status`; a fresh lock means that run is mid-edit, so re-read the file immediately before your change and keep the change small.** **GPU window for non-chain jobs (2026-11-22):** a chain hands off inside a point with no idle gap, so a non-chain GPU job (a corruption sweep, a one-off eval) cannot wait for a free GPU without racing the handoff. `corruption/gpu_hold.py` solves it: a CPU-only, self-expiring process whose command line carries a BUSY_PAT token (`--tag rust_grid.py`), so it holds the queue "busy" for a bounded window; run the job inside that window, then kill the holder. It never uses the GPU, so it cannot create a second GPU workload. **Git caveat (2026-11-22 00:07, found by the hourly supervisor):** this checkout's git rejects `git status --short --cached` (exit 129, "unknown option `--cached`"); verify the staged set with the read-only equivalent `git diff --cached --name-status` instead. **Wedge hazard (2026-11-22):** an F: (Disk 1) I/O error - System log events `[disk] 153` (I/O retried at LBA 0x131488190) and `[disk] 51` (paging error on `\DR1`) at 02:33:45-46 - wedged a training run for 8.7 h at 0.000 s CPU while the report still said `GPU BUSY`. `it36ladder_lib.run` now kills and retries a tree that shows neither log growth nor CPU for 20 min, and `status_report.py` prints a `CPU over 5 s:` line per workload process (a healthy 4-process tree reads about 14 s per 5 s of wall time). **Chain state (2026-11-22 11:21, Andrew's instruction "stop further runs for now and ONLY do corruption checks for (as of right now) passed runs"):** `it36corrupt` is the only chain allowed to run; `it36shrink`, `it36equalbits`, `it36combo` and `it36corrupt2` are PAUSED (killed, locks deleted, marked PAUSED in `QUEUE.md` sections 2 and 5) until Andrew lifts the instruction. **Update 2026-11-22 12:02:** Andrew resumed experiments ("Since corruption checks are done, we can resume experiments. Sure, try rank-2 note"), so all five chains of `QUEUE.md` section 2 are LIVE again; `it36noterank2` started at 12:05:28 and the other four wait behind it. **Second wedge, 2026-11-22 15:17:** the guard fired with NO disk event logged (System log clean for 90 min before it, F: Healthy) - a smoke trained and validated, then sat at 0 CPU - killed the tree, and the single retry finished that smoke in 1.6 min. `STALL_RETRIES` is per step, so every step (smoke, decay, eval) has one retry of its own. **Chains outside the supervisor's contract (first seen 2026-11-22 16:05):** `it36e1` (lock 15:22) and `it36e1shrink` (lock 15:59) are the interactive session's chains - they are not in `QUEUE.md` section 2, they wait behind `it36noterank2`, and a CPU-only `rwkv.train_rwkv` tree on config `scratchpad/quant_plan/qat_e1cpufake/` ran at about 1 core while they waited. The hourly supervisor must not start, kill, restart or score them. Also outside it (first seen 2026-11-22 17:05): a CPU-light `rust_grid.py` + `rwkv-infer.exe` tree (row 17 bundle) that the interactive session runs beside a chain eval - 7 workload processes with no `it36*_chain.py` owner is normal, not a second GPU workload started by the supervisor. Also outside it: `it36int6` (chain 17, the int5/int6 factor-precision ladder, first seen in the wait list 2026-11-22 17:05) is a seventh waiter that QUEUE.md section 2 does not list - do not start, kill, restart or score it. Further waiters outside section 2 seen in the 2026-11-23 wait lists - `it36int6card`, `it36int6note`, `it36int6combo`, `it36fixedbasis`, `it36shiftrung`, `it36e1b`, `it36e1combo` - are the interactive session's chains too, so the same rule applies to them. **Interactive-session RESULTS edits (2026-11-22 21:44-21:59):** RESULTS.md rows 39-40 appeared outside any hourly run (stamps in `RECORDED.txt` at 21:44, RESULTS.md written 21:59, no `AUTORUN_LOG.md` block, local == origin), together with `M scratchpad/quant_plan/it36noterank2_chain.py` and an untracked `corruption/_fix_row31_cell.py`. Read an uncommitted RESULTS.md like that as the interactive session's own work: QUEUE.md section 4 forbids a git command in a run that changed nothing, so the supervisor must not commit or push it - check row cell counts (12 per row) instead, log the observation, and let that session commit. That session did commit: by 2026-11-22 23:05 RESULTS.md was clean in git and rows 41-42 (`l3i4s4n8s3`, `l3i4s4n3`, landed 22:11:04 and 22:43:41) had been recorded the same way, so a report count of 0 `NOT RECORDED` beside rows that no hourly block mentions is normal here - verify the cells (12 per row) and the `RECORDED.txt` stamps, then do nothing.
- `scratchpad/quant_plan/corruption/shiftbasis_note16.npz|.txt` + `RWKV_SHIFT_FIXEDBASIS_PATH/MODES/SCOPE`
  - **the TOKEN-SHIFT CODEC** (2026-11-23, the shift half of the fixed-basis family). The two persisted
  shift vectors are 47% of a compressed note layer's bits (2*32*4 = 256 of 768 at int4). They are
  LayerNorm outputs with a near-shared mean, so a MODEL-SIDE `(mu, P)` - 32 + 32*m floats, ZERO state
  bits - codes them as m coefficients at the stream's shift level: 2*m*level bits/layer. Multi-user fit
  (6 users, 3,000 rows/key): **m=16 int4 = rel err 0.01119 at 64 bits/vec against raw int4's 0.01214 at
  128 bits/vec**, m=16 int6 = 0.00664 at 96, m=16 int8 = 0.00641 at 128, m=12/m=8 int4 = 0.01568/0.02486.
  The single-user figure that first motivated this (m=8 int8 = 0.0132) was OPTIMISTIC: pooled over 6
  users m=8 int8 = 0.0244, worse than raw int4 at half the bits. **The artifact is NOTE-ONLY** - the
  card's own shift subspace needs m=16 int8 merely to MATCH int4 (0.01666 vs 0.01738) and a note-fitted
  basis scores 0.10-0.25 on the card, so the bare `r0`/`r1` tiers were deleted from the .npz and the
  .txt, and `RWKV_SHIFT_FIXEDBASIS_SCOPE=note` pins the codec in the Python lookup
  (`rwkv_model._shift_fb_lookup`), the Rust tier resolution (`model.rs`, after the
  `RWKV_SHIFT_FIXEDBASIS_PATH` parse) and BOTH bit accounts (`quant_sweep.bits_for`, `rust_grid.bits_for`).
  Per-mode coefficient scales were priced and REJECTED (`corruption/price_shift_coders.py`). Chain 28
  `it36shiftsb` = the 5-point ladder (2,048 / 2,240 / 2,432 / 1,952 / 1,856 bits, all holding chain 26's
  PASSING row 61) and it measures the codec as FREE on speed (`rust_speed_c28.json`: 2,323.6 rev/s,
  identical to `l2i8fb4n` in the same run).
- **The Python<->Rust parity harness is a LOOSE witness** (2026-11-23): `corruption/export_parity_trace.py`
  exports through `export_rnn_trace.py`'s `SrsRWKVRnn` (`rwkv/model/srs_model_rnn.py`), which has its OWN
  mixers and therefore calls neither `fake_quant_shift`/`fake_shift_fixedbasis` nor
  `rwkv_ops.quant_aware_rwkv7` - the trace's `py_pred_*` are UNQUANTIZED while the Rust engine's are
  quantized, so `py_rust_parity.py` passes whenever the quantization is LIGHT (a synthetic 10%-error
  shift codec is what finally made the gap visible: `imm d -0.001557`, FAIL). The 750-user sweep is NOT
  affected: `quant_sweep.py` runs `optimization/eval_sharded.py`, i.e. the real `rwkv_model` mixer
  forwards, which apply the state codec, the shift codec and the int-N paths (that is why a codec point
  shows a nonzero delta at all).
- `scratchpad/quant_plan/corruption/` - **the corruption (finiteness) check** (Andrew's gate, 2026-11-21: a recipe fails if even ONE card/note state value or even ONE p(recall) is non-finite on the users with the longest histories, with those histories NOT truncated). `finite_probe.py` monkey-patches the eval path (no repo source is edited) to observe the RAW recurrent state before `_sanitize_state` caps it, the token-shift vectors, and `p_curve`/`p_imm`/`p_imm_all` of every row; `probe_get_result.py` is the entry point that installs it; `check_corruption.py` is the driver (one process, `RWKV_QAT_FUSED=0` -- the bit-parity Python twin of the fused CUDA kernels -- because the state is only visible in Python there; a run with 0 state calls is INCOMPLETE, never a pass); `corrupt50.json` is the 50-user roster (200,000-section-row VRAM cap, the too-big users recorded in `excluded_too_big`). Run it via `it36corrupt_chain.py`. Its raw output is `probe_out/` (gitignored); the verdict is `corruption_corrupt_<tag>.json`.
- `optimization/research_5k.md`, `optimization/research_log.jsonl`, `optimization/LEDGER.md` - results and loss changes.
- `optimization/DEPLOY_FUNCTIONS.md` - Anki deploy contract; `optimization/PROTOCOL.md` - experiment procedure.
- `optimization/CPU_INFERENCE.md`, `optimization/DATASETS.md`, `optimization/FUTURE_FEATURES.md` - topic-specific detail.
- `optimization/HISTORY.md` and `optimization/AGENTS_ARCHIVE_2026-10-12.md` - historical context, not active state. Verify history against source.
- `rwkv/`, `rust/rwkv-infer/`, and tests are the implementation source of truth.

## Research advisor (after the current queue)
- After all experiments already listed in `optimization/GPU_RUN_BOARD.md` have finished and have recorded verdicts, start a **new conversation** with the **`gpt-6.1-sol-high`** profile for research advice. Do not reuse a prior Sol conversation. Start another new conversation for every later request for research directions.
- Treat Sol as a consultant, not an agent. Give it the selected repo files as text context and instruct it not to use tools, run commands, edit files, delegate work, or take other agentic actions. It must return only a text response.
- Ask Sol to propose **5-10 next experiments**. For each proposal, require both (a) the conceptual mechanism and expected effect, and (b) concrete technical changes, including relevant files, functions, classes, configuration, and guards.
- Select the context files yourself. Always include this `AGENTS.md`, `optimization/research_5k.md`, `optimization/research_log.jsonl`, `optimization/LEDGER.md`, `optimization/PROPOSALS.md`, `optimization/GPU_RUN_BOARD.md`, `optimization/LOGLOSS_BREAKDOWN.md`, `optimization/DEPLOY_FUNCTIONS.md`, `optimization/PROTOCOL.md`, and `optimization/HISTORY.md`. Add the relevant current architecture, training, evaluation, feature, and test files for the proposals. Check every file against the current source; historical notes do not override code.
- Keep the complete prompt, including file contents and instructions, below **200,000 tokens**. Count it with an off-the-shelf tokenizer before sending. `tiktoken` with `o200k_base` is an acceptable estimate; leave a clear margin below the limit because Sol's exact tokenizer may differ. If the selected files exceed the budget, include the complete required core files first, then trim or summarize lower-priority history and unrelated source files. Report the tokenizer and estimated count in the prompt.
- Do not accept proposals that violate the current research gate, deploy constraints, run order, data-safety rules, or Andrew's approval requirements. The normal experiment loop and validation rules still apply to every proposal.

## Experiment smoke runs
- For every experiment that is **not simply hyperparameter tuning**, run a small end-to-end test at **`EPOCHS = 0.04` before the full training run**. This check must use the experiment's real code path and relevant flags. Confirm that training starts, the new path executes, checkpoints and required artifacts are written, no batches are skipped unexpectedly, and the runner's guards pass. Use a separate smoke tag, folder, log, and checkpoint prefix so it cannot overwrite the full run.
- A smoke run checks implementation and runner correctness. It does not measure accuracy, prove parity, establish an acceptance result, or replace required unit, parity, data-integrity, or full-run checks. For a new experiment, run this 0.04-epoch smoke in addition to any other dry-run requirement; a 0.05-epoch generic dry run does not replace it unless it exercises the same real code path and already uses at least 0.04 epochs.

## Session notes (2026-11-22, late)
- **Tool limit:** `execute_javascript` kills its shell call after **300 s**. Any command that can run
  longer (a deploy grid, a sweep, a trace export) must go through
  `powershell -NoProfile -File scratchpad\detach.ps1 -Command "<cmd> > <log> 2>&1"` and then be polled.
  A foreground grid killed at 300 s writes **nothing** (no preds, no JSON), so the whole run is lost.
- **The "3,000/3,000-bit run" is VETOED.** `scratchpad/quant_plan/autorun/QUEUE.md` section 6 forbids it
  by name, and the deploy bars are card <= 2,176 / note <= 6,528 bits. Any task-planning note that asks
  for a 3000/3000-bit run is stale: do not act on it, and do not change a state budget without Andrew.
- **Axis-1 frontier (2026-11-23 02:00):** the smallest PASSING row is **45 `l2i6s4` = 1,024/1,920 =
  **2,944 bits** (ub95 +0.000189/+0.000446, corruption PASS 158,986 state obs / 0 non-finite), i.e.
  row 17's layout with BOTH factor widths cut int8 -> int6. That cut costs only +0.000073/+0.000054 on
  top of row 17 for 640 bits. **int6 is a KNEE on both streams**: the next steps cost 6x more per bit
  (row 46 `l2i5s4` 2,624: +0.000160/+0.000184, FAIL imm 1.13x; row 47 `l2i4s4n4` 2,304: FAIL
  1.34x/2.25x), and the independent CPU screen agrees (`it36wide6`: the note's int6 -> int5 step
  costs +0.000605/+0.000747 for 192 bits). Every FACTOR rung below 2,944 is closed, so
  SHIFT width is the one remaining knob. The paired `it36wide7` screen
  (2026-11-23 02:45, base = row 45's own `l2i6s4`) priced the three rungs at +0.000071/+0.000497
  imm (card int4 -> int3, 2,880 bits), +0.000139/+0.000461 (both streams at int3, 2,688) and
  +0.000848/+0.002195 (the extra card int2 rung, 2,624) and closed them - but that closure rested on
  an INVERTED reading of its own number ("the note's int6 cut recovered 9% of its PTQ imm price"
  means QAT REMOVES ~90%, which is the condition that would let a rung PASS). Corrected 2026-11-23
  03:30, with the measured ratios: QAT's cost is 10-53% of a PTQ screen's price across the cuts
  measured so far (the note's int6 cut: screen +0.000268/+0.000307 vs QAT +0.000046/+0.000031), so
  **a PTQ screen does NOT predict a QAT cost.** Chain 25 (`it36shiftrung`, waiting behind chain 24)
  therefore MEASURES the two rungs (2,880 then 2,688) at 124 steps against `qnj_it36ctl`, each with
  its own finiteness check. Until they land, treat 2,944 bits as the frontier but NOT as a proven
  floor of the uniform design space. The last live sub-2,944 claim is chain 22's 2,752
  point (`l2i6s4n1i6s3` = row 45 minus the NOTE's 192 shift bits, which row 35 priced at
  +0.000011/+0.000037 on 750 users); the CARD owns the shift cost, not the note.
  **The fixed-basis CODEC (approach 1) is live and it BEATS the per-step factors at equal bits.** Row 50
  (`l2i8fb8n`: card rank-2 int8 + a codec NOTE at m=8) is 1,280/2,304 = 3,584 bits - row 17's EXACT size -
  and reads +0.000040/+0.000129 against +0.000052/+0.000327 for row 17's per-step rank-1 note, i.e. better
  on BOTH heads; its held-out energy capture is LOWER (0.876-0.921 at m=8 vs 0.904-0.951), so capture does
  not predict the loss tax. Its engine path FAILED the floor at 0.738x median per-user until 2026-11-23,
  when `fixedbasis_roundtrip` was rewritten to transpose the basis ONCE per call instead of walking it with
  a stride-32 offset in every inner loop: 1.70-1.80x faster per compression call and BIT-IDENTICAL
  (`rust_grid_fbpar_{before,after}.json` equal to the last bit, `compress_bench::bench_fixedbasis_layout`),
  which moves the same probe to 0.775x rev / 0.778x median per-user - so the codec is now deployable on
  both axes, and chain 19's second point (`l2i8fb6n`, codec note m=6 = 1,280/1,632 = 2,912 bits, 32 BELOW
  row 45's frontier, 0.797x/0.793 median) is the size claim to watch.
- **Deploy screening:** `scratchpad/quant_plan/corruption/wide_screen.py` turns any number of
  `rust_grid.py` grids into per-user PAIRED deltas against a base config (2000-resample bootstrap,
  seed 20261119). Its `tags` list takes several grids, so a screen can be run in batches under
  different `--tag`s without overwriting the per-config pred dirs. Both tools are CPU-only and run
  beside a chain safely; the 7-user screens are at the noise floor for deltas below ~0.002, so widen
  the user list before trusting such a number. The already-tested factor-precision knobs are Hadamard
  (hurts), per-column (`RWKV_LOWRANK_PERCOL`, neutral), ALS (neutral to slightly better), 4-level,
  mixed53, compand, VLEVEL (asymmetric U/V) and EF (error feedback, chains 15/16).
- **E1 tail-energy penalty, and its VRAM fix (2026-11-23).** The penalty (`RWKV_QAT_TAIL_PENALTY`,
  `rwkv/model/rwkv_ops.py:accum_tail_penalty`) adds `lambda * mean over streams of the tail fraction`
  to the training loss: zero state bits, zero deploy cost, the only zero-bit axis-2 lever. The first
  two points (`l2i8s4e1a` lambda=1.0, `l2i8s4e1b` lambda=10.0, chain 15) **OOM'd at 10.34 GiB of the
  11.00 GiB cap before their first step** - NOT a broken penalty, because it ran clean on CPU
  (`corruption/e1_cpu_smoke.py`), but a memory one: a differentiable `svdvals` per TIMESTEP per
  truncated layer keeps every step's graph alive until backward, so the cost scales with the training
  WINDOW. Row 44 (`l2i8s4pf`, the same layout with the penalty OFF, trained on the fused-off Python
  loop) is the control and PASSES at +0.000050/+0.000325, within 2e-6 of row 17's fused run - so
  `RWKV_QAT_FUSED=0` is bit-parity and an E1 delta cannot be blamed on the fused/Python difference.
  The fix is `RWKV_QAT_TAIL_STRIDE` (default **8**): accumulate on one step in 8 **per stream tag**, an
  unbiased sample of the mean tail at 1/stride of the graph memory. `stride<=1` is the old every-step
  behaviour and still OOMs. The counter is per TAG, so a 3-layer stream is not sampled 3x as often as a
  1-layer one (a global counter gave only a 2x reduction - the trap this avoids). Verified by
  `scratchpad/_test_e1_stride.py` (4 -> 2 of 8, 8 -> 1 of 8, 1 -> 8 of 8) and by the real trainer on
  CPU (`e1_cpu_smoke.py`, `E1SMOKE_LAMBDA` / `E1SMOKE_STRIDE`), which logs
  `[tail-penalty] lambda=1.0 stride=8 card=0.002546(n=7) note=0.011098(n=19)`. Chain 23
  (`it36e1b`) re-tests lambda = 1.0 / 3.0 / 10.0 at stride 8, ungated (three points on one curve), each
  with its own corruption check. **The default path is byte-identical when the lever is unset** - read
  `_TAIL_REG` is read once at import - so editing this module never disturbs a running chain.
- **The 496-step decay is NOT an axis-2 lever.** Rows 8/13/14/27 already show it: a longer decay makes
  the fp32 twin BETTER (row 0b is 0.0121/0.0116 below row 0) while the quantized model improves less,
  so the RELATIVE tax GROWS (+0.000715/+0.000793 for `l2i8` at 496 vs +0.0003-ish at 124). Do not
  queue another 496-step point as a fix for the tax; Andrew's "more than 124 steps" permission is for
  turning a failing recipe into a winning one, not for this.
- **The Rust deploy engine CLI must be started from the REPO ROOT (settled 2026-11-23).** `Model::load`
  (`rust/rwkv-infer/src/main.rs:1076`) reads `RWKV_WEIGHTS` (default `reference/rwkv_ref_558.safetensors`)
  and `ref_dir()` reads `RWKV_TRACE_DIR` (default `reference`) as RELATIVE paths, so a run with
  `cd rust\rwkv-infer` dies inside `Model::load` with `Error: The system cannot find the path specified.
  (os error 3)` before any mode is dispatched. That error was misread for a day as a `--verify-fast` bug:
  `--verify-fast 107` is FINE and prints `max|imm_fast - imm_candle| = 2.980e-7  PASS (<1e-4)` (B=1014)
  when the process starts in the repo root with the same `RWKV_WEIGHTS`/`RWKV_TRACE_DIR` the normal path
  uses (`rwkv-infer.exe 107` -> 2,815.8 rev/s). Diagnostic rule: set `RUST_BACKTRACE=1`; a backtrace whose
  frames name `Model::load` means the WEIGHTS path, not the mode.
- **The shell tool can run the SAME command TWICE, CONCURRENTLY (found 2026-11-23).** Three `git add`/`git
  commit` calls in a row died with `fatal: Unable to create '.git/index.lock': File exists` (and the stale
  lock carried the timestamp of the failing call). A direct probe confirmed the mechanism:
  `Add-Content C:\Temp\_tw.txt ('X'+...)` returned `n=2 :: START|X095542722` from ONE copy while the other copy
  died with `Add-Content : The process cannot access the file ... because it is being used by another process`,
  i.e. two live copies of one command racing; the same duplication created a commit while the second copy
  exited 1 with `no changes added to commit`. Rules that follow: (1) every mutating command must be
  IDEMPOTENT, and a stale `.git/index.lock` with no `git.exe` alive is safe to delete before a re-run (verify
  with `Get-CimInstance Win32_Process`); (2) NEVER append to a file from a shell one-liner - write the whole
  file, or guard the append with an "already present" check (this is why every `_record_chainN.py` carries an
  idempotence guard); (3) verify the result (`git log --oneline -1`, a line count) instead of trusting the exit
  code; (4) a launch line may appear once or twice in `detach_launches.log`, so check the job's own `.lock`
  before assuming a double start. Note: a shell call that FAILS also discards the `console.log` output already
  produced in that same `run()`.
- **The eval's per-head result files are named the other way round (settled 2026-11-23):**
  `result/adw_<tag>.jsonl` is the **AHEAD** head and `result/adwP_<tag>.jsonl` is the **IMMEDIATE** head
  (`quant_sweep.py` line 10 says so in its own docstring). Reading them the intuitive way swaps the two heads
  and fabricates a "0.005-0.09 ahead-head trace-vs-eval gap" out of nothing; with the correct mapping, on the
  40 long-history users of `trace_corrupt50`, the SAME weights and NO quantization, the deploy trace path and
  the evaluation path agree with mean|d| 0.0031 (imm) / 0.0032 (ahead), max 0.014 (see
  `scratchpad/quant_plan/trace_eval_gap.py`, `trace_eval_gap_plain.json`). So the residual is a small
  trace-maker-frame vs eval-dataset INPUT difference, present on BOTH heads - which is why a deploy tax must
  always be quoted WITHIN one path. Related tools: `scratchpad/quant_plan/bucket_tax.py` (the deploy tax split
  by a card's answer count 1-2/3-4/5-8/9-15/16-31/32-127/128+, plus the per-review |delta p| median/P99/max and
  the share of reviews and of cards moving >0.05 / >0.10) and `bucket_mix.py` (a roster's own answer-count mix;
  the 750 roster reads 25.7/19.4/22.9/15.2/10.4/5.9/0.4% of answers per bucket and P50 = 5 answers per card,
  i.e. the same mix the other team reports).
