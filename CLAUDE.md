# rwkv-anki-autoresearch — Claude handover

> **GitHub rule (always):** every GitHub comment posted on Andrew's behalf — PR
> descriptions, review replies, issue comments, anything — **must start with the line
> "Written by Claude".** No exceptions. (Also in the global `~/.claude/CLAUDE.md`.)

## 0. Who you are / read this first

You own **`C:\Users\Andrew\rwkv-anki-autoresearch`** (GitHub:
`Expertium/rwkv-anki-autoresearch`). (The repo started empty; the RWKV code has long since been
vendored in — see §2/§10 for what lives where. Roadmap steps 1–3 are DONE.)

- **Separate Claude instances own the sibling repos** `C:\Users\Andrew\srs-benchmark`
  (the upstream-clone benchmark + the original RWKV code) and
  `C:\Users\Andrew\fsrs-autoresearch` (the FSRS-7 GPU autoresearch). **Do not edit those
  repos** — they are your *read-only source*. One Claude per repo.
- **The user is Andrew** — hobbyist data scientist. He knows **Python/PyTorch and little
  else**, so spell out non-Python tooling (Rust, `candle`, CUDA toolchains, git internals,
  build systems) rather than assuming it. **He did NOT write this neural net** and can't
  answer deep questions about its internals — *the source code in `srs-benchmark/rwkv` is
  the ground truth; read it and be self-reliant.* When a design choice is yours to make,
  explain the trade-off and recommend, don't quiz him.
- Andrew sometimes hand-edits files; if you see an uncommitted change you didn't make,
  it's intentional — don't revert it, commit it if needed.

## 1. The end goal

A small **RWKV-7 neural network for Anki spaced-repetition scheduling** that:
- runs **CPU-only, inference-only**, with **frozen weights** (no per-user training — the
  net generalizes across users from one frozen checkpoint), and
- is **small, fast, and quantized** enough to **ship inside Anki**.

The path there is the roadmap in §8: reproduce → shrink the data loop → port to Rust →
speed up → reduce params → CPU inference → quantize.

## 2. What the model is, and where it currently lives

**"RWKV"** is the current top entry on the
[open-spaced-repetition/srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark)
leaderboard — a multi-stream RWKV-7 (RWKV is a *linear-attention RNN* LLM family; the "7"
is the architecture generation) that predicts recall probability from a user's full review
history. It is materially more accurate than FSRS-7 (numbers in §5).

**Architecture** (`rwkv/architecture.py`, `rwkv/config.py`, `rwkv/model/srs_model.py`):
- **Five parallel RWKV-7 stacks**, one per ID granularity: `card_id`, `deck_id`,
  `note_id`, `preset_id`, `user_id`. Each stack is `d_model = 128` (32 × 4 heads),
  **2–4 layers**, channel-mixer factor 1.5–2.0, with LoRA-style low-rank projections for
  the decay / `a` / `v0`-mix / gate terms. The streams let the net pool evidence at
  different scopes (this exact card vs. the whole deck/note/preset/user).
- **SRS heads on top** (`srs_model.py`): a **forgetting-curve** head = a softmax **mixture
  over 128 basis curves** (`w_head`) sampled at **128 points** (`ahead_head`), plus a
  **4-way per-rating head** (`p_head`, Again/Hard/Good/Easy). So the model outputs a
  flexible learned forgetting curve, not a fixed FSRS-style formula.
- **~2.76 M parameters total** (measured from the pretrained `.pth`). This is the baseline
  for the param-reduction work (step 6).

**Two benchmark scoring modes** (`rwkv/get_result.py`, names `FILE_AHEAD` / `FILE_IMM`):
- **`RWKV` ("ahead")** — predict the next review cold from history.
- **`RWKV-P` ("imm")** — the immediate-prediction variant; **the stronger one.**
  Read `get_result.py` + `srs_model.py::forward` for the *exact* definitions — you must
  understand this distinction to match the baseline and to port correctly.

**Source to vendor** (read-only, in `C:\Users\Andrew\srs-benchmark`):
- `rwkv/` — the whole subtree (model, CUDA/C++ kernel `rwkv/model/csrc/`, training,
  eval, data pipeline). See the file index in §10.
- **Cross-repo imports it depends on** (must come along or be re-implemented): `features/`
  (`create_features`), `utils.get_bin`, `config.py` (`Config`, `create_parser`), and
  `setup.py::get_rwkv_extensions` (builds the kernel — a **`CUDAExtension`** when CUDA is
  present, else a **CPU `CppExtension`** fallback). After vendoring, the package must
  `import` and run standalone — verify before anything else.
- **Pretrained weights:** `srs-benchmark/pretrain/RWKV_trained_on_101_4999.pth` and
  `RWKV_trained_on_5000_10000.pth` (2.76 M params each).

## 3. Data + preprocessing (HEAVY — plan storage first)

- **Dataset:** `C:\Users\Andrew\anki-revlogs-10k` — 10 000 users, ~745 M reviews
  (sibling, **read-only — never write there**). `anki-revlogs-3k` is the first 3 000 users.
  `user_order.jsonl` ranks user ids by size.
- The pipeline writes **LMDB** databases (`train_db`, `test_db`, `label_filter_db`):
  1. `python -m rwkv.find_equalize_test_reviews` — builds a ~7 GB helper db that precomputes
     RMSE(bins) bins and which reviews count in the benchmark (the "equalized" test set,
     aligned with the `--short --secs` feature settings).
  2. `python -m rwkv.data_processing --config rwkv/data_processing_config_{train,test}.toml`
     — the train + test LMDBs.
- **⚠ The full 10k preprocess needs ~400 GB of disk.** This is the binding constraint on
  Andrew's machine (§7). Working on the **2k subset (step 2) is the cheap iteration loop**
  for everything downstream — do that early.

## 4. Training

`rwkv/train_rwkv.py` + `rwkv/train_rwkv_config.toml`:
- **WSD LR scheduler** (Warmup–Stable–Decay): ~10 epochs warmup+stable, then ~2 epochs
  decay; switch phases via the config (`TRAIN_MODE = "WS"` / `"D"`).
- **bfloat16, CUDA**, peak LR `7e-4`, 20 000 warmup steps, `MAX_TRAIN_GLOBAL_LEN = 66000`.
  Requires the compiled CUDA kernel. CPU training is supported (`DEVICE = "cpu"`) but
  "dramatically slower" — not practical for full runs.
- The provided weights were trained on users **5000–10000** and **101–4999**, with
  **1–100 held out as validation**.

## 5. Evaluation + the baseline to reproduce

- `rwkv/get_result.py` (CUDA, fast) scores a trained model on a held-out user range.
  **Cross-validation:** model trained on 5000–10000 → evaluate users 1–4999; model trained
  on 101–4999 → evaluate 5000–10000; pool both → 10 000 users.
- **Metric = by-user mean `LogLoss`** (each user weighted equally), plus `RMSE(bins)` —
  same definitions as srs-benchmark.
- **★ BASELINE NUMBERS TO MATCH** (from `srs-benchmark/result_upstream/`, 10 000 users):

  | Variant | mean LogLoss | mean RMSE(bins) |
  |---|---|---|
  | `RWKV` (ahead) | **0.29743** | 0.05438 |
  | `RWKV-P` (imm) | **0.26600** | 0.03212 |

  (For scale: FSRS-7 is ~0.32 by-user on the 3k subset — the RWKV nets are clearly better.)
- **"Reproduce" = match the mean LogLoss within a small tolerance**, not bit-for-bit.
  Cross-version SGD + bf16 nondeterminism makes exact reproduction unrealistic; aim for
  ~matching aggregate LogLoss/RMSE(bins). (This is unlike the FSRS bit-exact speedup
  project — here parity is statistical.)
- **Plan of attack (confirmed with Andrew):** first **reproduce the table from the
  provided `.pth` weights** via `get_result.py` — no training. Once it matches, **move all
  further work to the 2k loop** (roadmap step 2: train ids 1–1000 → eval 1001–2000, then
  swap for full 2 000-user coverage) and don't routinely touch the 400 GB 10k pipeline again.

**Acceptance tolerances** — apply to **both** scoring modes (`ahead` *and* `P`)
independently; a change passes only if **both** pass:
- **Parity (Rust port, roadmap step 3):** each mode's mean LogLoss must be **within ±0.0005**
  of the Python reference. This is the gate for "the Rust port is correct."
- **Efficiency-regression budget (roadmap steps 4, 5, 7 — speedups, param reduction,
  quantization):** each mode's mean LogLoss may **rise by at most +0.0015** relative to its
  **parity-verified Rust baseline**. Within budget → keep; over → reject. (A pure speedup
  should cost ≈0; the +0.0015 is headroom for param-cutting and quantization to spend.)

## 6. CPU inference — already half-built (central to the goal)

RWKV-7 has an **exact RNN (sequential/recurrent) formulation** mathematically equivalent
to the parallel CUDA training kernel. It's already implemented:
- `rwkv/model/rwkv_rnn_model.py` (`RWKV7RNN`) + `rwkv/model/srs_model_rnn.py` +
  `rwkv/run_as_rnn.py` already run **a single user on CPU** from the saved weights
  (`run_as_rnn_config.toml`: `DEVICE = "cpu"`, `DTYPE = "float"`).
- This RNN-mode path is the **starting point for steps 6–8** (Rust port, CPU inference,
  quantization). Inference one-token-at-a-time needs no custom CUDA — pure tensor ops,
  ideal for `candle`/Anki. (The `CppExtension` CPU kernel build is a fallback if you need
  the chunked form on CPU, but RNN-mode is likely enough for inference.)

## 7. Host machine + build caveats

- **Andrew's PC:** RTX 4070 (**12 GB VRAM** — less than the 24 GB 3090 this code was
  developed on; the model is tiny so bf16 training should fit, but you may need to lower
  `MAX_TRAIN_GLOBAL_LEN`), Ryzen 9 5950X (16c/32t), 64 GB RAM, 1 TB M.2 SSD + 4 TB external
  USB HDD. **The ~400 GB preprocessed dataset is the storage constraint** — put the LMDBs
  where there's room (M.2 if it fits; otherwise the 4 TB USB, which is slower I/O).
- **CUDA build: RESOLVED long ago** (torch cu130 wheel + VS2022 vcvars64; the kernel builds and is
  the live production `.pyd`). Rebuild via `scratchpad/run_build_k16.cmd` — fails only if a process
  holds `RWKV_CUDA.*.pyd` (use `setup.py build_ext` WITHOUT `--inplace` for an isolated build then).
- **Native Python** here (no Docker, unlike fsrs-autoresearch). Later, **Rust** (step 3).
  Run from PowerShell. Use a venv; install torch matching your CUDA situation.

## 8. The roadmap (Andrew's plan)

1. **Reproduce existing results on 10k.** Train RWKV on the first 5 000 users (ids 1–5000),
   evaluate on the second 5 000 (5001–10000); then **swap** train/test and repeat. Match
   the §5 baseline. **Start by reproducing from the *provided* weights** (no training);
   a fresh exact-split train is optional after that.
2. **Move to a 2k loop** — train ids **1–1000 → evaluate 1001–2000**, then **swap**
   (train 1001–2000 → eval 1–1000) for full 2 000-user coverage. This is your **fast
   iteration workbench** for everything below; build it right after step 1 and don't
   routinely touch the 400 GB 10k pipeline again.
3. **Implement RWKV in Rust** (likely [`candle`](https://github.com/huggingface/candle),
   HF's minimalist Rust tensor/ML library). Port the **RNN-mode recurrence** (§6) — no
   custom CUDA needed. **Verify parity** with the Python implementation: both modes'
   LogLoss within **±0.0005** of Python (the §5 parity gate). This Rust engine is what
   ultimately runs inside Anki.
4. **Speed it up WITHOUT changing architecture/training** — pure-performance wins (op
   fusion, killing redundant recompute, better batching/memory layout, cutting allocation
   churn). A pure speedup should keep both LogLosses ≈unchanged; stay within the **+0.0015**
   regression budget (§5) vs the Rust baseline.
5. **Reduce the parameter count** while keeping LogLoss within the **+0.0015** budget (and
   ideally **improving** it) — via hyperparameter tuning, architecture search, pruning, or
   distillation. Baseline = 2.76 M params @ 0.266 (RWKV-P). The dream is an algorithmic
   change that *lowers* LogLoss while shrinking. The most "research-y" step — measure every
   change on the 2k loop, keep the wins.
6. **CPU-only, inference-only** (training stays on GPU). The RNN-mode path (§6) is the
   start. End state: usable inside Anki with **frozen weights**.
7. **Quantize.** Read the two papers and pick an approach (adapt — both target 14B-scale
   RWKV; ours is 2.76 M), keeping both LogLosses within the **+0.0015** budget:
   - **RWKV-edge** — *Deeply Compressed RWKV for Resource-Constrained Devices*
     ([arXiv 2412.10856](https://arxiv.org/abs/2412.10856)): a compression **suite**
     (architecture optimizations + post-training compression), **3.4–5× memory reduction**,
     edge-device focus.
   - **RWKVQuant** — *Quantizing the RWKV Family with Proxy-Guided Hybrid of Scalar and
     Vector Quantization* ([arXiv 2505.03803](https://arxiv.org/abs/2505.03803)): PTQ built
     for RWKV's quirks (non-linear ops that block fusion; near-uniform weights that hurt
     clustering) — a **proxy-guided hybrid of scalar + vector quantization** with codebook
     optimization, **~3-bit, <1% accuracy loss, 2.14× speedup**.
   Quantized weights pay off in the Rust/candle CPU path for Anki.

Steps 4, 5, and 7 are naturally **iterative** (propose a change → measure LogLoss + speed +
size on the 2k loop → keep it only if it passes the §5 tolerances) — i.e. a lightweight
autoresearch loop, hence the repo name. Keep an append-only log of what you tried and the
deltas so dead ends aren't re-run.

## 9. Working norms

- **Be self-reliant on RWKV internals** — Andrew didn't write the net. The source in
  `srs-benchmark/rwkv` is ground truth; verify facts against it, not memory.
- **Parity discipline:** verify against the reference on a **small fixed verification
  user-set** defined early (mirror srs-benchmark's `test_users.json`: a few small + a few
  large + some random, seeded) so checks are fast and comparable. The numeric gates live in
  §5: **±0.0005** for Rust-port parity (step 3), **+0.0015** regression budget for
  efficiency changes (steps 4/5/7) — and **both** scoring modes (`ahead` and `P`) must pass.
- **Git:** commit/push only when asked; for non-trivial pushes branch off `main`; end commit
  messages with the `Co-Authored-By` trailer. GitHub comments start "Written by Claude".
- When a step is ambiguous (exact split, quant target, candle vs other Rust ML lib), state
  the trade-off and your recommendation rather than guessing silently.

## 10. Key files (in `srs-benchmark/rwkv`, to vendor)

| Path | What |
|---|---|
| `architecture.py` | the 5-stream RWKV-7 config (d_model, layers, LoRA dims per ID module) |
| `config.py` | ID-encoding dims, time-feature periods, `RWKV_SUBMODULES` |
| `model/srs_model.py` | the SRS model (training mode): feature FC + the 5 RWKV stacks + curve/rating heads |
| `model/srs_model_rnn.py` | the SRS model in **RNN (sequential) mode** — CPU inference |
| `model/rwkv_model.py` | core `RWKV7` (parallel/CUDA training form) |
| `model/rwkv_rnn_model.py` | core `RWKV7RNN` (recurrent form) |
| `model/rwkv_ops.py` | kernel wrapper + a pure-PyTorch `reference_rwkv7` |
| `model/csrc/**` | the CUDA/C++ kernel (`rwkv7_cuda.cu`, `parallel_scan.cu`, `rwkv7.cpp`) |
| `train_rwkv.py` / `train_rwkv_config.toml` | training entry + config (WSD scheduler) |
| `get_result.py` / `get_result_config.toml` | evaluation (CUDA) — produces the `RWKV` / `RWKV-P` jsonls |
| `run_as_rnn.py` / `run_as_rnn_config.toml` | **single-user CPU inference** (RNN mode) |
| `data_processing.py`, `prepare_batch.py`, `data_fetcher.py` | dataset → LMDB → batches |
| `find_equalize_test_reviews.py` | builds the helper db (test-review alignment + RMSE bins) |
| `parse_toml.py`, `utils.py` | config + small helpers |
| *(parent)* `features/`, `utils.get_bin`, `config.py`, `setup.py` | shared deps to vendor |

## 10b. THIS repo's layout — the file map

> **⚠ KEEP THIS MAP CURRENT:** whenever files/folders are added, renamed, or deleted (and at
> every housekeeping pass), check this section and update it. Stale maps are worse than none.

- **Root:** `CLAUDE.md` (this handover) · `README.md` · `INPUT_FEATURES.md` (the 92-dim input
  table; future timestamp-features plan → `optimization/FUTURE_FEATURES.md`) · `.gitignore` ·
  `requirements.txt` · `setup.py` (CUDA/C++ kernel build) · `config.py` + `utils.py` +
  `features/` (vendored cross-repo deps — needed for imports) · `build_dataset.py` ·
  `test_users.json` · `verify_rust.py` (Rust-parity gate) + `export_rnn_trace.py` +
  `make_reference.py` (its trace/reference companions).
- **`rwkv/`** — the vendored+evolved package: `architecture.py` (5-stream config + env hooks +
  RWKV_ARCH_MODULE), `config.py`, `train_rwkv.py`, `get_result.py` (eval), `data_processing.py` /
  `prepare_batch.py` / `data_fetcher.py` / `find_equalize_test_reviews.py` (data pipeline),
  `run_as_rnn.py` (CPU RNN mode), `parse_toml.py`, `utils.py`; `model/` = `srs_model.py`,
  `srs_model_rnn.py`, `rwkv_model.py`, `rwkv_rnn_model.py`, `rwkv_ops.py`, `csrc/` (CUDA kernel;
  the built `RWKV_CUDA.pyd` is untracked). Live tomls only (the ~120 closed-era iterN run
  configs were git-rm'd 2026-07-15; git history keeps them).
- **`optimization/`** — tooling + the canonical record. Record: `research_5k.md` (front tables,
  4dp) · `research_5k_notes.md` (methodology) · `research_5k_verbose.md` (per-iter detail,
  AI-only) · `research_log.jsonl` (5k source of truth) · `log.md`/`log.jsonl` (regenerated
  canonical table — `python optimization/logbook.py rebuild`) · `research_log.md` (CLOSED
  100/100-era log) · `HISTORY.md` (superseded plans + archived CLAUDE.md live-state) ·
  `FUTURE_FEATURES.md` · `LIT_REVIEW.md` · `PROTOCOL.md` (iter0-era mirror of §11) ·
  `STATEFUL_BPTT_PLAN.md` (shelved). Champions: `champion_5k.json` (QAT deploy truth, FROZEN) ·
  `champion_5k_plain.json` (track-1 plain) · `champion_5k_track2.json` (A0 anchor) ·
  `champion_5k_history.jsonl`. Tools: `logbook.py`, `gate.py`, `paired_pvalue.py`,
  `promote_champion_5k.py`, `eval_sharded.py`, `hp_tuner_5k.py` (+ old `hp_tuner.py`),
  `model_stats.py`, `measure_throughput.py`, `wilcoxon_speed.py`, `count_groups_5k.py`,
  `entropy_floor.py`, `quant_ptq.py`, `soup.py`. Journals: `tuner_5k_log.jsonl`
  (+ `_2ep_era`/`_proxyera` archives), `tuner_log.jsonl`, `baseline_log.jsonl`, `qat_log.jsonl`,
  `quant_log.jsonl`, `cpu_speed_log.md`. `arch_snapshots/` = per-iter architecture.py snapshots
  (100/100 era).
- **`reference/`** — deploy + parity artifacts: `pq_cb_{wkv,shift}_q72u.txt` (the q72u deploy
  codebooks), `pq_cb_m2b8.txt`, `ref_metrics.json`, `weight_names.json`, `rpv_*.json`
  (Rust-parity vectors); `.safetensors` untracked by design.
- **`rust/rwkv-infer/`** — the Rust CPU inference engine (`src/{main,model,fast}.rs`,
  `BATCHING_PLAN.md`); K-dynamic + full PQ/joint-cb/norm-quant engine since `1d3b5b8`.
- **`scratchpad/`** — per-run pipelines + shared helpers. Tracked per run: `.cmd` + tomls +
  `*_ws_trace.jsonl` (+ champions' final cbs `cb_{wkv,shift}_final.txt`). Shared:
  `write_decay_setup.py`, `write_eval_toml.py`, `detach.ps1`, `liveplot/`,
  `architecture_old_d128.py`. Untracked on disk: ckpts (`*.pth`), logs, mid-run cb snapshots
  (gitignored since 2026-07-15). ⚠ Champion ckpts live here UNTRACKED (the champion jsons point
  at them) — single-machine artifacts; losing the disk loses the ckpts, not the record.
- **`result/`** — eval outputs, untracked (`RWKV-<tag>.jsonl`, `RWKV-P-<tag>.jsonl`,
  `*.nanskip.jsonl`).

## 11. Optimization loop (steps 4–5–7) — THE PROTOCOL (canonical; mirror in `optimization/PROTOCOL.md`)

> **⚠ SUPERSEDED GATE:** the work is now in the **research phase** — the live acceptance gate
> (both modes improve ≥0.0003 vs the current champion, params ≤225k, card/note state fixed) is in
> the **"Optimization state"** section below, NOT the iter0 +0.0015 gate described here. The rest of
> this section (logging discipline, the Wilcoxon speed protocol, Rust-parity invariant, the
> training-resume mechanism) is still current. Keep it for those; use the research gate for accept/reject.

Steps 4 (speed), 5 (param reduction), 7 (quantize) run as ONE iterative autoresearch loop.
Follow this exactly — Andrew has flagged sloppiness, so do every step every iteration.

**Scope / allowed changes:** both **exact** (float-noise) and **inexact** (accuracy-affecting)
changes — training, hyperparameters, AND architecture. Biggest wins first, but per Andrew
(2026-06-27): **bank cheap size/speed wins that barely move LogLoss first; don't push the
champion close to the +0.0015 threshold early** (the champion's distance from the threshold is
the remaining budget for ALL future iterations — burning it early starves them).

**Two hard INVARIANTS (never change):** (1) hierarchy `card→note→deck→preset→global` (5 chained
streams in that order); (2) inputs — the model must still run on the *same preprocessed 92-dim
data* / existing LMDBs. No new/changed inputs.

**The 5 gates — a change is KEPT only if ALL pass:**
1. **LogLoss (both modes):** ahead AND imm by-user-mean LogLoss not worse than **iteration 0**
   by >**+0.0015**. (A pure/exact change ≈0; a real rise is a red flag, not budget to spend.)
2. **Review count ("size"):** per-user equalized review count IDENTICAL to iter0 (it's a
   property of the data+filters; any change = a pipeline bug).
3. **State size:** per-card RNN state (card_id stream) **≤ iter0** (13,056 floats / 51.0 KiB).
4. **Hierarchy** preserved. 5. **Inputs** unchanged.
GPU training speed is **untimed** (prefer it not balloon, but it doesn't gate).

**Eval recipe (FIXED every iteration):** train users **1–100**, eval **101–200** (all 100),
bf16 CUDA `python -m rwkv.get_result --config rwkv/get_result_config_iterN.toml` → by-user mean
of `result/RWKV-iterN.jsonl` (ahead) + `RWKV-P-iterN.jsonl` (imm). Training recipe = **WSD**:
WS 18 epochs (558 steps, `train_rwkv_config_iterN.toml`) then **D** 2-epoch cosine decay
(`..._iterN_decay.toml`, loads the WS-final ckpt) — the decay phase matters (it's what landed
the iter3 champion). **Rust-parity invariant:** `verify_rust.py` (3-user float32) must pass for
the champion arch before "shipping" (re-export trace + match the trained model bit-exactly).
**RUN IT WITH `RWKV_WEIGHTS=reference/rwkv_iter36_124.safetensors`** -- the trace_user py_pred is the
iter36 champion's; verify_rust's DEFAULT (rwkv_ref_558) and other models (iter45 etc.) will MISMATCH
(that is wrong-weights, not a regression). Confirmed 2026-06-29 bit-exact: dpred ~3e-7, |rust-python| 0.000000.

**Speed = batch throughput via simultaneous paired Wilcoxon (protocol point 7–8):**
- **Lock CPU freq** (admin, once/session): `powercfg -attributes SUB_PROCESSOR
  75b0ae3f-bce0-45a7-8c89-c9611c25e100 -ATTRIB_HIDE` ; `powercfg /setacvalueindex SCHEME_CURRENT
  SUB_PROCESSOR PROCFREQMAX 3400` ; `... PROCTHROTTLEMIN 100` ; `... PROCTHROTTLEMAX 100` ;
  `powercfg /setactive SCHEME_CURRENT`. (`PROCFREQMIN` is not a valid alias — pin the perf
  state instead. Restore: `PROCFREQMAX 0`, `PROCTHROTTLEMIN 5`.)
- **One trial** = run *before* (champion) and *after* (candidate) **simultaneously**, each
  pinned to **3 threads**, each looping the **same frozen pre-chosen batch set** for a fixed
  wall-clock **T≈20–30 s**; count reviews each finishes → one paired point. Pairing the *trial*
  (not the batch) keeps pairs independent + cancels external load + avoids tail bias.
- Repeat **20 trials** (drop 1–2 warm-ups); accept the speedup only if **one-sided Wilcoxon
  signed-rank p < 0.01**. (**Andrew 2026-06-28: use 20 trials, not ~10** — `wilcoxon_speed.py`
  default is now `--trials 20`.) (Power: n all-same-sign pairs → p≈2⁻ⁿ, so 20 consistent trials
  clear p<0.01 with wide margin.) Batch throughput = stepping many *independent* card-streams in
  parallel (per-card is inherently sequential); batching is an exact, free speedup. Build via the
  config-driven Rust bench + a Python Wilcoxon driver.

**Logging — DO NOT BE SLOPPY (Andrew flagged this twice):** `optimization/logbook.py` appends to
`log.jsonl` and regenerates `log.md` (table excludes `comment`). EVERY iteration gets ALL fields:
`number, timestamp, logloss{ahead,imm}, params, state_kib, throughput, wilcoxon_p,
review_count_check, logloss_tolerance_check, state_size_check, summary(≤15 words, BEFORE),
comment(after; jsonl only)`.
- **Throughput (rev/s) is MANDATORY for every ACCEPTED iteration** — measure it then and there
  (`python optimization/measure_throughput.py <ckpt.pth>`); rejected → `n/a`. Never "pending".
- **`wilcoxon_p` is MANDATORY for every ACCEPTED iteration** — run the paired Wilcoxon trial
  (champion-vs-candidate) and record p; rejected → `n/a`.
- Plain ASCII in shell-written values (an em-dash mojibakes). Log dead ends with a why-comment.

**Tooling (`optimization/`):** `model_stats.py` (params + per-card state), `gate.py` (computes
the gates + appends a record; `--no-write` to dry-run), `logbook.py`, `measure_throughput.py`,
`PROTOCOL.md`. Use `.venv/Scripts/python.exe`, `OMP_NUM_THREADS=7`.

**Training survives the ~5-min session teardowns** (which kill bg/detached jobs) via
**foreground + resume-from-checkpoint**: ckpts every 100 steps; resume by copying
`{prefix}_optim_{step}.pth` → `{prefix}_{step}_optim.pth` and setting LOAD_MODEL /
LOAD_MODEL_NAME=`{prefix}_{step}` / STEP_OFFSET=step+1.

## Optimization state (5k phase: train 1-5000 / eval 5001-10000)

> Numeric record = `optimization/log.md` (the CANONICAL regenerated table -- now has a Research-phase
> section fed by `research_log.jsonl`; rebuild via `python optimization/logbook.py rebuild`) + the source
> jsonls (`research_log.jsonl`, `baseline_log.jsonl`, `log.jsonl`, `quant_log.jsonl`, `qat_log.jsonl`).
> 5k-phase verbose per-iteration notes = `research_5k_verbose.md` (AI-only; research_5k.md's summary
> column is capped at <=20 words, Andrew 2026-07-13; `research_log.md` = the CLOSED 100/100-era log);
> `HISTORY.md` = superseded plans + the full pre-2026-06-30 snapshot. **Log EVERY research experiment
> to `research_log.jsonl` + research_5k.md row + research_5k_verbose.md section + rebuild log.md.**
> This section keeps ONLY the current champion, deploy config, acceptance gate, lesson bank, live state, ops.
> **Every research_log.jsonl entry + research_5k.md row records `nan_users` / "NaN users"** (eval users
> skipped by the NaN guard; Andrew 2026-07-16) — backfilled for all prior iters (all 0 except iter19=1, A0=7).

### Workbench + baselines
- **5k phase (CURRENT):** train 1-5000, eval 5001-10000, budget 2 WS ep + tuned-ratio decay,
  MAX_TRAIN_GLOBAL_LEN=110000 (swept), quant-aware logloss. Baseline-to-beat = the old d=128 model
  (`pretrain/RWKV_trained_on_101_4999.pth`, unquantized) eval'd on 5001-10000 (PENDING, needs eval data).
  Front table `optimization/research_5k.md`; full methodology + status `optimization/research_5k_notes.md`.
- **Run env (all phases):** **augmentation OFF** (RWKV_AUGMENT_SEED=1234) + RWKV_DETERMINISTIC=1 +
  RWKV_EMPTY_CACHE_EVERY=0 -> run-to-run variance ~0. Eval `python -m rwkv.get_result` (CUDA, JIT-on ->
  REQUIRES the `@torch.jit.ignore` fix on `quant_aware_rwkv7`).
- **Historical 100/100 + 1500u workbench refs** (eval 101-200, MAX=66000, sc8k dbs): champion recipe was
  "1 ep on 1500 users (1000-2499) + decay" (data variety >> repetition; ~25 min/experiment -- still useful
  for cheap sanity checks). d=128-on-1-100 baseline = 0.320295/0.281913 (arch-swap
  `scratchpad/architecture_old_d128.py`); iteration-0 floor = 0.374046/0.319475.

### CHAMPION = H=2/K=16 on the 1500-user data-variety recipe  (d=32, 2 heads x K=16; 193,724 params)
- arch `[1,4,3,3,3]` (card,deck,note,preset,user), d_model=32 split as **2 heads x 16 (K=16)** via the NEW
  K<32 CUDA kernel -- this HALVES the per-card WKV state (1088->576 floats; model_stats confirmed) at ~same
  params, ~half the WKV-kernel work, and **~1.16x faster GPU training (WS 1.182 vs 1.020 steps/s)**. Trained on
  users 1000-2499 (`train_db_sc8k_1500`), 1 epoch WS (3351 steps) + 0.27-epoch cosine decay (904 steps). ckpt
  `scratchpad/exp_h2k16/h2k16d_904.pth`; weights `reference/champ_h2k16.safetensors`. Recipe env = RWKV_N_HEADS=2
  RWKV_HEAD_DIM=16 + HP {peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25} + RWKV_EMPTY_CACHE_EVERY=0.
- **fp32: ahead 0.309723 / imm 0.276566** (eval 101-200) -- accuracy PARITY with the prior H=1 champion
  (champ_1500d 0.309706/0.276357; both modes within 0.0002, far inside the +0.0015 efficiency budget), and BEATS
  the d=128 baseline by +0.0106 ahead / +0.0053 imm. Accepted as a **SIZE/SPEED win** (state halved + faster),
  NOT on the +0.0003 monotonic gate. HPs are re-tuned as part of the 5k phase (methodology d), not on 1500u.
- **★ KEY FINDINGS:** (1) DATA VARIETY beats repetition -- "1 epoch on ~1500 varied users" >> "15 epochs on
  100 users" (drove the prior champion jump; the d=32 model is DATA-limited, so the path forward is MORE DATA,
  scale toward 5k). (2) K<32 UNBLOCKED -- the WKV kernel is now K-dynamic (any K dividing 32), so H=2/K=16 gives
  the 2x-smaller-state + faster model that makes 5k-user training practical. PRIOR champions kept as refs:
  champ_1500d (H=1/K=32, 0.309706/0.276357), decay15 (100u, 0.314807/0.280200).
- **DEPLOY config (the sibling's FINAL locked recipe `q72u`, research CLOSED 2026-07-07; results ported
  here 2026-07-08) [[champion-logloss-deployed]]: 72 b/layer = 9-BYTE CARD, 27 B note, 256x compression.**
  Format per layer: m2b12L learnable shift catalog (2 chunks x 4096 entries, 48 b) + JOINT-UV b10 WKV
  catalog (per head ONE 10-bit code into a 1024-entry concat(u,v) 32-dim catalog, 20 b) + 1-bit norms (4 b).
  VAL penalty vs fp32 **+0.00114/+0.00021 (seed 1234) and +0.00115/+0.00040 (seed 4321)** — 2/2 seeds pass
  with margin; best-ever robustness (imm nbad 96-98/400); imm is ~seed-noise-FREE under joint coding.
  **Artifacts (ported to our `reference/`):** `qat_pq_q72u.safetensors` + `pq_cb_wkv_q72u.txt` +
  `pq_cb_shift_q72u.txt`. **Deploy env (Rust):** `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4
  RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q72u.txt
  RWKV_SHIFT_PQ=reference/pq_cb_shift_q72u.txt RWKV_PQ_NORM_BITS=1`. **QAT recipe:** warm-start champion,
  2.0-ep plain QAT (no rotation/anneal/KD), BOTH cbs learnable (`RWKV_QAT_PQ_LEARN=1
  RWKV_QAT_SHIFT_PQ_LEARN=1`), `RWKV_QAT_NORM_BITS=1 RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3`, NO_JIT.
  **The full engine (joint cb + warm search + norm quant) IS in OUR `rust/rwkv-infer` since `1d3b5b8`**
  (byte-identical champion eval verified from the parent build). Full detail: sibling
  `research_log_h2k16.md` + explainer `how_state_compression_works.md`.

### ACCEPTANCE GATE (research phase) -- accept iff ALL hold (record binary accepted/rejected per iter):
1. "size" (equalized review count, 101-200) IDENTICAL to champion (data-integrity; any change = pipeline bug).
2. params <= **225,000**.   3. card AND note per-entity state UNCHANGED (deck/preset/global MAY grow freely).
4./5. **(Andrew 2026-07-19 ~21:00, LOOSENED from >=0.0003): each mode's improvement vs the
   CURRENT champion, ROUNDED TO 4 DECIMALS, must be >= 0.0001 — i.e. raw delta >= 0.00005 —
   in BOTH modes** (so +0.000088 rounds to 0.0001 = PASS). First applied to iter 26.
6. **p-gate (Andrew 2026-07-08):** paired per-user one-sided Wilcoxon (candidate vs champion, same 5000
   eval users) gives **p < 0.0001 in BOTH modes** -- `python optimization/paired_pvalue.py` (zero GPU cost,
   reads the result jsonls; exit 0 = pass). Record both p-values in research_5k.md's `p-value` column.
   Applies to accuracy accepts only (SIZE/SPEED-exception accepts claim parity, not improvement -> exempt).
=> accept ONLY changes that improve BOTH modes (>=0.0001 after 4-dp rounding, 2026-07-19; was
>=0.0003) AND pass the p-gate (a monotonic champion).
[[research-acceptance-gate]]
**EXCEPTION -- SIZE/SPEED changes** (e.g. H=2/K=16): judged on the **efficiency budget** instead -- accept if
both modes stay within **+0.0015** of the champion AND the change shrinks state and/or speeds training (it
Pareto-dominates at accuracy-parity). H=2/K=16 was accepted this way (halved card state, 1.16x faster, accuracy
within 0.0002). Such a change MAY shrink card/note state (gate #3 is for accuracy-research iters, not these).
Two HARD INVARIANTS (never change): hierarchy card->note->deck->preset->global; same preprocessed 92-dim
inputs / existing LMDBs (no new/changed inputs).
**RESEARCH-PHASE CONDUCT (Andrew 2026-07-10) -- for the phase after HP tuning + the deck/preset/global
state-size ladders:** (1) try LOTS of different tweaks of both the ARCHITECTURE and the TRAINING
PIPELINE, from different FAMILIES of ideas (not many variants of one); (2) if an idea BARELY misses the
logloss threshold, don't give up early -- try a slightly different implementation of the same idea first;
(3) MIX literature review (optimization/LIT_REVIEW.md) with self-generated ideas; (4) spend AT LEAST 50
iterations (NOT counting HP-tuning trials) before even considering declaring "nothing left to improve";
(5) (Andrew 2026-07-13) NEVER declare a FAMILY "closed" after one iteration -- writing off a family
needs at least 3-5 distinct in-family variants; 1-2 rejects = "0/N so far, deprioritized", not closed.
[[research-phase-conduct]]
**5k-PHASE METHODOLOGY (Andrew 2026-07-01) -- full text in `optimization/research_5k_notes.md`:** the 5k
research phase (train 1-5000 / eval 5001-10000; old d=128 model eval'd on 5001-10000 as the target) keeps
the same >=0.0003-BOTH-modes gate + params <=225,000, and ADDS: (a) **LogLoss recorded WITH (fake)
card- AND note-state quantization** -- beat the old fp big model *while* quantized. Env UPDATED 2026-07-08
to the final q72u recipe (fixed champion codebooks, no cb-learning -- that upgrade needs per-run
cb-export->eval wiring, queued): `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt
RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3 RWKV_QAT_NORM_BITS=1 RWKV_QAT_FUSED=1 RWKV_NO_JIT=1` (JIT on the
grafted q72u paths unverified -- A/B once at champion-run launch); (b) card+note state sizes FIXED, but deck/preset MAY grow
~5-10x and global up to ~100x; (c) WS FIXED at **1 epoch** (2->1 Andrew 2026-07-09 via the champ5k_b1
budget A/B: 2nd epoch adds nothing -- ahead -0.00006 p=0.31, imm +0.00043 BETTER p=6e-62), decay = WS x
ratio, ratio in [1/10, 1/2.5] (ALSO quant-aware), decay_ratio is an `hp_tuner_5k.py` lever; (d) HP-tune FIRST,
then re-tune after accumulated small changes OR a major one; (e) every change must be Rust/CPU-deployable
in Anki -- no GPU-only tricks in the shipped model; (f) BEFORE HP tuning, sweep MAX_TRAIN_GLOBAL_LEN (the
WKV batch dim) over ~100 steps each and fix the largest batch that ALMOST maxes the 12 GB VRAM (fastest
training; batch size is structural so LR/warmup tune after it; don't go below 66000 = data drops) --
**DONE 2026-07-02: MAX=110000** (peak 38,968 rev/s @ 9.44 GB; 132k thrashes, -25%); (g) **Wilcoxon
early-pruning (2026-07-02):** run order = old-model eval -> ONE champion-HP run logging per-step WS train
logloss (RWKV_STEP_TRACE; NOT decay) -> HP tune; candidates then check one-sided Wilcoxon (candidate vs
champion, paired by step, **last-1500-paired-steps window** -- RWKV_PRUNE_WINDOW, 0=old full window;
changed 2026-07-08 after the 0p0014 audit: full-window drags stale early history -> ~2k-step lag on late
regressions AND would kill late-bloomer configs) every 300 steps and ABORT iff BOTH modes worse at p<1e-4
at TWO CONSECUTIVE checkpoints (RWKV_PRUNE_PERSIST=2, added 2026-07-09: the identical-config null control
champ5k_r1-ep1-vs-b1 showed autocorrelated drift transients hit imm p~1e-15 under the NULL -- single-mode
p is overconfident; the persist rule guards the joint test. No false fire in the control itself.)
⚠ SCOPE (2026-07-09 decay_ratio_0p1 FALSE-KILL audit): prune ONLY candidates at MATCHED regularization
vs the reference -- train-loss pruning is sign-biased against regularization levers (wd=0.1 ran train-hot
vs the wd=0.01 champion trace yet WON eval both modes; its WS-identical twin got killed at imm p=3e-45 --
drift scales with config, no fixed alpha calibrates across bases). HP-TUNER trials therefore run WITHOUT
train-loss pruning; they use the REPLACEMENT **VALIDATION-based prune** (Andrew 2026-07-09): validate
every 500 steps, die iff BOTH modes' val loss >= champion's val at the same step + per-mode delta
(RWKV_VPRUNE_DELTA_AHEAD=0.004 / _IMM=0.006) at 2 consecutive val checkpoints from step 1000
(RWKV_VPRUNE_MIN_STEP/PERSIST). EARLY window by necessity (Andrew's flat-curve catch: val curves are
~flat past 2500 -- ahead range only 0.004 -- so late thresholds catch nothing; at 1000-2000 curves drop
~0.01/1000 steps and disasters gap +0.004-0.011 vs twin-null <=0.0025/0.0029). Sign-correct for
regularization, magnitude-based; late-emerging regressions intentionally run to an honest eval.
RWKV_VPRUNE_REF=champion_5k.json (carries val_step/val_ahead/val_imm; promote_champion_5k --val-trace
embeds them; train_rwkv writes <trace>.val.jsonl when STEP_TRACE is on).
(exit 42 + .pruned.json with estimated finals = champ_final + mean(diff over last 300 paired steps) ->
front-table `logloss` column says exact|estimated). Champion accept = `python optimization/
promote_champion_5k.py` (auto-replaces optimization/champion_5k.json = the prune ref; never hand-edit).
Pairing needs identical db/MAX/seeds.
[[research-acceptance-gate]]

### LESSON BANK -- do NOT re-run these (full numbers in log.md / HISTORY.md)
- **TUNE-EVAL SUBSET OVERFIT (2026-07-12, champ5k_t1):** the 200-user tune-eval (5001-5200) is for
  COARSE ranking only -- a +0.0008/+0.0010 subset win (in-subset paired imm p=5e-8!) INVERTED to
  -0.0005/-0.0007 at n=5000. Sub-0.001 effects measured on 200 users do NOT transfer; confirm on the
  full eval before adopting. Champion HPs (wd 0.01, dropout 1.0, beta2 0.999, cb_lr 1x, peak_lr 1e-3,
  warmup 200, clip 0.25, decay_ratio 0.25) are CONFIRMED at 5k -- don't re-tune without new structure.
  **REMEDY ADOPTED (Andrew 2026-07-12): future HP tuning uses a 1000-user tune-eval (5001-6000)** --
  SE ~sqrt(5)x smaller, resolves ~0.001 effects. Wired: hp_tuner_5k EVAL range + trial template now
  passes the range explicitly; write_eval_toml default 5200->6000. When tuning reopens: re-record the
  tuner baseline on 5001-6000 FIRST (old journal rows are 200-user, not comparable); sub-0.001
  verdicts still need full-eval confirmation.
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
  the no-JIT body. `torch.compile` WORKS on Windows now (STALE-CLAIM FIX 2026-07-03: triton-windows 3.7.1
  is in the venv; smoke test compiles + runs bit-correct) -- but it was 0.79x on a mixer-like chain at our
  tiny C=32 sizes (per-call overhead; 4070 too few SMs for max_autotune_gemm), so it must EARN its way in
  via a real profile A/B, and needs RWKV_NO_JIT (Dynamo can't trace ScriptModules). Gate parallelism
  (run_qat_eval.sh NPROC) made the Rust gate ~8x faster.
- **DONE (was BLOCKED): K<32** -- the WKV kernel is now K-DYNAMIC (any K dividing 32; byte-identical at K=32,
  K=16 parity-verified) and H=2/K=16 is the champion. ~~OUR rust/rwkv-infer is still K=32-hardwired~~
  RESOLVED: `1d3b5b8` ported the sibling's full engine (K-dynamic + PQ + joint cb + warm search).
- **QUANT ENDGAME LESSONS (sibling, 2026-07-04..07, full ladder in its research_log_h2k16.md):** per-card
  cost is INDEX bits -- catalog size is FREE (amortized): fewer/bigger chunks + huge learnable catalogs beat
  the product form on BOTH shift (m2b12) and WKV (joint-uv b10) sides * JOINT coding of correlated vectors
  buys robustness + seed-stability more than mean * rotation lever CLOSED (absorbed by learnable m=1
  catalogs; negative on big catalogs; only "won" on capacity-starved rungs that died as seed luck) * EMA at
  decay-tail = nil (3 confirmations); 2-seed weight soup HURTS (breaks weight<->cb co-adaptation) * norm
  axis bottoms out at 1 bit (0-bit fixed norms = +0.004 cliff) * ⚠ SEED-PAIR DOCTRINE: at-the-gate passes
  with margin < ~0.001 imm / ~0.002 ahead are UNRESOLVABLE by one run (64-b and 56-b "wins" both died on the
  seed test); any thin-margin verdict needs the exact recipe re-run at a second RWKV_AUGMENT_SEED.
- STILL DEFERRED: CUDA graphs (variable shapes, ~1.1-1.3x only); Stateful-BPTT carry SHELVED (smaller chunks
  don't speed training; the verified stateful WKV kernel is done + committed) [[stateful-bptt-shelved]].
- **TIER 1 DEPLOYED (2026-07-01):** the cudaMalloc/cudaFree->`torch::empty` caching-allocator scratch (WKV
  fwd+bwd scan, kills the synchronizing `cudaFree`, bit-exact ~1.3-1.44x microbench) is now the LIVE production
  `rwkv/model/RWKV_CUDA.cp312-win_amd64.pyd` (SHA256 == the bit-exact-validated build). Real-world WS steps/s
  A/B still deferred (falls out of the next training run).
- **TENSOR CORES -- PROFILED + DEAD (2026-07-01, hard numbers, `scratchpad/prof_wkv.py`).** The ONLY matmuls
  (scan `rwkv7_scan_kernel`+`rwkv7_add_kernel`) are **<=1.1% of WKV GPU time, 0.74% at B16xT30000** (realistic
  5k shape); the other 96% is the per-timestep matrix-VECTOR warp-shuffle recurrence (backward `final` ~61%,
  fwd `final`/`base` ~12/11%, bwd `base` ~11%) which tensor cores CANNOT touch. Amdahl ceiling <1% => the cheap
  "tensor-core the scan" win is DEAD. The only TC path is a from-scratch chunked-matmul (fla delta-rule) rewrite
  of the recurrence -- multi-day + parity-risky (K=16 underfills TC tiles); revisit ONLY if 5k proves too slow.

### SPEED -- where GPU training time actually goes (RE-DIAGNOSED 2026-06-30) [[gpu-training-speed-levers]]
- **Fetching is already HIDDEN -- NOT a lever.** `data_fetcher.get()` waits ~2.5-3 s on the FIRST batch then
  ~3-7 ms/step (7 workers + FETCH_AHEAD=5 fully hide prep+IPC); the input `.to(device)` H2D is ~0 ms on the
  critical path (~21 MB batch). Async-pinned prefetch / mp.Queue swap / vectorizing prepare() buy ~nothing.
  (This CORRECTS the earlier "fetch overlap 1.5-1.85x" claim, which was wrong about the mechanism.)
- **Cheap win = `RWKV_EMPTY_CACHE_EVERY`** (env added; default 1 = byte-identical). The per-step
  `torch.cuda.empty_cache()` (first 1000 steps, an OOM-fragmentation guard) costs ~118 ms/step.
  **VALIDATED 2026-06-30 (scratchpad/run_ectest.cmd, 320-step WS on train_db_sc8k):** every=1 -> 0.932 steps/s,
  every=0 -> 1.047 steps/s = **1.12x, NO OOM** (exit 0). Numerics-neutral (allocator only). Full 1.12x only for
  runs <=1000 steps (only the first 1000 steps clear); for WS-15 (~2400 steps) ~5% overall. ADOPT every=0 for
  research runs (model is tiny ~6/12 GB -> no frag-OOM risk).
- **Real lever = the WKV-kernel compute floor (fwd 140 + bwd 403 = ~543 ms/step, ~80% of the step).** Only a
  smaller model / K<32 kernel / bigger batch moves it. **PARTLY BANKED:** H=2/K=16 (K<32, now champion) cut
  ~half the WKV-kernel work for a net 1.16x WS speedup; bigger effective batch is the remaining headroom.
  Param breakdown (~193k): 5 RWKV streams 75.5% (deck 4L 21.6%, note/preset/user 3L 16.2% each, card 1L 5.4%),
  SRS heads 16.0%, input FC 8.4%; ~10.4k params per d=32 layer.
- **RE-PROFILED 2026-07-03 at the 5k regime (H=2/K=16, MAX=110000, RWKV_PROFILE_STEP env hook in
  train_rwkv): the WKV floor is NO LONGER dominant.** Plain step = 578 ms GPU: elementwise/other 78%, WKV
  recurrence 18%, gemm 5% => the chunked-matmul (fla delta-rule) rewrite is DEAD as a priority (addresses
  <=18%); the new top surface is the PyTorch elementwise mass.
- **torch.compile: WORKS on Windows (triton-windows in venv; the old "blocked" claim was STALE — Andrew
  caught it 2026-07-03) but SHELVED at an honest 1.05x.** Whole-graph compile hits Python 3.12's fixed
  C-recursion cap in Dynamo (RecursionErrors swallowed by the NaN-except -> HOLLOW steps -> a fake 1.27x
  profile); mixer-scoped compile is clean + deterministic but only 365 vs 384 ms. Costs (NO_JIT switch,
  warmup, recompile risk, numerics break) outweigh 5%. Plumbing kept: RWKV_COMPILE=1 + RWKV_NO_JIT=1 +
  scratchpad/train_bigstack.py. LESSON: count "Exception caught" before trusting any run's numbers.
- **DETERMINISTIC-INDEXING SPEEDUP BANKED 2026-07-03 (1.5x plain step, BIT-EXACT):** RWKV_DETERMINISTIC=1
  cost 251 of the 578 ms (sort-based index_add from 2 gather sites). Fixes: **PermGather** (srs_model --
  stream gather is a permutation+pads -> backward = index_select by the inverse permutation; escape hatch
  RWKV_PERM_GATHER=0) + **flat-row time_shift_gather** (rwkv_model -- row index_select instead of
  gather-expand-C -> det backward sorts B*T keys not B*T*C). Both verified by 10-step E2E training traces
  BIT-IDENTICAL to the old path. Det step 578->384 ms (det tax now ~57 ms). **STACKED total: the full
  quant-aware deterministic step = 4,122 -> 450 ms (9.2x); a 5k champion run ~= 4-5 h.**
- **QAT KERNEL SPEEDUP BANKED 2026-07-03 (the big one): quant-aware training was 7.1x slower than plain**
  (4,122 ms/step, 87% in the qat_lr kernels -- would have made every methodology-(a) 5k run ~30-40 h).
  Three bit-exact fixes in `qat_lr_rank1` + call sites: (1) skip-step elision (~half of rows are query
  duplicates whose truncation result was computed then discarded), (2) BLOCK-PARALLEL PQ codebook search
  (was single-threaded ~8k serial FMAs/step while 255 threads idled) with first-strict-min tie-breaking,
  (3) warp-0-scoped power iteration (__syncwarp instead of ~6 block barriers x <=64 iters). Result:
  **QAT share 3,577 -> 96 ms/step (37x), full step 4,122 -> 651 ms (6.3x); quant-aware now costs ~13%
  over plain. BIT-EXACT verified** (32-tensor golden fwd+bwd, int-N + PQ paths, both shapes) + deploy
  parity re-run (max REL 3.2e-07). Goldens: `scratchpad/qat_speed/golden_gen.py gen|check`.

### CURRENT STATE (updated 2026-07-15 — KEEP THIS SECTION SHORT: champions, live run, queue, live rules. Superseded chronology moves to optimization/HISTORY.md "5k-era LIVE STATE archive"; per-iter detail lives in research_5k_verbose.md)

**Champions / anchors:**
- **Track 1 (d=32 plain) CHAMPION = iter 29 `iter29_muon` (accepted 2026-07-21 16:05):
  ahead 0.302033 / imm 0.271440 ON THE VAL HALF (5001–7500, n=2500, 0 nanskips — the
  FIRST val-split verdict; val-half absolutes are NOT comparable to full-range iters
  ≤28), 171,453 params** (`champion_5k_plain.json` = ckpt
  `scratchpad/iter29_muon/iter29d_1638.pth` + WS/val traces = the track-1 vprune ref).
  **= iter 26 + hybrid Muon+AdamW (rwkv/muon.py) — the first OPTIMIZER-family win:
  matrix wd-groups on Muon (lr 0.02, momentum 0.95 nesterov, NS5, aspect-scaled,
  decoupled wd at the AdamW-equivalent rate), rest bit-exact functional AdamW. vs
  iter 26 same-users: ahead +0.000143 (p=2.5e-06), imm +0.000485 (p=6.5e-71, the
  phase's largest imm gain).** Champion recipe env (set ALL in every future track-1 run
  + the final QAT run): RWKV_NO_AHEAD_RESIDUAL=1, RWKV_ZERO_FEATURES=22,
  RWKV_PAVA_LAMBDA=0.1, RWKV_PROBE_DENSITY=0.08, **RWKV_GRU_HEAD=3**,
  RWKV_STRIP_L0_VLORA=1, RWKV_STATE_CLAMP_TAU=300, RWKV_STATE_CLAMP_WINDOW=32768,
  **RWKV_MUON=1, RWKV_MUON_LR=0.02, RWKV_MUON_MOMENTUM=0.95** +
  H=2/K=16 + HP {peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25} + MAX=110000.
  Optimizer is train-time only — nothing ships to Rust. Val-lag lesson now
  BIDIRECTIONAL (Muon trailed the 10-user val all WS tail, won eval decisively).
  PAVA middle-junction power strongly negative in ALL GRU/PAVA iters (−1.44/−1.44/−1.59).
  **Deploy contract:** learned-power PAVA rectifier on the 4 counterfactual button
  predictions (duration imputed to the frozen train median `scratchpad/iter23_pava/
  duration_median.json`) + per-step state clamp — Rust ports queued. Lineage kept:
  iter 26 (0.303942/0.273353 full-range, GRU N=3) → iter 25 (0.304427/0.273441, N=2,
  size-exception accept) → iter 23 (0.304220/0.273423, PAVA champion, 64-basis head) →
  iter 22 (0.304497/0.273539, no-residual re-baseline) → iter 15 (0.303663/0.273227,
  last with-residual); iter 14 = QAT tax ref (+0.0029/+0.0044).
- **Track 2 CHAMPION = A9 `track2_a9` (accepted 2026-07-22 04:05, ratio gate — BETTER
  both modes): ahead 0.298625 / imm 0.267615 ON THE VAL HALF (5001–7500, n=2500,
  0 nanskips — first val-split track-2 verdict; not comparable to full-range rows ≤A8),
  1,468,724 params** (= A8 −9.22%, −46.8% vs the original 2.76M;
  `champion_5k_track2.json` = ckpt `scratchpad/track2_a9/t2a9d_5586.pth` + WS/val traces
  = the track-2 vprune ref). **= A8 + note 2L→1L (arch scratchpad/track2_a9/
  architecture_d128_cmix1_user3_card2_note1.py — HALVES per-note deploy state, the
  dominant deploy-memory term) + L0 mixer strips user_id:0 + preset_id:0 — full track-2
  env now: RWKV_ARCH_MODULE=<the current champion arch>, RWKV_GRU_HEAD=2,
  RWKV_STRIP_L0_VLORA=1, RWKV_STATE_CLAMP_TAU=300, RWKV_STATE_CLAMP_WINDOW=32768,
  RWKV_NO_AHEAD_RESIDUAL=1, RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,
  preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1.** vs A8 same-users: ahead
  +0.000098 / imm +0.000010 BETTER (p 0.35/0.60 — not individually significant; the
  ratio gate prices cuts and both signs are right). Saliency pruning 5/5 since A6.
  **Stability: cleanest run of the chain — ZERO training NaN activity (A8's watch item
  did NOT recur; shallow note appears to have helped).** Lineage: A8 (0.300380/0.269006
  full-range, card 3L→2L + card.L1 strip, per-card state −1/3, the NaN-transients run) →
  A7 (0.300365/0.268966, user 4L→3L + note.L1/deck.L2 strips, imm p=9.1e-118 the
  strongest of the phase) → A0 (d=128 1-ep retrain, 0.299857/0.269030, n=4993, 7
  nanskips — 1-ep budget tax +0.0037/+0.0044 vs the upstream 12-ep .pth) → A1
  (mixers→1.0) → A4 = A1 + NO_AHEAD_RESIDUAL. The d=128 residual price = ahead
  +0.000495 (p=1.0) but imm 0.000062 BETTER — cheaper + more asymmetric than d=32's.
- **QAT deploy truth (FROZEN until research closes) = champ5k_b1** (0.306629/0.277893 quant-aware;
  `champion_5k.json` + its own cbs). At research close the final champion gets ONE 2-ep
  confirmation run + ONE quant-aware run (q72u deploy env + the frozen NO_JIT family flags;
  plain-era vs QAT-era logloss are NOT comparable).

**Iters 17+19 REJECTED: the pbin lever (binary-recall loss term) is CLOSED by dose-response.**
Scale 0.5 (iter 17): imm +0.000387 / ahead −0.000222; scale 0.25 (iter 19, n=4999): imm
+0.000258 (p=1.6e-70, under the bar) / ahead −0.000101 (p=1.0). The trade is ~linear through
zero → NO scale can make both modes improve ≥0.0003. Real, reproducible effect; pure trade.
⚠ Iter 19 also produced the FIRST-EVER d=32 NaN-skip (user 8902, 2.0M-token mega user, on its
1M–2M-token chunk; finite in all prior track-1 runs) — fp32-probe verdict in
research_5k_verbose.md; watch future track-1 evals for nanskips (gate needs --intersect then).

**Iter 18 REJECTED (directed, 2026-07-15 23:45): duration ablation (ZERO_FEATURES=8,22) =
+0.0018 ahead / +0.0024 imm worse — 6-8x the ≤0.0003 tolerance. Review duration is REAL signal
(historical answer times predict retention; nothing else recovers it); deploy keeps feeding it.
Champion recipe stays RWKV_ZERO_FEATURES=22 only.** The honest persistent val deficit predicted
this one (consistent-all-run val gaps mean something; oscillating ones don't).

**Family scoreboard (track 1, plain+QAT eras; conduct rule 5 — 1-2 rejects = deprioritized, NOT
closed):** early-training-intervention 0/2 (shrink-perturb, warmup-KD — both led early val then
washed out; mid-WS val leads do NOT predict verdicts); grade-representation 0/1; capacity-at-5k
0/2 (head resolution 64→128, mixer 1.5 — the d=32 trunk is not capacity-limited at 5k);
state-size ladder 0/5 CLOSED (no stream is state-capacity-limited at d=32/H=2; iter 6's near-miss
died on the seed pair); readout 0/3 WITH SIGNAL (prehead gate null; iter 20's 64-param cross-head
mix improved BOTH modes at p 2e-10/2e-25 but ~2/3 of the bar; iter 21's KxK 16x-capacity variant
ERASED the gain, ahead −0.0009 — the channel is real but capacity-starved is the WRONG diagnosis;
v3 queued = v1 with the delta EXCLUDED from wd);
loss-reweighting 0/2 (pbin 0.5 + 0.25 = linear imm/ahead trade, the SCALE lever is closed by
interpolation — other reweighting ideas like recency/per-rating weights would be new family
members); HP tuning CLOSED (champion HPs confirmed
vs 19 alternatives at full eval); **optimizer 1/2 — Muon ACCEPTED iter 29 (the strongest
family start of the phase: imm +0.000485 p=6.5e-71); cautious wd REJECTED iter 30 (pure
trade: imm +0.00014 / ahead −0.00038 — the pbin shape again); micro-tuning NOT auto-queued;
NorMuon/Polar-Express = deprioritized in-family variants.**
All hooks stay in-repo env-gated, default off: RWKV_KD_DUMP_OUT/
RWKV_KD_MIX, RWKV_INIT_BLEND, RWKV_GRADE_EMB, RWKV_STREAM_HEADS/RWKV_STREAM_LAYERS,
RWKV_PREHEAD_GATE, RWKV_PBIN_SCALE, RWKV_ZERO_FEATURES, RWKV_ARCH_MODULE, RWKV_EVAL_CAST_FP32,
RWKV_MUON (now ON in the champion recipe).

**Live rules (5k phase, both tracks):**
- **⚠ VAL/TEST SPLIT (Andrew 2026-07-21, effective from iter 29 / post-A8): candidates eval
  ONLY the VAL half = users 5001–7500 (n=2500); all verdicts + p-gates run there, pairing vs
  the champion's existing jsonls via `paired_pvalue --intersect`. TEST = 7501–10000 is touched
  ONLY at each track's close (final champion + the 2-ep confirmation + QAT runs) for honest
  numbers — NEVER for decisions.** Delta bars/p-thresholds unchanged (expect ~1.4× noisier SEs
  at n=2500). Training-val 5001–5010 + tuner 5001–6000 already ⊂ val (vprune refs stay valid).
  Eval tomls: `write_eval_toml ... 5001 7500`. Bonus: eval wall-clock halves. Full text:
  research_5k_notes.md methodology amendment.
- **RWKV_NO_AHEAD_RESIDUAL=1 in EVERY future run, both tracks (Andrew 2026-07-16: the
  piecewise-linear curve correction is DISABLED)** — track-1 iters and track-2 A3+ alike;
  A2 grandfathered (mid-flight). Iter 22 measures the cost; re-baseline is Andrew's call.
- **Track-2 (d=128) runs: RWKV_EMPTY_CACHE_EVERY=1 + RWKV_EMPTY_CACHE_WINDOW=0** (whole-run
  per-step clears — allocator-envelope creep → WDDM paging → 4x slowdown otherwise; ~free under
  the ~1 s step). **MAX=32768 EVERYWHERE incl. `write_decay_setup.py` arg 10** (its 110000
  default THRASHED A0's decay; pairing needs MAX identical across all track-2 runs). d=128 evals
  UNSHARDED (`--shards 1 --solo-threshold 0`; one alone ~9 GB). Coverage fact: max single batch
  in train_db_5k_h1 = 16,384 tokens → zero data drop at any MAX ≥ 16,384.
- d=32 evals: phased `eval_sharded.py` (solo mega-users → 2 LPT shards → merge; ~1.9x over
  sequential, wedge-safe; completeness gate = merged+skipped == rostered or exit 3).
  Elevated-VRAM rungs (e.g. K=32 streams) → sequential shards.
- **No mid-epoch resume on 1-ep runs** (the train loop has NO group skip on STEP_OFFSET — a
  resume re-sees early groups, drops the tail, breaks pairing → restart from scratch). Vals are
  only comparable at the SAME step (a val fires at step 50 = standard early ckpt).
- **Seed-pair doctrine (research phase):** any single-run margin < ~0.0005 needs the exact recipe
  re-run at RWKV_AUGMENT_SEED=4321 before acting — cross-seed spread on the same recipe is
  ~0.0004 both modes; in-seed Wilcoxon p (even 1e-29) measures per-user consistency, NOT
  cross-seed robustness.
- **TorchScript hook rules (cost 2 hollow/dead launches in iter 16):** @torch.jit.ignore bodies
  must NOT call submodules (through scripted code they see the raw C++ ScriptModule → 'not
  callable' → the NaN-except turns the run HOLLOW) — use root Parameters + F.linear, names
  containing weight/bias for the wd groups; root-level Parameters are INVISIBLE to
  selective_cast's module walk (cast them explicitly); ScriptModule forbids persistent=False
  buffers (use plain tensor attrs). Smoke tests MUST exercise the SCRIPTED forward +
  selective_cast/copy_downcast_ chain, not direct Python calls. Gate every .cmd phase on exit
  codes AND artifacts (train_rwkv can swallow fatal errors to exit 0).
- FETCH WORKERS = 4 in every training/eval toml (Andrew 2026-07-08, RAM). Live loss plot:
  `detach.ps1 -Script scratchpad/liveplot/run_liveplot.cmd` (auto-discovers the newest
  `*_ws_trace.jsonl`, champion ref from champion json).

**★ anki-revlogs-10k-id DATASET DONE (2026-07-16 00:07, 16.2 GB at
`C:/Users/Andrew/anki-revlogs-10k-id`):** the 10k dataset rebuilt from the raw HF release with
**REAL Anki epoch-ms IDs** (card/note/deck/parent/preset — no factorize) **+ corrected
`review_time = revlog id − taken_millis`** (show time, Andrew's directive; raw answer id =
review_time + duration; day_offset/elapsed_*/sort all use the corrected time). User numbering
== published set (file stems). VERIFIED vs published: user 70 row set identical (720,110 rows,
ratings 1:1 aligned by answer time), day_offset differs on exactly 1 row (show-time crossed the
day rollover — the intended effect); 10,000/10,000 revlog+deck tables, 9,934 card tables (==
published exactly). Builder `scratchpad/dataset_id/build_parquet_id.py` (resumable). Staging
`...-10k-id-raw` (archive + extracted protobufs ~40 GB — deletable once the parquets are
trusted). Follow-on work: a NEW preprocessing pipeline deriving FUTURE_FEATURES.md features
from the real timestamps.

**★ TRACK-2 A1 ACCEPTED (2026-07-16 10:57) = NEW TRACK-2 CHAMPION: all channel mixers → 1.0.**
**2,320,516 params (−442,368 vs A0); intersection (n=4993) ahead 0.299768 = +0.000089 BETTER
(p=2e-4), imm 0.269070 = +0.000040 worse (p=1.0) ⇒ per-100k ratios −0.0000201 / +0.0000090 —
~50× inside the ≤0.0001 gate.** Full-5000 finals 0.300009/0.269324 with **ZERO NaN-skips** (A0
needed 7 — the instability is gone; future track-2 gates can pair on full n=5000).
`champion_5k_track2.json` = A1 (ckpt `scratchpad/track2_a1/t2a1d_5586.pth`, 24 val points = the
track-2 vprune ref). d=32's mixer lesson transfers to d=128; decay-end val was IDENTICAL to A0.
Detail: research_5k_verbose.md. **Track-2 A2 queue (next track-2 block):** user 4L→3L / deck
4L→3L (~149k each), LoRA-dim cuts, d_model 128→96. **A2+ runs must set
RWKV_GRAD_STATS=<out.json> (Andrew's directive 2026-07-16):** records per-param mean|grad| +
mean|grad·w| (SNIP saliency) across all steps + final near-0/near-1 no-op weight stats, to
rank ablation targets; recorder `rwkv/grad_stats.py` (unit-tested), report
`python optimization/grad_stats_report.py <json>` (layer ranking + type-aware no-op suspects).

**Iter 20 REJECTED (2026-07-16 17:55) but = the plain era's strongest positive signal:
cross-head readout mix v1 (RWKV_XHEAD_MIX=1, zero-init (H,H,K) delta on the WKV output
pre-GroupNorm, 194,620 params) improved BOTH modes — ahead +0.000178 (p=2.0e-10), imm
+0.000107 (p=2.0e-25), n=5000, 0 nanskips — first p-gate PASS since iter 15, but both
magnitudes miss the 0.0003 bar.** Smoke lesson: W_o is zero-init → nothing upstream of it is
observable at fresh init (randomize W_o before perturb/grad smoke checks).

**Iter 21 REJECTED (2026-07-16 21:12): cross-head mix v2 (full K×K, 208,060 params) —
ahead −0.000859 worse (p=1.0), imm tied. The 16× capacity erased v1's both-modes gain;
the readout channel is information-poor + regularization-hungry, not capacity-limited.**

**TRACK-2 A2 REJECTED (2026-07-17 07:25): deck 4L→3L = ahead +0.000180 worse (p=1.0) =
per-100k ratio +0.000155 = 1.55× the ≤0.0001 bar (imm +0.000020 = +0.0000172, passes).**
Full n=5000, 0 nanskips (2nd consecutive clean d=128 run). Deck DEPTH is load-bearing for
the ahead/curve pathway; d128-single-layer-cut family 0/1, deprioritized for BUNDLES (the
cut was exactly 5.0% and still failed the price check). ⚠ A2's grad-stats jsons are DEAD
(whole-step-skip bug: layer-0 v_lora_simple.A never receives grads → every step skipped;
FIXED `dcf11f5` — per-param subset accumulation, report refuses dead jsons + lists
never-grad tensors as FREE prune candidates (5×1,024 params at d=128); A3 records
correctly on the same A1 trunk). Detail: research_5k_verbose.md.
**ITER 22 ACCEPTED (Andrew 2026-07-17 ~10:50, directed re-baseline): no-residual cost
ahead +0.000834 / imm +0.000312 vs iter 15 = the price of monotone-in-t. NEW track-1
champion/reference = 0.304497/0.273539; champion_5k_plain.json re-pointed (promote
--val-trace done).**
**A3 (GRU curve head) COMPLETE 2026-07-17 21:12 — REJECTED on the drafted vs-A1 gate,
VERDICT DEFERRED to the no-residual re-anchor.** n=4871 intersection: **imm 0.268403 =
+0.000105 BETTER (p=1.6e-21, FIRST significant track-2 accuracy win)**; ahead 0.299964 =
+0.000443 worse → ratio +0.000228 (2.28× bar) — but CONFOUNDED (A1 is residual-ON; iter 22
priced residual removal alone at +0.000834 ahead at d=32; A3's deficit is ~half that).
**⚠ 129/5000 eval NaN-skips** (instability oscillates through training; deploy-side
state-norm clamp now load-bearing for d=128). Grad-stats (fixed recorder): 10,886
never-grad params (layer-0 v_lora ×5 = free strip); saliency bottom = ALL non-L0 channel
mixers + user.L3.time_mixer = A4 bundle shortlist. Detail research_5k_verbose.md.
**ITER 23 ACCEPTED (VERDICT CHANGED by Andrew 2026-07-18 ~12:55; auto-verdict 01:15 had
been reject-on-magnitude): learnable power-mean PAVA rectifier = NEW TRACK-1 CHAMPION —
adopted for the monotonicity constraint itself (ordered button intervals = product UX),
with accuracy ~free-to-mildly-positive: BOTH modes improved (+0.000278 p=1.3e-33 /
+0.000116 p=8.1e-15 vs iter 22), n=5000, 0 nanskips, 193,727 params. Curve-shape-
constraints family 1/1. Detail research_5k_verbose.md (incl. the changed-verdict
addendum).**
**TRACK-2 A4 RE-ANCHOR DONE + PROMOTED (2026-07-18 12:02): 0.300504/0.269262, n=5000,
0 nanskips, ZERO NaN val windows (the GRU head, not d=128/no-residual, was A3's
destabilizer). A3 DEFERRED VERDICT = ratio gate PASS both modes (−0.0000288/−0.0000221
vs ≤0.0001; A3 BETTER than the fair anchor: ahead +0.000056 p=0.107, imm +0.000043
p=7.6e-05) — but promotion stays BLOCKED by A3's 129-NaN instability (recorded
gate-PASS-unstable); the GRU head (−194,292 params) is VALIDATED as an A5-bundle
component once the state-norm clamp / train-time fix lands. Re-anchor grad-stats:
never-grad 142,592 (dead ahead head 131,712 + 5×L0 v_lora 10,880 = free strip);
saliency bottom = 8 non-L0 channel mixers (~265k = 11.4% of A1) then card.L1/user
time-mixers — consistent with A3's report = robust A5 menu. ⚠ NAMING: "A4 bundle" in
older notes = A5 now (A4 = the re-anchor). Detail research_5k_verbose.md.**
**ITER 24 REJECTED (2026-07-18 15:32): p-head-weighted PAVA pooling = NULL vs iter 23
(ahead +0.000035 p=0.54, imm +0.000002 p=0.03; n=5000, 0 nanskips) — uniform pooling
suffices, iter 23 stays champion, deploy keeps the simpler rectifier. CONFIRMATION
BONUS: vs iter 22 it scored +0.000312 (p=6e-35) / +0.000118 (p=7e-21) — the PAVA gain
reproduced across two independent trainings (~+0.0003 ahead / +0.0001 imm real).
Weighting sub-lever closed. Detail research_5k_verbose.md.**
**ITER 25 ACCEPTED (VERDICT CHANGED by Andrew 2026-07-19 ~10:35; auto-verdict 07:24 had
been reject-on-logloss): GRU power-curve head at d=32 = NEW TRACK-1 CHAMPION on the
SIZE/SPEED exception — parity inside the budget (ahead −0.000207 p=1.0, imm −0.000018
p=0.38 vs iter 23) at 171,066 params (−11.7%); n=5000, 0 nanskips. The d=128 imm win did
NOT transfer (the d=32 trunk is the binding constraint) but both tracks now share the
GRU head. Val-lead lesson strongest instance: led vals nearly all run, lost eval. PAVA
Hard–Good power −1.44 IDENTICAL to iter 23 under a different head. Detail
research_5k_verbose.md (incl. changed-verdict addendum).**
**MEME RUN DONE (2026-07-19 10:53, recorded in optimization/side_experiments.md SE-1):
BLIND RWKV LOSES to FSRS-7 decisively — ahead 0.351922 (+0.034, wins only 7.5% of
users), imm 0.341322 (+0.023, wins 25%); n=5000, 0 nanskips. Intervals+grades are worth
~0.048 ahead LogLoss (~3.5× the full model's margin over FSRS-7). NOT in
research_log.jsonl by design.**
**ITER 26 (GRU N=3) ACCEPTED (VERDICT CHANGED 2026-07-19 ~21:00 — Andrew LOOSENED the
gate to rounded-4dp ≥0.0001 both modes; auto-verdict 20:18 had been reject on the old
0.0003 imm bar): ahead +0.000485 (p=4.4e-42, largest ahead gain of the phase), imm
+0.000088→0.0001 (p=4.8e-09); n=5000, 0 nanskips, 171,453 params = NEW TRACK-1
CHAMPION (recipe now GRU_HEAD=3). Under the new bar iter 20 (xhead v1,
+0.000178/+0.000107, both p≪1e-9) would also have passed → xhead-mix v3 gains queue
priority. PAVA middle junction −1.59 (3rd straight strongly-negative). Detail
research_5k_verbose.md.**
**ITER 27 REJECTED (2026-07-20 00:01): GRU N=4 = ahead −0.000411 / imm −0.000172 worse
than N=3 (p=1.0 both); n=5000, 0 nanskips. THE N-SWEEP PEAKS AT 3 — closed, no N=5;
iter 26 stands. Val-parity lost eval again. Detail research_5k_verbose.md.**
**ITER 28 REJECTED (2026-07-20 14:38): xhead v1 on the iter-26 recipe = ahead −0.000114
/ imm −0.000160 worse (p=1.0 both); n=5000, 0 nanskips. Iter 20's old-recipe gain did
NOT transfer — the readout channel measures NEGATIVE under the GRU head. V3 (wd
exclusion) DEPRIORITIZED with inverted rationale; readout/xhead family 0/3 on current
lineages, closed pending new ideas. Transfer-failure ledger: never graft, re-measure.**
**→ GPU plan (updated 2026-07-22 11:40): A10 DONE/REJECTED — the chain's first floor
after 5 accepts (both ratios over the bar 1.96×/1.76×; prime suspect = the note_id:0
strip that left the 1L note stream a bare time-mixer). **A11 RUNNING (launched 11:35,
verdict ~21:00): the A10 bundle MINUS note_id:0 — user 3L→2L + deck.L3 mixer strip,
note.L0 mixer KEPT (same A10 arch module), 1,352,620 params (−7.9% vs A9, allowed
0.000116/mode, gate vs A9 val-half). PASS → banks the size + fingers note.L0; FAIL →
user depth floors at 3L.** ⚠ EVAL-PATH FETCH-WORKER LEAK IS SYSTEMATIC: every
eval/rerun leaves 1–2 orphan pythons, some spinning a FULL CORE (iter-29's for 14 h,
the A9-rerun's for 8.5 h) — the trainer kills its workers ("Killed processes.") but
the eval path doesn't; CHECK + KILL ORPHAN PYTHONS after every run (spare pythonw =
bridge/controller, ~80000s-CPU = FSRS, and Andrew's liveplot); fix candidate: worker
cleanup in eval_sharded/get_result. Track-1 queue: permutation init (LOW),
fresh-family planning (LIT_REVIEW + FUTURE_FEATURES). 2026-07-21: A8 + iter 29 (Muon)
ACCEPTED, iter 30 (cautious wd) REJECTED; A8's first launch died in the ~02:35
black-screen hang (zero telemetry precursor, driver 610.62; crash combo REMAPPED to
RIGHT Ctrl + SPACE ×2, registry armed + rebooted); A9's first eval WEDGED on user
5747 (transient fetch race; eval_sharded RESUME recovered it). **ITER 28 QUEUED (Andrew 2026-07-19 ~20:50: re-benchmark iter 20 on the new recipe):
xhead-mix v1 EXACT (RWKV_XHEAD_MIX=1, +896 params) on the iter-26 champion recipe —
the old +0.000178/+0.000107 (p 2e-10/2e-25, would pass the NEW gate) was measured vs
the stale iter-15 recipe and must be re-earned (transfer failures are precedented).
Parked pid 21048 on A6's DONE_EXIT (~12:00 tomorrow → verdict ~15:30); tail prints
paired vs BOTH iter 26 and iter 27. If it passes → v3 (wd exclusion) as a follow-up
lever; if it fails → v3 is the in-family retry.** Track-1 queue after: permutation
init (LOW).
⚠ ERRATUM (2026-07-19): module index 1 = the DECK stream (arch order card,deck,note,
preset,user — NOT the RWKV_SUBMODULES order); the A3/A5 "note.L2 diverges" narrative
should read **deck.L2** (CLAMP_NOTES.md corrected; grad reports were always right).
New env for the strip: RWKV_STRIP_CMIX (rwkv_model.py, name:layer list, dummy-mixer
pattern, default off = byte-identical; RWKV7Config gains stream_name, stamped in
SrsRWKV.__init__).
⚠ OPS (cost 2 launches 03:22): PowerShell Set-Content -Encoding utf8 writes a BOM →
tomli dies line 1 col 1 — write tomls via the Write tool or UTF8Encoding($false); and a
crashed run's DONE_EXIT_WSFAIL satisfies downstream waitloop greps → relaunch upstream
first (its cmd truncates its own log), THEN re-park dependents.**
**MEME RUN "BLIND RWKV" QUEUED (Andrew 2026-07-19 ~02:30, recorded SEPARATELY — new
`optimization/side_experiments.md` at verdict, NOT research_log.jsonl): train d=32
WITHOUT interval features and WITHOUT grades (RWKV_ZERO_FEATURES=0-7,9-12,22; duration
kept) — can blind RWKV still beat FSRS-7? TARGET = FSRS-7-sched_penalties-short-secs-
recency on users 5001-10000: by-user mean LogLoss 0.317933 (vs AHEAD mode; our champion
0.304220 → 0.0137 of margin). Parked pid 4460 on iter 25's DONE_EXIT (scratchpad/
meme_blind/, ~3.5h). Recipe deviations (forced): vprune OFF (champion val ref would
false-kill), PAVA OFF (grade probes meaningless), clamp ON (full-n insurance), standard
64-basis head. Cmd tail prints paired-vs-iter23 (the cost of blindness). Interpretation
caveat: day-resolution intervals remain PARTIALLY reconstructible from the cycle
features (rows 22-28 share a per-batch phase → day gaps recoverable) + rows 12/13
(activity since card's last review) — grades are truly gone (duration correlates only).
Andrew's queue order: meme BEFORE further experiments → if iter 25 passes, iter 26
(GRU N=3) parks on the MEME's DONE_EXIT, not iter 25's.**
**ITER 25 QUEUED (Andrew 2026-07-18 ~23:30: "Let's try power curves first, to see if they
improve log loss of the small model"): GRU-faithful power-curve head at d=32
(RWKV_GRU_HEAD=2 + RWKV_STRIP_L0_VLORA=1 + state clamp τ=300 as insurance; full iter-23
champion recipe incl. PAVA; 171,066 params = −11.7%; MIN_STEP=6000). Parked pid 36720,
waitloop on A5's DONE_EXIT (~03:00) → verdict ~06:30. Gate: ≥0.0003 both modes vs iter 23
+ p<0.0001. **If iter 25 PASSES: iter 26 = RWKV_GRU_HEAD=3 (Andrew 2026-07-18 ~23:55 —
"If iter 25 succeeds, try 3"); sweep upward while it keeps winning (ordered-S
cumsum-softplus anti-collapse insurance available if higher N label-switches).** If it
misses: variant A (fixed log-spaced S-grid, weights-only, N≈8–16) is the family sibling. By-construction button-ordering ideas (FOSD/CDF-power head,
shared-shape ordered-S) discussed with Andrew 2026-07-18 — candidate follow-ups in the
curve-shape-constraints family after the power-curve verdicts.**
**Iter 22 REDEFINED (Andrew 2026-07-16 ~23:00) = DISABLE THE PIECEWISE-LINEAR CURVE
CORRECTION, queued behind A2 (detached pid 20584, waitloop on A2's DONE_EXIT → self-starts
~08:30, verdict ~11:45; run dir `scratchpad/iter22_nores`).** Andrew's directive: "check if
RWKV-Curve is using a linear piecewise correction, and if so — disable it for both tracks."
Confirmed: `curve_logits = logit(mixture) + interp(out_ahead_logits, t)` — a learned
64/128-point residual linearly interpolated between log-spaced time points. New flag
**RWKV_NO_AHEAD_RESIDUAL=1** (srs_model + srs_model_rnn) zeroes the residual outside
autograd → curve = pure mixture-of-exponentials, monotone in t BY CONSTRUCTION (supersedes
the cummin variant, which never trained; the raw-mixture BCE term AHEAD_RAW_SCALE=0.5
already supervises the mixture directly). NaN probe moved to out_p_logits under the flag
(zeros can't NaN — eval nanskip + train guard key off that probe). Params unchanged 193,724
(~12.5k now dead at d=32; ~131.7k dead at d=128 — strippable at deploy/in a track-2 bundle).
Smoke ALL_PASS (zero-residual, grad isolation, off-path byte-identity, JIT + NO_JIT).
**MANDATORY RECIPE both tracks from now on: RWKV_NO_AHEAD_RESIDUAL=1 in every future run
(track-1 iters AND track-2 A3+); A2 grandfathered (mid-flight, residual-on — its gate vs A1
is within-family valid).** **Iter 22 gate = ANDREW DECIDES: report both modes' finals,
deltas vs iter 15, p-values, and nan_users to him and WAIT — no auto-accept/reject, no
promotion. Likely outcome: iter 22 becomes the new track-1 REFERENCE (directed re-baseline
à la iter 14/15) since with-residual champions aren't fair gates for no-residual candidates;
track 2 similarly needs a no-residual re-anchor decision at the A2 verdict.**
**Track-1 queue (Andrew 2026-07-16 late, FIXED ORDER — iter 23 DONE/rejected-near-miss):
iter 24 = learnable PAVA + pooling weights from the p-head's button-press probabilities
(Instant mode, RWKV_PAVA_PWEIGHT=1; λ/density unchanged — validated by iter 23).** Then:
xhead-mix v3 (v1 delta excluded from wd), permutation init (LOW). **Duration imputation for the counterfactual probes (Andrew
delegated): ONE shared value across all 4 buttons (causally correct — duration is spent
before the press, independent of which button), = a GLOBAL CONSTANT (train-set median)
frozen into the deploy contract; only duration is imputed (elapsed/etc. are real at both
train and deploy); upgrade path if the audit shows sensitivity = per-user EMA carried
next to the state. Build-time checklist: enumerate ALL outcome-dependent dims of the 92
(INPUT_FEATURES.md) — rating one-hot + duration + any derived — and swap/impute them
consistently in the probe rows.**
**Track-2 sizing recommendation (Andrew 2026-07-16, soft rule): aim for ≥5% param reduction
per iteration, ideally more** — single ~116k layer cuts are borderline (A2 = exactly 5.0%);
future candidates should BUNDLE cuts (e.g. deck+user layers together, LoRA-dim cuts folded
into a bigger ablation) or go structural (d_model 128→96 ≈ 40%+). Track-2 queue after A3:
grad-stats-ranked BUNDLES (single ~116k layer cuts are now proven under-priced — A2's deck
cut failed at exactly 5.0%): user-layer + LoRA-dim bundles, d_model 128→96, head_w squeeze
(~83k, once the GRU head proves N=2 suffices) — re-ranked per A3's (fixed-recorder) report;
now confirmed by A4's report (same bottom tier) — this bundle = **A5** (A4 = the re-anchor).
**+ POWER-CURVE BASIS (Andrew 2026-07-16 late, for A3 bundling): replace the 128 exponential
bases with a handful (N≈8–16) of FSRS-7-style power curves** `R_i(t) = (1 + f_i·t/S_i)^(−c_i)`,
`f_i = 0.9^(−1/c_i) − 1` (pins R_i(S_i)=0.9; form = srs-benchmark `models/fsrs_v7.py`
forgetting_curve), S_i = fixed log-spaced grid, c_i = N learnable decays sigmoid-clamped to
[0.01, 0.95] (init ~0.5). Why few can replace 128: a power curve IS an infinite Gamma-mixture
of exponentials — one basis covers the heavy-tail region that needed dozens of exponentials.
Monotone in t by construction (keeps the no-residual guarantee). **Params at d=128:
w_linear 512→N cuts 65,664 → ~4.1k (−61.5k); + stripping the DEAD ahead head (−131.7k,
zero-risk, residual already disabled) ≈ −193k ≈ 8.3% of A1 before any head_w shrink**
(head_w 82.8k is a further optional squeeze once N is tiny). d=32 port later if it works
(w_linear 64→8 saves ~7.2k ≈ 3.7%). Note for the future hard-ordering option: per-basis c_i
breaks total pointwise order of the basis (curves with different decays cross); a single
SHARED learnable c + S-grid keeps the basis totally ordered (FOSD trick compatible) —
measure both if cheap. **VARIANT B = GRU-FAITHFUL (Andrew 2026-07-17, srs-benchmark models/gru.py — his call,
A3 ANCHOR): predict w, S, AND decay per curve.** ⚠ NAMING (Andrew 2026-07-17): the
benchmark model is called **GRU** — the old GRU-P entry was REMOVED from srs-benchmark
(training-data remnant; never write "GRU-P"). Our env flag = RWKV_GRU_HEAD=N, params
gru_*. GRU uses n_curves=2 and THREE tiny
linears off the trunk feature — w_fc (N logits→softmax), s_fc (exp(clamp(·,−25,25))
stabilities), d_fc (same-form decays) — into R(t) = Σ wᵢ·(1 + t/(1e−7+Sᵢ))^(−dᵢ). Plain
form, no R(S)=0.9 factor pinning. exp ⇒ dᵢ>0 ⇒ EACH curve monotone in t even with
per-curve decays (time-axis monotonicity does NOT need a shared decay — shared d is only
for the future FOSD hard rating-ordering, where the basis must be totally
pointwise-ordered; keep as later variant). Plan: N=2 faithful first (proven on the
leaderboard, label-switching moot at N=2); if it holds, sweep N with ordered-S
(cumsum-softplus) as anti-collapse insurance. Init: zero-init the three head WEIGHTS
(input-independent start, like the current zero-init w_linear) + set BIASES to a sane
prior curve (spread log-S, moderate d). Reuse the head_w trunk; replaces w_linear
(65,664 → ~3.1k at N=2, d=128). NB the current head does NOT predict S at all — fixed
log-spaced S grid (0.1 s→~e^22 s), model predicts only the 128 softmax weights (a
distribution over grid stabilities); grid-power-basis (variant A) = fallback.** ⚠ TorchScript trap (cost smoke_mono v1): old-style ScriptModule bakes
the FIRST construction's env-flag into the compiled class — never two flag values in one
process; ahead_linear is zero-init (like W_o) — randomize before head perturb/grad smokes.

**Queued:** entropy-floor analysis (irreducible-LogLoss estimate from the two disjoint d=128
.pths on users 1-100; design in research_5k_notes.md; ~30 min GPU); future-input-features plan =
`optimization/FUTURE_FEATURES.md` (real-timestamp features; needs a new dataset export — Andrew
2026-07-15); **scheduling-monotonicity plan = `optimization/MONOTONICITY_PLAN.md`** (Andrew
2026-07-16: button intervals can invert, e.g. Again > Hard — constraint must live IN the model;
time-axis stage RESOLVED BY REMOVAL — the piecewise residual is disabled per Andrew's directive,
curve now monotone in t by construction; remaining: audit → counterfactual button-consistency
loss at segment-end states via the shelved stateful kernel → isotonic projection as part of the
model at deploy; = the "curve-shape constraints" track-1 family); `optimization/LIT_REVIEW.md` queue;
deploy-side state-norm clamp (NaN guard, MONOTONICITY_PLAN-adjacent ship-time work).

### Ops
- **Compaction (ONLY sanctioned way):** run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
  then yield idle and STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) +
  focus-bearing flag (stale/empty = purged). Never hand-create `pending_compact.txt`. The injector is 24/7
  (ClaudeLoopController every 3 min; acts only on a stale heartbeat) and may inject EXACTLY `/compact <focus>`
  or a short `Continue` -- nothing else Claude-originated. (Since 2026-07-03 the **Telegram bridge**
  (`claude-automation/telegram_bridge.py`, task `ClaudeTelegramBridge`) additionally injects messages
  AUTHORED BY ANDREW from his authenticated Telegram account + mirrors chat output to his phone -- human
  steering, not self-injection. Master switch `telegram_bridge_active.txt`; see automation README.)
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
  already an input. (Features that WOULD become possible with a real-timestamp dataset export:
  `optimization/FUTURE_FEATURES.md`.)
- Quant papers: `scratchpad/{rwkvquant,rwkvedge}.txt` (poppler installed; the Read tool handles PDFs). Use the
  CURRENT session's scratchpad dir for transient logs (it rotates on teardown -- check task-output paths).
