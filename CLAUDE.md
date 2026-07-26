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
- **★ THREE-WAY PARITY — ALWAYS CHECK TRAIN vs EVAL vs CPU INFERENCE (Andrew's standing
  directive, 2026-07-26).** Whenever you add or change anything that touches the model's
  inputs, outputs, or objective, explicitly ask: *what does training optimize, what does
  eval score, and what will CPU inference (Python RNN + Rust) actually compute?* All three
  must be the same quantity. Write the answer down in the iteration's notes; a mismatch is
  a silent correctness bug that no gate catches, because each path looks self-consistent in
  isolation. The three real cases that motivated the rule:
  1. **PAVA was trained but never evaluated** (found 2026-07-26). The rectifier lived only
     inside the loss — `curve_probs` was returned unrectified — so every reported ahead
     number scored a model that differed from the one we intended to ship. It survived from
     iter 23 to iter 30 unnoticed.
  2. **The probe duration disagreed with the pipeline's own convention.** Probes imputed
     the train-set median (`scale_duration(6433)` = −0.121) while query rows — the
     pipeline's existing "no press yet" encoding — carry a literal 0.0. Now unified on 0.0.
  3. **The rectifier does not exist in `rust/rwkv-infer` at all**, so the deploy path could
     not have reproduced either version. On the port plan.
  Practical prompts: does the eval path apply every train-time transform that belongs to
  the model (rather than to the loss)? Do train/eval/deploy feed the same value for inputs
  that are unavailable at deploy time? Does the Rust engine implement it? Note that `imm`
  comes from the rating head and `ahead` from the curve head, so a curve-side change moves
  only one of the two gate modes.
  **THE TOOL (2026-07-26): `scratchpad/parity3/parity_train_vs_rnn.py`** — feeds identical
  weights + inputs through RWKV7 (parallel/training) and RWKV7RNN (recurrent/deploy) and
  requires ~1e-6 agreement. CPU-only, seconds to run, one subprocess per env combination
  (ScriptModule bakes the first construction's flags). **Add a case to it whenever you add
  an arch env flag** — that is the cheap check that would have caught STRIP_CMIX /
  STRIP_L0_VLORA / STATE_CLAMP living only in `rwkv_model.py` for a whole track-2 phase.
  Two vacuity traps it guards and yours should too: randomize the zero-init params (W_o and
  the scale linears zero out most of the recurrence) and assert the output scale is
  non-trivial before comparing.
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
- **`optimization/`** — tooling + the canonical record. `DEPLOY_FUNCTIONS.md` (the 2 Anki-fork target functions: `review_features` = the 5 stream states, `retrievability_head` = 1-P(Again); + the AGPL caveat) · `DEPLOY_SPEED_LOG.md` (the SINGLE `review_features` states/s table: 200-user paired Wilcoxon, size-identical + logloss-within-0.0005 assertions; distinct from `cpu_speed_log.md`, which pairs trials) ·  Record: `research_5k.md` (front tables,
  4dp) · `research_5k_notes.md` (methodology) · `research_5k_verbose.md` (per-iter detail,
  AI-only) · `research_log.jsonl` (5k source of truth) · `log.md`/`log.jsonl` (regenerated
  canonical table — `python optimization/logbook.py rebuild`) · `research_log.md` (CLOSED
  100/100-era log) · `HISTORY.md` (superseded plans + archived CLAUDE.md live-state) ·
  `FUTURE_FEATURES.md` · `LIT_REVIEW.md` · **`CPU_INFERENCE.md`** (the deploy-speed
  scoreboard: why param cuts have NOT yet bought CPU rev/s, and the Rust port that gates
  the real answer; bench `cpu_infer_bench.py`) · `PROTOCOL.md` (iter0-era mirror of §11) ·
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
  (Rust-parity vectors); `.safetensors` untracked by design. ⚠ Its June trace is for the OLD
  d=128 `rwkv_ref_558` and is NOT reproducible by current Python — see §11.
- **`reference_a18/`** (NEW 2026-07-26) — the self-consistent track-2 parity trace: A18's
  weights + traces + `ref_metrics.json` (which now records the checkpoint AND arch module
  actually exported). Self-contained at exactly 0.000e+00; this is what `PARITY: PASS` was
  measured against. Regenerate with `RWKV_REF_DIR=<dir> python export_rnn_trace.py`; only
  `ref_metrics.json` is tracked, the bulk artifacts are gitignored like `reference/`'s.
- **`rust/rwkv-infer/`** — the Rust CPU inference engine (`src/{main,model,fast,pava}.rs`,
  `BATCHING_PLAN.md`, `TRACK2_PORT_PLAN.md`); K-dynamic + full PQ/joint-cb/norm-quant engine
  since `1d3b5b8`; track-2 arch (GRU head, per-layer cmix skips, state clamp) + the PAVA
  button API since 2026-07-26, parity-verified. `pava.rs` = the rectifier + interval solver
  and the crate's only unit tests.
- **`vendor/jschoreels_anki/`** (NEW 2026-07-25, Andrew's directive) — READ-ONLY reference
  copy of the RWKV code from `github.com/JSchoreels/anki`, an Anki fork shipping the OLD
  2.76M RWKV as a live scheduler: `rust/` (mod.rs = the whole engine incl. the 5-stream
  chain `review_features`, the p(recall) head `retrievability_head` = 1−P(Again), state
  serialization and a `StateCompression` scheme; bulk.rs; matmul.rs = macOS Accelerate
  BLAS; **x86_simd.patch = AVX2/FMA `dot_product`+`add_scaled_in_place`, the most portable
  speed win for our engine, which has NO SIMD**; two CPU benches) + `python/` (his shipped
  torch RNN package). **`NOTICE.md` = provenance + ⚠ AGPL-3.0-or-later: copying any of it
  into `rust/rwkv-infer` makes that AGPL-derived — fine for shipping inside Anki, but flag
  it to Andrew first and keep provenance comments.** `INDEX.md` = the function map.
- **`scratchpad/`** — per-run pipelines + shared helpers. Tracked per run: `.cmd` + tomls +
  `*_ws_trace.jsonl` (+ champions' final cbs `cb_{wkv,shift}_final.txt`). Shared:
  `write_decay_setup.py`, `write_eval_toml.py`, `detach.ps1`, `liveplot/`,
  `architecture_old_d128.py`. **`parity3/`** (2026-07-26) = the three-way-parity harnesses §9
  requires: `parity_train_vs_rnn.py` (RWKV7 parallel vs RWKV7RNN recurrent on identical
  weights — ADD A CASE PER NEW ARCH ENV FLAG), `trace_selfcontained.py` (is a parity trace
  reproducible by current Python? run this FIRST when a gate looks wrong), `buttons_py_vs_rust.py`
  (the 4 button intervals, Python vs Rust). **`eval_pava/`** = the rectified-eval pipeline +
  `check_imm_identical.py` (imm must be bit-identical rect vs unrect — proves the probes are
  non-perturbative). Untracked on disk: ckpts (`*.pth`), logs, mid-run cb snapshots
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
⚠ **CORRECTED 2026-07-26 -- the old instruction here was WRONG and self-confirming.** It said to run
with `RWKV_WEIGHTS=reference/rwkv_iter36_124.safetensors` and that the default `rwkv_ref_558` "will
MISMATCH (wrong-weights, not a regression)". In fact **`verify_rust.py` never runs the engine** -- it
scores `reference/rust_pred_<user>.json` left by an earlier manual run, so `RWKV_WEIGHTS` cannot affect
its verdict, and any weights argument "works" or "fails" identically. Those files were stale (Jun 30,
quant-ladder era), which is how a FAIL with identical dpred across 3 crate versions and 3 weight files
went unnoticed. Correct procedure: **run the binary from the REPO ROOT** (`RWKV_WEIGHTS=...
./rust/rwkv-infer/target/release/rwkv-infer.exe`; it resolves `reference/trace_user_*` relative to CWD)
-> it writes `preds/rust_pred_*.json` -> **copy those into `reference/`** -> `python verify_rust.py`.
`reference/ref_metrics.json` names the reference model: **`rwkv_ref_558.pth`**, not iter36.
**★ SOLVED + GREEN 2026-07-26 -- the ROOT CAUSE was that the June `reference/` trace is not
reproducible by current Python, so the gate was scoring the artifacts, not the port.** New tool
`scratchpad/parity3/trace_selfcontained.py` asks the question that settles it: feed the trace's own
92-dim features back through the Python RNN at review 0 (all states empty = pure forward pass) and see
if it reproduces the `py_pred` frozen in the same file. The June trace FAILS at |d| up to 3.4e-1 -- the
same magnitude as the Rust "error" -- because `architecture.py` has since moved from d=128 to d=32 and
the model code has evolved, so nothing can reproduce those numbers now. **Run this check FIRST whenever
a parity gate looks wrong; a stale reference is far likelier than a broken engine.**
**FIX = regenerate, do not archaeologise.** `export_rnn_trace.py` now honours `RWKV_REF_DIR` (and
`verify_rust.py` too, matching the engine's `RWKV_TRACE_DIR`), so a fresh trace lands beside the old one
instead of clobbering it; `ref_metrics.json` now records the checkpoint + arch module actually exported
(the old code hardcoded "rwkv_ref_558.pth" regardless -- the very thing that made CLAUDE.md's
instruction wrong). The fresh A18 trace is `reference_a18/` and is SELF-CONTAINED at exactly 0.000e+00.
**RESULT -- the first-ever track-2 parity verification: `PARITY: PASS`, imm 0.000035 / ahead 0.000044
vs tol 0.0005** (14x and 11x inside). Procedure:
`RWKV_WEIGHTS=reference_a18/track2_a18.safetensors RWKV_TRACE_DIR=reference_a18 RWKV_STATE_CLAMP_TAU=300
RWKV_PRED_DIR=preds_a18v ./rust/rwkv-infer/target/release/rwkv-infer.exe` -> copy `preds_a18v/*` into
`reference_a18/` -> `RWKV_REF_DIR=reference_a18 python verify_rust.py`.
⚠ Note max per-review |rust-python| = 9.6e-3 (one review in 5,229) even though the by-user means agree
to 3.5e-5. That is accumulated float divergence over a ~5,000-step recurrence in two independent
implementations, not a formula error -- the gate measures the mean and passes with wide margin. Do not
expect the old d=32 port's "dpred ~3e-7"; that model was shallower and its chain shorter.

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
  (`pretrain/RWKV_trained_on_101_4999.pth`, unquantized) eval'd on 5001-10000 = **ahead 0.296385 /
  imm 0.264905 (n=5000)**, measured 2026-07-03 by `scratchpad/run_base5k_eval.cmd`; restricted to the
  VAL half 5001-7500 (the only set candidates are scored on) it is **0.294612 / 0.263561**, `size`
  identical to ours on all 2500. (This line read "PENDING, needs eval data" until 2026-07-26 -- it was
  stale by three weeks. `result/RWKV-base5k.jsonl` had the answer all along.)
- **★ THE GAP TO THAT BASELINE IS THE TRAINING BUDGET, NOT THE ABLATIONS (decomposed 2026-07-26,
  after Andrew asked "how did this happen? We were doing much more efficient ablations").** At
  IDENTICAL architecture and IDENTICAL 2,762,884 params, our own retrain **A0** scores 0.298342 /
  0.267858 -- so **+0.00373 / +0.00430 was already spent before a single parameter was cut**. The
  entire 2.76M -> 558k ladder (A0 -> A18) cost only **+0.00096 / +0.00053**, and iter 31 handed back
  -0.00039 / -0.00075: the net price of being **4.95x smaller is +0.00057 ahead and -0.00022 imm**
  (i.e. on imm the small model BEATS the same-recipe d=128 anchor). Fair label for the remainder is
  "our 1.25-epoch recipe vs upstream's ~12", since A0 also differs in augmentation (off), peak LR
  (1e-3 vs 7e-4), warmup (200 vs 20,000) and MAX (32768 vs 66000).
  **Two consequences.** (1) The budget gap is SHARED by every iteration, so it cancels in every gate
  and no ranking is affected -- do not "fix" it mid-phase. (2) It is where ~0.004 lives, so the
  research-close plan's 2-ep confirmation run was undersized. **DECIDED (Andrew 2026-07-26): a
  10x-epoch-budget run, ONCE, at the very end — after the algorithmic loop AND after new input
  features. See "THE ENDGAME, ORDERED".** Do not run it earlier "just to see".
  ⚠ This check was DESIGNED IN and its answer was never promoted: `research_5k_verbose.md` planned
  "1-ep-budget check at d=128 rides along free: if A0 ~ the 12-ep upstream number...". It ran, came
  back 0.004 short, and the headline framing kept comparing a 1-epoch model to a 12-epoch one without
  saying so -- which is what makes the gap read as an ablation regression. It is not one.
  **NOT a data-drop bug (checked, dead end):** `get_groups` silently skips any chunk with
  `size > MAX_TRAIN_GLOBAL_LEN` (train_rwkv.py:247-250, the old MAX=20000 incident), but the largest
  chunk in `train_db_5k_h1` is 16,384, so MAX=32768 drops **0 of 46,062 chunks / 0 of 667,525,912
  rows**. MAX only controls how many chunks batch together here.
  Front table `optimization/research_5k.md`; full methodology + status `optimization/research_5k_notes.md`.
- **Run env (all phases):** **augmentation OFF** (RWKV_AUGMENT_SEED=1234) + RWKV_DETERMINISTIC=1 +
  RWKV_EMPTY_CACHE_EVERY=0 -> run-to-run variance ~0. Eval `python -m rwkv.get_result` (CUDA, JIT-on ->
  REQUIRES the `@torch.jit.ignore` fix on `quant_aware_rwkv7`).
- **Historical 100/100 + 1500u workbench refs** (eval 101-200, MAX=66000, sc8k dbs): champion recipe was
  "1 ep on 1500 users (1000-2499) + decay" (data variety >> repetition; ~25 min/experiment -- still useful
  for cheap sanity checks). d=128-on-1-100 baseline = 0.320295/0.281913 (arch-swap
  `scratchpad/architecture_old_d128.py`); iteration-0 floor = 0.374046/0.319475.

### HISTORICAL CHAMPION (SUPERSEDED -- the live champion is A18, see CURRENT STATE) = H=2/K=16 on the 1500-user data-variety recipe  (d=32, 2 heads x K=16; 193,724 params)
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

### CURRENT STATE (trimmed 2026-07-26 — KEEP THIS SECTION SHORT: champions, live run, queue, live rules. Superseded chronology is in `optimization/HISTORY.md` "5k-era LIVE STATE archive"; per-iteration detail in `research_5k_verbose.md`; numbers in `research_5k.md` + `log.md`)

**★ THE TWO TRACKS HAVE MERGED (Andrew 2026-07-26).** The track-2 A-series is CLOSED at A18.
Work continues as ONE lineage on the A18 trunk, numbered as track-1 iterations in
**research_5k.md's FIRST table**. ⚠ The old track-1 `params <= 225,000` cap is RETIRED — it
belonged to the d=32 track; this lineage's size story is the 4.95x reduction (flagged to
Andrew, not silently dropped).

#### CHAMPION = iter 31 `iter31_algo` (A18 trunk + PAVA + GRU N=3 + Muon)
Accepted 2026-07-26, the FIRST merged-lineage iteration. **ahead 0.298909 / imm 0.267637** on the
VAL half (5001-7500, n=2500, 0 nanskips) = +0.000393 / +0.000753 vs A18 at p=6.0e-26 / 1.5e-209;
`size` identical (0/2500 mismatches). **558,212 params** (+966 vs A18 = GRU N=2->3 + `pava_theta`);
per-card state 2,880 floats and note 1,440, both UNCHANGED (PAVA and Muon are train-time only; the
GRU head is a head, not a stream). ckpt `scratchpad/iter31_algo/iter31d_5586.pth`;
`champion_5k_track2.json` now points at it (= the vprune ref). Env = A18's full env below PLUS
`RWKV_GRU_HEAD=3`, `RWKV_PAVA_LAMBDA=0.1`, `RWKV_PROBE_DENSITY=0.08`, `RWKV_MUON=1`,
`RWKV_PROBE_DUR=0.0`.
⚠ A BUNDLE of three changes -- it establishes that the graft transfers to d=80, NOT which part
carries it. Ablation = 3 more runs, deferred pending Andrew.
⚠ Third confirmation that VAL LAG IS BIDIRECTIONAL: iter 31 trailed A18 on val all through WS and
won both modes on eval. Record val lag; never act on it.

#### PREVIOUS CHAMPION = A18 `track2_a18` (the trunk iter 31 builds on)
Accepted 2026-07-26 by Andrew's directed verdict change over an auto-reject at 108%/111% of
the ratio bar: *the >=5x product goal outranks a marginal RATE missed by ~10%*, costing only
+0.000960 ahead / +0.000532 imm cumulative vs A0 (~1/3 of what the matched-param GRU baseline
gave up). Precedent = iters 23/25/26.
- **ahead 0.299302 / imm 0.268390** on the VAL half (5001-7500, n=2500, 0 nanskips);
  **557,246 params = 4.95x below the original 2.76M** (79.8% cut); per-card state 2,880 floats.
- ckpt `scratchpad/track2_a18/t2a18d_5586.pth`; `champion_5k_track2.json` = the vprune ref.
- arch `scratchpad/track2_a18/architecture_d80_lora4.py`: d_model 80 (5 heads x K=16), LoRA
  decay/a/gate 4, v0-mix 2.
- **FULL ENV (set all of these in every run on this trunk):** `RWKV_ARCH_MODULE=<that file>`,
  `RWKV_GRU_HEAD=2`, `RWKV_STRIP_L0_VLORA=1`, `RWKV_ZERO_FEATURES=22`,
  `RWKV_STATE_CLAMP_TAU=300`, `RWKV_STATE_CLAMP_WINDOW=32768`, `RWKV_NO_AHEAD_RESIDUAL=1`,
  `RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1`.
- Fallback = A15 (0.299031/0.268111, 808,762 params, 3.41x — the gate-clean one).
- **THE WIDTH LADDER IS CLOSED:** two independent draws at d=80 (A17 112%/83%, A18 108%/111%)
  = a genuine accuracy floor; d=64 (A16) is ~1.8x the bar. 4.95x is the end of the width road.
  All depth floors are mapped too (card=2, deck=4, note=1, preset=3, user=3) — ladder exhausted.
  **Side-finding that sets the agenda:** the second LoRA halving is NOT free at d=80
  (+0.00002/+0.00009 for -27.5k) whereas A14's first halving IMPROVED both modes at d=128 —
  the lever flips sign as the trunk narrows, so the model is now genuinely capacity-limited and
  further gains must come from **ALGORITHMS, not shape**.

**Track-1 (d=32) lineage — closed, but its wins carry over.** Last champion iter 29
`iter29_muon` (0.302033/0.271440 val half, 171,453 params). Its three algorithmic wins are
exactly what iter 31 grafts onto the A18 trunk: **PAVA** (iter 23), **GRU N=3** (iter 26),
**hybrid Muon+AdamW** (iter 29, `rwkv/muon.py` — train-time only, nothing ships to Rust).
**Deploy contract:** learned-power PAVA rectifier on the 4 counterfactual button predictions
(current-row duration zeroed on all four) + per-step state clamp.

**QAT deploy truth (FROZEN until research closes)** = champ5k_b1 (0.306629/0.277893
quant-aware; `champion_5k.json` + its own codebooks). At research close the final champion gets
the LONG RUN below + ONE quant-aware run (q72u deploy env + the frozen NO_JIT family flags).
Plain-era and QAT-era logloss are NOT comparable.

### ★ THE ENDGAME, ORDERED (Andrew 2026-07-26 — this is a SEQUENCE, not a menu)
> *"We'll do a run with 10x the current epoch budget, but only at the end, after algorithmic
> improvements and adding new input features."*
1. **Algorithmic improvements** — the current research loop, unchanged (gate, families, conduct
   rules all as-is). iter 32 onward.
2. **New input features** — previously filed LOW; this directive puts it ON the critical path.
   Design in `optimization/FUTURE_FEATURES.md`; **needs an LMDB rebuild**, so it is the long-lead
   item — start scoping it before the algorithmic loop runs dry, not after. ⚠ It also breaks
   research-phase INVARIANT 2 ("same preprocessed 92-dim inputs / existing LMDBs") — that
   invariant was for the shrink phase and Andrew has now superseded it; the hierarchy invariant
   stands. Every candidate after the rebuild re-bases against a champion re-run on the new inputs.
3. **THEN the 10x-budget run — ONCE, on the final champion only.** 1.25 ep -> ~12.5 ep, i.e.
   parity with what upstream trained (~12), which is where the +0.0037/+0.0043 measured in
   "Workbench + baselines" actually lives.
   **Do NOT run this earlier "just to see".** Changing the shared budget re-bases every
   iteration's comparison, and mid-phase it would cost days of GPU for a number that cannot be a
   champion accept.
   **Cost to plan for, at iter 31's measured 1.44 steps/s:** WS 223,460 steps ~= 43 h, decay at
   the 0.25 ratio ~55,900 steps ~= 11 h, plus eval ~= **2.5 days of continuous GPU**. Use the
   mid-epoch resume (`RWKV_RESUME_SKIP_GROUPS=1` + `make_resume.py`) — over that span a crash or
   reboot is likely, and losing it whole is the only real failure mode.
   **⚠ AND IT IS QUANT-AWARE (Andrew 2026-07-26, "It will also be with QAT").** Two open items,
   both to settle BEFORE launching, neither expensive:
   - **THE QAT TAX IS ~1.7x WALL-CLOCK, NOT 13% — do not quote the 13% (Andrew corrected this
     2026-07-26: "IIRC it ended by being like 3x slower").** Both numbers are in the record and
     they measure different things:
     * **+13%** (`research_5k_notes.md` 2026-07-03) is a PROFILED GPU STEP, 651 vs 578 ms, kernel
       share only, at MAX=110000 on `train_db_sc8k_1500`.
     * **1.7x** is WALL-CLOCK at the real recipe: iter 14 `champ5k_plain` is champ5k_b1's exact
       recipe with the QAT env stripped and ran **WS 0.82 s/step = "1.7x faster than
       quant-aware"** (`research_5k_verbose.md` iter 14). Its eval was 75 min phased vs the
       **145-min sequential QAT eval**.
     * **The gap between them is almost all LOST JIT:** QAT forces `RWKV_NO_JIT=1`, and the banked
       JIT restoration is worth ~1.38x. 1.13 x 1.38 = 1.56, vs 1.7 observed. The profiled figure
       never included it.
     **=> the 10x run costs ~4 DAYS of training, not 2.5** (WS 43 h -> ~73 h, decay 11 h -> ~19 h),
     and that is still at d=32 state sizes; iter 31's per-card state is **2,880 floats vs 576**, so
     the kernel share should be re-measured at d=80 (100-step A/B, plain vs q72u env, ~5 min GPU).
   - **★ HIGH-VALUE, CHEAP: test whether QAT can run JIT-ON.** If the tax is mostly lost JIT rather
     than kernel work, it is recoverable — CLAUDE.md already flags "JIT on the grafted q72u paths
     unverified -- A/B once at champion-run launch", and the `torch.linalg.svd` that originally
     broke TorchScript was fixed by `@torch.jit.ignore` on `quant_aware_rwkv7`. Worth **~1.38x, i.e.
     roughly 1.5 days off a 4-day run**, for what is a smoke test. Do this BEFORE the long run.
   - **WHERE the QAT sits in the 10x is a real fork,** because `QAT from scratch = +0.0118` (iter
     40) — it MUST warm-start. Reading A: 10x plain -> warm-start QAT for the existing 2.0-ep
     fine-tune. Reading B: 10x budget with QAT active throughout, warm-started from the current
     champion. A is much cheaper and matches how QAT has always behaved here (a fine-tune);
     B is the literal reading of "the long run is quant-aware". **ASK Andrew which, with the
     measured overhead in hand — do not assume.**
   **Three things that are tuned for 1 epoch and must be RECONSIDERED at 12.5 — write the answers
   down before launching:** (a) **warmup 200 steps** is 0.9% of a 1-ep run but 0.09% of this one;
   upstream used 20,000. (b) **augmentation is OFF** (`RWKV_AUGMENT_SEED=1234`) — a deliberate
   workbench choice for ~0 run-to-run variance, but at 12.5 epochs over the SAME 5,000 users it is
   repetition, not variety, and augmentation is exactly the regularizer that regime wants; the
   variance argument no longer applies to a one-off final run. (c) **wd/dropout** were tuned where
   overfitting was impossible; at 10x they are live levers. None of this is a reason to delay —
   just do not assume the 1-ep recipe transfers.

#### LIVE — a 3-job GPU chain, each parked on the previous one's `DONE_EXIT_`
1. **RUNNING: rectified evals** — `scratchpad/eval_pava/run_rect_evals.cmd`, detached pid 33012.
   Started 19:22 after iter 31 finished; A18 leg ~2.4 h then the iter-31 leg, so the pair lands
   ~00:10. Log `scratchpad/eval_pava/rect_evals.log`.
   **WHY TWO METRICS:** iter 31's own eval leg is UNRECTIFIED (its `.cmd` predates
   `RWKV_EVAL_PAVA`, and a RUNNING `.cmd` must never be edited — cmd.exe re-reads it at a saved
   byte offset). That is fine and is the **PRIMARY gate**, being directly comparable to A18's
   existing jsonls; the rectified pair is the **deploy metric**. **If the two disagree, report
   both to Andrew — do not pick.**
2. **PARKED: mode-2 duration diagnostic** — `scratchpad/eval_pava/run_mode2_diag.cmd`, pid 17928,
   log `scratchpad/eval_pava/mode2_diag.log`. Answers Andrew's question about zeroing the current
   review's duration. `RWKV_EVAL_PAVA=2` substitutes the pressed probe WITHOUT pooling, so modes
   0/1/2 give an additive split: `m2-m0` = duration zeroing, `m1-m2` = PAVA pooling, `m1-m0` = the
   total a rect-vs-unrect run reports. 500 users (5001-5500, ~30 min) — a paired diagnostic on one
   model with one flag moving, not a cross-training ranking, so the 200-user subset-overfit lesson
   does not bite. `scratchpad/eval_pava/decompose_duration.py` (verified: it reproduces the
   recorded iter31−A18 delta and p exactly).
3. **PARKED: iter 32 = full-run DISTILLATION** — `scratchpad/iter32_kd/run_iter32_kd.cmd`, pid
   25348, log `scratchpad/iter32_kd/iter32_kd.log`. ~10 h (teacher dump ~3 h + WS + decay + eval).
   Teacher = `pretrain/RWKV_trained_on_101_4999.pth` under `scratchpad/architecture_old_d128.py`;
   student = the iter-31 recipe unchanged plus `RWKV_KD_MIX` + **`RWKV_KD_ALPHA=0.5`** (new flag,
   2026-07-26: holds alpha FIXED = the classic form; unset keeps iter 10's linear 1->0 ramp
   byte-identical). Decay runs on hard labels. Gate = ordinary accuracy iter vs iter 31; the
   `.cmd` also reports the candidate against the d=128 teacher, i.e. how much of the 0.004 closed.
   **Three things to know if it misbehaves:** (a) **vprune is deliberately OFF** — the
   decay_ratio_0p1 FALSE-KILL scope rule says prune only at MATCHED regularization, and KD
   replaces the target wholesale while validation still scores HARD labels; (b) the teacher must
   set `RWKV_PROBE_DENSITY=0.08` + `RWKV_PROBE_DUR=0.0` even though it has no PAVA, because probes
   are a DATA-side row-layout change and teacher/student must agree or the per-step shape check
   exit-43s; (c) a smoke dump of 5 steps gates the full one on `check_dump.py`, which tests that
   p_curve is inside (0,1) and p_imm_all sums to 1 — the student's checksum proves ALIGNMENT but
   nothing else proves the tensors are teacher outputs at all, and a wrong arch/flag yields
   perfectly aligned garbage. It also projects the dump's disk footprint before committing.
- ⚠ **WAITLOOP TRAP, cost one wrongly-started co-tenant eval (2026-07-26):** `findstr /C:"DONE_EXIT"`
  matches a log line that merely MENTIONS the token — including the waiter's own
  `=== WAIT for ... DONE_EXIT ===` message — so the loop fires instantly. **Anchor it:
  `findstr /B /C:"DONE_EXIT_"`** (terminal lines start with the token; prose never does) and do not
  write the token in non-terminal log lines. This is distinct from the known
  `DONE_EXIT_WSFAIL`-satisfies-the-grep gotcha.
- ⚠ **`detach.ps1` needs an ABSOLUTE path.** `Win32_Process.Create` starts in System32, so a
  relative script path exits instantly, silently, and still returns a pid.

#### QUEUE
1. **Record iter 32 at verdict** (~06:00-11:00): `research_log.jsonl` + `research_5k.md` FIRST
   table + `research_5k_verbose.md` + `python optimization/logbook.py rebuild`. File it under the
   **distillation** family (iter 10 was mis-filed under early-training-intervention, which is why
   the scoreboard shows distillation nowhere). Read the mode-2 decomposition out of
   `scratchpad/eval_pava/mode2_diag.log` at the same time.
2. **If iter 32 lands well, the cheap follow-ups reuse the SAME dump** (the expensive part): the
   annealed-alpha variant (unset `RWKV_KD_ALPHA`, window = full run) and an alpha sweep are
   student-only re-runs. Curve-level distillation (variant 3) needs new dump code.
3. **RUST PORT** (`rust/rwkv-infer/TRACK2_PORT_PLAN.md`) — the highest-value non-research work.
   Steps 1-4 DONE and `PARITY: PASS` for A18 (§11). Remaining: port the button API to `fast.rs`
   (model.rs has it; fast.rs is the DEFAULT runtime path, so deploy cannot serve intervals at
   speed until it lands), then **measure** — the experiment that says whether the ablations bought
   user-visible speed. First data point already in `CPU_INFERENCE.md`: 2.39x on the Rust path.
4. **NEW INPUT FEATURES — now on the critical path** (Andrew 2026-07-26; see "THE ENDGAME,
   ORDERED"). `optimization/FUTURE_FEATURES.md` + the deck-tree features. **Needs an LMDB
   rebuild**, which is the long-lead item in the whole plan, so scope it BEFORE the algorithmic
   loop runs dry. Budget question CLOSED: the 10x run happens once, last, after this.
5. Entropy-floor analysis (~30 min GPU; design in `research_5k_notes.md`); permutation init (LOW).
   `pava_loss_avg` / `pava_pool_frac` step-trace fields: DONE (train_rwkv.py, keyed on enablement).

**⚠ CPU-INFERENCE REALITY CHECK (Andrew 2026-07-25: "I told you to do ablations hoping that
fewer params -> faster CPU inference in Anki").** Measured in `optimization/CPU_INFERENCE.md`:
in the PYTHON RNN path a 4.5x arithmetic cut buys only **1.24x** wall-clock and PLATEAUS after
A14 — that path runs at 0.08-0.30 GMAC/s vs a core's 5-20, so it is OVERHEAD-bound and cost
tracks op count (layers x streams), not width. **1 thread beats 3 and 6 → deploy
single-threaded.** The deploy path is Rust (~10x faster, far less per-op overhead) where width
SHOULD pay off — which is why the port is the gating work for whether the ablations bought
user-visible speed. Bench: `python optimization/cpu_infer_bench.py`.
(Training speed IS monotone in width — median steps/s A0 0.933 -> A16 1.746 = 1.87x faster at
7.11x fewer params, sublinear as the elementwise-dominated profile predicts.)

#### FAMILY SCOREBOARD (conduct rule 5: 1-2 rejects = deprioritized, NOT closed)
curve-shape constraints **1/1** (PAVA) · optimizer **1/2** (Muon ACCEPTED iter 29, the phase's
largest imm gain; cautious wd REJECTED iter 30 — a pure trade) · GRU-head N-sweep **peaks at
N=3** (N=4 worse, closed) · readout/xhead **0/3** with real signal but negative under the GRU
head (iter 28), closed pending new ideas · loss-reweighting **0/2** (pbin scale lever closed by
dose-response — a linear imm/ahead trade through zero) · early-training-intervention **0/2** ·
grade-representation **0/1** · capacity-at-5k **0/2** · state-size ladder **0/5 CLOSED** · HP
tuning **CLOSED** (champion HPs confirmed vs 19 alternatives at full eval).
All hooks stay in-repo, env-gated, default off.

#### LIVE RULES (both tracks)
- **⚠ VAL/TEST SPLIT (from iter 29 / post-A8):** candidates eval ONLY the VAL half = users
  **5001-7500** (n=2500); all verdicts + p-gates run there, pairing vs the champion's jsonls via
  `paired_pvalue --intersect`. **TEST = 7501-10000 is touched ONLY at each track's close** —
  never for decisions. Eval tomls: `write_eval_toml ... 5001 7500`.
- **`RWKV_NO_AHEAD_RESIDUAL=1` in EVERY run** (Andrew 2026-07-16): the piecewise-linear curve
  correction is disabled, so the curve is monotone in t by construction.
- **d=128/d=80 runs:** `RWKV_EMPTY_CACHE_EVERY=1` + `RWKV_EMPTY_CACHE_WINDOW=0` (allocator creep
  -> WDDM paging -> 4x slowdown otherwise). **MAX=32768 EVERYWHERE** incl. `write_decay_setup.py`
  arg 10 — pairing needs MAX identical across runs. Evals UNSHARDED (`--shards 1
  --solo-threshold 0`). d=32 evals use phased `eval_sharded.py`.
- **MID-EPOCH RESUME:** `RWKV_RESUME_SKIP_GROUPS=1` + `python scratchpad/make_resume.py
  <run_dir> <prefix> <ws_toml>`, then rerun the WS phase with the run's FULL env, WITHOUT
  deleting step-trace files. Crash recovery loses <=1000 steps. The resumed tail's dropout draws
  differ (weights/optim exact) — statistically equivalent, not bit-identical.
- **⚠ NO co-tenant GPU work during gate-critical runs** — cuBLAS algo selection under memory
  pressure breaks bit-replay (~1e-4 val drift), and at 11.6/12 GB two processes deadlocked in
  WDDM paging for 2.7 h. Smokes wait for a free GPU or run tiny/CPU.
- **Seed-pair doctrine:** any single-run margin < ~0.0005 needs the exact recipe re-run at
  `RWKV_AUGMENT_SEED=4321` first — cross-seed spread on the same recipe is ~0.0004 both modes;
  in-seed Wilcoxon p (even 1e-29) measures per-user consistency, NOT cross-seed robustness.
- **TorchScript hook rules** (cost 2 dead launches): `@torch.jit.ignore` bodies must NOT call
  submodules (scripted code sees the raw C++ ScriptModule -> 'not callable' -> the NaN-except
  turns the run HOLLOW) — use root Parameters + `F.linear`, names containing weight/bias for the
  wd groups; root-level Parameters are INVISIBLE to `selective_cast`'s module walk (cast them
  explicitly); ScriptModule forbids `persistent=False` buffers. Old-style ScriptModule bakes the
  FIRST construction's env flags into the compiled class — never two flag values in one process.
  Smoke tests MUST exercise the SCRIPTED forward. Gate every `.cmd` phase on exit codes AND
  artifacts (train_rwkv can swallow fatal errors to exit 0).
- **`RWKV_GRAD_STATS=<out.json>` on every ablation run** (Andrew 2026-07-16) — per-param
  mean|grad| + SNIP saliency, to rank targets. Report: `python
  optimization/grad_stats_report.py <json>`.
- FETCH WORKERS = 4 in every toml (RAM). Live loss plot: `detach.ps1 -Script
  scratchpad/liveplot/run_liveplot.cmd`.
- **⚠ Eval-path fetch-worker leak is SYSTEMATIC:** every eval leaves 1-2 orphan pythons, some
  spinning a full core for hours. **CHECK + KILL orphan pythons after every run** — but inspect
  command lines first: the spare `pythonw` are the bridge/controller, the ~80000s-CPU python is
  Andrew's FSRS benchmark, and he also runs a Reddit bot + liveplot. **Do not kill those.**
- **OPS gotcha:** PowerShell `Set-Content -Encoding utf8` writes a BOM -> `tomli` dies at line 1
  col 1. Write tomls with the Write tool or `UTF8Encoding($false)`. A crashed run's
  `DONE_EXIT_WSFAIL` satisfies downstream waitloop greps — relaunch upstream FIRST, then re-park
  dependents.

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
