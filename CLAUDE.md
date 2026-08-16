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
  `user_order.jsonl` ranks user ids by size. **Everything built so far uses this set.**
- **★ SECOND DATASET — `C:\Users\Andrew\anki-revlogs-10k-id` (REAL epoch-ms IDs + corrected
  `review_time`), built by `scratchpad/dataset_id/`.** Same layout + same 1:1 user numbering, so it
  is a drop-in source for a future preprocessing pass. This is what unblocks every timestamp
  feature in `optimization/FUTURE_FEATURES.md`; full detail in the Ops section's DATA FACT bullets.
  Also read-only. Staging copy: `anki-revlogs-10k-id-raw`.
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

- **★ STANDING AUTHORIZATION: RUN EXPERIMENTS WITHOUT ASKING (Andrew 2026-08-15, verbatim:
  *"you don't have to ask for permission to run more experiments"*).** Launching the next iteration
  off the ranked queue -- writing its runner, training, evaluating, logging the verdict, and moving
  to the next -- is ORDINARY WORK, not a decision needing sign-off. Do not end a turn with "say the
  word and I'll launch"; launch it and report the result. This is what [[work-autonomously]] already
  said and the loop had drifted from.
  **What still needs Andrew:** (a) anything that changes the DEPLOY CONTRACT or state-size budget
  (e.g. spending bits on the +1 norm/index bit), (b) the ~4-day 10x endgame run, (c) deleting the
  LMDBs / starting the features rebuild, (d) anything that breaks a stated constraint of his
  ("keep the current quantization recipe", the two hard invariants), (e) a genuine fork in research
  DIRECTION where the queue is empty or exhausted. Cost alone does not require asking -- a 5.5 h or
  13 h queued iteration is routine.

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
- **`rwkv/`** — the vendored+evolved package: **`id_features.py`** (NEW 2026-08-16 — `RWKV_ID_FEATURES=1`,
  default OFF and structurally inert: the 21 real-timestamp columns, the measured normalization
  constants, the negative-gap clamp, and `input_width()`, which `srs_model.py` + `srs_model_rnn.py`
  now BOTH call instead of each hardcoding `card_features_dim = 92`); `deck_tree.py` (NEW 2026-08-16 — `RWKV_DECK_TREE=L`: the
  parent-map loader, the ancestor walk, and the shared `build_module_data` grouping helper that
  `prepare_batch.insert_probes` now also calls); `architecture.py` (5-stream config + env hooks +
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
  `FUTURE_FEATURES.md` · `LIT_REVIEW.md` · **`DATASETS.md`** (which review dataset to train on:
  the four sets on this machine, the FSRS-Anki-20k VERDICT = do NOT train on it (disk 1.5 TB,
  no note/deck/preset, 4.3% leakage), the card-id-is-not-a-fingerprint lesson, the
  augmentation-off/byte-identical-epochs finding, and the 2-line users-vs-epochs ablation that
  settles data-limited-or-not) · **`CPU_INFERENCE.md`** (the deploy-speed
  scoreboard: why param cuts have NOT yet bought CPU rev/s, and the Rust port that gates
  the real answer; bench `cpu_infer_bench.py`) · `PROTOCOL.md` (iter0-era mirror of §11) ·
  **`PROPOSALS.md`** (the RANKED research queue + the 3-agent generation protocol + the standing
  constraints a proposal must satisfy; write rankings here immediately -- a compaction ate the
  2026-08-10 list's tail) ·
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
  reproducible by current Python? run this FIRST when a gate looks wrong — ⚠ it honours
  `RWKV_REF_DIR` only since the 2026-07-27 fix; before that it hardcoded `reference/`),
  `buttons_py_vs_rust.py` (the 4 button intervals, Python vs Rust), `smoke_qat_jit.py` (CPU,
  seconds: proves QAT compiles as a ScriptModule, dispatches to the jit-ignored kernel, and
  matches eager bit-for-bit — i.e. `RWKV_NO_JIT=1` is not structurally required by QAT).
  `smoke_id_features_width.py` (2026-08-16 — the §9 three-way check for `RWKV_ID_FEATURES`, which
  CANNOT live in `parity_train_vs_rnn.py` because that harness is single-stack: asserts the training
  class, the deploy RNN class and `CARD_FEATURE_COLUMNS` agree on the width at BOTH 92 and 112).
  `smoke_deck_tree_rnn.py` (2026-08-16 — the deck tree in the RNN DEPLOY path: an all-inactive
  parent map must reproduce the tree-off forward exactly, a real one must not).
  **`deck_tree/`** (NEW 2026-08-15/16) = the deck-hierarchy lever's own tooling:
  `build_parent_maps.py` -> **`parent_maps.parquet` (TRACKED, 4.6 MB — a RUN DEPENDENCY of any
  deck-tree iteration; users 1-7500, 94.55% resolve, 0 cycles)**, `verify_lmdb_link.py` (do the
  LMDB's stored deck ids resolve? 49.21% of reviews can walk up >=1 level), `level_reach.py` (the
  per-level reach + depth histogram that set L), `shape_cost.py` (padded kernel volume per stream
  — ⚠ it does NOT count B, which is what the singleton blunder turned on), `smoke_inert.py` (the
  lever is byte-identical with the flag off), `smoke_tree.py` + `run_smoke_tree.cmd` (off / null /
  real; `parent_maps_null.parquet` is derived scratch, gitignored).
  **`id_features/`** (NEW 2026-08-16) = `smoke_id_features.py`, the inertness + leakage smoke for the
  `-id` feature rebuild (prefix invariance at 0.000e+00 is the one that catches a whole-table
  statistic). **`optimizer_regime/`** (NEW 2026-08-16) = `muon_gap_over_training.py` (Muon-vs-AdamW
  train gap by decile on the iter-29/iter-26 MATCHED pair) + `ns_steps_dose.py` (the NS step-count
  screen). **`iter50_decktree/`** = the finished run.
  **`eval_pava/`** = the rectified-eval pipeline +
  `check_imm_identical.py` (⚠ its premise is WRONG in bf16 — see below) + `decompose_duration.py`
  (splits the rect-vs-unrect ahead delta) + `run_mode3_noise.cmd` (the noise control).
  **PROBE-INSERTION NOISE (the one rule to carry):** probes are +4 rows per scored review, which re-buckets sequences by length and reorders bf16 reductions, so a rectified eval is NOT numerically comparable to an unrectified one -- **compare rect-to-rect only**. The magnitude is channel- and model-dependent and on `ahead` it is ZERO (iter 31, n=500, `RWKV_EVAL_PAVA=3`: +0.000000 +/- 0.000014, p=0.33, an exact coin flip), so **measure the control for the model in hand** (~30 min at n=500) rather than quoting a fixed number. Mode semantics: `m3-m0` = noise, `m2-m3` = the clean duration cost, `m1-m2` = pooling. Full measurements (A18's imm +0.000280 scaling with recurrence length, and why LogLoss convexity makes the bias one-signed): `HISTORY.md`.
  **`dataset_id/`** (2026-07-15/16, was MISSING from this map —
  Andrew flagged it 2026-07-26) = **the builder for the real-timestamp `anki-revlogs-10k-id`
  dataset**: `run_build_id.cmd` (download -> extract -> build) + `download_from_hf.py` +
  `extract_7z.py` + `build_parquet_id.py` (the real work; `build_parquet_upstream.py` is the
  anonymizing original it was adapted from) + `stats.proto`/`stats_pb2.py` (locally compiled
  protobuf) + `parent_id_probe{,2,3,4}.py` and `deck_depth_by_review.py` (the deck-tree evidence
  quoted in `FUTURE_FEATURES.md`). **`iter32_kd/`** = the distillation run + `check_dump.py`.
  **`iter33_dur/`** = the duration-fix run (`RWKV_AHEAD_PROBE_ONLY=1` + `PROBE_DENSITY=1.0`,
  MAX 16384, RECTIFIED eval). **`ds20k/`** (2026-07-27) = the FSRS-Anki-20k evaluation:
  `scan20k.py` (hand-decodes each user's protobuf header, 64 B instead of ~2.7 MB) +
  `overlap2.py` (review-timestamp fingerprint, the CORRECT one) + `overlap.py` (the card-id
  version, kept only as the counter-example — shared decks propagate card ids, so it reports
  64% overlap instead of 4.3%). Verdict + method: `optimization/DATASETS.md`.
  Untracked on disk: ckpts (`*.pth`), logs, mid-run cb snapshots
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

**Two hard INVARIANTS (never change):** (1) hierarchy `card→deck→note→preset→global` (5 chained
— ⚠ CORRECTED 2026-08-09, Andrew's catch: this doc said card→NOTE→DECK for weeks, but every
architecture file incl. the vendored original executes card→DECK→NOTE; the code is and always
was the ground truth (`RWKV_SUBMODULES` in config.py orders feature columns only, not
execution). Note the actual order is NOT monotone fine-to-coarse (notes ≈ 0.9× cards, decks
≈ 56/user) — a "true fine-to-coarse swap" is a legitimate cheap future candidate. —
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
the iter3 champion).
**Rust-parity invariant:** `verify_rust.py` must pass for the champion arch before "shipping". **⚠ IT DOES NOT RUN THE ENGINE** -- it scores `reference/rust_pred_*.json` left by an earlier manual run, so `RWKV_WEIGHTS` cannot affect its verdict. Correct procedure: run the binary **from the repo root** (it resolves `reference/trace_user_*` relative to CWD) -> it writes `preds/rust_pred_*.json` -> copy those into the reference dir -> `python verify_rust.py`.
**★ WHEN A PARITY GATE LOOKS WRONG, RUN `scratchpad/parity3/trace_selfcontained.py` FIRST.** It feeds a trace's own features back through the Python RNN at review 0 and checks the stored `py_pred` reproduces -- a stale reference is far likelier than a broken engine, and that is exactly what a red gate meant in July (the June trace predated the d=128 -> d=32 move, so nothing could reproduce it). **Fix by REGENERATING, not archaeologising:** `RWKV_REF_DIR=<newdir> python export_rnn_trace.py` lands a fresh trace beside the old one. ⚠ That tool itself had a bug until 2026-07-27 (it honoured `RWKV_CHAMP_CKPT` but hardcoded `reference/`), so distrust any verdict from before that date.
**STATUS: PARITY PASS, twice.** A18 (imm 0.000035 / ahead 0.000044) and the iter-31 champion (imm 0.000008 / ahead 0.000001, i.e. 62x and 500x inside the 0.0005 tolerance), against `reference_a18/` and `reference_iter31/` respectively -- both self-contained at exactly 0.000e+00. The iter-31 run was the first to exercise GRU N=3 and a real learned PAVA end-to-end; the engine loaded both from the checkpoint unaided. **★ THIRD PASS 2026-08-11 -- the INTERLEAVED champion is now certified too** (`reference_iter41/`, exported from `i41_d_10935.pth` with `RWKV_INTERLEAVE=1` + the `_cnd` arch; self-contained at 0.000e+00): imm 0.000000 / ahead 0.000000 on BOTH engines, max per-review 4.78e-06 (fast) / 1.25e-06 (candle), and the sequential path stayed BIT-IDENTICAL to iter-31's green preds through the refactor. So the old warning that "both traces are SEQUENTIAL" is discharged. ⚠ Front-loaded placement only; a future `RWKV_ILV_SPREAD` adoption would need a fresh port + trace.
Exact env strings and the full debugging narrative: `HISTORY.md` (archived 2026-08-10).

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

### HISTORICAL CHAMPION (d=32 era, fully superseded)
The H=2/K=16 / 193,724-param champion on the 1500-user data-variety recipe, its q72u deploy config and the two findings that came out of it are archived to `HISTORY.md` (2026-08-10). The two that still matter are stated where they are used: **DATA VARIETY BEATS REPETITION** (1 epoch on ~1500 varied users >> 15 epochs on 100) and the WKV kernel is **K-dynamic**. The frozen QAT deploy truth (`champion_5k.json`, 0.306629/0.277893 quant-aware, 9-byte card / 27-byte note at 256x compression) is unchanged and is what the endgame's quant arm re-bases against; its recipe and artifacts are in the DEPLOY block of `HISTORY.md`.

### ACCEPTANCE GATE (research phase) -- accept iff ALL hold (record binary accepted/rejected per iter):
1. "size" (equalized review count, 101-200) IDENTICAL to champion (data-integrity; any change = pipeline bug).
2. params <= **225,000**.   3. card AND note per-entity state UNCHANGED (deck/preset/global MAY grow freely).
4./5. **★ CURRENT RULE (Andrew 2026-08-10, TIGHTENED): each mode's RAW improvement vs the CURRENT
   champion must be >= 0.0001 in BOTH modes.** No rounding step -- a raw +0.000088 now FAILS.
   **WHY it moved:** the previous rule (>=0.0001 *after 4-dp rounding*, i.e. raw >= 0.00005) put
   the bar BELOW the measured noise floor. Iters 41/43/44 are three structurally different
   schedules at identical capacity, mutually indistinguishable at |delta| <= 7.5e-5 -- so a raw
   0.00005 bar could accept a difference the data cannot resolve. 0.0001 sits above that floor.
   History: >=0.0003 (original) -> raw >=0.00005 via 4-dp rounding (2026-07-19, first applied to
   iter 26) -> raw >=0.0001 (2026-08-10).
   ⚠ NO PAST ACCEPT IS INVALIDATED -- checked: the smallest accepted margins are iter 39
   (+0.000158/+0.000153) and iter 35 (+0.000153/+0.000271), both clear the new bar; iter 38 was
   already rejected for missing the OLD bar. And iter 36 stands as a directed accept regardless.
   ⚠ The floor is BUDGET-DEPENDENT: if research moves to the 1/3 training budget, the calibration's
   c41-vs-c43 null measures the noise floor THERE, and the bar should be re-derived from it rather
   than carried over.
6. **p-gate (Andrew 2026-07-08):** paired per-user one-sided Wilcoxon (candidate vs champion, same 5000
   eval users) gives **p < 0.0001 in BOTH modes** -- `python optimization/paired_pvalue.py` (zero GPU cost,
   reads the result jsonls; exit 0 = pass). Record both p-values in research_5k.md's `p-value` column.
   Applies to accuracy accepts only (SIZE/SPEED-exception accepts claim parity, not improvement -> exempt).
=> accept ONLY changes that improve BOTH modes (RAW >=0.0001 in each, Andrew 2026-08-10; was
raw >=0.00005 via 4-dp rounding, and >=0.0003 before that) AND pass the p-gate (a monotonic
champion).
[[research-acceptance-gate]]
**★ EXCEPTION -- CURVE-SIDE changes (Andrew 2026-08-12): "ahead better, imm not (statistically)
significantly worse."** For levers that touch ONLY the curve/ahead objective, requiring imm to also
improve by >=0.0001 demands a side effect the mechanism cannot produce. Verified for iter 46's
self-distillation: the teacher is `.detach()`ed so no gradient reaches the rating head, and the imm
objective is `p_loss` = cross-entropy on **`label_rating`**, which the lever never touches -- it
rewrites `label_y`, which feeds only `curve_loss` / `curve_raw_loss` / the PAVA probe target. imm can
therefore move ONLY through the shared trunk. (§9 already said this in general: "a curve-side change
moves only one of the two gate modes".)
⚠ **`label_y` DOES reach one imm-side term** -- `p_binary_loss`, srs_model.py:1128 -- but
`pbin_scale=0` in this recipe so it is skipped. **If `RWKV_PBIN_SCALE` is ever turned on, a
curve-side lever starts softening the imm objective too and this exception no longer applies.**
**THE RULE:** accept iff ahead improves by **raw >=0.0001 with p<0.0001**, AND imm is **not
significantly worse** = NOT (imm mean declines AND the one-sided paired Wilcoxon for "candidate
worse" gives p < 0.05). **Both halves of the harm test are load-bearing, and iter 44 is why:** its
imm mean moved -0.000001 (nominally worse) while the RANK test said candidate BETTER at p=1e-4 --
most users improved slightly, a few worsened a lot. A rank-only guard fails a magnitude-null
iteration; a mean-only guard fires on noise. Tool: `paired_pvalue.py --curve-side` (exit 0 = pass;
`--harm-alpha` tunes the 0.05). It also NOTES, without failing, an imm decline larger than the
7.5e-5 noise floor that the rank test did not call worse -- look before promoting.
**SCOPE -- use it only for curve-side levers** (self-distillation, PAVA lambda, ahead-target and
monotonicity changes, duration handling). Trunk / optimizer / capacity / topology changes keep the
BOTH-modes rule, because those genuinely can move both. Precedent: iter 36 (PAVA lambda 0.1->0.2)
was this exact shape and was directed-accepted on a 5.9:1 ahead-for-imm trade.
**EXCEPTION -- SIZE/SPEED changes** (e.g. H=2/K=16): judged on the **efficiency budget** instead -- accept if
both modes stay within **+0.0015** of the champion AND the change shrinks state and/or speeds training (it
Pareto-dominates at accuracy-parity). H=2/K=16 was accepted this way (halved card state, 1.16x faster, accuracy
within 0.0002). Such a change MAY shrink card/note state (gate #3 is for accuracy-research iters, not these).
Two HARD INVARIANTS (never change): hierarchy card->deck->note->preset->global (the CODE's order,
corrected 2026-08-09 -- the doc had note/deck swapped vs what every arch file executes); same preprocessed 92-dim
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
cb-export->eval wiring, queued).
**★ BOTH CODEBOOKS REPLACED 2026-08-12 -- THE q72u PAIR IS d=32-SHAPED AND MUST NOT BE USED ON THIS
TRUNK.** `pq_cb_shift_q72u.txt` is C=32 and hard-FAILS a shape assert at C=80; `pq_cb_wkv_q72u.txt`
stays dimensionally valid (K=16 either way) and so fails SILENTLY -- it measures **worse than random**
here (held-out 1.0107 vs 0.9576 for 1024 random directions, against the 1.0 encode-to-zero bound).
Refits: **`reference/pq_cb_wkv_c80_b10.txt`** (same header/1024 rows -> identical deploy state size;
error 1.0107 -> 0.3973) and **`reference/pq_cb_shift_c80_m2b12.txt`** (24 b/vector, same bits as
q72u). The 4-arm probe matrix priced the WKV swap at **+0.003235 ahead / +0.004183 imm of recovered
PTQ cost for zero extra bytes**, and showed the WKV side is ~14x the shift side (whole shift cost
+0.000365/+0.000720), so m2b12 is the deploy choice and quantization work belongs on the WKV half.
**CURRENT ENV:** `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
RWKV_QAT_PQ=reference/pq_cb_wkv_c80_b10.txt RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_c80_m2b12.txt
RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3 RWKV_QAT_NORM_BITS=1 RWKV_QAT_FUSED=1 RWKV_NO_JIT=1
**RWKV_QAT_PQ_LEARN=1 RWKV_QAT_SHIFT_PQ_LEARN=1**` (JIT on the grafted paths unverified -- A/B once at
champion-run launch).
**★ THE TWO LEARN FLAGS ARE ADOPTED AS DEFAULT (2026-08-13) -- put them in EVERY quant-aware run
including the endgame's arm 2.** They cut the measured QAT tax **45.4% / 43.9%** (+0.004185/+0.006219
-> **+0.002286/+0.003486**, n=2500) for **zero deploy bytes and zero wall-clock** (0.3333 steps/s,
identical to frozen catalogs -- the learning is kernel-side atomicAdd). Mechanism: most of the "tax"
is the catalog going stale *within the run* as QAT moves the weights, which a frozen catalog cannot
track. ⚠ Shape invariant enforced at upload: role-mode learnable catalogs need `m <= 4` (the kernel's
`rec_idx_chunk` slot stride is 8 ints); joint mode is `m == 1` and exempt.
⚠ **A fitted codebook is validated by SHAPE and used on CONTENT** -- re-fit whenever d_model, H, or
the state distribution changes, and check it against a random-catalog control (CPU, minutes:
`scratchpad/qat_tax/wkv_cb_staleness.py`). Nothing in the pipeline warns you.
(b) card+note state sizes FIXED, but deck/preset MAY grow
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

#### CHAMPION = iter 45 `iter45_kddecay` (KD kept through the DECAY phase, on the iter-41 recipe) -- promoted 2026-08-11 23:45
**RECTIFIED (the gate basis): ahead 0.297697 / imm 0.265375** on the VAL half (n=2500) =
+0.000192 / +0.000104 vs iter 41 at p=3.9e-47 / 1.4e-82. size 0/2500, nan_users 0, **558,212
params EXACT, card 2,880 / note 1,440 / deck 5,760 state floats ALL unchanged** (a schedule of
teacher signal has no weights), throughput 1833.5 rev/s (vs 1849.8 -- identical within noise, as a
training-only change must be). ckpt `scratchpad/iter45_kddecay/i45_d_10935.pth`;
`champion_5k_track2.json` points at it.
**THE LEVER, and it is ZERO CODE:** the runner does NOT clear `RWKV_KD_MIX`/`RWKV_KD_ALPHA` before
the decay phase; WS keeps the tuned `alpha=0.9`, **decay gets `alpha=0.5`**. Since iter 34 adopted
decay_ratio=1.0, decay is HALF of all training and had been running on pure hard labels.
**★ THE CHAMPION ENV = iter 41's, PLUS KD surviving into decay at 0.5.** So every run on this trunk
sets: `RWKV_INTERLEAVE=1`, `RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py`
(card→note→deck→preset→user, depths [2,1,4,3,3]), seed 4321, KD alpha 0.9 WS **and 0.5 through
decay**, PAVA lambda 0.2, tuned HPs, and the speed stack during training only.
**PERFECTLY CONTROLLED, verified not assumed:** the WS step trace is IDENTICAL to iter 41's for all
10,935 steps (`scratchpad/iter45_kddecay/extract_trace.py --compare`), so the whole gain is
attributable to the DECAY phase alone -- and run-to-run reproducibility at seed 4321 is re-confirmed
free.
**⚠ MARGIN:** imm clears the raw 0.0001 bar by only **4%** (+0.000104) -- ~1.4x iter 44's ±7.5e-5
same-capacity noise floor and inside the ~0.0004 cross-seed spread. Accepted because the written
gate passes both modes and single-run-at-4321 is the practice since iter 35; a second-seed
confirmation is still the rigorous move before leaning on this number.
**DEPLOY: nothing to port** -- training-only, no arch change. (And the interleave/order the recipe
depends on ARE now in `rust/rwkv-infer`, parity-verified 2026-08-11.)
**Open in-family, cheap (same dump):** alpha_decay 0.9 and 0.25 -- the WS curve peaked at 0.9 and
bracketed at 1.0, so decay's shape is not implied. Detail: `research_5k_verbose.md` iter 45.

#### THE CHAMPION LINEAGE (full blocks archived to `optimization/HISTORY.md` 2026-08-10 -- this table replaces ~150 lines of superseded champion detail)

| iter | what changed | rect ahead / imm (VAL half) |
|---|---|---|
| A18 | the d=80 trunk; width ladder closed at 4.95x smaller | 0.299302 / 0.268390 |
| 31 | + PAVA + GRU N=3 + Muon (3-change graft) | 0.298909 / 0.267637 |
| 32 | + full-run distillation from the d=128 teacher | 0.300268 / 0.267262 |
| 34 | + the MAX=65536 tuned recipe (HP tuning; the phase's largest gain) | 0.298970 / 0.266217 |
| 35 | + KD restored at seed 4321 (the seed pair) | 0.298816 / 0.265946 |
| 36 | + PAVA lambda 0.1 -> 0.2 (directed accept, a 5.9:1 trade) | 0.298338 / 0.266027 |
| 39 | + KD alpha 0.5 -> 0.9 | 0.298180 / 0.265875 |
| 41 | + interleaving (and a reorder that iter 43 later showed is free) | 0.297889 / 0.265479 |
| **45** | **+ KD kept through the decay phase (alpha 0.5), zero code** | **0.297697 / 0.265375** |

⚠ iters 32 and 34 are not directly comparable to their neighbours: the gate basis changed to the RECTIFIED metric at iter 33, and iter 34 changed the training budget. Per-iteration detail: `research_5k_verbose.md`. Full superseded champion blocks (env strings, ckpt paths, caveats): `HISTORY.md`.

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
   - **★ QAT CAN RUN JIT-ON — the CPU half is DONE and GREEN (2026-07-27).** `RWKV_NO_JIT=1` is
     **not structurally required** by quant-aware training. `scratchpad/parity3/smoke_qat_jit.py`
     (CPU, seconds, no GPU) shows, under the full q72u QAT env on the A18/iter-31 trunk: (1) the
     whole `SrsRWKV` **compiles as a ScriptModule** — PQ codebooks, low-rank scope and norm bits
     included; (2) scripted code **does dispatch** at runtime to the `@torch.jit.ignore`'d
     `quant_aware_rwkv7`; (3) eager and scripted give **bit-identical** CPU checksums
     (-394.3387028575). The belief that it could not was an untested ASSUMPTION written into
     `rwkv_ops.py:520-522` ("this path only runs under RWKV_NO_JIT") — that `jit.ignore` was added
     to un-break JIT for NON-QAT runs, and nobody re-asked the question for QAT itself.
     ⚠ **What is NOT yet shown, and needs a free GPU (~15 min):** bit-exactness of a real training
     step vs the NO_JIT path (the CUDA `qat_lr_rank1` kernels, not the CPU reference), and the
     actual steps/s A/B. Also note the dispatch test exercised the int4/rank-1 call variant, not
     the PQ-codebook variant — the PQ config was covered only by the compile half. Do both BEFORE
     the long run; worth **~1.38x, i.e. roughly 1.5 days off a 4-day run**.
   - **★ RESOLVED — ANDREW 2026-08-01: "we'll need to do the 10x budget run WITH AND WITHOUT QAT,
     to answer two questions: how much 10x-ing epochs reduces log loss, and how much QAT increases
     it."** So the 10x is TWO ARMS, and each arm IS one of the two measurements:
     * **Arm 1, PLAIN 10x** vs the current (1.25-ep) champion  ->  *what does 10x the epoch budget
       buy?*  This is where the +0.0037/+0.0043 vs upstream is supposed to live.
     * **Arm 2, QAT** vs arm 1  ->  *what does quantization-aware training cost?*  Reported as a
       clean delta at matched budget, which no previous number gives (plain-era and QAT-era
       loglosses have never been comparable).
     ⚠ **ONE THING STILL TO CONFIRM BEFORE LAUNCH, and it is a 2x cost difference:** arm 2 can be
     (A) the standard **warm-started ~2.0-ep QAT fine-tune on arm 1's final** — cheap, and it
     measures the QAT cost *as actually deployed*, since that is how QAT has always been applied
     here; or (B) a **second full 10x run with QAT active throughout**, warm-started from the
     current champion. **A is the recommendation** — it answers exactly the question asked, adds
     ~2 epochs instead of ~12.5, and B additionally risks the iter-40 lesson (`QAT from scratch =
     +0.0118`; it MUST warm-start). B is only worth it if the separate question "would QAT-aware
     training from the start be BETTER at 10x?" is also wanted. Put it to Andrew when the run is
     actually scheduled, not before — this is after the features rebuild.
   **★ WS:DECAY SPLIT DECIDED = 10+2 (Andrew asked Claude to decide, 2026-08-13).** ⚠ And it
   corrects a belief the record was carrying: **iter 34's `decay_ratio 0.25 -> 1.0` gain (+0.00145)
   is CONFOUNDED** — it also took total training 1.25 -> 2.0 epochs, and the log-linear budget curve
   explains +0.00084 of it. So "our tuning prefers a 1:1 ratio" is NOT matched-budget evidence; the
   ratio itself is worth at most ~+0.0006 from one confounded point. **QAT length does not constrain
   the split** (closure saturates by ~0.37 epochs, so even 1.5 ep of decay is 4x what QAT needs; the
   apparent coupling is only that the runners enable QAT for exactly the decay phase). Decided on
   COST, since arm 2 scales with decay length at the measured 3.76x: 6+6 = 89 h, 8+4 = 71 h,
   **10+2 = 53 h**, 11+1 = 44 h. 10+2 is 36 h cheaper than 6+6 for a benefit we cannot demonstrate,
   is standard WSD practice, and is a known-good upstream point. The +0.0006 is being SPENT, not
   disproven — the de-confounding test (1+1 vs 1.6+0.4 at a FIXED 2-ep budget, ~10 h) is Andrew's
   call. Full reasoning: `research_5k_notes.md`.
   **Three things that are tuned for 1 epoch and must be RECONSIDERED at 12.5 — write the answers
   down before launching:** (a) **warmup 200 steps** is 0.9% of a 1-ep run but 0.09% of this one;
   upstream used 20,000. (b) **augmentation: DECIDED — IT STAYS OFF (Andrew 2026-08-16: "screw
   augmentation (at least that particular kind)").** This item previously recommended turning it ON
   for the 10x run; that recommendation is WITHDRAWN, not merely qualified. `RWKV_AUGMENT_SEED=1234`
   stays. Scope of the decision: the per-batch **random ID codes + random cycle phase** specifically
   — Andrew's parenthetical leaves other regularizers open, so do not read this as closing
   regularization as a family.
   Three things made it a bad trade even before the preference: it carries ~0.0024 run-to-run
   variance against a 0.0001 gate (so it could never be validated by an ordinary iteration), it is
   structurally incompatible with KD-from-dump (see below), and dropping KD to buy it would forfeit
   the ~0.0019 that iters 32/35/39/45 banked.
   **★ CONFIRMED IN CODE 2026-07-27, and it is stronger than "repetition": epochs are BYTE-IDENTICAL
   replays.** `prepare()` calls `torch.manual_seed(seed)` per batch (`prepare_batch.py:210-211`) and
   `prepare_data_train_test` passes the SAME constant every batch (`:655`) — so the two augmentations
   (per-batch random ID codes, per-batch random cycle phase) are drawn identically in every epoch;
   only dropout differs. **=> the `champ5k_b1` budget A/B that fixed WS at 1 epoch ("2nd epoch adds
   nothing", ahead -0.00006 p=0.31) was measured in the one configuration where extra epochs CANNOT
   help. It says "more IDENTICAL epochs don't help" — never quote it as "more epochs don't help".**
   ⚠ The obvious worry — "with augmentation off, does 10x epochs buy anything at all, or does it
   reproduce that null at 40x the cost?" — **is already answered by our own data, and the answer is
   that it buys plenty.** The 2026-08-11 budget calibration ran at 1/3 budget WITH AUGMENTATION OFF
   and measured a **3x-budget step worth +0.002**, which projects to **+0.0042 at 10x** against the
   +0.0040 upstream gap — corroborating the endgame premise to 4%. So byte-identical replays are not
   the blocker; what more epochs buy here is optimization steps under the WSD schedule (and dropout
   does still differ per epoch), not data variety. The `champ5k_b1` null remains correctly read as
   "a 2nd IDENTICAL epoch adds nothing at THAT budget", and it does not generalise to 12.5.
   **★★ BUT AUGMENTATION-ON AND KD-FROM-DUMP ARE MUTUALLY EXCLUSIVE (found 2026-08-16; the plan as
   written above would hit this SILENTLY).** `RWKV_AUGMENT_SEED=none` makes the fetch children
   UNSEEDED, so the ID-encoding and time-phase draws are not reproducible run to run. The KD dump
   stores the teacher's OUTPUT LOGITS plus `labels_sum` as its ONLY identity check -- and
   augmentation changes INPUTS, not labels. A KD run with augmentation on therefore PASSES the
   checksum while distilling toward teacher predictions computed on different inputs, and
   regenerating the dump cannot fix it (the next run draws differently again). **RESOLVED (Andrew 2026-08-16): option (c) --
   augmentation OFF, KD KEPT.** The alternatives were (a) augmentation ON with no KD, forfeiting the
   ~0.0019 iters 32/35/39/45 banked, and (b) a LIVE teacher forward per step instead of a dump --
   exact, but adds the teacher's forward to every step. Neither is needed now. This entry stays
   because the TRAP is still live for any future dump-based KD: the checksum proves LABEL alignment
   and says nothing about inputs, so ANY input-side change (not just augmentation) silently
   invalidates a dump while the checksum keeps passing. Same family as the QAT-inert bug: the checksum proves LABEL alignment and was being
   read as proving BATCH alignment.
   ⚠ Augmentation-on also carries **~0.0024 run-to-run variance against a 0.0001 accept gate** (24x
   the bar), so it can never be validated as an ordinary research iteration -- it is an endgame-only
   decision, and it will have to be taken on reasoning rather than on a measured A/B.
   (c) **wd/dropout** were tuned where
   overfitting was impossible; at 10x they are live levers. None of this is a reason to delay —
   just do not assume the 1-ep recipe transfers.

#### LIVE
**★★★ ANDREW 2026-08-17 — AT LEAST 10 MORE ALGORITHMIC ITERATIONS, AND THE FEATURES PHASE WAITS.**
*"It seems a bit too early to give up on algorithmic improvements, give it at least 10 more iters.
There is no way the current architecture and training are so optimal that no improvement is possible."*
This OVERRIDES the reading that closed families + a 0-for-6 run meant the loop was done. The features
code stays implemented-and-inert; the GPU goes back to algorithmic iterations. **Ranked queue of 5
mechanism-motivated candidates is in `optimization/PROPOSALS.md`** (Muon on the excluded LoRA
matrices; ensemble teacher; spacing-effect monotonicity; the fixed-budget WS:decay de-confound; hint
distillation). Re-rank after the cheap ones report.
**★ AND HIS SECOND POINT IS ALREADY SATISFIED — VERIFIED TWO WAYS, NOT QUOTED FROM THE DOC.**
*"make sure in the new dataset review duration is subtracted from review ID, so that review ID is
time when the user glanced at the card's front."* It is: `build_parquet_id.py:70` computes
`review_time = entry.id - entry.taken_millis` with `duration = entry.taken_millis`. Confirmed IN THE
DATA (user 333, 296,001 rows): SHOW times are monotone while ANSWER times (`review_time + duration`)
are NOT -- which can only happen if the stored column is the show time and durations vary.
**⚠ ONE CONSEQUENCE WORTH KNOWING, found while checking:** `elapsed_seconds` is a diff **in protobuf
order** (per-card blocks), and the frame is sorted by `review_time` only afterwards -- so the
show-time correction can reorder two adjacent reviews of one card and leave a NEGATIVE
`elapsed_seconds`. Example: a 56.3 s review's show time lands 45 s before its predecessor's. That is
the true origin of the NaN landmine (0.04% of rows on user 333), and it confirms the clamp-to-0
choice: the real gap is tens of seconds, not "unknown". Inherited from upstream's formula, so NOT
being changed unilaterally.
**⚠ REBUILD COST IS NOW UNCERTAIN, NOT CHEAPER -- do not quote either number.** A 2-point fit
(20 and 100 users, back to back) gives fixed 5.3 s + 84,666 rev/s => **1.8 h for train+test**,
against the recorded **~23 h**. But the two disagree ~8x **on the identical 20 users**, and I have
not identified why; my own cache-cold control was INVALID (I read the parquet to count reviews
*before* timing, warming the cache for both arms). The honest statement is that the rebuild's real
cost is unmeasured at scale -- 5 M reviews fit in page cache and 372 M do not. It gates nothing now
that the features phase is deferred.
**▶ THE FEATURES PHASE IS NOW CODE, NOT A PLAN (2026-08-16, CPU-only, zero GPU).** The endgame's
step 2 -- previously "scoped" -- is implemented behind **`RWKV_ID_FEATURES=1`, default OFF and
structurally inert**: `rwkv/id_features.py` + four hooks in `data_processing.py`. 21 real-timestamp
columns replace Anki's card-state column, width **92 -> 112**. Nothing changes for a run that does
not set the flag (verified: original 24-column list, width 92, all seven CPU parity cases still pass).
**WHY NOW:** the algorithmic loop is visibly running dry -- **0-for-6** since iter 45 (46, 47, 48, 49,
50, 51), with ahead-vs-imm CLOSED on mechanism, topology CLOSED, capacity 0/3, low-rank-reg CLOSED and
the optimizer remainder demoted. CLAUDE.md's own instruction was to "start scoping it BEFORE the
algorithmic loop runs dry, not after". ⚠ This is preparation only -- **starting the ~23 h rebuild is
still Andrew's call** and is step 1 of the FEATURES phase, not a preparatory step.
**★ THREE PLAN CORRECTIONS, each found by running it, full detail in `FUTURE_FEATURES.md`:**
(1) the documented "FIX (one line)" for the NaN landmine **does not work** -- clamping
`elapsed_seconds` to the -1 SENTINEL moves the NaN into `elapsed_seconds_cumulative` (a per-card
cumsum; a second sentinel cumulates to -2 and takes the same `log(negative)` branch). Measured on the
page's own index case, user 486. **Clamp to 0**: more faithful (the overlap is bounded by the review's
own duration, so the gap really is ~0) and self-limiting (`-1 + sum(nonneg) >= -1` by construction,
now asserted). Counterfactual: **4 of 60 stride-sampled train users (6.7%) would have NaN'd**.
(2) **NOT `elapsed_days`** -- `is_first_review` IS `elapsed_days == -1`, so clamping there re-labels a
mid-card review as a first review and poisons the label machinery. Assert instead.
(3) **The reference derivations in `feature_stats_id.py` LEAK** -- they count a user's WHOLE card
collection, correct for the marginals they were written for and wrong as a feature
(`creation_batch_1d` would reveal cards created later that day). Production counts are clipped at
`review_time`; **289 of user 1's 22,430 rows** would otherwise have leaked.
**★ AND A TWO-COPIES-OF-ONE-NUMBER BUG REMOVED:** `card_features_dim = 92` was hardcoded separately in
`srs_model.py` AND `srs_model_rnn.py`. Both now call `id_features.input_width()`; `RWKV_ZERO_FEATURES=22`
is REFUSED under the new layout rather than silently masking `day_of_week`.
**Smokes, both green:** `scratchpad/id_features/smoke_id_features.py` (60 users / 6.3 M rows, zero NaN,
plus **prefix invariance at exactly 0.000e+00** -- truncate a user's history and the surviving rows are
unchanged, which is what catches ANY accidental whole-table statistic) and
`scratchpad/parity3/smoke_id_features_width.py`.
⚠ **STILL OWED before the rebuild:** `smoke_scripted_eval.sh` (GPU-gated, waiting on QAT#2 -- mandatory
after touching `srs_model.py`), the 100-user de-risk build (ON-vs-OFF on `-id`; the `-id`-vs-published
comparison is INVALID since `size` moves for ~30% of users from the dataset swap alone), a rebuilt
`label_filter_db`, and the Rust input-width port.
**✗ ITER 51 FAILED 2026-08-16 20:31 -- NOT a reject, no number produced** (`iter51_muon`,
`RWKV_MUON_POLAR=1`: a per-step Polar-Express Newton-Schulz schedule replacing Muon's single fixed
triple). Died hollow -- 410 good steps, then `Nan from RWKV-7` on all 3,684 remaining batches. ~0.5 h
GPU, no eval, no champion impact. **Optimizer family stays 1/2** (a failed launch is not a rejected
iteration).
**★ THE MECHANISM INVERTS HOW THE RECORD DESCRIBED THE PRODUCTION CONSTANTS:** `a+b+c = 0.7010`,
so **p(1) = 0.70 < 1 -- the production triple CONTRACTS anything at or above 1**. That is a STABILITY
GUARANTEE, not the "deliberate sloppiness" I called it in the runner header. A (3,320) momentum
matrix of effective rank 1 has all its energy in sigma_max after Frobenius normalisation, so
sigma_max ~ 1 and bf16 rounding puts it at **1.0012** -- just past the fixed point. The fitted
schedule then ran 1.229 / 1.835 / 2.860 / 47.9 / **1.9e8**; production runs 0.696 / 1.118 / 0.726 /
1.082 / 0.701. **Accuracy at the top of the spectrum IS p(1)->1, so any revival must keep p(1)
strictly below 1 -- forfeiting accuracy exactly where the energy is.** A constrained refit (peak
capped at production's own 1.20) ALSO diverged (2.75e8), promoting the diagnosis to structural. The
flag now raises at import with this reason inline.
**★★ THE VALIDATION ERROR IS THE REUSABLE LESSON: I FITTED AND CHECKED ON THE WRONG DISTRIBUTION.**
Fitted on step-10935 momentum, deployed from step 1; early momentum is differently conditioned
(median sigma_min 2.8e-6 vs 6.6e-5 late) and produced an update **1.76e7x** baseline. Two compounding
mistakes: (1) a late-training checkpoint is NOT a sample of the training run; (2) I reassured myself
with a **MEDIAN** ||O||_F ratio (+2.6%) when the early-buffer **MAX** was 1.76e7. **A MEDIAN CANNOT
SEE A BLOW-UP; ONLY A MAX CAN -- report the max in every future numerical-stability check.**
**★★ AND THE FOLLOW-UP MEASUREMENT DEMOTES THE REST OF THE FAMILY (no GPU, from Andrew's own
recollection):** on the matched pair iter 29 (Muon) vs iter 26 (AdamW) -- the only difference is the
three `RWKV_MUON_*` vars -- the **TRAIN**-loss advantage decays over 6,554 paired steps from
+0.01446/+0.09809 to **-0.00058/+0.00097** (it INVERTS on ahead), while the **HELD-OUT** advantage
holds at **+0.001909/+0.001913**. **=> at our budget Muon is a REGULARIZER, not a faster optimizer.**
PolarExpress and NorMuon both refine the DESCENT, i.e. the half that has stopped paying, so **NorMuon
is demoted on mechanism** and any future optimizer proposal must name which half it attacks. Tool
`scratchpad/optimizer_regime/muon_gap_over_training.py`; detail `research_5k_notes.md`.
⚠ The obvious AdamW control (`champ5k_plain`) is the WRONG one -- it also differs in PAVA lambda and
the GRU head. Diff the runners, do not read the labels. Third instance of that shape after iter 47's
wrong checkpoint and iter 50's compile-warmup timing.
**✓ ITER 50 DONE 2026-08-16 19:20 -- REJECTED as an EXACT TIE** (deck tree, `RWKV_DECK_TREE=2`):
ahead **+0.000007 at p=0.52** (a literal coin flip) / imm **-0.000024 at p=0.86**, both 3-10x INSIDE
the +/-7.5e-5 floor. size 0/2500, nan_users 0, 558,292 params, 12.5 h.
**★ THE DIAGNOSTIC IS THE RESULT:** the zero-init level embedding TRAINED to L2=1.766 (max|.|=0.591)
vs a `features2card` row-L2 median of 0.829 -- ~2x a typical input-projection row. The model marked
the parent-deck level distinctly, USED it, and gained nothing (same shape as iter 48).
**★★ MECHANISM: the 5-stream hierarchy already BRACKETS that scope** -- `deck_id` pools per-deck below
it, `preset_id`/`user_id` pool more broadly above it -- so a parent-deck level is INTERPOLATION
between scopes the model already has, not new evidence.
**⚠ THIS DEMOTES L=3 RATHER THAN MOTIVATING IT:** deeper ancestors interpolate even CLOSER to
preset/user, so if the parent level (most distinct scope, widest reach at 49.21%) is a coin flip,
levels 3-4 are a worse bet. Do not run L=3 on "we only tested the shallow case"; it needs a new
argument, and it costs 95% VRAM.
**Original launch note (kept for the config):** Andrew's long-standing ask: `card->note->deck->preset->global`
becomes `card->note->(deck, depth_level)->preset->global`. The deck stream runs once per ancestor
level, grouping reviews by the deck's k-th ancestor, reusing the SAME module object -- depth is a
LOOP COUNT over the user's tree, not an arch constant. Chain
`card_id, note_id, deck_id, deck_id@1, preset_id, user_id`; **558,292 params** (+80, the level
embedding); 17 layer-steps. **NO LMDB REBUILD** -- `data_processing` drops `parent_id` but never
factorizes `deck_id`, so `scratchpad/deck_tree/parent_maps.parquet` (TRACKED; a run dependency)
applies to stored ids directly.
**Bypass is EXACT and verified bit-for-bit:** rows with no k-th ancestor are grouped by negated leaf
deck id, marked inactive, and never scattered back. An all-inactive map reproduces the tree-off
forward exactly in BOTH Python paths (`scratchpad/deck_tree/smoke_tree.py`,
`scratchpad/parity3/smoke_deck_tree_rnn.py`); a real map moves it.
**★ ONE REAL COSTING LESSON, AND ONE WRONG CONCLUSION I HAD TO RETRACT WITHIN THE HOUR.**
(1) **REAL -- the WKV state is per SEQUENCE, so a new stream's MEMORY lives on B, not rows.**
Inactive rows were singletons first ("T=1 is the cheapest thing the kernel can be handed" -- true of
the sequential axis, and irrelevant): 13,533 + 24,646 singleton sequences x 1280 floats x 4 layers =
**~780 MB of extra WKV state fp32**, before backward saves. Grouping them by negated leaf deck
instead gives 64 / 57 sequences and total state +1.4%. Padded volume (1.60x) and kernel launches
(42->74) BOTH looked fine; **neither counts B**, and I had written the high sequence count up as
*reassuring* ("parallelism ROSE"), which was the alarm read backwards.
**Cost every future "add a coarse stream" proposal on B as well as on rows.**
(2) **⚠ RETRACTED -- "low-B/high-T is the worst shape for this kernel" is NOT supported.** I measured
0.16 then 0.238 then 0.360 steps/s and built a mechanism story on the gap to the 1.31x shape
prediction. **All three windows were inside `torch.compile` warmup.** True steady state is
**0.664 steps/s = 1.35x**, i.e. the shape prediction was right all along and there is no anomaly to
explain. **The rule: never time a run against a steady-state baseline until compile warmup is
provably over** -- on this stack that is several hundred steps, not tens. Same family as iter 47's
three method failures (metric / trajectory / signal each held fixed).
**=> the L=3 -> L=2 re-scope now rests ONLY on memory**, which is the part that was never a timing
artifact: L=3 sits at 11.6-11.8 of 12.3 GB (95%), and this machine has a documented WDDM cliff and a
GPU co-tenant (Andrew's FSRS benchmark), so a 10 h run there is fragile. L=2 sits at 10.8 GB and
still puts a parent level on **49.21% of reviews**. Depth histogram PEAKS AT 4 (reach 49.2 / 38.3 /
31.2 / 20.9%), so L=3+ remains the better test in principle, worth the memory work if L=2 shows
signal. Detail + the L=3 pre-registration kept verbatim: `research_5k_verbose.md` iter 50.
**✓ ITER 49 DONE 2026-08-16 06:32 -- REJECTED** (`iter49_cmix`, restore the user/preset LAYER-0
channel mixers by dropping `user_id:0,preset_id:0` from `RWKV_STRIP_CMIX`; +26,070 params, +4.7%).
ahead 0.297630 = **+0.000067 at p=0.113** (INSIDE the +/-7.5e-5 floor -- a coin flip); imm 0.265288 =
**+0.000087 at p=5.3e-16** (real by rank, still under the 0.0001 bar). size 0/2500, nan_users 0.
Both-modes rule (a trunk capacity change can move either mode).
**★ THE FINDING: capacity at the general streams' ENTRY layer is not the bottleneck.** 4.7% more
params, placed exactly where the cmix ablations had removed the most, buys noise on ahead and a
sixth of the bar on imm. **capacity-at-5k goes 0/3** -- consistent with the 100-user era's
"DATA-limited, not capacity-limited", now confirmed at 5k on a 4.95x smaller trunk.
**✓ ITER 48 DONE 2026-08-15 20:48 -- REJECTED as an exact TIE** (`iter48_rcouple`,
`RWKV_RCOUPLE=1`: the curve logit R(t) added to the 4 rating logits via 4 zero-init coefficients,
before the softmax so `out_p_binary` is coupled; undetached). ahead +0.000009 (p=0.19), imm
+0.000013 (p=0.37) vs iter 45 -- both ~7x INSIDE the +/-7.5e-5 floor. size 0/2500, nan_users 0.
**★ THE DIAGNOSTIC, not the verdict, is the result: the coupling WAS learned and sign-correct**
(Again **-0.0138** -- higher retrievability lowers P(Again) -- then +0.0074/-0.0044/+0.0043) **but
negligible**: max shift 8*|w| = 0.110 vs a `p_linear` bias spread of 0.772, and only at the clamp.
So the model used R(t) and gained nothing => **the trunk already carries the retrievability
information the rating head needs; supplying it explicitly is redundant, not missing.**
**★★ WITH ITER 46 THIS CLOSES THE AHEAD-VS-IMM-GAP FAMILY (0/2).** Two structurally different ways
to move information between the heads both return exact nulls -- soft targets (iter 46, -0.000023/
+0.000016) and an architectural path (iter 48). **The 0.032411 gap is NOT an information-routing
deficiency**, exactly as `PROPOSALS.md` warned in advance ("an UPPER BOUND, not a target": the query
row sees the intervening reviews and the exact lag, the ahead row structurally cannot, and predicting
cold from history IS the task). Demonstrated twice now, not merely argued -- **do not propose a third
routing variant.** Still open: changing what the ahead path is FED (new input features), which
attacks information CONTENT rather than routing.
⚠ OPS -- **COMPILING IS NOT RUNNING, and it cost this iteration's eval.** The first eval died on a
TorchScript RUNTIME bug unrelated to the lever: `take_rank1_penalty()` (iter 47) had
`@torch.jit.ignore` with no return annotation, so TorchScript typed it `-> Tensor`; returning `None`
handed scripted code an UNDEFINED tensor and the next `float * Tensor` aborted with
`op.is_output INTERNAL ASSERT FAILED ... Found type undefined input tensor!` pointing at a builtin,
nowhere near the cause. The COMPILE half of the same bug was fixed a day earlier and reported as
done. **Training was intact; only the eval re-ran.** Guard: **`scratchpad/parity3/smoke_scripted_eval.sh`**
(~90 s, one user, refuses to run under `RWKV_NO_JIT=1`) -- run it before ANY launch touching
`srs_model.py` / `rwkv_model.py`. Note a PLAIN eval is the ONLY path that scripts the model, so this
bug class is invisible to training AND to QAT evals. Detail: `research_5k_verbose.md` iter 48.
**✓ ITER 47 DONE 2026-08-15 10:15 -- REJECTED** (`qtaxf_r1reg`, the rank-1-friendly regulariser
`RWKV_QAT_RANK1_REG=0.05`; single-variable vs `qtaxd_cblearn`, both quant-aware). ahead 0.300018 vs
0.299983 = **-0.000035** (inside the +/-7.5e-5 noise floor, a tie); imm 0.269041 vs 0.268861 =
**-0.000180** (outside it, a small REAL regression). size 0/2500, nan_users 0. Both-modes gate
(pre-registered; the curve-side exception does NOT apply -- this lever changes the WKV state, i.e.
the shared trunk).
**★★ THE FINDING IS WORTH FAR MORE THAN THE VERDICT: THE FLOOR MOVED 43%/75% AND THE LOSS DID NOT.**
Matched finals, same tool/users/entities: **card 0.3594 -> 0.2043 (-43.2%), note 0.2689 -> 0.0660
(-75.4%, MEDIAN 0.0152 = essentially exactly rank-1)**. So **the reconstruction ladder's ranking of
rank-1 truncation as the LARGEST term (53% card / 39% note) does NOT survive as a logloss ranking** --
the remaining QAT tax (+0.002286/+0.003486) lives in the CODEBOOK and NORM terms. Fourth
reconstruction-vs-logloss misprediction of the week, and the first about the term the ladder was
built to prioritise. **Do NOT attack the rank-1 term by any further route** (rank-2 states, softer or
scheduled lambda): the step-50 check proved the lever ENGAGES hard (-13%/-22% after fifty steps), so
dose is not the issue. **Family CLOSED.**
**★ ANDREW'S OBJECTION CONFIRMED, and the CONTROL arm is the strongest evidence:** with NO regulariser
the control drifts 0.3831 -> 0.3594 card and 0.3556 -> **0.2689** note (**-24.4%**) over the same
decay -- the model moves toward rank-1 unaided, as far on note as the regulariser managed in its
first 50 steps. Because a small REGRESSION appeared rather than an exact tie, the rank-1 constraint
costs the model slightly MORE than the reduced truncation damage repays. (The stronger claim
"rank-2+ components carry nothing" is NOT established and its fp32 disambiguation is blocked -- a
QAT-trained model is not a valid fp32 model when the quantisation is STRUCTURAL; see
`research_5k_notes.md`.)
**★ THREE METHOD FAILURES, ALL THE SAME SHAPE -- a difference is attributable only if the METRIC, the
TRAJECTORY and the SIGNAL are each held fixed.** (1) *Wrong metric:* the ladder's "0.4353 / 0.3049"
is not reproducible and disagrees in OPPOSITE directions per stream; the saved tool
`scratchpad/qat_tax/rank1_floor.py` (verified against explicit truncation to 2.4e-07) reads
0.3733 / 0.3729 on the same corpus -- comparing to the old value would have printed a ~0.06 FAKE
improvement next to a null. (2) *Wrong checkpoint:* the first run compared step-50 against iter45's
FINAL, differing by 10,885 training steps as well; it also spawned a confident side-claim about
training raising state rank that the matched control reversed. (3) *Wrong signal:* the penalty
climbing 0.025 -> 0.077 was read as the model surrendering structure and drove a WRONG prediction --
proxy and target had DECOUPLED via the blunt edge documented when the penalty was built (it ignores
the decay weighting). **A proxy that can diverge from its target is an ENGAGEMENT DETECTOR, not a
progress signal.**
⚠ Knock-on: card and note in fact have the SAME rank-1 floor (0.3733 vs 0.3729), so the old "visibly
different distributions (0.435 vs 0.305)" argument for per-stream catalogs is unsupported; that lever
stays closed on the disjoint-centroid evidence instead.
⚠ Ops: the ~10 h quant-aware eval is NORMAL (the control's was 10h18m) -- the "~2.9 h" in the queue
is a PLAIN eval. Detail: `research_5k_verbose.md` iter 47.
**✓ BUDGET CALIBRATION DONE 2026-08-11 14:01 -- VERDICT: gating STAYS at full budget; and screening is NOT worth it either (Andrew, follow-up): keep doing FULL RUNS.** Three arms, 15.3 h, `DONE_EXIT_0`. Measured short-budget noise floor (c41 vs c43, a pairing verified null at full budget): **ahead |delta| 9.0e-5, imm 3e-6** -- against the PRE-REGISTERED bar of 4.9e-5, imm passes and **ahead fails by ~1.8x**. Mechanism: on ahead the floor got 1.2x WORSE than full budget's 7.5e-5 while signal compressed to 65%, so signal-to-noise falls ~1.9x and the effective accept bar would become 1.84e-4 vs the 1.0e-4 we accept today -- i.e. short budget would silently make us 2x stricter and discard real candidates. **Screening was checked separately and REJECTED for our pool:** a short run is 54% of a full one (the eval is a fixed 2.9 h), so screening only breaks even at K>2.2 candidates -- and its PAIRWISE noise is 1.96e-4 in full-budget effect size, which leaves **6 of the last 10 iterations inside its noise, including two ACCEPTED champions (35, 39)**. It would pay only on batches with wide spread (new arch family, coarse HP grid), never on near-bar work. **Banked and reusable:** effects are SCALED not scrambled (~65% compression, both p<1e-33) and the **0.65 constant** converts a short delta to full-budget size; the **3x-budget step = +0.002** projects to +0.0042 at 10x vs the +0.0040 recorded upstream gap, corroborating the endgame premise to 4%. ⚠ All three arms were SCHEDULE changes, so this does NOT measure how short budget treats regularization or capacity. Detail: `research_5k_notes.md`.
**✓ ITER 45 DONE 2026-08-11 23:30 -- ACCEPTED, NEW CHAMPION** (KD through the decay phase; see the
champion block above). Distillation is now 4/4.
**✓ ITER 46 DONE 2026-08-12 09:28 -- REJECTED as a TIE** (privileged self-distillation imm->ahead,
beta 0.7): ahead -0.000023 / imm +0.000016, both inside the noise floor; imm was NOT significantly
worse (p_worse 0.986) so the curve-side gate failed purely on ahead.
**★ THE FINDING IS WORTH MORE THAN THE ITERATION: the 0.032 ahead-vs-imm gap is NOT transferable by
soft targets.** The teacher shares the trunk AND the forward pass -- it is a different head on the
same representation, not a different function like the d=128 teacher that made iters 32/35/39 work.
So the soft target only re-expresses what the student already computes. Closing that gap requires
changing what the ahead path COMPUTES or is FED, not what it is FIT to. Before the self-distillation
sub-family is deprioritized, the literature-supported variant is a teacher that is NOT the same
forward pass (past checkpoint / mean-teacher, or a different augmentation view).
Detail: `research_5k_verbose.md` iter 46.
**▶ THE QAT TAX WAS THE RIGHT TARGET, AND IT IS NOW 45% SMALLER (Andrew 2026-08-12, "let's re-measure
the QAT tax and work on reducing it").** Runner: `scratchpad/qat_tax/run_qat_tax.cmd`.
**WHY IT JUMPED THE QUEUE -- the stopping-point balance sheet** (`research_5k_notes.md`):
`still_needed = (champion - old model) - budget_credit + QAT_tax`. At the recent algorithmic rate
(+0.000112 ahead / +0.000057 imm per attempted iteration) every 0.001 of tax is ~9 ahead-iterations
or ~18 imm-iterations, i.e. days of GPU -- so the tax was worth more than the entire remaining loop,
and it had never been a research target only because plain-vs-QAT numbers were never comparable.
**THE LEDGER, in order:**

| when | QAT tax (ahead / imm) | still_needed (ahead / imm) |
|---|---|---|
| pre-measurement, using the d=32 placeholder | +0.00290 / +0.00445 | +0.00225 / +0.00196 |
| MEASURED on this trunk (n=2500) | +0.004185 / +0.006219 | +0.00360 / +0.00374 |
| **+ learnable catalogs (adopted)** | **+0.002286 / +0.003486** | **+0.00165 / +0.00100** |

So the placeholder understated the real tax by ~1.5x, and one adopted change then more than repaid
it: the remaining requirement fell from ~32 and ~66 iterations to **~15 and ~18**, and **imm stopped
being the binding mode**. ⚠ Neither the tax nor `still_needed` is a gate -- they are the
stopping-point estimate, i.e. how much further the loop must run before this model beats the old
d=128 one *as deployed*.
**★ THE MEASUREMENT IS ANDREW'S THREE CELLS (2026-08-12):** (1) no QAT, full precision = iter 45,
already have; (2) QAT-trained, evaluated QUANTIZED = the deploy number; (3) QAT-trained, evaluated
at FULL PRECISION = same checkpoint, QAT env off at eval. **(2)-(1) = the full tax. (2)-(3) =
PRECISION DEGRADATION** (what quantization costs a model trained for it). **(3)-(1) = MODEL DRIFT**
(what training under fake-quant costs by itself). The d=32 `qat_log` already carries this
decomposition and backs Andrew's recollection that drift dominates: warm-started decay-QAT #39 =
precision degradation -0.000127/+0.000018 (nothing) vs drift +0.001129/+0.002456.
Cells 2/3 are a SINGLE-VARIABLE A/B reusing iter 45's WS unchanged: re-run iter 45's DECAY from the
same WS-final with the q72u QAT env added and nothing else changed (KD stays alpha 0.5). A PTQ arm
(no training, 500 users) additionally gives the untrained-quantization cost, i.e. what the QAT
fine-tune recovers.
**★★ BLOCKER FOUND AND FIXED 2026-08-12 (`70185c7`) -- THE QAT ENV WAS SILENTLY INERT UNDER
`RWKV_ARCH_MODULE`.** `architecture.py` applied the `RWKV_QAT_*_SCOPE` vars to the DEFAULT config's
layers, then the arch-module override replaced `DEFAULT_ANKI_RWKV_CONFIG` wholesale and discarded
them -- **every track-2 run since A0 ignored the QAT env**, symptomless except for a zero
quantization cost (both PTQ arms returned +0.000001/-0.000000 with m2b12 == m5b12 EXACTLY). No
recorded number is invalidated (no track-2 iteration claimed quant-awareness), but methodology (a)'s
"quant-aware logloss" has been unsatisfiable on this trunk. Fix = `_apply_qat_scopes()` called LAST,
on the FINAL config. Verified: the same 10-user probe went +0.000001/-0.000000 -> **+0.009276 /
+0.012690**.
**⚠ THE LESSON, worth more than the bug: a banner proves a value was COMPUTED, never that it was
USED.** `[QAT-LOWRANK] set:` was truthful; the object it mutated was thrown away one line later.
**Guard added: `scratchpad/qat_tax/assert_qat_live.py`** imports the arch under the run's own env and
exits 44 unless every scope-named stream is really quantized in the FINAL config -- phase 0 of
`run_arm.cmd`, ~2 s, before any GPU. Any future env-driven setting should be gated the same way:
inspect the CONSUMED state, never the parsing log. (Also: eval banner guards must grep the SHARD
logs -- `eval_sharded`'s parent log never has them, which cost a spurious rc 41 on both arms.)
**★ THE LIVE HYPOTHESIS:** the +0.00290/+0.00445 on record came from `champ5k_b1`, which ran QAT
THROUGHOUT WS+decay -- not how QAT is deployed here. The qat_log's decay-only rows say placement
dominates: #39 (decay-only, warm-started) cost **-0.000127 ahead / +0.000018 imm, essentially
FREE**, while #40 (from scratch) cost +0.00534/+0.00446. So much of the "tax" may be WHEN QAT is
applied. If ARM A is bad and ARM B recovers it -> the fine-tune does the work; if both are bad ->
the q72u codebooks (learned on d=32 states, 5x smaller card state) are the suspect.
**BOTH suspicions were confirmed and BOTH are now fixed** -- the catalogs were stale (refit, above)
AND staleness kept accruing during the run (learnable catalogs, adopted). ⚠ The old note that
"`RWKV_QAT_PQ_LEARN` exists but its export->eval wiring does not" was WRONG: the wiring is complete
on both catalogs and was exercised end-to-end; that line cost a day of treating the lever as blocked.
⚠ Python maps QAT scopes BY NAME (verified: `card_id=rank1/fq7.0, note_id=rank1/fq7.0` under the
_cnd arch), so the Rust-only positional bug fixed 2026-08-11 does not affect these numbers.
**★★ SECOND BUG, FOUND THE SAME DAY AND BIGGER: THE WKV CODEBOOK IS WORSE THAN RANDOM ON THIS
TRUNK.** `reference/pq_cb_wkv_q72u.txt` (`1 10 32 16 1024`, joint-uv) was fitted on the **d=32/H=2**
model. K=16 is unchanged at d=80, so it stays DIMENSIONALLY valid and every assert passes -- the
shift catalog got refitted only because it hard-FAILED a shape check; this one fails **silently, by
being aimed at the wrong subspace**. Held-out mean relative L2 on this trunk's own card/note WKV
states: **OLD 1.0107 cross-user / 1.0026 random-split -- at or past the encode-everything-to-ZERO
bound (1.0), and worse than 1024 RANDOM directions (0.9576)**. Mechanism is subspace, not per-head:
the data's top-8 PCs carry 63.8% of the DATA's variance but only 22.6% of the OLD CATALOG's.
**✓ REFITTED: `reference/pq_cb_wkv_c80_b10.txt`** (`f3cc719`; ~3 min CPU from a 787 MB corpus =
47,700 states, 7 train-range users, via the Rust engine's `--dump-corpus`). **Byte-compatible
drop-in -- same header, same 1024 rows, so deploy state size is UNCHANGED**; error 1.0107 -> 0.3973
cross-user.
**=> THE RECORDED TAX MEASURES THE WRONG THING.** The +0.00290/+0.00445 on record and the
+0.009276/+0.012690 PTQ probe were both taken with this catalog, so they are substantially the cost
of DESTROYING the card/note WKV state, not of quantizing it. Re-measuring before refitting would
have spent ~11 h of GPU characterizing a broken config.
**NO HISTORICAL NUMBER IS INVALIDATED (checked):** every runner using q72u is d=32-era where it
matches its model; track-2 never produced a QAT number (the env was inert); `CPU_INFERENCE.md`'s
q72u figures are SPEED, and search cost depends on catalog size, not fidelity.
**WKV CAPACITY CURVE (cross-user, user 102 held out) -- my "flat curve" prediction was REFUTED:**
bits 8/10/12/14 = 0.4580 / **0.3776 (shipped)** / 0.3224 / 0.2844. The WKV joint-uv scheme is NOT
saturated, unlike the shift scheme (where 2.5x the bits bought ~9%). ⚠ **NOT FREE and NOT adopted:**
index bits are per head per layer, so +2 bits ~ **+1.25 B on the frozen 9 B/card budget (+14%)**.
**★★ AND THIS WHOLE CURVE IS RECONSTRUCTION-ONLY, WHICH 2026-08-15 SHOWED CANNOT EVEN RANK CATALOGS**
-- the learned catalog that cut the QAT tax 45% reconstructs WORSE than the frozen one it started
from (0.513 vs 0.334 on champion states), because centroids train on the TASK loss and nothing
optimizes them for reconstruction. So neither the bits ladder NOR the "free axis" (oracle 0.2044 vs
refit 0.3224 => fit on more users) has any demonstrated link to logloss. **Both are UNMOTIVATED
rather than refuted: do not spend an ~11 h run on either without a logloss A/B justified on its own
terms.** Same trap as the 2-bit NORM, which was PTQ-motivated at +0.0023/+0.0028 and measured a null
under learnable catalogs -- **DO NOT RE-PROPOSE THAT ONE; it is tested and closed** (Andrew caught it
being re-proposed 2026-08-15).
Shift catalogs available: **m2b12** (24 b/vector, same bits as q72u, ~23 B card, held-out
0.1902/0.1601) vs **m5b12** (60 b/vector, same bits PER DIM, ~37 B card, 0.1734/0.1465); on that side
capacity is chunk-limited, not bit-limited (at a fixed 24 b, m4b6 is 1.9x WORSE; 5x the bits barely
moves it).
⚠ Stale result jsonls MUST be deleted before any re-run -- `eval_sharded` skips completed users, so
old numbers get silently reused.

**★ FOUR NORM/CATALOG LEVERS ARE CLOSED ON MECHANISM, NOT ON NULL RESULTS -- do NOT rebuild them
(2026-08-13):** 2-bit norm, per-stream norm ranges, learnable norm levels, per-stream catalogs. Each
wanted to do a job the LEARNED catalog already does. Two measurements settle all four: (1) learned
centroids are **not unit-norm** -- they absorb magnitude in their length (spread widened **2.43x**),
so extra norm bits re-encode information already carried, and 1-bit norm is an INTERIOR optimum, not
a floor we were pinned against (this independently explains the d=32 sibling's 1-bit choice, reached
from the opposite direction); (2) card and note already occupy **disjoint** regions of the shared
catalog -- 501 vs 220 centroids with 25 shared (3.6%), Bhattacharyya 0.0219, **0.78% shared mass** --
so splitting the catalog per stream buys nothing, and with only 721/1024 centroids in use capacity is
not binding either. **The generalizable lesson: a learnable component silently absorbs the levers
around it, so measure what it has ALREADY absorbed before adding a lever beside it.**

**Iters 35-44 are COMPLETE and their narratives are archived** to `HISTORY.md` (2026-08-10). Verdicts: 35 seed pair ACCEPTED · 36 PAVA lambda DIRECTED-ACCEPTED · 37 by-user weighting REJECTED (mechanism refuted in every size quartile) · 38 KD alpha 0.75 rejected (missed by 2e-6) · 39 KD alpha 0.9 ACCEPTED · 40 alpha 1.0 rejected (brackets the peak; lever closed) · 41 interleave+reorder ACCEPTED (champion) · 42 order-only rejected · 43 interleave at the original order rejected as a TIE · 44 spread placement rejected as a TIE. Detail: `research_5k_verbose.md`.
**★ THE TOPOLOGY FINDING (iters 41-44 together), which supersedes the individual verdicts:** interleaving is worth +0.000216..+0.000611 in both modes, but THREE structurally different arrangements of it (stream order, layer placement) are mutually indistinguishable at |delta| <= 7.5e-5. So the EXISTENCE of a cross-scope information path is what pays; the choreography is not. The rearrangement sub-family is EXHAUSTED -- further topology work must change WHAT is computed (extra rounds via layer reuse, cross-stream fusion), not when. That ±7.5e-5 same-capacity spread is also the measurement that moved the accept bar to a raw 0.0001.

### ★★ THE DEPLOY CONTRACT -- ONE QUANTITY IN ALL THREE PATHS (Andrew, 2026-07-27)
> *"Everywhere (train+eval+CPU inference): duration of the most recent review zeroed out + PAVA + no piecewise correction. And yes, train with zeroing as iter 33."*

**The contract:** 1. the most recent review's duration zeroed, 2. PAVA rectification applied, 3. no piecewise ahead correction (`RWKV_NO_AHEAD_RESIDUAL=1`, in every run).
**The gate is therefore the RECTIFIED metric** (`RWKV_EVAL_PAVA=1`) from iter 33 on. Pre-iter-31 rows are unrectified and NOT comparable; do not retro-score them (the rect-vs-unrect delta is model-dependent: A18 +0.003588 vs iter 31 +0.001893).

**Three measured facts that still drive decisions:**
- **Training under PAVA halves the deploy rectification cost** -- A18, never trained under the constraint, pays +0.003588 on ahead; iter 31 pays +0.001893.
- **~70% of the deploy penalty is the lost current-row duration, ~30% is PAVA pooling** (+0.001451 vs +0.000611, from the mode-2/mode-3 decomposition). So PAVA lambda can only ever attack the smaller half.
- **Probe-insertion noise is channel- and model-dependent, and ZERO on ahead** (iter 31, n=500, `RWKV_EVAL_PAVA=3`: ahead +0.000000 +/- 0.000014, p=0.33). Measure the control for the model in hand rather than quoting a fixed magnitude; the "never compare rectified to unrectified at the 0.0001 gate" rule stands for **imm**.

**⚠ `RWKV_ZERO_FEATURES` WAS MISSING FROM `rust/rwkv-infer` -- FOUND AND FIXED 2026-07-27, AND IT WAS A LIVE BUG.** The mask lives inside the Python module, so the exported trace carries RAW features and the engine consumed columns Python had thrown away. With the mask applied, iter 31's max per-review |rust-python| falls from 1.59e-3 to **2.28e-6** (~700x). It survived the gate because the gate scores MEAN LogLoss and the column's weight is small. **This CORRECTS section 11's old explanation of the per-review spread as "accumulated float divergence" -- it was a formula error, and a large per-review spread should be read as a SIGNAL that the two paths compute different formulas.** Fix: `model.rs::load` zeroes the named input columns of `features2card.0.weight` once at load (zeroing input column j == zeroing feature j, since `y = Wx+b` is linear).
**→ OPEN RECOMMENDATION (Andrew's call):** bake the mask into the exported safetensors at export time instead of applying it from an env var at load, so the deploy artifact is correct for any consumer -- Anki will not be setting `RWKV_ZERO_FEATURES`.

Full narrative (the global-vs-surgical zeroing design, the iter-33 implementation fork, and the superseded gate-vs-deploy question): `HISTORY.md` (archived 2026-08-10).

#### QUEUE
**★★★ SPEEDUP PHASE CLOSED 2026-07-30 — ADOPTED STACK = 1.68x, and MAX=65536 IS ACCEPTED.**
Andrew 2026-07-30: *"Accept it, do compaction and then run the HP tuner."* Full measurements in
`optimization/TRAINING_SPEED.md`; the operative facts:

**THE STANDARD TRAINING ENV — put ALL of these in every new run `.cmd` from now on:**

    set RWKV_MUON_BATCHED=1     REM batched Newton-Schulz, 35x fewer matmul dispatches
    set RWKV_NO_JIT=1           REM required by torch.compile (worth ~0 alone: 1.003x)
    set RWKV_QAT_COMPILE=1      REM fuses the 26 mixer forwards

plus **`MAX_TRAIN_GLOBAL_LEN = 65536`** and **`NUM_FETCH_PROCESSES = 2`** in the toml.
Defaults stay OFF in code, so these must be set EXPLICITLY; old runs stay reproducible.
Result: WS 4h23m -> **2h37m**, decay 63 -> **40 min** (1.68x). Eval: use `--fetch-per-shard 2`.

⚠ **MAX=65536 COSTS ~0.0003 IN BOTH MODES at the OLD LR** (ahead -0.000264, imm -0.000307 vs
iter 31 rectified). That is a real-but-small systematic loss, not noise: both modes moved the
SAME direction, whereas the accuracy-neutral combo went +0.000064 / -0.000047 (one up, one down).
Mechanism: groups 22,346 -> **10,935**, i.e. HALF the optimizer steps per epoch at unchanged LR.
**Andrew accepted it ANYWAY and directed HP tuning to recover the 0.0003** — batch size is
structural and LR/warmup are tuned after it (methodology (f)). Do NOT treat the -0.0003 as
permanent; it is the tuner's target.

**▶ LIVE: THE HP TUNER IS REBUILT AND RUNNING** (launched 2026-07-30, detached pid 32352 via
`scratchpad/tuner65k/run_tuner_loop.cmd`; loop log `scratchpad/tuner65k/tuner_loop.log`, per-trial
`scratchpad/tuner65k/<name>.log`). Target = recover the -0.0003 that MAX=65536 cost.
`optimization/hp_tuner_5k.py` was rewritten wholesale (the old one targeted d=32 H=2/K=16,
MAX=110000, QUANT-AWARE, WS 2 epochs, eval 101-200 — every one of those wrong). What it does now:

  * recipe = **`scratchpad/maxval/run_maxval.cmd` with the HPs swapped** — the d=80 A18 trunk env,
    PLAIN (no QAT), WS 1 epoch, MAX=65536, NUM_FETCH_PROCESSES=2, the three speed flags during
    training and **cleared before eval**, RECTIFIED eval (`RWKV_EVAL_PAVA=1`) on **5001-6000**.
  * **★ THE BASELINE COST ZERO GPU:** `maxval` IS the default config, and restricting its existing
    rectified jsonls to 5001-6000 gives **ahead 0.299250 / imm 0.266335**, seeded into the journal.
    That subset also RANKS maxval-vs-iter-31 the same way the full VAL half does (+0.000113/+0.000309
    vs +0.000264/+0.000306), so it is a usable proxy — unlike the 200-user one that inverted.
  * **LEVER ORDER LEADS WITH THE LEARNING RATES**, because that is what the batch change implicates.
    Lever 1 is a joint **`lr_mult`** [1.0, 1.41, 2.0, 2.8] scaling **BOTH** `PEAK_LR` (1e-3, the
    AdamW group = 57,412 params) **and `RWKV_MUON_LR`** (0.02, the Muon groups = 500,800 params).
    ⚠ Tuning `peak_lr` alone would have moved only ~10% of the weights — Muon has its own base LR
    and the schedulers scale it proportionally (`train_rwkv.py:188-196`). Then `warmup_steps`
    [200,400,800], `muon_lr_mult` [1.0,0.5,2.0] (re-balance Muon vs AdamW after the joint move),
    `weight_decay` [0.01,0.05,0.1], `clip` [0.25,0.5], `decay_ratio` [0.25,0.4].
  * **11 non-default points x ~4.0 h = ~44 h** if nothing prunes — MEASURED on trial 1, not
    projected: **1.253 steps/s** steady state (5-min window past compile warmup), so WS 10,935
    steps = 2.42 h, decay 2,733 = 0.61 h, rectified eval on 1000 users ~1.0 h. Trials are named
    `t65_*`, trial dir `scratchpad/tuner65k/`, journal `optimization/tuner_5k_log.jsonl` (the old rows were archived to
    `tuner_5k_log_d32qat_era.jsonl` — different arch AND batch, not comparable).
  * **Val-based early pruning is ON** against **`optimization/tuner65k_vprune_ref.json`** (built from
    maxval's own val trajectory + its 5001-6000 finals = a matched reference on this exact trunk and
    batch). `RWKV_VPRUNE_MIN_STEP = max(1000, 2 x the trial's warmup)` so a long-warmup trial is not
    killed for being slow by construction. It matters most for the LR grid, where 2.8x can diverge.
  * **Three guards worth keeping in any future runner:** (1) a 40-step sanity phase that greps the
    sanity log for BOTH `BATCHED Newton-Schulz` and `[compile] torch.compile` — an env typo that
    silently disables a speed flag would cost ~2 h *per trial* across 11 trials; (2) stale-result
    deletion happens in **Python at trial-generation time, not in the `.cmd`**, so the `.cmd`'s
    **three eval attempts with NO `del` between them** keep `eval_sharded`'s resume property for the
    giant-user OOM; (3) the WS exit-code guard, because `write_decay_setup` takes the LATEST ckpt and
    would silently decay+evaluate a half-trained one.
  * **★ THE BAR, STATED CONCRETELY so trials are judged not eyeballed:** "recover what MAX=65536
    cost" means reaching **iter 31's numbers ON THE SAME 1000-user subset = ahead 0.299137 /
    imm 0.266026**. Against the seeded baseline (0.299250/0.266335) that is **+0.000113 ahead and
    +0.000309 imm** — note the two are NOT equal, because MAX hurt imm ~2.7x more than ahead on
    this subset. Anything beyond that bar is net new gain on top of the 1.68x speedup.
  * A sub-0.001 winner still needs **confirming on the full VAL half (5001-7500)** before it becomes
    the recipe — the subset is a ranking proxy, not a gate.

### ★ THE ORDER FROM HERE
The 2026-08-01 ordering (finish HP tuning -> seed pair -> PAVA lambda) is **COMPLETE** -- those became iters 34, 35 and 36. The ~340-line queue that tracked it, including the speedup phase's ranked list and the per-item DONE annotations, is archived to `HISTORY.md` (2026-08-10). What remains live:

1. **The algorithmic loop** (the endgame's step 1) -- **the ranked proposal queue now lives in `optimization/PROPOSALS.md`**, along with Andrew's 3-agent generation protocol (three subagents with DIFFERENT priors -- literature / domain / reject-log steelman -- each write 5 proposals from >=2 families; rank all 15; implement the top). ⚠ **WRITE THE RANKED LIST TO THAT FILE THE MOMENT IT IS PRODUCED:** the 2026-08-10 ranking lived only in the transcript and a compaction destroyed items 7-15 permanently.
2. **NEW INPUT FEATURES -- the long-lead item. ⚠ ONLY THE PREPROCESSING IS CPU-ONLY (corrected by Andrew 2026-08-12: "Pre-processing is CPU-only, sure, but training is obviously not").** This line used to claim features "do not compete with the GPU loop" -- WRONG, and it would have led to planning them as a free parallel track. Only the ~2-4 day LMDB rebuild overlaps the loop. Everything that makes features *count* -- re-basing the champion on the new inputs, then training + evaluating each candidate -- is GPU work on the same single 4070, and every pre-rebuild iteration is gated against a champion the rebuild invalidates. Features are a PHASE that largely DISPLACES the algorithmic loop, not a parallel one. Fully scoped in `optimization/FUTURE_FEATURES.md`: the four code sites, the F:-side-by-side disk plan (605 GB against 889 GB free -- no delete needed), the measured constants, the ~23 h build, the NaN-clamp landmine, and Andrew's directive that the rebuild DROP Anki's card-state input (dim 22). ⚠ It moves the `size` gate: the filter amplifies a 0.001% raw-row difference into ~30% of users getting a different equalized count, so gate #1 must be read as *within a rebuild generation*.
3. **THEN the 10x-budget run, ONCE, on the final champion** -- see THE ENDGAME above for the two arms (plain, then warm-started QAT), the ~4-day cost, and the three 1-epoch assumptions (warmup 200, augmentation off, wd/dropout) that must be reconsidered first.
4. **Rust port** (`rust/rwkv-infer/TRACK2_PORT_PLAN.md`) -- **★ GAPS 7 + 8 CLOSED AND PARITY-VERIFIED 2026-08-11** (`276f379`): both engines (candle + the default fast path) now run the interleaved schedule (`RWKV_INTERLEAVE=1`) and the reordered stream list (`RWKV_STREAM_ORDER=card,note,deck,preset,user`), against a fresh `reference_iter41/` trace that is self-contained at exactly 0.000e+00 -- interleaved PARITY PASS on both paths (max per-review 4.78e-06 / 1.25e-06) and the sequential path BIT-IDENTICAL to the green iter-31 preds on both. **Gap 8 was a LIVE cross-wiring bug the gate caught** (states were assembled positionally, so `_cnd` fed DECK's state into the NOTE module; `name_to_idx` likewise would have quantized card+DECK for `card,note` scopes). ⚠ Front-loaded placement only -- fine today (iter 44's spread was rejected), but a future spread adoption needs `interleave_schedule()`'s table, not `r < depth[m]`. Remaining measured items and the AGPL/SIMD note are in that plan.

**⚠ BIG-EVAL OPS RULE (learned 2026-07-29/30):** giant users (5002/5905/5995, 266k-367k reviews) OOM the 12 GB card **iff the DESKTOP holds several GB of VRAM** (4.6 GB during three failures vs ~0.5 GB when the same users cleared three evals overnight). `expandable_segments` does NOT help. **Never `del` the result jsonls between eval attempts** -- `eval_sharded` skips completed users, so a relaunch only re-risks the remainder. Check `nvidia-smi` before starting a big eval.

**⚠ CPU-INFERENCE REALITY CHECK:** in the PYTHON RNN path a 4.5x arithmetic cut buys only **1.24x** wall-clock and plateaus -- that path is overhead-bound, so cost tracks op count (layers x streams), not width. **1 thread beats 3 and 6 -> deploy single-threaded.** The Rust path DOES convert the cut: **2.39x** measured. Full numbers: `optimization/CPU_INFERENCE.md`.

#### FAMILY SCOREBOARD (conduct rule 5: 1-2 rejects = deprioritized, NOT closed)
**capacity-at-5k 0/3** (iter 49 added the user/preset L0 channel mixers back, +4.7% params, for +0.000067 ahead at p=0.11 and +0.000087 imm -- both under the bar). Three placements now agree: this model is not capacity-limited at 5k. Do not propose a fourth width/depth add without a mechanism argument that distinguishes it from these three.
**ahead-vs-imm-gap exploitation 0/2 -- CLOSED ON MECHANISM** (iters 46, 48). Both attempts to route
the better-conditioned imm signal into the ahead/rating path returned exact nulls, by structurally
DIFFERENT routes (soft targets; an architectural coupling), and iter 48 showed the coupling was
learned yet negligible -- the trunk already carries the information. The 0.032 gap is intrinsic
difficulty, not a routing deficiency. Attack CONTENT (new input features), never routing again.
**low-rank-friendly regularization 0/1 -- CLOSED ON MECHANISM, and the single reject is enough**
(iter 47). Conduct rule 5 normally forbids closing a family on one result; this is the exception it
allows for, because the iteration did not merely fail, it **measured that the target term has almost
no logloss in it**: the exact rank-1 truncation error fell 43% card / 75% note (note median 0.0152 =
essentially exactly rank-1) and the deployed loss did not improve. Any other route to the same term
-- rank-2 states, softer/scheduled lambda, a different proxy -- is aimed at the same empty term, and
the step-50 check already proved engagement is not the bottleneck. **The QAT tax lives in the CODEBOOK
and NORM terms.**
**TOPOLOGY 1/4 -- and iter 50 CLOSES the remaining direction.** Iters 41-44 showed that the
EXISTENCE of a cross-scope information path pays (interleaving) while its CHOREOGRAPHY does not
(three arrangements indistinguishable at |delta| <= 7.5e-5). **Iter 50 (the deck tree) shows that
adding more SCOPES does not pay either** -- an exact tie at p=0.52/0.86 with the level embedding
demonstrably learned. The scope ladder card->note->deck->preset->user is SUFFICIENT: the
productive lever was moving information between the levels that exist, and it is banked.
Do not propose a new intermediate scope without a mechanism that distinguishes it from a level
the ladder already brackets.
**TOPOLOGY (the 41-44 detail), and both rejects are CONTROLS that changed what we believe** (iter 41 ACCEPTED
— interleave + reorder bundle, the phase's largest architectural gain; iter 42 REJECTED —
order-alone is a small NEGATIVE, so INTERLEAVING carries all of it; iter 43 REJECTED AS A TIE —
interleave at the original order equals the champion (p=0.42/0.098), so the reorder's cost
VANISHES under interleaving. **ORDER lever CLOSED; SCHEDULE is the productive one** and the
2×2 is complete) ·
**distillation 4/5** (external-teacher sub-family 4/4; SELF-distillation 0/1 -- iter 46, a clean null that explains why: a same-forward-pass teacher carries no independent information) (iter 32 ACCEPTED, the d=128 teacher; iter 35
the seed pair; iter 39 alpha 0.9; **iter 45 KD through DECAY, the current champion** — teacher signal
pays in BOTH phases, so the "anneal onto the true objective" intuition is wrong here. Open and cheap
on the same dump: alpha_decay 0.9 / 0.25. ⚠ iter 10 was mis-filed under early-training-intervention,
which is why this family once read as absent) ·
curve-shape constraints **2/3** (PAVA ACCEPTED iter 23; lambda=0.2 DIRECTED-ACCEPTED iter 36 on a
5.9:1 ahead-for-imm trade; lambda=0.3 rejected as the worse point of the same lever) ·
objective-alignment **0/1 mechanism-refuted** (iter 37 by-user weighting: worse in every size
quartile incl. its intended beneficiaries — do not retry milder doses) · **optimizer 1/2, and the
REMAINDER IS DEMOTED ON MECHANISM** (Muon ACCEPTED iter 29, the phase's largest imm gain; cautious wd
REJECTED iter 30 — a pure trade; iter 51 PolarExpress FAILED structurally, p(1)<1 is load-bearing.
**Muon is a REGULARIZER here** — its train-loss edge decays to −0.00058/+0.00097 while eval holds at
+0.0019 — so descent-quality refinements incl. NorMuon target the half that stopped paying) · GRU-head N-sweep **peaks at
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
- ⚠ **WAITLOOP TRAP, cost one wrongly-started co-tenant eval (2026-07-26):** `findstr /C:"DONE_EXIT"`
  matches a log line that merely MENTIONS the token — including the waiter's own
  `=== WAIT for ... DONE_EXIT ===` message — so the loop fires instantly. **Anchor it:
  `findstr /B /C:"DONE_EXIT_"`** (terminal lines start with the token; prose never does) and do not
  write the token in non-terminal log lines. This is distinct from the known
  `DONE_EXIT_WSFAIL`-satisfies-the-grep gotcha.
- ⚠ **`detach.ps1` needs an ABSOLUTE path.** `Win32_Process.Create` starts in System32, so a
  relative script path exits instantly, silently, and still returns a pid.
- ⚠ **NO `< > & | ^` IN `REM` COMMENTS** — cmd.exe processes REDIRECTION *before* it honours `REM`,
  so a comment containing an arrow (`->`) or a usage line with placeholder brackets is parsed as a
  redirect. Symptom is baffling and points nowhere near the comment: `'M' is not recognized as an
  internal or external command` (cmd resumes mid-`REM`) plus `< was unexpected at this time`. Cost
  one dead launch 2026-08-14. Write `CKPT_PATH LABEL`, not bracketed placeholders, and `==` not
  `->`. Same family as the backslash-in-generated-content and BOM traps: **content destined for a
  shell needs escaping discipline even when it is "just a comment".**

### Ops
- **Compaction (ONLY sanctioned way):** run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
  then yield idle and STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) +
  focus-bearing flag (stale/empty = purged). Never hand-create `pending_compact.txt`. The injector is 24/7
  (ClaudeLoopController every 3 min; acts only on a stale heartbeat) and may inject EXACTLY `/compact <focus>`
  or a short `Continue` -- nothing else Claude-originated. (Since 2026-07-03 the **Telegram bridge**
  (`claude-automation/telegram_bridge.py`, task `ClaudeTelegramBridge`) additionally injects messages
  AUTHORED BY ANDREW from his authenticated Telegram account + mirrors chat output to his phone -- human
  steering, not self-injection. Master switch `telegram_bridge_active.txt`; see automation README.)
- **★★ NEVER TOUCH A RUNNING `.cmd` -- AND `git checkout` IS NOT A SAFE UNDO (cost iters 43 AND 46).**
  cmd.exe re-reads a batch file from a saved BYTE OFFSET every time a command returns, so any edit that
  shifts bytes past that offset makes it resume mid-garbage. Three things follow, learned the expensive way:
  (1) A chain's LATER phases are new processes that import whatever is on disk THEN -- so editing
  `rwkv/*.py` mid-chain silently changes the next phase too (found during iter 45; mitigate by gating new
  code on its env flag so it is inert when unset).
  (2) **Reverting an accidental edit with `git checkout --` DOES NOT RESTORE THE BYTES.** git normalizes
  line endings: a runner written LF (python `newline='\n'`) comes back CRLF, +1 byte per line. Iter 46's
  runner grew 222 bytes that way; cmd.exe resumed at the wrong offset, re-ran a fragment that re-opened the
  SAME ws log with `>`, and TRUNCATED the training log to 44 bytes. The WS checkpoint survived, so only the
  chain and the log were lost -- recovered with a phase-2 runner, same as iter 43.
  ⚠ The tell was visible and dismissed: `md5sum` of the file vs `git show HEAD:` differed, and it was waved
  off as "just line endings". It WAS just line endings, and that was exactly the failure.
  **If a running runner has already been touched, restore from a BYTE-EXACT copy (keep one before editing)
  or leave it alone and write a phase-2 runner. Never `git checkout` it.**
  (3) Annotate a running experiment in a SEPARATE file (e.g. `GATE.md`), never in its runner.
- **ESC-PROOF detached launches:** Esc / session teardown tree-kills Claude's Bash/PowerShell bg jobs INCLUDING
  training. Launch each training as a self-contained `.cmd` via `scratchpad/detach.ps1` (WMI Win32_Process ->
  parented to WmiPrvSE, survives); log to a STABLE repo path (`scratchpad/*.log`, NOT the rotating session
  temp); end the .cmd with `echo DONE_EXIT_%ERRORLEVEL%`. MONITOR via OS truth (poll the log / Get-Process /
  ckpt mtime) -- detached runs give NO tool-completion event. A Bash watcher gives notifications but is itself
  Esc-killable (re-arm it each turn; the training survives). Beat the heartbeat each working turn
  (`claude-automation/beat.ps1`). **Do NOT kill the FSRS benchmark PIDs (the ~80000s-CPU python procs).**
- **DATA FACT (SUPERSEDED 2026-07-26 -- read the next bullet before acting on it):** the PUBLISHED
  `anki-revlogs-10k` has NO absolute timestamp / review-id (anonymized; raw `revlogs` parquet = card_id,
  day_offset [integer DAY counter], rating, state, duration, elapsed_days, elapsed_seconds). Time-of-day is
  unrecoverable **from that set**. elapsed_seconds (time-since-last) is already an input.
- **★ THE REAL-TIMESTAMP DATASET EXISTS AND IS BUILT — `C:\Users\Andrew\anki-revlogs-10k-id`** (Andrew
  2026-07-26: *"we should have code for making it, so idk why CLAUDE.md doesn't mention it"* — it didn't;
  fixed). Built 2026-07-15/16 by **`scratchpad/dataset_id/`** (`run_build_id.cmd` -> `build_parquet_id.py`,
  adapted from the upstream anki-revlogs-dataset-builder), staging copy `anki-revlogs-10k-id-raw` (38.7 GB,
  keeps `revlogs.7z`). 15.8 GB; **10,000 user dirs in revlogs + decks, 9,934 in cards**; same layout and same
  1:1 user numbering as the published set, so results are comparable.
  - **IDs stay RAW Anki epoch-ms** = creation timestamps (`card_id`/`note_id`/`deck_id`/`parent_id`/
    `preset_id`), instead of upstream's per-user factorized small ints.
  - **`review_time` is CORRECTED to SHOW time** = `revlog.id - taken_millis` (the row is written on ANSWER),
    which is the right base for elapsed/time-of-day. Everything downstream (day_offset, elapsed_days,
    elapsed_seconds, sort order) is recomputed from it. Raw answer time = `review_time + duration`.
    ⚠ So day_offset can differ by one from the published set for reviews spanning the day rollover.
  - Spot-checked 2026-07-26 (user 1): `review_time` = 2021-05-22 15:31:47 UTC, `card_id` = 15:14:10 UTC —
    the card was created 17 min before its first review, i.e. "first review - card creation" reads directly.
  - **=> every HIGH-priority feature in `optimization/FUTURE_FEATURES.md` is derivable TODAY** (time-of-day
    + circular-mean deviation, true calendar phase, creation->first-review, seconds-resolution
    time-since-any-review, creation-batch size, tenure, note/deck/preset ages). No export is blocked.
  - **What IS still needed for them:** they are per-review FEATURE COLUMNS, so they need a preprocessing
    change + an **LMDB rebuild** sourced from `-id`. **Unlike the DECK TREE**, which needs NO rebuild at
    all — see the correction in `FUTURE_FEATURES.md`.
  - **★ THE DELETE IS PROBABLY UNNECESSARY — BUILD ON F: (measured 2026-07-27).** The "must delete
    first" conclusion assumed the rebuild lands on C:. It does not have to. `train_db_5k_h1` is a
    BARE RELATIVE path (`data_processing_train_5k_h1.toml:10`), i.e. repo root on C:, which is the
    only reason it is competing for C:'s 242 GB. Retarget it at F: and the new train (372.5 GB) +
    new test (232.8 GB) = **605 GB against F:'s 889.5 GB free** — both fit BESIDE the originals with
    ~284 GB spare, so a bad rebuild is `rm -rf` of the new dir instead of a 2-4 day re-run. Reclaim
    candidates if F: gets tight, both Andrew's call and neither needed to start: `train_db_5k_h2`
    (372.5 GB on F:, the swap half, referenced by NO live toml) and the closed-era `train_db_sc8k`
    + `train_db_sc8k_1500` + `test_db` (101 GB on C:). ⚠ **The TEST db must be rebuilt too** — eval
    feeds the same feature vector, so a train-only rebuild silently scores a mismatched layout.
    Full plan + the four code sites + the 100-user de-risk build: `optimization/FUTURE_FEATURES.md`
    "IMPLEMENTATION PLAN". Andrew's delete authorization stands as a fallback; prefer not to use it.
  - **DISK / DELETE-THE-OLD-DB — AUTHORIZED, WITH A SEQUENCING CONSTRAINT (Andrew 2026-07-26).**
    `train_db_5k_h1` is 372.5 GB and C: has 229 GB free, so a side-by-side rebuild does NOT fit ON C:. Andrew:
    *"We can delete the current copy, sure. It's strictly more data, not less, so nothing will be lost."*
    **Verified, and he is right:** published vs `-id` over 6 users (1/2/3/17/101/555, 363,598 reviews) —
    row counts IDENTICAL user-for-user, and `day_offset` differs on **4 of 363,598 reviews = 0.001%**
    (the show-time correction moving a review across a day rollover). So the rebuild is additive in
    columns and ~identical in rows.
    ⚠ **BUT DO NOT DELETE UNTIL THE REBUILD IS READY TO RUN.** The endgame order puts the algorithmic
    phase FIRST, and every run in it reads `train_db_5k_h1`; the rebuild is 2-4 days of CPU. Deleting
    early = a dead GPU and a killed run for zero gain. **The delete is step 1 of the FEATURES phase, not
    a preparatory step.** Do it when the preprocessing change is written and smoke-tested, not before.
    Three things to settle at that moment, none now: (1) re-run the champion on the new DB to re-base —
    at 0.001% it should be ~free, but cross-rebuild numbers are otherwise not comparable; (2) confirm
    the `size` gate still holds (row counts say yes, but the equalize filter is derived); (3) decide
    whether `label_filter_db` (37.3 GB, the "permanent deterministic cache") needs rebuilding too.
    `test_db_5k` (232.8 GB) lives on F: with 890 GB free, so it CAN be built side-by-side.
- Quant papers: `scratchpad/{rwkvquant,rwkvedge}.txt` (poppler installed; the Read tool handles PDFs). Use the
  CURRENT session's scratchpad dir for transient logs (it rotates on teardown -- check task-output paths).
