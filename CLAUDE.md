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

## 0a. Simplified Technical English -- how to WRITE to Andrew (2026-08-18)

**Write chat replies in ASD-STE100 Simplified Technical English.** The rules:

- Write short sentences. Use 20 words maximum for an instruction, 25 for a description.
- Write one idea in one sentence.
- Use the active voice. Use simple tenses.
- Use the same word for the same thing every time. Do not change words for variety.
- Do not use idioms, metaphors, or slang.
- Do not use a noun cluster of more than three words.
- Write six sentences maximum in a paragraph.
- Keep the articles "a" and "the". Put complex data in a list or a table.

**Scope: CHAT REPLIES ONLY.** This file, `research_5k_verbose.md`, `PROPOSALS.md`, the run notes and
the commit messages keep their current style. Those documents must carry mechanism, caveats and
numbers, and the density is deliberate. The [[claude-md-bloat]] problem is length, not sentence style.

**One limit, stated honestly:** STE also fixes an approved vocabulary of approximately 900 words,
one meaning per word. That list is licensed, so I cannot check words against it. I follow the writing
rules fully and only approximate the vocabulary rule. The standard permits technical names and
technical verbs, so `logloss`, `checkpoint`, `decay`, `quantize` and `Wilcoxon` stay.

**If a rule would drop a caveat, keep the caveat and shorten the sentence instead.** Accuracy first.

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
  **★★ TIGHTENED 2026-08-30, and this is the operative form: "not to stop unless there is a
  decision for *me* to make or unless the GPU/CPU is already busy and there is nothing to do but
  wait."** Exactly TWO reasons may end a turn without continuing: (1) a decision that is genuinely
  Andrew's -- the list below; (2) compute is busy AND nothing else can be advanced. **Both halves
  of (2) are required.** A busy GPU does not license stopping while CPU work exists, and it almost
  always does: the next runner, a CPU screen that could redirect a queued run, a guard, a smoke, a
  verdict to log, the record to update. Everything else is ordinary work -- do it and report it.
  Never end a turn with "say the word and I'll launch" or a menu of options; launch it. When
  something IS his call, ask it while continuing everything that does not depend on the answer.
  [[work-autonomously]]
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
  4. **★ THE INTERVAL ITSELF DIVERGES (found 2026-08-30, and it is live in the champion).**
     TRAIN and EVAL use the dataset's `elapsed_seconds` = `answer(k) - answer(k-1)`, i.e.
     **end-to-END**. A live Anki scheduler computes `now() - last_review_time` (jschoreels fork,
     `rust/rwkv.rs:322`), which is **end-to-START** and structurally cannot be anything else --
     `duration(k)` has not happened when the prediction is made. The two differ by exactly
     `duration(k)`. **It is sharper for us than for anyone else: `duration` is input feature 7 and
     the deploy contract already zeroes the most recent review's duration** for precisely this
     reason -- so we remove it from the features and hand it back inside the interval.
     Measured leak size: at a FIXED end-to-start gap, `duration(k)` still predicts failure at
     **AUC 0.618** against a shuffled-within-bin floor of 0.4996 (`duration_leak_probe.py`,
     2.18 M gaps); it moves the interval by >=10% on 11.1% of same-day rows and 0.00% of longer
     ones. Being measured on the arm that matches deploy: `scratchpad/features_ab/e2s`, control
     `featA2`. Full write-up `scratchpad/hybrid100k/INTERVAL_HANDOFF.md` sections 8-10.
     ⚠ **A cross-project lesson came with it: an interval change can move the REVIEW COUNT, and
     then the comparison is confounded before any model runs.** srs-benchmark's `delta_t > 0`
     (`features/base.py:284`) deleted 0.172% of reviews whose corrected gap floored to zero, and
     those rows were **2.7x easier than average** (6.09% failure vs 16.14%) -- deleting the easiest
     rows raises mean logloss by itself, and it was **two thirds** of the effect being reported
     (+0.000331 -> +0.000111 once sizes matched). Our pipeline is immune (no such filter; rows are
     kept and only marked via `label_is_equalize`) and it is **CHECKED, not assumed**:
     `scratchpad/features_rebuild/compare_db.py` asserts entry-count equality and is phase 0 of the
     arm. Both pass -- train 1,483,984 / test 170,384, identical in both arms.
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
  `FUTURE_FEATURES.md` · `LIT_REVIEW.md` · **`CONTENT_EMBEDDINGS.md`** (CONTINGENT, 2026-08-19 --
  if a dataset with CARD CONTENT ever appears: Andrew's decision is
  `paraphrase-multilingual-MiniLM-L12` at **int8, 118 MB, NO pruning**, on-device at all times
  because users edit cards on mobile. Carries why multilingual beat the AnkiHub/med skew concern,
  why pruning to ~29 MB was declined, that a LEARNED reduction is forbidden by Andrew's own
  anti-skew argument while a fixed random projection is not, and the +5.5% param cost of a 384-dim
  input to our 558k trunk. Nothing is scheduled) · **`DATASETS.md`** (which review dataset to train on:
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
  **`smoke_id_identity.py`** (2026-08-21 -- do TRAINING and DEPLOY agree on WHICH ROWS ARE THE SAME
  ENTITY? Compares the actual PARTITION, not merely entity counts, and **proves its own
  non-vacuity** by re-running against a simulated int32 store and requiring >=1 case to detect it;
  8 do, and they are exactly the two real bugs. Users are picked by NaN-metadata rate (0.0 / 66.8 /
  99.6%) -- a smoke sampling only user 1 would have passed on the broken build. ⚠ The simulation
  MUST use `torch.tensor(float64, dtype=int32)`, which SATURATES; `numpy.astype(int32)` WRAPS and is
  nearly injective, so a wrap-based simulation reports the guard vacuous everywhere. Getting that
  backwards was this file's first version.)
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
1. "size" (equalized review count) IDENTICAL to champion (data-integrity; any change = pipeline bug).
   **★★ THE GATE IS LINEAGE-SCOPED, AND THE LINEAGE IS DEFINED BY THE LABEL FILTER DB (Andrew
   2026-09-01: "with e2s we need to start counting size from a new baseline" -- measured 2026-09-02,
   and the trigger is NARROWER than the dataset generation).**
   **`size` IS the stored `label_is_equalize` count**, which comes from `LABEL_FILTER_LMDB_PATH`.
   Verified against the db itself: per-user equalized counts read out of `test_db_5k_e2s` match the
   `size` field in `RWKV-e2sc.jsonl` exactly (users 5001/5137/5613/6104/7499).
   **=> THE e2s SWITCH DID *NOT* MOVE `size`, so the published baseline carries across it
   UNBROKEN.** Four eval dbs spanning the whole published lineage -- `test_db_5k` (July), `_fix`
   (08-21), `_fixc` (08-31), `_e2s` (08-30) -- give **0 per-user mismatches out of 2,500 and an
   identical 128,800,080 total**. Mechanism: all four set `LABEL_FILTER_LMDB_PATH = label_filter_db`,
   and our pipeline has **no `delta_t > 0` filter** (srs-benchmark does, which is why the same
   interval change deleted 0.172% of *their* reviews; we keep every row and only mark it).
   So the gate still BRIDGES the e2s transition and can still catch a pipeline bug there -- more
   useful than a re-base would have left it.
   **What DOES move the baseline is swapping the LABEL FILTER, i.e. moving to `-id`**
   (`label_filter_db_id`). Measured: user 5001 scores 12,625 on `test_db_5k_id3` vs 12,615 on
   `test_db_5k_e2s`; 3 of 6 sampled users differ. That is why featB cannot be gated on `size`
   against featA2 and is not a champion candidate.
   **★ COROLLARY, and it is a free integrity check: gen 3 and gen 4 SHARE `label_filter_db_id`, so
   their sizes MUST be identical.** A difference is a build bug, not a dataset property. Wired as
   phase 3 of `run_rebuild4.cmd` via `scratchpad/features_rebuild/compare_equalize.py`
   (non-fatal -- by then the dbs are built and verified, so a mismatch is information for a human).
   **TOOLS:** `optimization/size_baseline.py snapshot|check <lineage> <result.jsonl>`, with
   `optimization/size_baseline_published.json` already snapshotted from e2sc (2,500 users,
   128,800,080). Both tools refuse to pass vacuously and both were proven able to FAIL: the size
   check catches a 1-review perturbation on 3 of 2,500 users; `compare_equalize` returns 1 across
   label filters. **Snapshot the `-id` baseline from featB when it lands** -- a lineage with no
   baseline must not be gated by treating its first candidate as the reference.
   **⚠⚠ CORRECTION 2026-09-01, and I told Andrew the wrong thing first: WE ALREADY HAVE
   `delta_t > 0`.** I wrote "we have no `delta_t > 0` filter", generalising this file's own line
   about the LMDB builder (`data_processing` genuinely has none -- every review row is stored) to
   the whole pipeline. But `find_equalize_test_reviews.py` calls **`create_features`**, i.e.
   srs-benchmark's own code, and `features/base.py:284` is `df[df["delta_t"] > 0]`. That is
   precisely why our `size` reproduces their published jsonls. **Two different filters, one name:
   the SCORED set has always honoured `delta_t > 0`; the STORED rows never did.**
   **THE REAL GAP, and it is narrower and live:** `find_equalize` reads the parquet DIRECTLY and
   never went through `get_rwkv_data`, so the filter was evaluated on **END-TO-END** gaps while
   training and eval had moved to end-to-start. With `SECS = true` that filter is **not**
   interval-independent (`delta_t := elapsed_seconds / 86400`, base.py:127/227), so WHICH rows
   floor to zero depends on the definition. Consequence: rows whose end-to-start gap is zero
   stayed in the scored set, so the model was scored on reviews srs-benchmark's own rule deletes.
   **FIXED (Andrew 2026-09-01: "We should have delta_t > 0 though, to make our methodology closer
   to that of srs-benchmark").** `find_equalize_test_reviews.py` now applies the SAME two
   functions `get_rwkv_data` calls, in the same order, gated on the DATASET -- not a second
   implementation, because the two datasets need different formulas (published subtracts THIS
   review's duration, `-id` the PREVIOUS one) and the wrong one is silently wrong.
   **MEASURED, 60 published eval users / 3,151,582 rows: 0.1907% of rows leave, and they are
   1.46x EASIER than average (7.07% vs 10.31% failure, -9.8 sigma)** -- so removing them RAISES
   mean LogLoss on its own. Consistent with Andrew's 0.172% on the full 10k in srs-benchmark.
   ⚠ **Small samples flip the sign of the difficulty claim**: 8 users said 1.24x easier, 24 users
   said 0.86x *harder*, 60 users says 1.46x easier at -9.8 sigma. Do not quote a direction from a
   handful of users; the probe now prints a sigma and refuses a verdict under 2.
   **★★ AND THE SCORED SET IS RE-SELECTED, NOT MERELY SHRUNK.** The folds come from
   `TimeSeriesSplit` over the SURVIVING rows, so dropping rows shifts every boundary and reviews
   ENTER the scored set too -- measured **-1437 / +87** on user 5402. I expected a subset and the
   smoke caught me. **=> a per-user LogLoss is computed over a DIFFERENT set of reviews, not a
   smaller one, so old numbers cannot be corrected proportionally; the lineage needs its own
   baseline.**
   **WHAT GEN 4 DOES (my call, delegated by Andrew: "I'll let you figure out what size number to
   consider 'correct'"):** gen 4 builds against a NEW **`label_filter_db_id_e2s`**
   (`rwkv/find_equalize_id_e2s.toml`, phase 0d of `run_rebuild4.cmd`). `label_filter_db` and
   `label_filter_db_id` are **NOT** touched -- featB is scored against the latter and rebuilding
   in place would silently re-base a finished experiment.
   **=> THE "CORRECT" SIZE FOR FUTURE new-features + e2s RUNS IS WHATEVER GEN 4 PRODUCES.**
   Snapshot it with `size_baseline.py snapshot id_e2s` from the first gen-4 eval, and gate every
   later candidate against that.
   **⚠⚠ AND THE CONSEQUENCE FOR THE CURRENT CHAMPION, WHICH SHOULD NOT BE LEFT IMPLIED: `e2sc`
   HAS THE SAME GAP.** It trains and evaluates on end-to-start intervals but is scored on an
   **end-to-end-selected** set, because `label_filter_db` is deliberately untouched. So the
   champion's 0.297888 / 0.265676 includes ~0.19% of reviews srs-benchmark's own rule would
   delete, and those reviews are 1.46x EASIER than average -- i.e. the published number is
   slightly FLATTERING as a leaderboard-comparable figure.
   It is not wrong as a GATE: every published-lineage run shares the same filter, so candidate
   comparisons are unaffected, and that is why this is not being fixed reflexively. Fixing it
   means rebuilding `label_filter_db` and re-basing the whole published lineage -- **Andrew's
   call, and only worth making if published-lineage work continues.** If the features are adopted
   (Andrew 2026-09-01: "we will almost certainly adopt timestamp features"), all future work is
   `-id` and the question retires with the lineage. Flagged so the endgame's honest-number step
   does not inherit it silently.
   **THE TRADE I ACCEPTED, stated rather than buried:** this forfeits the clean gen3-vs-gen4 Bug C
   measurement, because gen 4 now differs from gen 3 in two ways instead of one. It is unavoidable
   at acceptable cost -- `label_is_equalize` is **baked into the LMDB at build time**, so adopting
   the filter later means a second ~4 h rebuild. Methodology alignment was a directive; the Bug C
   number was a nice-to-have I had invented.
   Verified by EXECUTION, not by reading: the modified filter ran on users 5001/5002 and moved
   them **12,625 -> 12,595 (-0.238%)** and **229,050 -> 227,995 (-0.461%)** against
   `label_filter_db_id`. Smoke: `scratchpad/features_rebuild/smoke_equalize_e2s.py`.
   **⚠ COST: the label-filter phase is ~3 h, NOT the "~1 h" this repo has assumed since gen 2.**
   Measured on a representative STRIDE of 20 users (not the first 20, which are
   unrepresentatively small), under featB's eval load: PROCESSES=4 gives 1.80 s/user = **5.00 h**,
   PROCESSES=8 gives 1.05 s/user = **2.92 h**. Set to 8; both are upper bounds since the real run
   starts after featB. **So the full gen-4 chain is ~3 h label filter + ~4 h dbs = ~7 h**, not the
   ~4 h the rebuild alone suggests -- budget accordingly. 8 workers are safe here despite gen 3's
   warning, which is about `data_processing`'s whole-user matrices, not this; and the phase is
   RESUMABLE, so an OOM costs a restart rather than the phase.
   **✓ ACTUAL: 49 min 41 s (06:34:12 -> 07:23:53), i.e. 3.5x FASTER than the 2.92 h projection.**
   The projection was measured while featB's eval was competing for CPU and was flagged as an
   upper bound, so the caveat held -- but **the contention factor is ~3.5x, which is the number
   worth carrying**: a CPU cost measured beside a live run overstates the free-machine cost by
   roughly that much on this box. Measure timings on a quiet machine or label them as ceilings;
   the same correction already applies to the CPU-side profiling numbers in the speed section.
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
⚠ **`label_y` DOES reach one imm-side term** -- `p_binary_loss`, srs_model.py:1128 (**stale: it is
now :1365**) -- but `pbin_scale=0` in this recipe so it is skipped. **If `RWKV_PBIN_SCALE` is ever
turned on, a curve-side lever starts softening the imm objective too and this exception no longer
applies.**
**⚠⚠ AND THAT CAVEAT NAMES THE WEAKER PATH WHILE OMITTING THE STRONGER ONE (found 2026-08-19, while
pre-registering kdalpha025's gate rule). EXTERNAL-TEACHER KD IS *NOT* CURVE-SIDE.** `RWKV_KD_ALPHA`
rewrites **BOTH** objectives from the same `kd_mix` tuple, gated on the same
`if kd_mix is not None`, with the same alpha:
* `srs_model.py:1263` -- `label_y = alpha*teacher_curve + (1-alpha)*hard`  ->  curve / **ahead**;
* `srs_model.py:1354` -- `_km2_target = alpha*teacher_p + (1-alpha)*one_hot(label_rating)`, and
  `p_loss` is then REPLACED by soft-target CE against it  ->  rating / **imm**.
So a KD-alpha lever is a **direct** lever on the imm objective, not an indirect one through the
trunk. **The exception's verification is correct FOR ITER 46** (self-distillation rewrote only
`label_y`); the trap is generalising "KD rewrites `label_y`" to "KD is curve-side". Any KD-alpha
iteration -- iter 55, kdalpha025, and any successor -- gets the **BOTH-MODES** rule.
This also gives iter 55 a cleaner mechanism than the record has: its imm **-0.000116** did not have
to travel through the shared trunk, which would be a surprisingly large indirect effect. alpha 0.9
replaced 90% of the imm TARGET with teacher probabilities, so the rating head inherited the teacher's
miscalibration head-on. **Before applying the curve-side exception to any lever, grep for every site
that consumes `kd_mix` / the lever's tensor -- do not reason from which tensor it is named after.**
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
**★ THE NUMBERS THE FOUR IN-FLIGHT RUNS GET (decided 2026-08-18):** `kdalpha` -> **52** (still
vacant, and its own slug), `cmixpow` -> **55**, `rgate` -> **56**, `decayshape` -> **57**. QAT#2 was
renumbered **56 -> 54** to close the last gap -- free ONLY because its number was in no path
(`scratchpad/qat_tax/`, `qtaxg_i45kd_*`). **⚠ So `iter54_cmixpow/` and `iter55_rgate/` now have
LYING slugs** -- `exp` in `research_log.jsonl` is the identity, directory digits are not. Result is
a contiguous 45-57 with no permanent hole.
**★ ITERATION NUMBERING = COMPLETION ORDER (Andrew 2026-08-18):** *"it's better to just order
iterations by the time they finished rather than by the time they were queued"*. **Assign the number
when the VERDICT is recorded**, so `iter N` means the Nth result and the log reads as a history of
what was known when. The old queue-time convention broke whenever runs finished out of order --
QAT#2 became 56 because 52-55 were reserved for queued runs, then finished before iter 53. **That
is the only violation in the log and it is GRANDFATHERED**; history is not renumbered, because the
numbers are load-bearing in run dirs, checkpoint prefixes, `champion_5k_track2.json` and commits.
**★ THE STRUCTURAL FIX -- new runs must NOT put the number in the directory or checkpoint prefix.**
Name the run for its lever (`rgate`, `cmixpow`); `exp` in `research_log.jsonl` is the stable
identity and `number` is assigned at verdict. Baking a number into a path is precisely what forced
the old convention: once `scratchpad/iter55_rgate/i55_*.pth` exists the number cannot move.
(No tooling change needed -- `logbook.py`/`gate.py` record the number, they never assign it. The
four runs in flight need no exception: their chain order 52 -> 54 -> 55 -> 57 is already ascending.)

**RESEARCH-PHASE CONDUCT (Andrew 2026-07-10) -- for the phase after HP tuning + the deck/preset/global
state-size ladders:** (1) try LOTS of different tweaks of both the ARCHITECTURE and the TRAINING
PIPELINE, from different FAMILIES of ideas (not many variants of one); (2) if an idea BARELY misses the
logloss threshold, don't give up early -- try a slightly different implementation of the same idea first;
(3) MIX literature review (optimization/LIT_REVIEW.md) with self-generated ideas
-- **★ TIGHTENED TO STRICT ALTERNATION (Andrew 2026-08-19): "1 invented, 1 adopted, 1 invented,
1 adopted".** Not a ratio to satisfy on average -- the provenance column must alternate row by row.
**WHY IT WAS NEEDED: iters 51, 54, 55, 56, 57, 58 are SIX CONSECUTIVE `invented` rows**, i.e. "mix"
had silently degraded to "self-generated only" while the doc still claimed a mix was happening.
**The next iteration is therefore `adopted`** -- sourced from a paper or a real RWKV repo, with the
source named in the `provenance`/`change` fields, not merely inspired by one. Andrew also framed
this as a change of pace, so treat it as a search-diversity mechanism rather than bookkeeping: the
invented ideas come from this trunk's own measurements and therefore inherit its blind spots, which
is exactly what an external source does not. ⚠ Pair it with his 2026-08-19 steer -- **stop chasing
0.0001** -- so an `adopted` slot means a LARGE-EFFECT idea from outside, not a small one merely
because it has a citation.
(4) spend AT LEAST 50
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
- **★★ CURRENT STATE, MEASURED 2026-08-31/09-01 ON THE d=80 INTERLEAVED TRUNK (quiet machine).
  Everything above this bullet is the d=32 era; the numbers below are the live ones.**
  Full narrative + every closed lead: `optimization/DISPATCH_PLAN.md`.
  * **BANKED, BIT-EXACT: `perm_gather` on the INTERLEAVED path (`srs_model.py:1249`). +5.5%
    throughput, GPU kernel 1,213 -> 892 ms/step (-26.4%).** It was wired on the SEQUENTIAL gather
    (`:1080`) and missed on the interleaved one, which has been the champion's path since iter 41.
    Verified BIT-IDENTICAL over 40 steps (`scratchpad/dispatch/cmp_traces.py`), so **no re-base and
    no seed pair** -- and the champion's recorded numbers stand.
    **⚠ SUPERSEDED BY UPSTREAM, AND MY VERSION WAS NEVER COMMITTED (2026-09-02).** Xemorr's
    **PR #3 (`f6d7505`, 2026-08-23)** had already done this **a week before I "found" it**, and
    does strictly more: `perm_gather` AND `perm_scatter` on the interleaved path. Merged into
    `main`; my redundant hunk was removed surgically rather than reverted, because the same file
    carried unrelated uncommitted work. **The lesson is cheap and I paid full price for it:
    check the remote before writing the fix** -- `git fetch` costs seconds and I did not do it
    until the commit step.
    Verified on OUR champion db before letting `main` move: forward bit-identical over 1,141,200
    elements, gradients bit-identical for all 354 params
    (`scratchpad/parity3/smoke_interleave_permscatter.py`). ⚠ That smoke was **skipping silently**
    here -- it hardcodes `train_db_5k_h1`, deleted 2026-08-30 -- so it now honours `RWKV_SMOKE_DB`
    with the old path as the default. **A guard that cannot run is not a guard.**
  * **⚠ `train_rwkv.py`'s own profiler comment ("237 ms of GPU kernel time inside a ~1450 ms step")
    IS STALE -- it is dated 2026-07-27 and interleaving landed 2026-08-11.** Do not build anything
    on it; it is what made the whole phase open on "85% dispatch-bound, CUDA graphs first".
  * **Interleaving costs 372 ms/step of GPU time** (1,213 with, 841 without) -- real, but only ~38%
    of the gap to that stale figure, so it does not fully explain it. NOT a revert proposal: iter 41
    is an accepted accuracy win and the protocol leaves GPU speed untimed.
  * **`RWKV_EMPTY_CACHE_EVERY=1` IS CORRECT AND IS WORTH 26%** -- `=0` measured 17.8k vs 24.0k
    rev/s, with GPU kernel time 1,369 -> 4,321 ms/step. ⚠ My argument for turning it off was
    backwards: I measured an 8.2 GB peak on a 12.28 GB card **while the flag was active** and read
    that as proof it was unnecessary. **Before removing a control, ask whether the evidence against
    it was produced by it.**
  * **The 273 `cudaStreamSynchronize`/step are CAUSED BY `empty_cache`** (they vanish at `=0`), and
    are not independently removable: `=0` replaces them with 80 `cudaFree` calls at ~15 ms EACH plus
    `Command Buffer Full` stalls.
  * **CUDA graphs are CONDITIONAL, not queued.** 18,756 launches/step cost 218 ms of a 1,121 ms
    step, but graph capture needs STABLE ADDRESSES and `empty_cache` every step is the opposite.
    Test coexistence FIRST; if they cannot, it is 19% against a measured 26% and graphs lose.
  * MAX downward is closed (65536 already optimal). MAX=98304 is +14.3% rev/s but **-34% optimizer
    steps** -- a phase-5 tuning lever, not a speedup.
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

#### CHAMPION = iter 53 `iter53_muonlora` (Muon extended to the LoRA matrices) -- promoted 2026-08-18 08:04
**ahead 0.297523 / imm 0.265191** on the VAL half (n=2500) = **+0.000174 / +0.000184** vs iter 45 at
p=3.5e-08 / 2.7e-54. size 0/2500, nan_users 0, **params 558,212 EXACTLY unchanged**, card/note/deck
state 2,880/1,440/5,760 all unchanged (an optimizer change has no weights), throughput 1860.3 rev/s
(vs 1833.5 -- identical within noise). Training cost ~1% (0.907 vs 0.920 steps/s).
ckpt `scratchpad/iter53_muonlora/i53_d_10935.pth`; `champion_5k_track2.json` points at it.
**THE LEVER, ~2 lines:** `RWKV_MUON_INCLUDE_LORA=1`. The Muon grouping excluded any param whose name
contains `lora` or `scale`, so the **27,520** rank-4/rank-2 LoRA projections (104 tensors, 4.9% of the
model) had always run on AdamW. They move to Muon **in their own group at `weight_decay = 0.0`** --
the value they already had in `other_params` -- so the optimizer is the ONLY variable (verified in
code: the group carries an explicit `weight_decay: 0.0`).
**★ THE MECHANISM IS CONFIRMED, NOT ASSUMED, and it is why this worked.** The 2026-08-16 finding that
**Muon pays here as a REGULARIZER rather than a faster optimizer** predicted this directly, and the
paired WS traces reproduce its signature: over deciles 2-10 iter 53's **TRAIN**-loss advantage on
ahead oscillates around **ZERO** (mean ~+0.00001) while the **HELD-OUT** gain is **+0.000174**. A
generalization gain with no optimization gain behind it. (imm: train ~+0.00012 vs +0.000184 held out.)
**=> THE PRODUCTIVE OPTIMIZER AXIS IS COVERAGE, NOT DESCENT QUALITY.** The 2026-08-16 note demoted
PolarExpress/NorMuon because they refine the DESCENT -- the half that stopped paying. This confirms
the other half pays. The only 2-D params still on AdamW are the 26 `*scale*` tensors.
**★ OPEN, and this result promotes it -- the LoRA norms have NO BRAKE.** Deployed `||W||_F` is **+62%**
over the champion's and does NOT saturate (+22.9 -> +36.4 -> +70.6% at steps 1k/2k/10.9k). It stops
only because the LR schedule anneals to zero: Muon's step is fixed-norm x LR, this group has `wd=0`,
nothing restores it. **At the 10x endgame the same mechanism runs ~6x longer, so that run must either
carry weight decay on the LoRA group or re-measure.** PROPOSALS rank 8 is now the natural follow-up,
as **Muon PLUS decay**.
**DEPLOY: nothing to port** -- training-only, forward pass untouched, params identical.
**⚠ PRE-REGISTERED PREDICTION WAS WRONG** (`scratchpad/iter53_muonlora/PREREG.md`): I predicted
null-or-harm because a rank-4 bottleneck exists *to* concentrate. It does overshoot (LoRA goes from
0.52 to 0.83 of shape-matched-random spread, past the 0.81 where the rest of the model sits) and it
helps anyway. Detail + the two measurement lessons: `research_5k_verbose.md` iter 53.

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
| 45 | + KD kept through the decay phase (alpha 0.5), zero code | 0.297697 / 0.265375 |
| **53** | **+ Muon extended to the LoRA matrices (own wd=0 group), ~2 lines** | **0.297523 / 0.265191** |

⚠ iters 32 and 34 are not directly comparable to their neighbours: the gate basis changed to the RECTIFIED metric at iter 33, and iter 34 changed the training budget. Per-iteration detail: `research_5k_verbose.md`. Full superseded champion blocks (env strings, ckpt paths, caveats): `HISTORY.md`.

### ★★★ THE PLAN, ORDERED (Andrew 2026-08-30) — THIS SUPERSEDES EVERY EARLIER ORDERING

> *"Let's stop making the model smaller and just focus on speedups -> then finish eval of featA ->
> then do featB -> then more algorithmic improvements -> then final HP tuning **with QAT on** ->
> then the final run with QAT and larger epoch budget -> then we'll see if there is anything left
> to squeeze from CPU inference."*

His reason, and it is the honest one: *"it's getting hard to keep track of all the experiments."*
So this list is the single source of order. Anything not on it is not scheduled.

| # | phase | state |
|---|---|---|
| 1 | **Training speedups** (dispatch-bound) | **DONE** — +5.5% bit-exact (PermGather on the interleaved path); four leads closed. `optimization/DISPATCH_PLAN.md` |
| 2 | **Finish the features CONTROL eval** | **DONE** — featA2 0.298186 / 0.265588; prices the Bug A fix at +0.000148 / +0.000169 |
| 3 | **featB** — the new-features arm | **RUNNING** since 2026-09-01 20:36 (~10 h, 0.931 steps/s). Died TWICE on the `insert_probes` KeyError (08-21, 09-01); root-caused and fixed 09-01 — see BUG B. **Gen-4 rebuild is armed behind it** (Bug C fix; see GEN 4) |
| 4 | **More algorithmic improvements** | the research loop, gate unchanged. **★ STOP CRITERION (Andrew 2026-09-02): keep going -- adopted AND invented, alternating -- until LogLoss is at most 0.2960 ahead \| 0.2650 imm.** On featB's basis imm (0.263217) is already under the bar and **ahead (0.297884) is the binding constraint, ~0.0019 away**; ⚠ gen 4 moves the absolute basis (e2s-selected equalize set removes ~0.19% of rows that are 1.46x easier), so read the target on the gen-4 lineage's numbers. First lever queued: `realcyc` (real-time cycles replace the pseudo day-offset ones, Andrew's directive) |
| 5 | **Final HP tuning, WITH QAT ON** | ⚠ NEW — all prior tuning was PLAIN |
| 6 | **The final run: QAT + larger epoch budget** | the old "10x endgame", both arms |
| 7 | **CPU inference** — whatever is left to squeeze | `optimization/CPU_INFERENCE.md` |

**★ "STOP MAKING THE MODEL SMALLER" CLOSES A WHOLE LINE OF WORK.** The parameter-ratio gate, the
<=100k hybrid arms, and the FSRS-core V1/V2 experiments are all DONE or SHELVED. Do not propose a
size reduction as an iteration. Size is now whatever falls out of phases 4-6.
* iters 60 (arm A) and 61 (arm B) are the recorded verdicts; both rejected, and together they
  showed the parameter-efficiency curve has a knee below 558k and that widening the feature
  pathway does not substitute for recurrent capacity.
* **V1 (FSRS-7 card core) is SHELVED, wired and verified but never trained.** 16/16 parity checks
  green, inert when off, 488,858 params. It is shelved on SPEED, not on doubt: 0.166 steps/s,
  ~9x slower than an arm (~32 h/run), because it replaces a fused CUDA kernel with a Python loop
  of ~40 ops per review into a step that is 85% dispatch-bound. Full state:
  `optimization/HYBRID_100K.md` sections 13-14. If phase 1 lands a large dispatch win, V1 becomes
  cheap enough to reconsider — that is the only condition under which it returns.

⚠ **PHASE 2 NAMING: Andrew said "finish eval of featA", and `featA` is COMPLETE.** The arm with no
number is **featA2** — the control retrained on the Bug-A-FIXED published dbs
(`train_db_5k_h1_fix`), trained but never evaluated. That is what phase 2 means; `run_featA2_evalonly.cmd`
exists. featA's own numbers are on the OLD dbs and are not a valid control for featB.

⚠⚠ **PHASE 3 IS CONFOUNDED AND ANDREW SHOULD SEE THIS BEFORE IT RUNS.** featB changes TWO things at
once, because `elapsed_end_to_start` (landed 2026-08-19) is gated on the DATASET, not on a flag: it
runs whenever `review_time` is present, which is true of every `-id` build. `train_db_5k_h1_id3` was
built 08-24. **So featB = new features AND end-to-start together, and no `-id` database with
end-to-end intervals exists.** The features A/B cannot separate them.
**The cheap resolution needs no rebuild:** the interval question is a ONE-LINE transform on the
PUBLISHED set (`elapsed_seconds - duration(k)/1000`), because `duration` and `elapsed_seconds` are
both public columns. That isolates the interval definition with features held fixed, and it is the
same experiment as the srs-benchmark hand-off. Details + the per-dataset formula trap:
`scratchpad/hybrid100k/INTERVAL_HANDOFF.md`. The alternative -- a 4th `-id` generation with the
correction disabled -- is ~23 h of preprocessing to un-bundle what one line answers for free.

**★ PHASE 5 IS NEW AND IS NOT A REPEAT.** Every previous HP tune was PLAIN; the champion HPs were
confirmed against 19 alternatives without QAT. QAT changes the loss landscape (the tax is
+0.002286/+0.003486 even with learnable catalogs), so the plain optimum is not known to be the
quant-aware optimum. Budget it as a real phase, not a re-confirmation.

**What phase 6 inherits, unchanged from the old plan:** two arms (plain 10x, then a warm-started
QAT fine-tune on its final -- NOT QAT from scratch, iter 40 measured that at +0.0118), the 10+2
WS:decay split, augmentation stays OFF, and the three 1-epoch assumptions (warmup 200, wd,
dropout) to reconsider first. The detail is preserved below.

---

### (superseded) THE ENDGAME, ORDERED (Andrew 2026-07-26)
> Kept for the phase-6 detail it carries -- the two arms, the cost model, the QAT-tax
> decomposition and the augmentation/KD-dump interaction. Its ORDER is superseded by the table
> above.

> **★★ ORDER AMENDED (Andrew 2026-08-29): a TRAINING-SPEED phase is inserted before step 1.**
> *"Let's do it after the three arms but before continuing with new features and algorithmic
> improvements."* So: **three hybrid arms -> DISPATCH SPEEDUP -> algorithmic loop -> features ->
> 10x run.** Plan: `optimization/DISPATCH_PLAN.md`.
> **THE FINDING THAT MOTIVATED IT:** the step is **CPU-DISPATCH-BOUND, not kernel-bound** -- 237 ms
> of GPU kernel time in a ~1,450 ms step (16%), 90,576 op dispatches, 199 ms of pure
> `cudaLaunchKernel`. Confirmed live on hybrid arm A: **mean GPU utilisation 31%, 17% of samples at
> literal 0%.** Amdahl ceiling ~6x, realistic target 2-3x.
> ⚠ **Do not repeat the "9x from a smaller model" framing.** 9x is an ARITHMETIC ratio and
> arithmetic is 16% of the step; arm A's real 1.41x comes from having 9 layer-steps instead of 13,
> i.e. fewer DISPATCHES, not from the width cut.
> **Why after the arms specifically:** the profile is d=80. If an arm promotes, the trunk becomes
> d=32-shaped and the dispatch profile changes, so profiling first would characterise a trunk we may
> not keep.
> **The constraint that shapes the work: BIT-EXACT OR RE-BASE.** Every banked speedup so far was
> bit-exact by design. A numerics-changing training speedup forces a champion re-run (~9.5 h) plus a
> seed pair before any later candidate can be attributed. CUDA graphs with padded static-shape
> buckets are the top candidate and can be bit-exact; **their old "variable shapes" blocker has
> inverted, because padding wastes GPU COMPUTE, which is only 16% of the step.**
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

#### ★★★ NEXT AFTER THE CHAIN DRAINS: A BUG HUNT, THEN RESUME (Andrew 2026-08-18)

> *"once GPU is free, do some bug hunting. Obviously not every bug will be caught just by staring at
> the code, so temporarily reduce the number of epochs to 0.05 and the number of eval users to 20,
> and use that for diagnostics. Once you are reasonably confident that there are no bugs left,
> resume the autoresearch loop normally."*

**This comes BEFORE the next research iteration.** Do not treat the queued candidates as the next
task when the chain finishes; the bug hunt is the next task.

**THE METHOD HE SPECIFIED, and it is the point of the exercise:** build a **FAST DIAGNOSTIC CONFIG**
-- `EPOCHS = 0.05` and an eval over **20 users** -- so a whole train -> decay -> eval cycle runs in
minutes instead of ~9.2 h. Reading code cannot find the failures that matter here; **executing the
path can**, and at 0.05 epochs it is cheap enough to execute every path. Every failure of
2026-08-17/18 would have been caught by one cheap end-to-end run: the `endlocal` marker, the
`LOAD_MODEL_FOLDER` omission, the alpha/guard mismatches, the smoke inheriting its own control.

**WHY (the audit, `HISTORY.md` "BUG-RATE AUDIT"):** the bug RATE is flat (21% of commits in both
halves of the last 24 days, ratio 1.01x). The recent spike is EXPOSURE -- orchestration edits ran
3.5x baseline after the outage, and model-code edits were zero. The root cause is **cloning runners
across lineages**: each new run inherits its nearest ancestor's defects (`run_iter45.cmd` carries the
`endlocal` bug and its descendants do; `run_iter52.cmd` does not and its descendants are clean).
**27 historical runners carry that one defect.** So the durable fix is **ONE CANONICAL RUNNER
TEMPLATE with the guards baked in** -- build that as part of the hunt, not another clone.

**Already done and not to be redone:** every smoke audited for the false-green class (a control arm
inheriting the treatment from the ambient env) -- 12 of 49 inherit `os.environ`, only `rgate` was a
live risk and it is fixed; no past verdict is affected (iter 48's `rcouple_w` is learned non-zero,
so its lever was live); `preflight_runner.py` now asserts the `endlocal` ordering and all four live
runners pass.

#### LIVE
**>>> THE GPU IS BOOKED ~31 h DEEP: ITER 53 IS DONE AND ACCEPTED, FOUR JOBS REMAIN CHAINED**
(re-armed 2026-08-17 21:47; iter 54 launched 2026-08-18 08:04). Each waiter is detached via WMI so
Esc cannot kill it, and each polls the previous job's log with an ANCHORED
`findstr /B /C:"DONE_EXIT_"` (the unanchored form matches its own progress line and fires
instantly). **Costs below are MEASURED on iter 53, and are ~40% higher than this table used to say.**

**⟶ STATE AS OF 2026-08-19 04:45. Four of the six are DONE; two remain plus a retry.** Numbers are
assigned at VERDICT time (completion order), so the reservations two paragraphs above are superseded
by what is in `research_log.jsonl`: `decayshape` took **56**, not 57, and `cmixpow`/`rgate` are
UNNUMBERED until they report. Directory digits bind nothing.

| order | run | lever | cost | state |
|---|---|---|---|---|
| ~~1~~ | **iter 53** `iter53_muonlora` | `RWKV_MUON_INCLUDE_LORA=1` | 9.2 h | **DONE -- ACCEPTED, CHAMPION** |
| ~~2~~ | **iter 55** `iter52_kdalpha` | KD `alpha_decay` 0.5 -> 0.9 | 6.1 h | **DONE -- REJECTED** (confirms the KD-calibration mechanism) |
| ~~3~~ | **iter 56** `iter57_decayshape` | `RWKV_DECAY_SHAPE=linear` | 6.1 h | **DONE 04:40:48 -- REJECTED.** Sub-bar vs iter 45 (+5.7e-5/+1.04e-4), loses to iter 53. **Stacking priced under perfect additivity: still FAILS the ahead bar -> do NOT queue it.** |
| 4 | **`cmixpow` PHASE 2B** `run_iter54_phase2b.cmd` | the owed decay + eval | 6.1 h | **RUNNING** since 04:41:45 (WS done pre-outage) |
| 5 | `kdalpha025` | KD `alpha_decay` 0.5 -> 0.25 | 6.1 h | waiter armed on `iter54_phase2b.log` |
| 6 | `rgate` `iter55_rgate` | `RWKV_RGATE=card` | 9.2 h | waiter armed on `kdalpha025.log`; its 22:25 smoke failure is STALE (fix landed 22:31) and the smoke now PASSES with the contaminating var deliberately set |

**★ ORDER CHANGED 2026-08-18 15:14 (Andrew asked why 54 was scheduled after 55/57).** iter 54's
phase 2 was originally queued LAST, because iter 55's waiter already polled `iter52.log` and a
second waiter there would have fired both at once. But iter 54 is the OLDEST incomplete work and
its owed decay+eval is only 6.1 h, so finishing it a day later left a half-done iteration exposed
to another outage for nothing. Resolved by re-pointing iter 55 at **`iter54_phase2.log`** -- a
FRESH file. ⚠ `iter54.log` could NOT be used: it already carries `DONE_EXIT_TOMLFAIL_1` from the
12:44 failure and would fire a waiter instantly.
⚠ COSMETIC: `wait_then_iter55.cmd`'s startup echo still says "waits on iter 52 re-run" -- its
`PREVLOG` is correct (`iter54_phase2.log`) and the file must NOT be edited while its waiter loops
(cmd.exe re-reads batch files from a byte offset).

**★★ THE CHAMPION MOVED, SO THE REMAINING FOUR NOW HAVE A HIGHER BAR.** All four were built on the
ITER-45 recipe, which keeps their comparison CONTROLLED against iter 45 -- but the gate is always vs
the CURRENT champion, so to be ACCEPTED they must now beat **iter 53's 0.297523 / 0.265191**, i.e.
their iter-45 deltas must exceed +0.000174 / +0.000184 before clearing the 0.0001 bar. **Report BOTH
numbers at verdict time** (delta vs iter 45 = the controlled effect of the lever; delta vs iter 53 =
the gate). The levers are orthogonal to iter 53's, so nothing needs re-running.

**★ COSTS (measured on iter 53, not projected): full iteration 9.2 h, decay-only 6.1 h.** Decay is
**10,935 steps, the SAME as WS** (`decay_ratio = 1.0` since iter 34; the champion checkpoints are
named `*_d_10935`). The 2,733-step figure once quoted here is the tuner-era ratio 0.25 and had this
chain costed ~40% low. WS 0.907 steps/s, decay 0.957, plain eval ~2.9 h.

**⚠⚠ OPS -- `endlocal` BEFORE THE `DONE_EXIT_` ECHO STRANDS THE WHOLE CHAIN (cost 45 min of idle GPU,
2026-08-18).** iter 53 finished cleanly at 07:11 (exit 0, 2,500 users in both result jsonls) and
**never announced it**: its runner ended `... endlocal / echo DONE_EXIT_0 >> "%LOG%"`, and `endlocal`
restores the pre-`setlocal` environment, so `%LOG%` expanded to EMPTY and the append target became
`""`. Nothing was written, and four chained waiters polled forever for a line that could not appear.
**`run_iter54.cmd` and `run_iter55.cmd` had the identical tail** -- all three were generated from
`run_iter45.cmd`, which has it too and never revealed it because iter 45 was LAST in its chain. Both
were patched before the chain was released (marker moved inside the `setlocal` scope) and the fix was
proven **by executing it in cmd.exe**, not by reading it. iters 52/57 came from a different generator
and were already correct.
**THE RULE: write the terminal marker BEFORE `endlocal`, and have `preflight_runner.py` assert it** --
its existing "declared before first use" check cannot see `endlocal`, because that invalidates
variables mid-file rather than at a line the parser can point at. Same family as the mk53/mk54
deletion bug: `%LOG%` silently empty, failure maximally quiet.


**DONE: QAT#2 `qtaxg_i45kd` = iter 54, REJECTED as an exact tie** (ahead +0.000084 p=6.2e-04, imm
-0.000070 p=1.0, both inside the +/-7.5e-5 floor). **The QAT tax does not live in the teacher**, and
a minutes-of-CPU screen had predicted it that morning (the two teachers agree at r=0.9460 because
iter 45 IS the d=128 teacher's own student). Detail: `research_5k_verbose.md` iter 54.

⚠ **ITER 52's FIRST LAUNCH DIED IN 0.07 s AND IS RE-QUEUED AS ORDER 3.** Its phase-0 guard called
`Git\usr\bin\bash.exe` -- the RAW MSYS binary, which from cmd.exe has no MSYS PATH, so the script
died on `dirname: command not found` -- AND passed no argument to `smoke_scripted_eval.sh`, which
takes a required `<eval_toml>`. Both fixed (`Git\bin\bash.exe` + iter 45's toml, which is the right
choice anyway since the guard tests whether the CURRENT CODE scripts, not iter 52's weights). Its
original log was RENAMED to `iter52_failed_smoke_2135.log`: the runner APPENDS, so a stale
`DONE_EXIT_45` would make any waiter on `iter52.log` fire instantly.

⚠ **BASIS SUBTLETY, accepted deliberately:** iters 53, 54 and 55 are all built on the ITER-45 recipe,
so if iter 52 wins and promotes, their controlled comparison stays vs iter 45 while the champion has
moved. The levers are orthogonal and chaining keeps the GPU busy rather than idling on a human
reading a verdict; report both numbers at verdict time.

**★★ TWO QUEUED CANDIDATES SCREENED ON CPU WHILE THE GPU RAN (2026-08-17).** **★★ FOUR CPU SCREENS HAVE NOW CHANGED THE RANKING OF SIX CANDIDATES** (four killed outright, one demoted, one re-specified), each for minutes-to-an-hour of CPU against the 5.5-13 h GPU runs they redirect. **Run one before every build.** Detail in `PROPOSALS.md`; both cost minutes.
* **Ensemble teacher: DEMOTED (rank 4 -> 7).** The proposed 2nd teacher is the 1st teacher's own
  STUDENT -- iter 45 ends a 4-iteration lineage (32/35/39/45) each trained against the d=128 dump
  (verified in `run_iter45.cmd:43,82`). Measured r=0.9460. Priced in the same units as an accepted
  change: the target shift is 0.0117 vs 0.0568 for iter 39's alpha 0.5->0.9, i.e. **21% of a change
  worth +0.00016**. Not killed only because 74.9% of the disagreement sits on uncertain rows.
  ⚠ **`RWKV_trained_on_5000_10000.pth` is DISQUALIFIED as the obvious 2nd big teacher -- it trained
  on our entire VAL+TEST halves and no gate would catch the leak.** Use iter 31 / A18 (verified
  KD-free in their runners).
* **Spacing-effect: RE-SPECIFIED, still queued.** The constraint BINDS (violation rates by button at
  30d: Again 59.3 / **Hard 65.9** / Good 39.7 / Easy 38.3%) but the blanket `rating>=2` form is
  WRONG -- it would fight the model on 66% of Hard transitions, where lowering retention is correct
  inference. Must be Good/Easy-conditional and pitched as a regularizer (PAVA's shape), because the
  "structural fact" is FSRS's fixed-DECAY assumption and our learnable-`d` mixture declines to obey
  it on ~40% of Good transitions. Instrument verified at **0.000e+00** vs the certified iter-41
  trace first; the alarming "R falls over a card's life" is difficulty SELECTION, shown from data
  alone (per-card lapse rate 1.9% -> 46.4% with review count, rho 0.4867).

**★★ ANDREW 2026-08-17: AT LEAST 10 MORE ALGORITHMIC ITERATIONS BEFORE THE FEATURES PHASE.**
*"There is no way the current architecture and training are so optimal that no improvement is
possible."* Ranked plan + what would change it: `optimization/PROPOSALS.md`.

**★★ AND HE OPENED A FAMILY THAT WAS MISSING: EXPRESSIVENESS != CAPACITY.** The queue had no
architecture entries because "capacity-at-5k is 0/3" was standing in for an argument it cannot
support -- all three of those rejects added **more of the same functional form**; none tested a
RICHER form at fixed parameter count. Iter 54 is the first of the new family.
**★ THE SCREEN THAT MAKES THE FAMILY CHEAP: the REDUNDANCY TEST.** If an adjacent FREE LINEAR can
absorb the new parameter, it adds exactly zero expressiveness. That kills learnable slopes on
tanh/sigmoid (all sandwiched between free linears) and cross-head mixing (`W_o` already mixes the
full `H*K` dimension) by algebra alone. A learnable EXPONENT survives -- curvature is not absorbable.
**★ FOUR ARCHITECTURE PROPOSALS KILLED BY MEASUREMENT, ~1 h of CPU, zero GPU** (probes in
`scratchpad/expressiveness/`, detail in `LIT_REVIEW.md`): the hardcoded `-0.5` decay floor is not
binding (median fastest reachable `w` 0.954-0.994 vs a floor of 0.545; 0.3% of channels get within
0.05); the LoRA `tanh` is not saturating (inputs 1.08-1.50); `a` sits in a median band of [0.41,0.60];
and the delta-rule authority lever died on its own follow-up measurement.

**★★ LIT REVIEW 2026-08-17 (Andrew's ask) -- one real lead, and it did not survive contact.**
RWKV-8 is NOT APPLICABLE (DeepEmbed/MoE, KV-cache streamlining, a token suffix automaton -- we have
none of those). fla's 2026 work is distributed-training plumbing. The lead was
**arXiv 2411.12537 (ICLR 2025), negative eigenvalues**: linear RNNs whose state-transition
eigenvalues are all positive provably cannot solve parity, and extending to [-1,1] costs zero params.
**Measured, not adopted, and it failed twice over:** (1) the naive `a = 2*sigmoid` port is nearly
INERT here -- our eigenvalue along the delta direction is `w - a*||kappa||^2` = 0.837-0.864, and
doubling `a` only reaches 0.67, because `||kappa||^2 ~ 0.24` (RWKV-7 rescales the normalised key by
`k_scale`) does most of the blocking, whereas DeltaNet has `||k||=1` exactly; (2) BOTH factors are
already freely learnable toward more delta authority -- the model can reach ~0.95 and operates at
~0.13, so nothing blocks it and raising the range only extends what is unused. **DEAD, do not run.**
**★ THE FINDING THAT SURVIVES, and it is a characterisation of this model:** the delta term moves the
state-transition eigenvalue by only **~0.15** against a decay of ~0.98, i.e. **our trunk uses its WKV
state almost as a pure exponential-decay accumulator with a small rank-1 correction -- RWKV-7's
headline innovation is barely engaged.** Whether that is the TASK's nature or our 1.25-epoch budget
is a FREE check on the 10x endgame checkpoint: re-run `scratchpad/expressiveness/decay_floor_probe.py`
and the eigenvalue probe on it.
**⚠⚠⚠ "BARELY ENGAGED" IS REFUTED AS A STATEMENT ABOUT FUNCTION -- MEASURED 2026-08-19, AND THE
EIGENVALUE NUMBER IS STILL CORRECT.** Andrew proposed simplifying the delta rule to cut parameters,
which is exactly what the sentence above invites. Screened on CPU first
(`scratchpad/bughunt/delta_ablate_screen.py`, ~25 min, zero GPU): **zeroing `a` on the champion --
which deletes the delta term and nothing else -- costs `+0.208 imm / +0.060 ahead`.** For scale, the
accept bar is 0.0001 and the ENTIRE A0->A18 ladder that made the model 4.95x smaller cost +0.00053
imm. The ablation is **~390x that whole ladder**. The delta rule is massively load-bearing.
**THE RECONCILIATION, and it is the reusable part: a small EIGENVALUE perturbation is not a small
FUNCTIONAL contribution.** The delta term is not tuning the decay rate -- it performs the
key-selective REMOVAL that makes the state an associative memory (erase the value bound to this key
before writing the new one). It is rank-1 and aimed at exactly the direction being overwritten, so
0.15 of eigenvalue movement, spent precisely where it is needed, does work that no amount of uniform
decay reproduces. **A scalar summary of a mechanism's MAGNITUDE says nothing about its SELECTIVITY,
and selectivity is the whole point of the delta rule.** Same family as the median-vs-max error of
iter 51: the wrong summary statistic, confidently applied.
⚠ The ablation is an INFERENCE-TIME upper bound (the weights co-adapted), so a retrain would recover
some of it -- but at ~390x the full param ladder, no retrain turns this into a free simplification.
**=> DO NOT propose delta-rule removal or `a`-simplification on the "barely engaged" argument.**
The measured prize was small anyway and it is the wrong KIND of prize: `a_lora` + `k_scale` are
14,625 params (2.62%) and they are **WEIGHTS, not STATE** -- the binding deploy budget is per-card
state (9 B/card, frozen), which they do not touch. Nor are they free to bias-replace: 36.4% (a_lora)
/ 20.0% (k_scale) of their variance is token-to-token, and `a_lora` needs a mean of 3.1 of its 4 rank
components for 95% of variance, so even rank 4->3 is not free
(`scratchpad/bughunt/delta_rule_screen.py`).
**⚠⚠ AND THE THIRD INSTANCE OF ONE FAILURE SHAPE, this time in my own writeup an hour after writing
it:** the first stability numbers for that lever used the RESTING `||kappa||^2 = 0.24` as if it were a
bound. It is not -- `k_scale = sigmoid(Linear(x))` has an UNBOUNDED input, so `||kappa||^2 -> 1` is
reachable, and only `a` has a true envelope (its LoRA passes through `tanh`). Redone with genuine
maxima the safe range is `c <= ~1.5`, not `c <= 8`; `c=2` already gives a worst-case eigenvalue of
-1.365. Iter 51 fitted on a median and missed a 1.76e7 blow-up; the `a`-is-dead probe used "not
pressed against the bound" where a REPRESENTATIONAL argument was needed; now a resting value stood in
for a maximum. **The rule is not "report the max" as a style preference -- a typical-case statistic
cannot bound a worst case, ever.**
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
`label_filter_db`, and ~~the Rust input-width port~~.
**✓ THE RUST INPUT-WIDTH PORT IS DONE (2026-09-02).** `FEATURE_DIM = 92` was a compile-time const
in `model.rs` -- the ONE dim the engine hardcoded, while the file's own header says H/K/C/
stream_layers/arch are derived from weight shapes "so the engine auto-adapts to any arch". It is
now derived from `features2card.0.weight` at load: `Model.feature_dim` (field) and
`FastModel::feature_dim()` (reads the same tensor, so the two engines cannot disagree -- the
divergence class this crate is most prone to, and why `arch`/`stream_slot` are threaded).
A hardcoded 92 would have **refused to load a 114-dim features model at all**.
**VERIFIED BEHAVIOUR-PRESERVING, not argued:** re-running the engine on `reference_iter41/`
reproduces all three certified prediction files **BIT-IDENTICALLY** (md5, users 107/136/156 --
those files were generated by the OLD code), and `verify_rust.py` gives **PARITY: PASS** at
imm 0.000000 / ahead 0.000000, max per-review 4.78e-06, matching the recorded figure exactly.
`COL_DUR=8` / `COL_R1=9` stay constants and that is CHECKED: `RWKV_ID_FEATURES` drops index 22 and
appends, so only indices AFTER 22 move -- 8 and 9 are before the drop point in both layouts.
Also added `check_trace_width()`, which fails loudly when a trace's width does not match the
checkpoint's. A silent mismatch is not obviously wrong -- it reads misaligned rows and still
produces plausible numbers, which is exactly how the missing `RWKV_ZERO_FEATURES` mask survived a
green mean-LogLoss gate until a per-review comparison caught it.
⚠ **The 114-dim path itself is UNTESTED and cannot be tested yet** -- no 114-dim checkpoint exists
until featB finishes. What is proven is that the hardcode is gone and that 92 is unchanged.
**★★★ 2026-08-21 -- ONE `dtype=torch.int32` WAS DESTROYING ENTITY IDENTITY, AND ONE HALF OF IT IS
PRE-EXISTING. `data_processing.create_sample` stored every id stream as int32. FIXED to int64
(`167fa15`) with a value-comparison guard, because a narrowing cast is invisible to every banner and
shape check -- saturation is not an error, it is a silent value change.**
* **BUG A, PRE-EXISTING, AFFECTS THE CHAMPION.** The NaN-metadata fill is written
  `ID_PLACEHOLDER + card_id` precisely to give each such card a UNIQUE placeholder. ID_PLACEHOLDER
  is 3.14e17, so int32 SATURATED them all to INT32_MIN and they collapsed into **ONE note and ONE
  deck**. Measured on the PUBLISHED set: **19.28% of all reviews carry a NaN note_id** (per-user
  median 4.1%, mean 22.1%, p90 70.0%; 4 users in 60 above 95%). Published user 101 goes from **1
  distinct note id to 3,277** once widened. So ~a fifth of the training data has had no note/deck
  stream identity for the entire project.
* **BUG B, NEW, INVALIDATES BOTH -id REBUILDS.** `-id` keeps RAW epoch-ms ids (~1.7e12): card_id is
  int64 so it WRAPPED (7,693 of 15,191 values negative); note/deck/preset go through the NaN-fill as
  float64 so they SATURATED -> **n_unique == 1 for every user tested, in gen 1 AND gen 2**. A wrap
  collision also makes a card's genuine FIRST review probe-eligible while `add_queries` gave it no
  query row -> `KeyError` in `insert_probes`, which killed featB's fetch worker and DEADLOCKED the
  run (GPU 0% for 69 min).
  **⚠⚠ THAT LAST SENTENCE IS WRONG, AND BELIEVING IT COST A SECOND 10 h RUN (2026-09-01). The
  `KeyError` IS NOT THE ID COLLISION AND WAS NEVER FIXED BY THE int64 CAST.** featB died at step
  ~939 on the *gen-3* dbs -- built after every id fix -- with the identical
  `KeyError` at `prepare_batch.py:123`. Diagnosed properly this time
  (`scratchpad/features_rebuild/probe_query_mismatch.py`), and the real mechanism is ROW ORDERING:
  * `insert_probes` eligibility is POSITIONAL -- the first real row of each `card_id` in the chunk.
    A query row exists iff `is_first_review == False`, and `is_first_review` is `elapsed_days == -1`,
    which `build_parquet_id.py:138` sets from **`state == 0`** and only THEN sorts the frame by
    `review_time` (`:142`). Nothing keeps the two aligned.
  * Anki caps `taken_millis` at **60 s**, and `review_time = id - taken_millis` subtracts that CAP,
    so a capped neighbouring review can acquire a show time up to a minute early and sort AHEAD of
    the card's genuine first review. That first review then passes the positional mask, has no
    query row, and raises. Ground truth, user 477 card 1708127478116: `review_th` 73724 is the first
    review (`elapsed_days` -1, 11.5 s) yet 73723 (`duration` exactly 60000) sorts before it and
    carries `elapsed_seconds` **-17**.
  * **The tell that separates the two stories is `shortfall = n_real - 1 - n_query` per card.** A
    collision or a double-first gives 1; REORDERING gives **0** -- `add_queries` emitted exactly
    the right number of query rows, they are just not on the row the positional mask spared. All
    measured cases are 0, so the collision story is refuted, not merely unproven.
  * **-id ONLY, and PROVEN so:** neither `elapsed_end_to_start*` re-sorts, so the published/e2s
    lineage cannot produce it -- 0 unpairable in **3,769,040** eligible targets over 796 chunks of
    `train_db_5k_h1_e2s`, vs hits in `train_db_5k_h1_id3` AND `test_db_5k_id3` (the eval phase
    would have died too).
  * **⚠ THE RATE IS HEAVILY SKEWED BY USER, and a stride sample badly understates it.** Two
    stride samples over users 1-694 found 1 unpairable target per 2,654-22,201 -- and the live
    featB log then reported **user 1503 dropping 22 of 331 picked targets in ONE chunk (6.6%)**,
    i.e. ~275 unpairable eligible rows in a single chunk. Most users have 0-1; a few have
    hundreds. Quote the per-user distribution, never the sampled mean, and note that a stride
    sample is the wrong instrument for a skewed rate.
  * **The dropped rows are genuine FIRST reviews, so the filter RESTORES the intended semantics
    rather than costing signal** -- `first_mask` exists precisely to exclude first reviews and
    was failing to. (Proven, not assumed: `shortfall == 0` means the card has exactly one
    is_first_review row and it is the unpairable one.) The converse row -- positionally first but
    NOT a genuine first -- is still skipped, which is a small pre-existing loss on every db and
    is unchanged.
  * **FIXED 2026-09-01 in `insert_probes`:** unpairable targets are now FILTERED OUT (a first
    review cannot be probed -- the imm task needs a prior review, so there is nothing to pair
    against; dropping it is the correct semantics, not a workaround) and REPORTED via
    `_note_unpairable` rather than dropped silently. The rng draw happens BEFORE the filter and the
    filter is a no-op wherever the old implication held, so **every published/e2s number is
    bit-identical**. Guard: `scratchpad/features_rebuild/smoke_probe_pairing.py` -- it replays the
    legacy indexing and REQUIRES it to raise, so it cannot pass vacuously on a clean db.
  **THE REUSABLE LESSON: a crash was attributed to the bug being fixed that week, the fix shipped,
  the crash was never re-run, and "That is fixed" entered the record as fact.** The two bugs shared
  a symptom and nothing else. A diagnosis that is never re-tested against the failure it explains
  is a hypothesis wearing a verdict's clothes.
* **★★ AND IT WAS A TRAIN-vs-DEPLOY DIVERGENCE -- EXACTLY WHAT THE THREE-WAY-PARITY RULE EXISTS
  FOR, AND NO GATE CAUGHT IT.** TRAINING grouped rows by the **int32-truncated** id stored in the
  LMDB. DEPLOY (`run_as_rnn`) keys its state dicts on the **raw frame value**
  (`self.note_states[row["note_id"]]`, :152), i.e. full precision. Measured on published user 101:
  training saw **1** note entity, deploy saw **3,277**. So for every NaN-metadata card the two paths
  computed a different quantity -- training pooled them into one note state, deploy gave each its
  own -- and each path was self-consistent in isolation, which is precisely the failure mode §9
  predicts. The fix makes both full precision. **Add an id-identity case to the parity harness**;
  `parity_train_vs_rnn.py` is single-stack and structurally cannot see this, exactly like the
  `RWKV_ID_FEATURES` width check that needed its own smoke.
* **★★ BUG C (2026-08-26, AUDIT): THE int64 FIX DID NOT FINISH THE JOB, AND IT CREATED A NEW
  TRAIN/DEPLOY DIVERGENCE. Full report + probes: `scratchpad/hybrid100k/ID_FILL_BUGS.md`.**
  (a) `ID_PLACEHOLDER + card_id` is computed in a **float64** column (`note_id` holds NaN at
  that moment), and 3.14e17 is far past 2^53, so float64 spacing there is **64** and the low
  bits of card_id are rounded away BEFORE create_sample's int64 cast. Intended-vs-actual
  distinct placeholders over 49,186 cards: **published 49,186 -> 812 (98.3% lost)**, **-id
  49,186 -> 30,869 (37.2% lost)**. The int64 fix widened the DESTINATION; the VALUE was already
  destroyed upstream.
  (b) **TRAINING fills `note_id` with `ID_PLACEHOLDER + card_id` (one note per card); DEPLOY's
  `run_as_rnn.add_id` fills it with the bare CONSTANT (ALL such cards share one note).** Bug A's
  shape with the direction reversed -- before the int64 fix both paths collapsed, so the fix is
  what made them diverge. Affects the 19.28% of reviews with a NaN note_id. TRAINING is the
  correct side. `deck_id`/`preset_id` use a constant on both sides and are fine.
  (c) **`smoke_id_identity.py` cannot see this**: it models deploy as `df[name]`, i.e. the frame
  AFTER the training-side fill, so both sides of its comparison inherit the same fill rule. It
  catches STORAGE truncation downstream of the fill and is blind to a fill-RULE difference.
  **A parity guard that MODELS the other path only tests what it already assumes is shared.**
  ⚠ **THIS GATES THE PUBLISHED-DB REBUILD.** Rebuilding with only the int64 fix still loses
  98.3% of note identity on 19% of reviews. Both fixes must land first, or the rebuild banks a
  smaller version of the same bug and re-bases the champion for nothing.
* **GEN 1 AND GEN 2 ARE DELETED / SUPERSEDED. GEN 3 (`*_id3`) is the first correct -id build.**
* **★★ GEN 4 (`*_id4`) IS ARMED AND CHAINED BEHIND featB (Andrew 2026-09-01: "since we will
  almost certainly adopt timestamp features, we need the fixes").** Gen 3 was built **2026-08-24,
  two days BEFORE the `nan_id_fill` fix**, so it still carries **Bug C** -- verified in the
  artifact, not inferred from dates: **39,599 NaN-note cards -> 24,707 distinct placeholders
  (ratio 0.6239, 37.2% of note identity lost)**. Gen 4 is gen 3 **plus that single fix**; the only
  other data_processing change since is `elapsed_end_to_start_published`, which is inert on an
  `-id` frame.
  **✓ THAT MAKES gen3-vs-gen4 THE CLEAN ID-FIX MEASUREMENT THE RECORD HAS NEVER HAD** -- same
  generation, same code, one variable -- i.e. the shape whose absence forced the featA2
  retraction. If featB's recipe is re-run on gen 4, the re-base and the measurement are the same
  run, at no extra GPU cost.
  **⚠ NOT STARTED IMMEDIATELY, AND THE PRECEDENT DOES NOT TRANSFER.** Gen 2 ran beside featA
  because featA's fetch workers were ~9.7 GB; **featB's measured 18.3 GB each with 10.9 GB of 63.9
  free**, and gen 3's own config header records a rebuild exhausting 64 GB and dying beside a
  training run. Starting early buys nothing (nothing consumes gen 4 until featB reports) and risks
  a 10 h run. `wait_then_rebuild4.cmd` therefore waits on **TWO** conditions -- featB's terminal
  marker **AND >=25 GB free RAM** -- because a marker alone is satisfied by a dead-and-relaunched
  featB.
  **Guards, and the first two are the point of the build:** `assert_bugc_fixed.py` (phase 0b --
  `nan_id_fill` is exact, and it **proves its own non-vacuity** by simulating the float64 path and
  requiring it to collapse: 4096 -> 65) and **`check_db_idfill.py`, which reads the FINISHED
  database back**. "The fix is live in code" and "this database was built with it" are different
  claims, and conflating them is exactly what produced the retracted featA2 number.
  Built on **F:** (615 GiB free; C: has 138 and already holds the e2s pair and gen 3), **beside**
  gen 3 -- nothing is deleted, and `train_db_5k_h1_id3` is featB's live database. ⚠ Training from
  F: costs ~2.2x per step, so if gen 4 becomes a training target it should move to C: behind a
  junction, which needs gen 3 deleted first -- **Andrew's call, and only after featB reports.**
  **✓ MEASURED 2026-09-02 on gen4base, which trains from F: directly: 0.689 steps/s instantaneous
  (0.579 cumulative incl. warm-up) vs featB's 0.931 on C: = 1.35x slower, NOT 2.2x.** The 2.2x
  figure is real but was measured on the KD teacher DUMP (1.40 vs 0.63 steps/s), a forward-only
  pass that is far more read-bound than a training step. Two workloads, two penalties; quote the
  one that matches. At 1.35x gen4base's WS is ~4.4 h instead of 3.3 h and the whole run ~12-13 h
  -- a cost worth paying once rather than interrupting a gate-critical run to move a database.
  **Also owed on `run_gen4base.cmd` once it finishes (cannot touch a running `.cmd`):** its log
  header echoes `FEATB START` and line 3's REM still describes featB -- cloned strings; the tag,
  dbs and label filter are all correctly gen 4. Same class as the fixc "e2s test db" prose.
* **★★ THE KD TEACHER DOES NOT SURVIVE THE 114-DIM LAYOUT -- SCREENED 2026-09-02, RE-LAY-OUT IS
  DEAD (`scratchpad/teacher_114/`, pre-registered in its `PLAN.md` before launch).** featB ran
  KD-OFF because the d=128 teacher's `features2card` in_dim is 92 and cannot forward 114 dims.
  The cheap fix was to re-lay-out its input projection by column NAME into the 114 layout, which
  costs exactly one thing: the teacher stops seeing `scaled_state`, the column the rebuild drops.
  Measured directly on the published data the teacher was trained for, 300 users, size 0/300
  mismatches, one variable (`RWKV_ZERO_FEATURES=22`, arch via `RWKV_ARCH_MODULE` -- NOT the file
  swap `run_base5k_eval.cmd` uses, since gen 4 was building in the same tree):
  **ahead 0.298203 -> 0.318650 (+0.020447), imm 0.268191 -> 0.276121 (+0.007930).** The
  pre-registered abort line was 0.004 on imm; this is 2x past it, and on ahead the crippled
  teacher (0.3187) is far WORSE than the student it would be teaching (featB 0.2979). A teacher
  worse than its student cannot pay through target-variance reduction. **=> do NOT re-lay-out the
  d=128 teacher.** The features phase therefore either (a) runs KD-OFF, featB's way, forfeiting the
  ~0.0019 KD is worth to this lineage (iters 32/35/39/45), or (b) gets a teacher that natively
  takes 114 dims -- a d=128 model retrained on gen 4 (a full big run, the honest option) or a frozen
  gen-4 checkpoint used as a same-size teacher (cheap, but iter 46 showed a teacher that is not a
  bigger/different function distils nothing). **That choice is Andrew's; it is the first open
  decision of phase 4 and gen4base (KD-off) is the baseline either way.**
  ⚠ The screen bounds the TEACHER's degradation, not the KD gain -- but at this size the bound is
  decisive on its own. Verdict + both arms' jsonls: `scratchpad/teacher_114/t114.log`,
  `result/RWKV{,-P}-t114{a,b}.jsonl`.
* **★★ GEN 5 (`*_id5`) = REAL-TIME CYCLES, `RWKV_REAL_CYCLES=1` -- Andrew 2026-09-02: "use real
  features for 3 days/week/month/year/decade/century, so that every pseudo feature is replaced with
  its real counterpart. If it requires an LMDB rebuild, ok." + "11 is also a pseudo-calendar
  feature, so make sure it also gets replaced."** The pseudo cycles (`prepare_batch.add_encodings`,
  `DAY_OFFSET_ENCODE_PERIODS`) survived the -id rebuild by SCOPE -- it changed only the card-feature
  block -- and were never calendar duplicates: an arbitrary seeded phase from the USER's first day,
  so relative position only, plus a first-review-day half per period (4 dims x 7 = 28).
  **THE FLAG (default OFF, needs `RWKV_ID_FEATURES=1`, needs a rebuild):** same math on the
  epoch-anchored UTC day index of `review_time`, no baseline, so the phase means the same thing for
  every user; each period keeps its first-review half; the review-time 7 d / 365 d halves are NOT
  duplicated (they are dow/doy). **24 card-feature columns** (`id_features.CYCLE_COLUMNS`, the tail
  of `CARD_FEATURE_COLUMNS`) replace 28 encoding dims, and **`day_of_week` (row 11) is dropped** too.
  **Layout: 69 card features + 40 ID dims = 109 input; params 565,252 -> 563,652.** Three-way parity:
  `prepare_batch.add_encodings`, `run_as_rnn.get_tensor` and the width contract
  (`id_features.input_width` / `id_encoding_dims`, both model files' asserts) gate on the same flag;
  Rust derives width from the weights. Because the cycles are card-feature columns they are
  **name-ablatable** via `RWKV_ABLATE_FEATURES`, which the pseudo ones never were.
  **VERIFIED BEFORE ANY CHAIN WAS ARMED, in three processes (the flag is read at import):** flag off
  -> `prepare().start` BIT-IDENTICAL to a snapshot taken BEFORE the edits, on two users (that is what
  protects gen4base, whose decay re-imports these files); flag on -> encoding block is IDs only, all
  24 columns present as unit-circle pairs (2.2e-16), two users on the same UTC day agree on
  `cyc3_sin`/`cyc36500_cos` to 1e-12, first halves constant within a card. Width smoke has the 109
  case. ⚠ TWICE an edit to `id_features.py` failed on a multi-line anchor while a DEPENDENT edit
  succeeded, leaving the -id import path broken for minutes -- caught by the verification, each time.
  **Anchor on one certain line, then verify in a fresh process.**
  **CHAIN:** `wait_then_rebuild5.cmd` fires on `gen4base DECAY_OK` (or a terminal marker) + >=25 GB
  RAM -> `run_rebuild5.cmd` (preflight_gen5 -> width 69 -> Bug C guard -> targets -> train db +
  check_db + idfill + **`check_db_cycles.py`** -> test db, same -> phase 3 compare vs gen 4, now
  REQUIRED IDENTICAL since both share `label_filter_db_id_e2s`). Then `wait_then_realcyc.cmd`
  fires on the ablation chain's marker AND `rebuild5.log` **DONE_EXIT_0** (success, not merely a
  marker) -> `run_realcyc.cmd` = gen4base's recipe + the flag, guard 563,652, tag `realcyc`,
  **control = gen4base, size-gated, single-variable, KD-off both.** Detail: `INPUT_FEATURES.md`
  "Simplified view -- 114-dim layout" (the superseding note).
* ⚠ **THE FEATURES A/B IS BLOCKED ON A DECISION, NOT ON COMPUTE:** featA ran on published dbs that
  still carry Bug A, so it is no longer a clean control for a featB built on fixed dbs -- the fix
  would enter the bundle as a fifth component, and at 19% of reviews it could dwarf the features.
  Rebuilding the published dbs re-bases the champion and is **Andrew's call**.
  **★★ THE FIX IS NOW MEASURED, 2026-08-30, and it is worth more than most accepted iterations.**
  featA2 finished (ahead **0.298186** / imm **0.265588**, n=2500, nan_users 0) against featA
  (0.298334 / 0.265757). Same recipe, same seed, same KD-off env -- the ONLY difference is the
  id-fixed dbs. **Bug A costs +0.000148 ahead / +0.000169 imm, p=4.6e-13 / 2.3e-27**, i.e. it
  clears the 0.0001 accept bar in BOTH modes on its own.
  **=> The champion lineage trained on `train_db_5k_h1`, which still carries Bug A, so ~0.00015 in
  both modes is sitting on the table unclaimed** -- larger than iters 39, 45 or 53 individually.
  That re-prices the rebuild decision: it is no longer "a re-base for cleanliness", it is a re-base
  that BUYS a measured gain. Still Andrew's call, but the cost/benefit is now known rather than
  assumed, and this number is what the "could dwarf the features" worry above was guessing at.
  **⚠⚠ RETRACTED 2026-09-01 -- THE "ONLY DIFFERENCE" SENTENCE ABOVE IS FALSE, AND THIS NUMBER MUST
  NOT PRICE THE REBUILD.** featA trained on `train_db_5k_h1`, **built 2026-07-03**
  (`data_processing_train_5k_h1.toml` + commit `ed0400e`); featA2 trained on `train_db_5k_h1_fix`,
  built **2026-08-21**. The two straddle commit **`c7883dc` (2026-08-19)**, which stopped the -1
  sentinel being SUMMED into the cumulative elapsed columns -- a **GLOBAL** input change hitting
  every card's second review onward (3.9% of rows on an exact collision, 15.4% distorted by
  >0.05 sigma). So `+0.000148 / +0.000169` is **Bug A PLUS the sentinel fix**, not Bug A.
  `data_processing.py:373` says so in as many words: *"Every model trained before this date learned
  the buggy column, so cross-generation comparisons were already invalid."* The warning was in the
  codebase before either arm ran and was not applied to the arms.
  **Corroborated independently, by concentration rather than by dates** -- Bug A only changes
  grouping for cards with missing note metadata, so its effect must live in users who have them.
  It does not (`scratchpad/features_rebuild/nan_note_concentration.py`, on the fixc/iter-53 pair
  which has the same flaw): Spearman rho **-0.019** on ahead; imm runs the **WRONG WAY** (top
  NaN-note quartile **-0.000067**, i.e. the fixed db is BETTER there); and the **621 users with
  <0.5% NaN-note reviews -- for whom the id fixes can do literally nothing -- show the full effect
  anyway** (+0.000047 / +0.000160). A global input change predicts exactly that; an id fix cannot.
  **=> The id fixes remain justified on CORRECTNESS (Bug A collapsed ~19% of reviews into one note;
  Bug C was a train/deploy divergence), and their ACCURACY value is UNMEASURED. Nothing in the
  record isolates it.** Measuring it needs two dbs of the SAME generation differing only in the id
  fill -- the shape the `fixc`/`e2sc` pair used for the interval, which is why that one is valid.
  **THE GENERAL RULE, and it is the third instance this week: two runs are comparable only if their
  DATABASES are, and a db is dated by when it was BUILT, not by what it is named.** Any future A/B
  across a rebuild boundary must either rebuild both arms together or state the bundle explicitly.
* Guards: **`smoke_id_identity.py`** (this doc said `smoke_id_dtype.py`; no such file exists -- corrected 2026-08-26) asserts ENTITY IDENTITY SURVIVES (distinct ids in the sample ==
  distinct in the frame), not merely "no exception" -- a no-exception smoke passes on the broken
  build. `scratchpad/chain_watch.sh` follows whichever ARM is live and alarms on a STALL as well as
  a death; the old watcher followed featA by name and exited when featA finished, which is why featB
  sat deadlocked unnoticed.

**★★ GENERATION 2 DONE 2026-08-20 21:15:05 (`DONE_EXIT_0`, 3 h 57 m, zero GPU cost -- it ran inside
featA's runtime). `train_db_5k_h1_id2` 1,483,984 entries / `test_db_5k_id2` 170,384, BOTH width 46
and BOTH entry counts IDENTICAL to gen 1 -- the integrity check, since chunking is unchanged and a
different count would mean row filtering had moved. Gen 1 kept as a fallback (414 GiB free).**
**(Andrew: "we still have to do another LMDB rebuild") -- 21 -> 23 columns,
input 112 -> 114, params 558,212 -> 565,252 (verified by constructing the model). Adds the two
features his coverage audit found designed-but-never-implemented: `scaled_sibling_gap` and
`card_predates_first_review`.** New dbs `train_db_5k_h1_id2` / `test_db_5k_id2`; `label_filter_db_id`
is REUSED (it selects WHICH reviews count, not what they contain). Runner
`scratchpad/features_rebuild/run_rebuild2.cmd`, tomls `*_id2.toml`. Ran CPU-only inside featA's own
runtime, so featB measures the COMPLETE bundle instead of a 21-column one -- ~11 h saved versus
rebuilding after featB.
**★ THE DISK BUDGET WAS WRONG BY 2.5x: LMDB map_size is SPARSE on Windows, so the file LENGTH is the
RESERVATION, not the allocation.** Gen 1 occupies **115.8 + 115.9 GiB**, not the 372.5 + 232.8 every
plan quoted. Found by accident -- deleting two finished 27.9 GiB de-risk dbs returned **3 GiB**. Use
`GetCompressedFileSize` (or measure free space before/after) for any sparse store; `Get-ChildItem |
Measure Length` is fiction there.
**★ COVERAGE WAS MEASURED BEFORE COMMITTING, AND IT LOWERS THE PRIOR:** the sibling gap is defined on
only **~10-16% of rows**, and the CEILING (rows whose note had another card created earlier) is
**~17%**. `preset_age` was dropped from gen 1 at 7.1%; the deck-tree level reached 49.2% and tied
(iter 50). **Pre-registered: this column is unlikely to move the gate.** It ships anyway on the
asymmetry -- **a column IN the db can be ablated without a rebuild, a column OUT cannot** -- which
is also why the redundancy screen is now interpretation, not a gate.
**✓ THE ABLATION MECHANISM NOW EXISTS -- `RWKV_ABLATE_FEATURES`, landed 2026-08-29, CPU-only,
default OFF and inert** (both model files, plus `scratchpad/parity3/smoke_ablate_features.py`,
7 checks green). Comma-separated COLUMN NAMES resolved through the LIVE `CARD_FEATURE_COLUMNS`, so
one name denotes the same column in both layouts; it unions with the index-based mask and prints
the dims it resolved. It exists because `RWKV_ZERO_FEATURES` is HARD-REFUSED under
`RWKV_ID_FEATURES=1` (`srs_model.py`, `srs_model_rnn.py`) -- correctly, since the rebuild drops the
card-state column so `=22` would mask `day_of_week`.
**★ AN UNKNOWN NAME RAISES, and that is the design point, not a nicety:** a typo that silently
ablated nothing would yield a candidate identical to the champion and a clean null, which reads as
"the feature does not matter" when it means "the experiment did not run". Same family as the rgate
control that inherited its treatment from `os.environ`.
**Rust needs no change:** the engine already honours `RWKV_ZERO_FEATURES`, which takes DIMS, and the
banner prints the resolved dims -- so an ablated model deploys by passing that list. Only if an
ablated model ever becomes champion does the standing recommendation apply (bake the mask into the
exported safetensors instead of an env var).
⚠ The smoke proves the COMPILE half in 4 env combinations but runs no scripted FORWARD; run
`smoke_scripted_eval.sh` before any launch that sets the flag. ⚠ Two of the smoke's own
expectations were stale on first run (it asserted gen-1's 44 cols / 112 width, and picked
`day_of_week` for its "the name really moves between layouts" check -- index 17, which sits BEFORE
the dropped column at 22 and therefore does not move). Both were the TEST being wrong, and the
second would have passed vacuously: **a moves-between-layouts check must pick a column after the
drop point.**
**★ A THIRD COLUMN WAS REJECTED BY A RULE WRITTEN DOWN FIRST:** `scaled_sibling_count` ships only if
>=30% of rows have >=1 prior sibling card; measured 17%, so it does not. Pre-registering the
threshold is what makes that a decision rather than a rationalisation.
⚠ **`mk_features_ab.py` NOW TAKES AN ARM FILTER** (`python mk_features_ab.py featB`). It rewrites BOTH
runners and featA's was RUNNING -- cmd.exe re-reads a batch file from a saved byte offset, the trap
that cost iters 43 and 46.
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

> ★ **2026-09-01: nothing to add here for the PermGather speedup — it is a CODE fix, on by
> default** (`RWKV_PERM_GATHER` defaults to "1"; `=0` is the escape hatch). It is bit-identical, so
> runs before and after it are directly comparable and no env var records which side you are on.
> Worth +5.5% throughput. Detail in the speed section above and `optimization/DISPATCH_PLAN.md`.

    set RWKV_MUON_BATCHED=1     REM batched Newton-Schulz, 35x fewer matmul dispatches
    set RWKV_NO_JIT=1           REM required by torch.compile (worth ~0 alone: 1.003x)
    set RWKV_QAT_COMPILE=1      REM fuses the 26 mixer forwards

**★★★ AND THE END-TO-START DBS, IN TRAIN *AND* EVAL (Andrew 2026-08-30, verbatim: "e2s should be
used both in train AND eval. That should be the new default for all future runs").**

    TRAIN_DATASET_LMDB_PATH = "F:/rwkv_lmdb/train_db_5k_h1_e2s"
    set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_e2s
    set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_e2s

**WHY IT IS A DEFAULT AND NOT AN EXPERIMENT: it closes a train/deploy divergence** (§9 case 4). A
live Anki scheduler computes `now() - last_review_time` = **end-to-START**, and structurally cannot
do otherwise, because `duration(k)` has not happened when the prediction is made. Training on
end-to-END fed the model a quantity deploy can never supply -- and one that correlates with the
outcome (at a fixed gap, `duration(k)` predicts failure at AUC 0.618). We already zero the most
recent duration as a FEATURE for exactly that reason, then handed it back inside the interval.
**Set them EXPLICITLY; the defaults in `write_eval_toml.py` / `write_decay_setup.py` stay on the old
paths so existing runners remain byte-reproducible** -- the same convention as the speed flags above.

⚠ **THREE CONSEQUENCES, none optional:**
1. **THIS RE-BASES THE CHAMPION.** Every number in the record -- iter 53 included -- is end-to-END.
   An e2s run is NOT comparable to them, so the champion recipe must be re-run on the e2s dbs to
   establish the new baseline before any candidate is gated against it.
2. **★ THE KD DUMP MUST BE REGENERATED FIRST, and nothing will tell you if it is not.**
   `C:\rwkv_kd_dump\t128_seedpair_65k` holds teacher logits computed on end-to-END inputs. Its only
   identity check is `labels_sum`, and labels are RATINGS, which the interval does not touch -- so a
   champion re-run on e2s dbs would PASS the checksum while distilling toward predictions for a
   different input. Identical in shape to the augmentation/KD incompatibility already recorded: the
   checksum proves LABEL alignment and gets read as proving BATCH alignment. Regenerate the dump
   with the teacher forwarding the e2s batches.
3. **The `-id` datasets already do this automatically** -- `elapsed_end_to_start` is gated on the
   presence of `review_time`, not on a flag -- so a future `-id` rebuild needs no change, and the
   featB confound noted elsewhere in this file is resolved in the direction of e2s.

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

**HP TUNING IS CLOSED** -- the 2026-07-30 tuner run became **iter 34**, which recovered the
MAX=65536 cost and was the phase's largest gain; the champion HPs have since been confirmed
against 19 alternatives at full eval. The ~44-line LIVE description of that tuner (recipe, lever
order, val-prune reference, per-trial bar) is archived to `HISTORY.md`. **Four things from it stay
operative:**
* **⚠ Tuning `PEAK_LR` alone moves only ~10% of the weights.** Muon has its OWN base LR
  (`RWKV_MUON_LR`) and the schedulers scale it proportionally (`train_rwkv.py:188-196`), so the
  AdamW group is 57,412 params against Muon's 500,800. Any future LR work must move BOTH.
* **1.253 steps/s** is the reference rate for this trunk at MAX=65536 -- but it is the **PLAIN,
  no-KD** recipe. A KD run is ~0.92 steps/s (measured on iters 45 and 53). Do not use one as the
  other's baseline; that mistake read a free flag as 27% slower on 2026-08-17.
* Journal `optimization/tuner_5k_log.jsonl`; the d=32/QAT-era rows are archived to
  `tuner_5k_log_d32qat_era.jsonl` (different arch AND batch -- not comparable).
* Its three runner guards are now generalized in **`scratchpad/preflight_runner.py`** -- run that
  before arming ANY runner.

### ★ THE ORDER FROM HERE
The 2026-08-01 ordering (finish HP tuning -> seed pair -> PAVA lambda) is **COMPLETE** -- those became iters 34, 35 and 36. The ~340-line queue that tracked it, including the speedup phase's ranked list and the per-item DONE annotations, is archived to `HISTORY.md` (2026-08-10). What remains live:

1. **The algorithmic loop** (the endgame's step 1) -- **the ranked proposal queue now lives in `optimization/PROPOSALS.md`**, along with Andrew's 3-agent generation protocol (three subagents with DIFFERENT priors -- literature / domain / reject-log steelman -- each write 5 proposals from >=2 families; rank all 15; implement the top). ⚠ **WRITE THE RANKED LIST TO THAT FILE THE MOMENT IT IS PRODUCED:** the 2026-08-10 ranking lived only in the transcript and a compaction destroyed items 7-15 permanently.
2. **NEW INPUT FEATURES -- the long-lead item. ⚠ ONLY THE PREPROCESSING IS CPU-ONLY (corrected by Andrew 2026-08-12: "Pre-processing is CPU-only, sure, but training is obviously not").** This line used to claim features "do not compete with the GPU loop" -- WRONG, and it would have led to planning them as a free parallel track. Only the ~2-4 day LMDB rebuild overlaps the loop. Everything that makes features *count* -- re-basing the champion on the new inputs, then training + evaluating each candidate -- is GPU work on the same single 4070, and every pre-rebuild iteration is gated against a champion the rebuild invalidates. Features are a PHASE that largely DISPLACES the algorithmic loop, not a parallel one. Fully scoped in `optimization/FUTURE_FEATURES.md`: the four code sites, the F:-side-by-side disk plan (605 GB against 889 GB free -- no delete needed), the measured constants, the ~23 h build, the NaN-clamp landmine, and Andrew's directive that the rebuild DROP Anki's card-state input (dim 22). ⚠ It moves the `size` gate: the filter amplifies a 0.001% raw-row difference into ~30% of users getting a different equalized count, so gate #1 must be read as *within a rebuild generation*.
3. **THEN the 10x-budget run, ONCE, on the final champion** -- see THE ENDGAME above for the two arms (plain, then warm-started QAT), the ~4-day cost, and the three 1-epoch assumptions (warmup 200, augmentation off, wd/dropout) that must be reconsidered first.
4. **Rust port** (`rust/rwkv-infer/TRACK2_PORT_PLAN.md`) -- **★ GAPS 7 + 8 CLOSED AND PARITY-VERIFIED 2026-08-11** (`276f379`): both engines (candle + the default fast path) now run the interleaved schedule (`RWKV_INTERLEAVE=1`) and the reordered stream list (`RWKV_STREAM_ORDER=card,note,deck,preset,user`), against a fresh `reference_iter41/` trace that is self-contained at exactly 0.000e+00 -- interleaved PARITY PASS on both paths (max per-review 4.78e-06 / 1.25e-06) and the sequential path BIT-IDENTICAL to the green iter-31 preds on both. **Gap 8 was a LIVE cross-wiring bug the gate caught** (states were assembled positionally, so `_cnd` fed DECK's state into the NOTE module; `name_to_idx` likewise would have quantized card+DECK for `card,note` scopes). ⚠ Front-loaded placement only -- fine today (iter 44's spread was rejected), but a future spread adoption needs `interleave_schedule()`'s table, not `r < depth[m]`. Remaining measured items and the AGPL/SIMD note are in that plan.

**★★ THE 2026-08-17 HARD FREEZE HAS A DIAGNOSED CAUSE, AND IT IS THIS RULE FIRING (first time with
telemetry).** At 12:53 the box froze mid-eval; Andrew forced a dump with **RightCtrl + Space x2** and
Windows captured it -- bugcheck **0xE2 MANUALLY_INITIATED_CRASH** (all four params zero = the
keyboard-forced path, NOT the underlying fault) plus a 4.4 GB `MEMORY.DMP` and minidump
`081726-8234-01.dmp`. **The flight recorder caught the approach**, which the July hangs did not have:
VRAM pinned at **11,981 of 12,282 MiB (97.6%)** for the last three samples, then the recorder
**stopped dead at 12:53:01**.
⚠ **CORRECTION, measured the same day: low power with high util is NOT the tell.** This entry first
read the accompanying 42-51 W (at util 99%) as "stalled in paging, not computing", and the very next
giant user REFUTED that: on the resume, user 6104 ran at **11,864 MiB / util 97-100% / 51-53 W for
minutes and COMPLETED NORMALLY**, then VRAM fell to 6,756 MiB and power returned to 110 W. Low power
at high util is just what a memory-bound giant user looks like on this card, and it triggered a false
alarm within the hour. **The only reliable freeze signal is the ABSENCE OF FORWARD PROGRESS** -- the
flight recorder ceasing to log, or the shard log's byte count not growing. Check `stat` on
`shard_s0.log` over ~25 s before concluding anything from a power reading. ⚠ Do NOT generalize this to the July black-screens, which
are recorded as having zero telemetry precursor; same symptom, not yet the same proven cause.
**THE TRIGGER: giant user 6104, work 1,274,765** -- ~3.5x the 5002/5905/5995 trio this rule names --
hit inline in a process that had already run 1,103 users, with a heavy desktop (Anki, 2x Edge,
2x Chrome, Word, Excel, Telegram, Steam, Razer). Desktop VRAM was ~1.1 GB after reboot vs the 4.6 GB
this rule cites for its three failures.
**★ AND THE CONFIGURATION IS THE FIXABLE PART: the runners pass `--solo-threshold 0`, which DISABLES
the power-user solo phase** Andrew approved 2026-07-14 for exactly these users. A dry run shows
**25 of the 2500 VAL users are >= 1,000,000 work (11.2% of total work)**, 6104 among them; with the
solo phase on they run alone, first, in their own process. The live rule above says d=80 evals use
`--shards 1 --solo-threshold 0`, and those are INDEPENDENT axes -- `--shards` controls parallel
shards, `--solo-threshold` controls giant isolation. **Turning the solo phase back on is not free
mid-run**: `merge_jsonl` ASSERTS no duplicate users across phases, and 7 of the 25 giants
(5414/5626/5835/5859/5900/5991/6007) are already banked in `-s0.jsonl`, so they would be re-run and
then collide at merge. Fix it at LAUNCH time, not on a resume.
**RECOVERY, and it was cheap:** the 13 h decay phase survived entirely (`qtaxg_i45kd_d_10935.pth` +
both exported catalogs), and `get_result` skips users already present in the output jsonls, so a
plain relaunch resumes at 6104 **in a fresh process** -- most of what the solo phase would buy (clean
allocator, giant first). Runner `scratchpad/qat_tax/run_i45kd_evalresume.cmd` is phase-B-only, sliced
from the chain runner so the env is byte-identical, with asserts that no training/dump phase leaked
in and that nothing deletes the result jsonls.
⚠ **A dirty shutdown truncates the chain log's last write to NUL bytes** (76 of them here, from a
second incident the same afternoon when the PC was switched off). Harmless to the anchored
`findstr /B /C:"DONE_EXIT_"` waiters, but strip them before relaunching or the next append lands
after the padding.

**⚠⚠ POWER OUTAGE 2026-08-18 ~10:32 -- RECOVERED BY MID-EPOCH RESUME, and it exposed a latent bug
in `make_resume.py`.** The box lost power mid-WS; boot 10:33, all four waiters dead, GPU idle.
iter 54 had reached step **8110 of 10,935** with a clean checkpoint pair at 8000 and **zero NUL
bytes** in its log, so ~2.4 h of training was salvageable for ~110 lost steps. Recovery ran the
documented path (`make_resume.py` + `RWKV_RESUME_SKIP_GROUPS=1`) and is confirmed live:
`[resume-skip] epoch 0: skipping the first 8000 already-trained groups`.
**★ THE BUG, and it is the day's recurring shape: `make_resume.py` REPLACED ONLY KEYS THAT
ALREADY EXISTED.** Every runner's WS toml is cloned from a from-scratch config, which declares
`LOAD_MODEL` and `STEP_OFFSET` but NOT `LOAD_MODEL_FOLDER` / `LOAD_MODEL_NAME` -- so those two
were silently never written and the resume died on `AttributeError: 'Namespace' object has no
attribute 'LOAD_MODEL_FOLDER'`. **`train_rwkv` SWALLOWED it and exited 0**, so the runner logged
`WS OK` after 8 seconds and marched on to decay a half-trained model; it was killed within two
minutes and no artifact was lost. Fixed to REPLACE-OR-APPEND with an assert on the OUTPUT toml.
**TWO RULES:** (1) a config transformer must assert the keys are present in what it WROTE, not
trust that they were there to rewrite -- same family as the mk53/mk54 slice and the `endlocal`
marker; (2) **a runner phase must gate on the ARTIFACT, not the exit code** -- the resume runner
now refuses to decay unless `i54_ws_10935.pth` exists, because CLAUDE.md's own warning that
"train_rwkv can swallow fatal errors to exit 0" is exactly what happened.
⚠ Also: `wait_then_iter52.cmd` polls the **QAT#2** log, which already carries a terminal marker,
so re-arming it would have fired instantly and run iter 52 BESIDE iter 54. Replaced by
`wait_then_iter52_v2.cmd`, pointed at `iter54.log`. **After any outage, re-check what each
waiter polls before re-arming it** -- a waiter is only as correct as the log it watches.
⚠ The resumed tail's DROPOUT DRAWS differ from an uninterrupted run (weights/optimizer exact),
so iter 54's number is a fair measurement but the run is not bit-reproducible.

**⚠⚠ THE SINGLE-WITNESS RULE RECURRED 2026-08-20, IN A WATCHER WRITTEN THE SAME DAY -- SO STATE IT
SHARPLY: NEVER IDENTIFY A CHAINED RUNNER BY PID.** A featA watcher pinned the WS phase's pid. A
runner is a CHAIN (WS -> decay -> eval) and every phase is a NEW process, so the normal transition
two minutes after `featA WS_OK` read as `featA DOWN`. Nothing was wrong; the alert measured its own
witness. **Identify the runner by COMMAND LINE** (the `cmd.exe` wrapper spans all phases) **and
require TWO witnesses**: process gone AND no terminal marker in its log. A finished chain writes the
marker, so gone+marker is SUCCESS and only gone+no-marker is a death. Knowing the rule below was not
enough to avoid re-implementing the bug -- which is the argument for the two-witness pattern being
the DEFAULT shape of any monitor here, not a fix applied after a false alarm.

**⚠ THE FLIGHT-RECORDER HANG SIGNAL BREAKS AT MIDNIGHT (2026-08-18, one false alarm).** The
recorder writes `flight_YYYYMMDD.csv`, so at 23:59:47 it stops appending to yesterday's file and
starts today's. A monitor that resolves the filename ONCE at launch then watches a file nothing
will ever write to again and fires ~10 min later -- 23:59:47 + 624 s, exactly when the alert came,
while training was advancing at 0.967 steps/s. **Chains here run ~25 h, so every one crosses
midnight.** Fixed in `scratchpad/chain_monitor.sh`, which (1) re-resolves the newest `flight_*.csv`
each poll and (2) **never declares a hang from the recorder alone** -- a hang stops the BOX, so it
must also stop the training log; requiring BOTH witnesses means one signal failing costs a log
line instead of a false alarm. That second fix is the general one: **an alert built on a single
witness reports the witness's health, not the system's.**

**⚠ BIG-EVAL OPS RULE (learned 2026-07-29/30):** giant users (5002/5905/5995, 266k-367k reviews) OOM the 12 GB card **iff the DESKTOP holds several GB of VRAM** (4.6 GB during three failures vs ~0.5 GB when the same users cleared three evals overnight). `expandable_segments` does NOT help. **Never `del` the result jsonls between eval attempts** -- `eval_sharded` skips completed users, so a relaunch only re-risks the remainder. Check `nvidia-smi` before starting a big eval.

**⚠ CPU-INFERENCE REALITY CHECK:** in the PYTHON RNN path a 4.5x arithmetic cut buys only **1.24x** wall-clock and plateaus -- that path is overhead-bound, so cost tracks op count (layers x streams), not width. **1 thread beats 3 and 6 -> deploy single-threaded.** The Rust path DOES convert the cut: **2.39x** measured. Full numbers: `optimization/CPU_INFERENCE.md`.

#### FAMILY SCOREBOARD (conduct rule 5: 1-2 rejects = deprioritized, NOT closed)

**★★ THE CROSS-FAMILY PATTERN, and it is now THREE independent confirmations (2026-08-19): THE MODEL
USES ANY NEW DEGREE OF FREEDOM IT IS GIVEN, AND *USE IS NOT EVIDENCE OF NEED*.**

| iter | lever | the parameter demonstrably moved | held-out result |
|---|---|---|---|
| 48 | `rcouple_w`, R(t) into the 4 rating logits | learned, **sign-correct** (Again -0.0138) | exact tie, p=0.19/0.37 |
| 50 | deck-tree level embedding | zero-init trained to **L2=1.766**, ~2x a typical input-projection row | exact tie, p=0.52/0.86 |
| 57 | learnable channel-mixer exponent | all 4 live exponents moved **2.0 -> 1.26-1.86**, same direction | exact tie, both inside the floor |

Three different mechanisms -- an architectural coupling, a new scope, a functional form -- one
signature. **=> A "the parameter trained, so the lever engaged" check proves only that the lever is
NOT INERT. It says nothing about whether the loss had anything to gain, and the two questions need
separate evidence.** Report the engagement diagnostic in every such iteration (it is what makes a
null interpretable rather than ambiguous), but never read it as a partial success or as grounds to
retry the same lever harder. **The productive inference runs the other way: when a model moves
decisively into a new freedom and gains nothing, the constraint that freedom removed was not
binding** -- so look for a DIFFERENT constraint, not a bigger dose of the same one.

**expressiveness-vs-capacity 0/1 -- DEPRIORITIZED, NOT CLOSED** (iter 57, the learnable channel-mixer
exponent; the family Andrew opened 2026-08-17). ⚠ **The lever reached 4 of 13 channel mixers** -- 9
`cmix_pow` params get no gradient and the dead set is EXACTLY `RWKV_STRIP_CMIX` (verified as a set
equality). So the honest claim is "null on card:0, note:0, deck:0, deck:3", not "learnable exponents
do not help". **But it is a STRONGER null than that caveat suggests: at the four sites it reached the
lever was FULLY engaged (up to a 37% move), so this is not a too-weak-to-matter result** -- decisive
where tested, silent elsewhere. A second variant must target a richer form at a site that SURVIVES
`RWKV_STRIP_CMIX`, and must clear the redundancy test. ⚠ Do not close this family on one run: it was
opened precisely because "capacity-at-5k is 0/3" had been standing in for an argument it could not
support. Note the overlap with iter 49 from the opposite direction -- it ADDED the user/preset L0
mixers back and got nothing; iter 57 made the SURVIVING mixers richer and got nothing.

**LR-schedule shape 0/1 -- and the follow-up is CLOSED BY ARITHMETIC, not by a second run** (iter 56,
`RWKV_DECAY_SHAPE=linear`). Real but sub-bar vs iter 45: ahead +0.000057 (INSIDE the +/-7.5e-5 floor,
so its reality rests on rank consistency at p=6e-12, not magnitude) / imm +0.000104 (clears both).
Loses to iter 53 at -0.000117/-0.000080. **The obvious follow-up -- does it STACK on iter 53, since
the levers are orthogonal -- was priced BEFORE queueing: under PERFECT additivity the stacked run
sits at +0.000057 ahead, which FAILS the 0.0001 bar. Even the best case cannot clear the gate, so do
NOT spend 6.1 h on it.** ★ The iteration also refutes a general claim: the "same-capacity
rearrangement is indistinguishable" result of iters 41/43/44 is NOT a law about this trunk -- it held
for the curve head and FAILED for the rating head at p=3e-161.

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
quartile incl. its intended beneficiaries — do not retry milder doses) · **optimizer 2/3 -- and iter 53 SPLITS the family in two**
(COVERAGE pays, DESCENT QUALITY does not) (Muon ACCEPTED iter 29, the phase's largest imm gain; cautious wd
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
  **★ AND AN ORPHAN HOLDS ITS LMDB OPEN INDEFINITELY (2026-08-24).** A featA2 fetch worker from
  08-21 was still holding `F:/rwkv_lmdb/test_db_5k_fix` **three days later** (10 CPU-seconds
  total, parent long dead), which made the directory un-renameable with a bare `Access is
  denied` and no indication of why. Diagnose in this order, because the obvious suspect is
  usually wrong: rename a fresh dir on the same volume (rules out permissions), then
  `[IO.File]::Open(path,'Open','ReadWrite','None')` on `data.mdb` (proves a handle exists), then
  find the process by START TIME + near-zero CPU + dead ParentProcessId. My first two theories --
  my own verification handle, then F: permissions -- were both wrong.
- **★★ SEVERAL `F:/rwkv_lmdb/*` PATHS ARE JUNCTIONS TO `C:\rwkv_lmdb\`. DO NOT DELETE
  `C:\rwkv_lmdb` — it is not scratch, it is those databases.**
  **UPDATED 2026-08-30 (the 08-24 list is stale):** `test_db_5k_fix` was DELETED with the other
  two superseded dbs, so its junction is gone too. The current junctions are
  **`train_db_5k_h1_id3`**, **`train_db_5k_h1_e2s`** and **`test_db_5k_e2s`** — the last two moved
  after Andrew made e2s the default, because reading a db from F: costs **2.2x per step**
  (the C:-hosted teacher dump ran 1.40 steps/s, the same dump on F: 0.63, GPU utilisation 8% =
  starved on reads, not computing).
  **✓ CONFIRMED BY THE OUTCOME, not just the microbenchmark:** after the move, the e2s teacher
  dump ran **2 h 03 m** for 10,935 steps -- against 2 h 10 m for the original C:-hosted dump and a
  **4.7 h projection while it was on F:**. The WS phase then resumed the champion's normal
  **0.905 steps/s** (reference 0.92). So the penalty was I/O and the move removed it entirely.
  ⚠ **Deleting a junction is not deleting a copy, and the two paths need different tools.**
  `Remove-Item -Recurse` on the F: path deletes THROUGH the link and destroys the C: data. Safe
  order: delete the real C: directory first, THEN remove the now-dangling link with
  `[IO.Directory]::Delete(path, $false)`. (`cmd /c rmdir` also removes a link without following,
  but the harness blocks that invocation.)
- **★ DELETED 2026-08-30, with Andrew's authorization, to make room:** `train_db_5k_h1` (91.0 GB),
  `train_db_5k_h1_fix` (103.4 GB) and `C:\rwkv_lmdb\test_db_5k_fix` (103.0 GB) — **297.4 GB freed,
  C: 52.2 -> 328 GB.** All three are end-to-END and so superseded by the e2s default; the first two
  additionally carry Bug A / Bug C. **They are REBUILDABLE in ~80 min each** from the read-only
  `anki-revlogs-10k`, so what was given up is bit-reproducibility of iter 53 and featA2 *until* a
  rebuild, not data. Their results are already in the record.
  ⚠ Measure LMDB sizes with `GetCompressedFileSize` — these are SPARSE, and `Get-ChildItem |
  Measure Length` and `Scripting.FileSystemObject.Size` BOTH report the map_size RESERVATION
  (372.5 GB), which is fiction. Free-space before/after is the other reliable method. They were moved to the SSD for speed: measured random-read throughput went
  25.1 -> 643 MB/s and 8.1 -> 346 MB/s (**25x and 43x**); junction overhead is ~10% vs a direct
  C: path. The F: paths are junctions precisely so nothing had to be edited — the db paths are
  hardcoded absolute strings in runners *and inside their `findstr` guard assertions*, so a path
  edit is the clone-a-runner failure mode waiting to happen. `test_db_5k_id3` (37 MB/s) and
  `test_db_5k` (35 MB/s) are still real directories on F:.
  ⚠ **Move them with `lmdb.Environment.copy(dst, compact=True)`, never a file copy** — these are
  SPARSE, so a plain copy materialises the reservation (`test_db_5k_fix` would land as its
  232 GB apparent size, not 103 GB). And budget ~7% GROWTH, not a saving: compaction wrote
  110.6 GB from a 103.0 GB source, because the sparse source's allocated extents undercount what
  a densely-written copy needs. Tools: `scratchpad/workload/move_lmdb.py` (copy + verify) and
  `finalize_lmdb.py` (rename -> junction -> verify through it -> delete original). They are two
  scripts because verifying and renaming in one process fails: the verifier's own handle blocks
  the rename.
- **⚠⚠ OPS -- CLONING A RUNNER MEANS UPDATING EVERY STRING THAT DEPENDS ON THE LEVER, NOT JUST
  THE LEVER. Three failures on 2026-08-18, same shape, one of them caught live.**
  * **iter 54 phase 2a: the ENV was wrong, the guard right.** The champion uses KD alpha **0.9
    for WS** (iter 39) and **0.5 for DECAY** (iter 45); the reset line sits INSIDE the WS phase,
    which a decay-only generator slices away. Phase 2a decayed 3.3 h at 0.9 -- iter 55's lever --
    and its own guard rejected it (`DONE_EXIT_WRONGALPHA_DECAY`). **The guard saved the
    iteration**: the number would otherwise have been a mixture of two experiments. A guard
    DETECTS, it cannot REPAIR.
  * **`decayshape`: the guard was wrong, the env right.** `mk57.py` set alpha to 0.5 but left
    `findstr /C:"alpha FIXED at 0.9"`, so a correct 3.3 h decay would have been rejected at the
    end. **Caught 90 s into the run** by reading the runner rather than trusting it; killed,
    fixed, relaunched for ~1 min of lost GPU. Both generators now assert the guard matches the
    value the runner SETS.
  * **`rgate`: the smoke's control inherited the lever.** `run_iter55.cmd` does
    `set RWKV_RGATE=card` BEFORE calling the smoke, and the smoke built arms with
    `dict(os.environ, **extra)` -- so the OFF arm was gated too. Its param check caught it
    (`rgate keys present with the flag OFF`), but note the inertness check had passed
    **VACUOUSLY at 0.000e+00 while comparing two gated models**. **A test that reads its
    CONTROL's configuration from the ambient environment is not a control.** Fixed by stripping
    the smoke's own vars before applying each arm's.
- **⚠ OPS -- A DECAY-ONLY RUN WRITES ITS CHECKPOINTS INTO THE *SOURCE* RUN'S DIRECTORY.**
  `write_decay_setup.py` takes the dir holding the WS-final checkpoint, so iter 52's decay landed
  in **`scratchpad/iter45_kddecay/i52_d_10935.pth`**, not in `scratchpad/iter52_kdalpha/`. The eval
  toml's `MODEL_PATH` points there and is correct. **Two consequences:** (1) `ls` in the run's own
  directory shows NO decay checkpoint, which looks like a failed decay and is not; (2) deleting an
  old champion's directory during housekeeping would silently take later runs' checkpoints with
  it. iter 57 (`decayshape`) will land there too, since it also decays from `i45_ws_10935`.
  Check `MODEL_PATH` in the eval toml before concluding anything about where a checkpoint is.
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
  or a short `Continue` -- nothing else Claude-originated. **The Telegram bridge is RETIRED (Andrew 2026-08-30)** --
  superseded by Dispatch in the Claude app; task `ClaudeTelegramBridge` is Disabled at the scheduler
  and its processes stopped. ⚠ Removing its master switch had NOT stopped it: the task kept firing
  every 5 min and idling on the absent flag. Reversible (code + config untouched). So the injector is
  now the ONLY injection source, and the two-form limit governs all of it.
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
- **★★ A RUNNER GENERATOR THAT CAN DELETE NEEDS OUTPUT GUARDS, NOT JUST CONTAMINATION GUARDS
  (2026-08-17; caught iters 53 AND 54 armed and broken, ~12 h of GPU saved).** `mk53.py`/`mk54.py`
  build a runner as `HEADER + s[s.index("setlocal"):]`. In the iter-45 runner they copy from, `cd /d`
  and the whole `DIR/LOG/STAMP/DUMP/WSSTEPS/MAXSTEPS` block sit **before** `setlocal` -- so the slice
  silently threw all of it away. (iter 52, generated by a different script that puts them AFTER
  `setlocal`, was unaffected -- which is why this was invisible until all three were compared.)
  **Failure mode is maximally quiet:** `%LOG%` expands to empty, so phase 0's `>> "%LOG%"` is a
  syntax error, the guard fires, tries to log to `""` as well, and exits 37 **without ever writing a
  `DONE_EXIT_` line** -- so a downstream waiter hangs forever and nothing in any log explains it.
  Missing `cd /d` compounds it: `Win32_Process.Create` starts in System32, where
  `.venv\Scripts\python.exe` does not exist.
  **THE RULE:** every generator assert in those files checked that stale text did not leak **IN**
  (no `iter45`, no `i53_`, KD schedule preserved). None checked that required setup **SURVIVED**.
  **★ THE MIRROR-IMAGE GAP, hit 2026-09-01 in my own generator.** `mk_fixc_arm.py` asserted that
  every line it CHANGED carried a db/tag token -- a check against stale text leaking in. It cannot
  catch a line that SHOULD have changed and did not, because no substitution fires on it and the
  line reads as unmodified. Result: `run_fixc_arm.cmd` logs "PHASE 4: rectified VAL-half eval on
  the **e2s** test db" while correctly evaluating fixc. Harmless here only because the GUARD is
  derived from cfg (`findstr /C:"...test_db_5k_fixc"`) while the prose was hardcoded -- so the
  runner cannot actually act on the wrong db, it can only describe itself wrongly.
  **A substitution-based generator needs BOTH directions: no stale token survives, AND every line
  mentioning the old identity was visited.** Grep the output for the SOURCE arm's name and require
  zero hits outside deliberate provenance comments. Same asymmetry as the bullet below (assert
  what leaked IN vs assert what SURVIVED), which is why it was easy to repeat.
  Assert on the OUTPUT: every `%VAR%` the runner references must be declared before its first use,
  and `cd /d` must be present. Same family as the QAT env that was parsed-then-discarded -- the
  banner was truthful and the object it mutated was thrown away.
  **AND VERIFY BY EXECUTION:** the repair was confirmed by running phase-0-only copies (everything
  up to the first training call, log redirected) to exit 0 -- which is also what re-proved iter 54's
  558,225 param count. Reading a `.cmd` does not tell you cmd.exe agrees with you.
  ⚠ Patching an ARMED runner is safe **only** while its waiter is still looping: a `call`ed `.cmd`
  is not open until the call, so the byte-offset hazard above does not apply. Check the waiter log
  says only "waiter armed" first, and keep a byte-exact backup.
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
