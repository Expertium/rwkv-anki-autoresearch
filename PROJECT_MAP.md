# Project map

This map describes the maintained source tree and the main tracked artifacts. It does not list every generated file in `scratchpad/`, prediction dump, or run directory. Those files are experiments and artifacts, not separate subsystems. Update this map when the top-level layout or a major subsystem changes.

## Root

| Path | Purpose |
|---|---|
| `AGENTS.md` | Short active rules for agents working in this repository. |
| `PROJECT_MAP.md` | This map of the repository. |
| `README.md` | Repository purpose, setup notes, and single-collection scoring guide. Check status claims against current code; some setup text may be old. |
| `RESUME_AFTER_PAUSE.md` | Notes for resuming work after a project pause. |
| `INPUT_FEATURES.md` | Definitions and status of model input columns. |
| `build_dataset.py` | Dataset preparation entry point. |
| `config.py`, `utils.py` | Shared configuration and utility code used by the vendored pipeline. |
| `features/` | Minimal feature engineering code needed to define benchmark rows and the equalized evaluation set. |
| `rwkv/` | RWKV model, training, evaluation, feature builders, and data pipeline. See below. |
| `setup.py`, `requirements.txt` | CUDA/C++ extension build and Python dependencies. |
| `export_rnn_trace.py`, `make_reference.py`, `verify_rust.py` | Export model traces and weights, create reference data, and check Rust parity. |
| `test_users.json` | User selection for reference and parity tests. |
| `rust/rwkv-infer/` | Rust CPU inference engine. |
| `optimization/` | Research plans, experiment records, gates, evaluation tools, and current run board. See below. |
| `scratchpad/` | Experiment-specific runners, smokes, profiles, temporary tools, and run artifacts. See below. |
| `reference/`, `reference_*/` | Parity traces, metrics, and quantization codebooks. Large trace and weight files can be ignored or local-only. |
| `preds*/` | Rust prediction outputs used for parity comparisons. |
| `vendor/jschoreels_anki/` | Read-only reference copy of an Anki fork. Check `NOTICE.md` before reusing its code. |
| `2412.10856v4.pdf`, `2505.03803v1.pdf` | Local copies of the RWKV-edge and RWKVQuant research papers. |

## `rwkv/`

| Path | Purpose |
|---|---|
| `architecture.py`, `config.py` | Model architecture and project-specific configuration. |
| `model/` | Parallel and recurrent RWKV cores, SRS heads, PAVA, and CUDA/C++ kernels in `csrc/`. |
| `train_rwkv.py`, `train_rwkv_config.toml` | Training entry point and default training configuration. |
| `get_result.py`, `get_result_config*.toml` | CUDA evaluation and evaluation configurations. |
| `run_as_rnn.py`, `run_as_rnn_config*.toml` | Single-user recurrent inference and smoke configuration. |
| `data_processing.py`, `data_fetcher.py`, `prepare_batch.py` | Convert review data into LMDB rows and batches. |
| `find_equalize_test_reviews.py`, `find_equalize_*.toml` | Build the equalized evaluation label filter and related bins. |
| `id_features.py`, `extra_features.py`, `deck_tree.py` | Optional timestamp, row-level, and deck-hierarchy feature builders. |
| `clock_fix.py`, `probe_impute.py`, `self_feedback.py` | Optional input transformations and feedback paths. |
| `muon.py`, `sam.py`, `fsrs_core.py`, `fsrs_stream.py` | Optimizer, training, and experimental model components. |
| `model/rwkv_ops.py`, `model/csrc/` | RWKV kernel wrapper and CUDA/C++ implementation. |
| `data_processing_*`, `get_result_config*`, `run_as_rnn_config*` | Dataset-generation and evaluation configurations. Check each file before use; names include old lineages. |

## `optimization/`

| Group or path | Purpose |
|---|---|
| `GPU_RUN_BOARD.md` | Current active and queued GPU work. Read this first before scheduling experiments. |
| `PROPOSALS.md` | Ranked research queue and proposal-generation rules. |
| `research_5k.md`, `research_log.jsonl` | Current experiment table and structured experiment source of truth. |
| `research_5k_notes.md`, `research_5k_verbose.md` | Methodology and detailed experiment notes. |
| `LEDGER.md`, `LOGLOSS_BREAKDOWN.md` | Measured changes in LogLoss and their decomposition. |
| `log.md`, `log.jsonl`, `logbook.py` | Regenerated canonical experiment table and its builder. |
| `PROTOCOL.md`, `gate.py`, `paired_pvalue.py`, `size_baseline.py` | Experiment procedure and acceptance checks. |
| `DEPLOY_FUNCTIONS.md`, `DEPLOY_SPEED_LOG.md`, `CPU_INFERENCE.md`, `cpu_infer_bench.py` | Deployment contract, speed records, and CPU inference tools. |
| `DATASETS.md`, `FUTURE_FEATURES.md`, `INPUT_FEATURES.md` | Data selection and feature definitions/plans. |
| `HISTORY.md` | Archived plans and historical live-state notes. Verify old claims against current code and run records. |
| `CLOUD_COMPUTE.md`, `TRAINING_SPEED.md`, `DISPATCH_PLAN.md`, `WORKLOAD_EFFICIENCY.md` | Compute and training-speed investigations. |
| `PHASE5_PLAN.md`, `WS_TUNE_PLAN.md`, `LIT_REVIEW.md`, and other `*.md` plans | Phase plans, literature notes, and closed or contingent directions. Check `GPU_RUN_BOARD.md` for current state. |
| `champion_*.json`, `*_log.jsonl`, `*_baseline*.json` | Champion pointers, experiment journals, and review-count baselines. Do not treat all old journals as active. |
| `gate.py`, `eval_sharded.py`, `model_stats.py`, `measure_throughput.py`, `promote_champion_5k.py` | Main research-gate, evaluation, model-size, throughput, and promotion tools. |

## `rust/rwkv-infer/`

- `src/main.rs` — command-line entry point.
- `src/model.rs` — model loading, state, and core inference logic.
- `src/fast.rs` — optimized inference path.
- `src/pava.rs` — monotonic curve correction and interval logic.
- `Cargo.toml`, `Cargo.lock` — Rust crate definition and locked dependencies.
- `BATCHING_PLAN.md`, `TRACK2_PORT_PLAN.md` — batching and port design notes.

## `scratchpad/`

This directory is large by design. Most tracked subdirectories are named for one run or one short investigation. Use the matching record in `optimization/research_5k.md`, `optimization/research_5k_verbose.md`, `optimization/HISTORY.md`, or `optimization/GPU_RUN_BOARD.md` to understand a run. Do not load every run directory into context.

| Area | Examples | Purpose |
|---|---|---|
| Runner safety | `preflight_runner.py`, `mk_dryrun.py`, `detach.ps1`, `make_resume.py`, `count_skips.py` | Validate, start, resume, and check long-running experiments. |
| GPU/RAM monitoring | `gpu_cold_check.py`, `gpu_cold_watch.cmd`, `ram_guard.py`, `ram_guard.ps1` | Local resource monitoring and notifications. |
| Parity and model smokes | `parity3/`, `qat_parity/`, `id_features/`, `deck_tree/` | Focused correctness checks across training, evaluation, and inference paths. |
| Data and feature work | `dataset_id/`, `features_rebuild/`, `rowtables/`, `newcards/`, `hybrid100k/` | Dataset builders, feature tables, rebuild checks, and input experiments. |
| Quantization and QAT | `lfqat/`, `qat_tax/`, `qat_gen5/`, `pqcap/`, `s64hrefine/` | Codebook fitting, quantization, kernel capacity, and QAT research. |
| Architecture and optimization studies | `shrink/`, `siblingrating/`, `wstune/`, `optimizer_regime/`, `arch_2026-09-07/` | Model size, feature, optimizer, and architecture experiments. |
| Run directories | `iter*`, `track2_*`, `w10*`, `s64*`, `wt*`, and other named experiment folders | Runner configs, logs, checkpoints, and result-specific artifacts. Their names are not a live queue. |
| Profiles and benches | `bench/`, `dispatch/`, `evalspeed/`, `kernel_ab/`, `prof_*`, `cpu_bench/` | Timing, profiling, and performance experiments. |

## `reference*/`, `preds*/`, and `vendor/`

- `reference/` contains shared traces, reference weights metadata, prediction vectors, and quantization codebooks.
- `reference_a18/`, `reference_iter*/`, `reference_realcyc/`, and `reference_s64h/` identify parity baselines for specific model lineages. `ref_metrics.json` records the associated export metadata.
- `preds/`, `preds_a18/`, `preds_a18v/`, and `preds_candle/` contain engine prediction outputs for parity checks. Treat them as generated outputs.
- `vendor/jschoreels_anki/` is a read-only external implementation reference. Its license and reuse conditions are in `vendor/jschoreels_anki/NOTICE.md`.

## Maintenance rule

Update this map when a tracked top-level path or a major subsystem changes. Do not document every per-run file here; document its purpose in the run record instead.
