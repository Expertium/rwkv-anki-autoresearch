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

## Current references (check these before acting)
- `PROJECT_MAP.md` - folders and key files. Update it when the tracked project structure changes.
- `optimization/GPU_RUN_BOARD.md` - active runs and queue. Read this first before scheduling experiments.
- `scratchpad/adaptive_wd/NOTES.md` - adaptive-WD lineage, gates, and resume hazards; use the run board for live status.
- `scratchpad/ta3/NOTES.md` - TA3 (Training-Time Testing) arm: pre-registered endpoints, gate history, epoch plan, monitoring, resume hazards.
- `scratchpad/quant_plan/RESULTS.md` + `scratchpad/quant_plan/NOTES.md` - card/note state-quantization arm (anchor `iter36`): results table with bit/speed columns, the QAT-vs-PTQ tax decomposition, live status and the queued screening chains. **Gate (Andrew, 2026-11-20): the anchor is the run that is identical EXCEPT for quantization**, so deltas are quoted against `qnj_it36ctl` (124-step decay) or `qnj_it36ctl496` (496-step decay), never against the iter36 champion. Screening chains take a lock file (`scratchpad/quant_plan/it36qatsmall*_chain.lock`); delete a stale lock by hand after an outage, and do not add a chain's own script to its busy pattern (the venv `python.exe` is a shim, so one chain shows two processes).
- `optimization/PROPOSALS.md` - ranked research queue and proposal rules.
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
