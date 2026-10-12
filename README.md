# rwkv-anki-autoresearch

A small **RWKV-7 neural network for Anki spaced-repetition scheduling**, on the path to
running **CPU-only, inference-only, quantized inside Anki**. See [`AGENTS.md`](AGENTS.md)
for the project guidance, roadmap, and baseline numbers.

This repo vendors the RWKV model + data pipeline from
[`open-spaced-repetition/srs-benchmark`](https://github.com/open-spaced-repetition/srs-benchmark)
(the current top leaderboard entry) and nothing else — the FSRS/LSTM/DASH/… models and
the benchmark harness were intentionally left out.

## Layout

```
rwkv/                 The RWKV-7 model + SRS heads + data pipeline (vendored verbatim).
                      model/   core RWKV7 (parallel CUDA + RNN forms), SRS model, CUDA kernel (csrc/)
                      *.py     data_processing, get_result (eval), run_as_rnn (CPU inference), train_rwkv
config.py             Benchmark Config + CLI parser (needed by the data pipeline).
utils.py              TRIMMED: only get_bin / count_lapse / cum_concat (RMSE-bin bucketing).
features/             TRIMMED to the FSRS feature engineer only (create_features for the
                      "equalized" test-set definition). Other engineers + models/ removed.
setup.py              Builds the RWKV CUDA/C++ kernel (rwkv.model.RWKV_CUDA).
pretrain/             Pretrained weights (git-ignored; copy from ../srs-benchmark/pretrain).
```

Sibling read-only repos on this machine: `../srs-benchmark` (vendor source + benchmark),
`../anki-revlogs-10k` (dataset), `../fsrs-autoresearch`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv -r requirements.txt
```

Verify the package imports standalone:

```bash
.venv/Scripts/python.exe -c "import rwkv.get_result, rwkv.run_as_rnn; print('ok')"
```

## Status / notes

- **Vendored + imports standalone.** ✅
- **CUDA kernel not yet built.** The parallel training/eval form (`get_result.py`) needs the
  compiled `rwkv.model.RWKV_CUDA` extension (`torch.ops.rwkv.*`); the pure-PyTorch CPU path
  raises "Not supported". Building it is blocked on a CUDA-toolkit-vs-torch-wheel mismatch
  (system CUDA 13.2 vs the cu126 torch wheel). The **RNN inference path** (`run_as_rnn.py`,
  CPU, pure tensor ops) sidesteps the kernel and is the proof-of-life route.
- **Storage:** the full 10k preprocess won't fit on the SSD; work on subsets. On Windows,
  LMDB reserves its full `map_size` on disk immediately — always shrink `LMDB_SIZE` for
  subset runs. (Details in the project memory.)
- **Threads:** cap data-pipeline parallelism at 7 on this machine.

## Scoring ONE foreign collection against the benchmark (added 2026-09-16)

Every stage takes `--config <toml>` on the command line and reads it at IMPORT time (`rwkv/parse_toml.py`,
`parse_known_args`, so the flag passes through the srs-benchmark parser too). Pass it on the command line,
never by injecting settings into the parent process: on Windows the `multiprocessing` pool spawns fresh
interpreters that re-import the module and read `--config` from `sys.argv`; with nothing there the workers
die or block at import and the parent waits forever (a silent hang at ~4 s of CPU -- seen by the Clanki
session 2026-09-16 on a single-user run).

Single user, single process, no dependency on the 10k dataset (each user is read from its own parquet):
copy `rwkv/find_equalize_id_e2s.toml`, set `USER_IDS = [<id>]` (a list; takes precedence over the range),
`PROCESSES = 1`, `DATA_PATH` to an export in the dataset layout (`revlogs/user_id=<id>/`, `cards/`, `decks/`
partitioned by user_id), and a NEW `LABEL_FILTER_LMDB_PATH`. Then:

    .venv\Scripts\python.exe -m rwkv.find_equalize_test_reviews --config <toml>
    .venv\Scripts\python.exe -m rwkv.data_processing --config <test-db toml>
    .venv\Scripts\python.exe -m rwkv.get_result --config <eval toml>

`get_result`'s per-user jsonl carries LogLoss, RMSE(bins) and AUC on the benchmark's equalized set (first
reviews and zero-gap rows out, `max_seq_len * 2 = 128` rows per card, the last 5/6 of the surviving rows).
The interval branch is chosen by the DATASET: an export with `review_time` takes the -id end-to-start
correction, one without takes the published end-to-end one; match the checkpoint's training.
