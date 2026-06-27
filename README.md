# rwkv-anki-autoresearch

A small **RWKV-7 neural network for Anki spaced-repetition scheduling**, on the path to
running **CPU-only, inference-only, quantized inside Anki**. See [`CLAUDE.md`](CLAUDE.md)
for the full goal, roadmap, and baseline numbers.

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
