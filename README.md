# rwkv-anki-autoresearch

A small **RWKV-7 neural network for Anki spaced-repetition scheduling**, on the path to
running **CPU-only, inference-only, quantized inside Anki**. See [`AGENTS.md`](AGENTS.md)
for the active project rules and [`PROJECT_MAP.md`](PROJECT_MAP.md) for the repository layout.

This repo uses the RWKV model + data pipeline from
[`open-spaced-repetition/srs-benchmark`](https://github.com/open-spaced-repetition/srs-benchmark)
(the current top leaderboard entry) and nothing else — the FSRS/LSTM/DASH/… models and
the benchmark harness were intentionally left out.

This repo is mostly inactive, work is continued in [https://github.com/JSchoreels/anki-rwkv](https://github.com/JSchoreels/anki-rwkv).
