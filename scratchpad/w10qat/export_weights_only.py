"""Convert a .pth checkpoint to the safetensors the Rust engine loads -- WEIGHTS ONLY.

`export_rnn_trace.main()` also replays three users through the Python deploy RNN at ~25 rows/s,
which is minutes-to-hours and is only needed for a PARITY trace. Fitting a PQ catalog needs the
weights and nothing else: the per-user trace files are model-independent INPUTS, so the existing
`reference_realcyc/trace_user_*` drive any gen-5 checkpoint.

Env is the caller's (RWKV_CHAMP_CKPT, RWKV_REF_DIR, the arch flags) -- this file only skips the
expensive half.
"""
import os
import sys

sys.path.insert(0, "C:/Users/Andrew/rwkv-anki-autoresearch")
import export_rnn_trace as ex  # noqa: E402  (env must already be set)

ex.OUT_DIR.mkdir(parents=True, exist_ok=True)
print("checkpoint:", ex.MODEL_PATH)
print("out dir:   ", ex.OUT_DIR)
ex.export_weights()
sft = ex.OUT_DIR / ex.WEIGHTS_SFT
assert sft.exists() and sft.stat().st_size > 1_000_000, "safetensors missing or implausibly small"
print("OK", sft, sft.stat().st_size, "bytes")
