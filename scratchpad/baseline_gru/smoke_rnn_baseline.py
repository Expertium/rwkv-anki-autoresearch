"""RNN-baseline smoke (CPU, RWKV_NO_JIT=1 required — set by the caller).

A. Masking-semantics unit test: RNNStream (compact -> pack -> cuDNN -> scatter-back)
   vs a slow per-step reference loop sharing the SAME weights, on random data with
   INTERIOR skips, tail pads, a leading skip, and one all-skip row. Semantics:
   state advances only on non-skipped steps; every position reads the current state;
   positions before the first real token are zero. Tolerance 1e-5 (packed vs stepwise
   kernels differ in accumulation order).
B. Construction + params under the champion depths (card2/deck4/note1/preset3/user3,
   arch = track2_a9's module): GRU ~1.55M, LSTM ~1.52M — the ~1.5M Andrew asked for.
C. Optimizer partition: GRU/LSTM matrix weights land in the main wd group.
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
assert os.environ.get("RWKV_NO_JIT"), "run with RWKV_NO_JIT=1"

from rwkv.model.rnn_baseline import RNNStream

torch.manual_seed(0)

# --- A: masking semantics --------------------------------------------------------
B, T, C, H, L = 4, 23, 16, 16, 2
for cell in ("gru", "lstm"):
    stream = RNNStream(cell, C, H, L, dropout=0.0)
    stream.eval()
    x = torch.randn(B, T, C)
    skip = torch.zeros(B, T, dtype=torch.bool)
    skip[0, 5] = True          # interior skip
    skip[0, 6] = True          # consecutive interior skips
    skip[1, -8:] = True        # tail padding
    skip[2, 0] = True          # leading skip (before first real token)
    skip[2, 10:15] = True      # interior block
    skip[3, :] = True          # all-skip row (fully masked)
    sel = torch.zeros(B, T, dtype=torch.int32)  # ignored by the baseline

    with torch.no_grad():
        out = stream(x, sel, skip)

        # reference: per-row stepwise loop with the same nn.GRU/LSTM weights
        ref = torch.zeros(B, T, C)
        for b in range(B):
            hx = None
            cur = None  # projected output of the last real step
            for t in range(T):
                if not skip[b, t]:
                    o, hx = stream.rnn(x[b : b + 1, t : t + 1], hx)
                    cur = stream.proj(o)[0, 0]
                ref[b, t] = cur if cur is not None else torch.zeros(C)

    d = (out - ref).abs().max().item()
    print(f"A. {cell}: max |vectorized - stepwise ref| = {d:.2e}")
    assert d < 1e-5, f"{cell} masking semantics mismatch"

# --- B: full-model construction + params ------------------------------------------
os.environ["RWKV_ARCH_MODULE"] = "scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py"
os.environ["RWKV_GRU_HEAD"] = "2"
os.environ["RWKV_NO_AHEAD_RESIDUAL"] = "1"
os.environ["RWKV_ZERO_FEATURES"] = "22"
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

for cell, lo, hi in (("gru", 1_500_000, 1_600_000), ("lstm", 1_460_000, 1_560_000)):
    os.environ["RWKV_BASELINE_CELL"] = cell
    m = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    n = sum(p.numel() for p in m.parameters())
    print(f"B. {cell}: params = {n}")
    assert lo < n < hi, f"{cell} params {n} outside [{lo}, {hi}]"
    assert all(type(mod).__name__ == "RNNStream" for mod in m.rwkv_modules)

    # C: optimizer partition — matrix weights must reach a wd group
    n_wd = sum(p.numel() for name, p in m.named_parameters()
               if ".rnn.weight" in name or ".proj.weight" in name)
    print(f"C. {cell}: rnn/proj matrix params (should be wd-grouped): {n_wd}")
    assert n_wd > 1_000_000

os.environ["RWKV_BASELINE_CELL"] = ""
print("SMOKE_ALL_PASS")
