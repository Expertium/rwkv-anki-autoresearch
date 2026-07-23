"""Classic-RNN stream baselines (GRU / LSTM) — Andrew's directive 2026-07-23.

Drop-in replacements for the per-stream RWKV7 stacks: same 5-stream hierarchy, same
input FC, same heads, same pipeline — ONLY the recurrent cell is swapped, to measure
whether RWKV-7's complexity is needed. Env: RWKV_BASELINE_CELL=gru|lstm (consumed in
srs_model.py), RWKV_BASELINE_HIDDEN=<int> (default: gru 128 / lstm 104 — sized so the
full model lands ~1.5M params, matching the track-2 champion scale).

Skip semantics (must match the WKV kernel; interior skips exist — e.g. track-1's
button-probe rows): at skip positions the recurrent state must NOT advance, but the
position still reads the CURRENT state (probe semantics). Implemented vectorized:
  1. stable-argsort each row so non-skipped positions form a dense prefix in original
     order; gather the inputs into that compact layout;
  2. cuDNN GRU/LSTM over pack_padded_sequence(lengths = #non-skipped);
  3. scatter back via one gather: position t reads compact output index
     cumsum(~skip)[t]-1 = its own step if real, else its last real predecessor;
     positions before the first real token are zeroed.
The time_shift_select_BT input is accepted for interface parity and ignored — the
token-shift input mix is RWKV machinery; classic cells read only x_t (the point of
the baseline).

REQUIRES RWKV_NO_JIT=1 (pack/pad + cuDNN RNN under TorchScript is not worth fighting;
the constructor raises otherwise). cuDNN RNN determinism under RWKV_DETERMINISTIC=1
needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the env (set in the run cmd).
"""

import os

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rwkv.model.rwkv_model import ModuleType, time_shift_gather


class RNNStream(ModuleType):
    def __init__(self, cell: str, d_model: int, hidden: int, n_layers: int,
                 dropout: float, stream_name: str = ""):
        super().__init__()
        if ModuleType is not torch.nn.Module:
            raise RuntimeError(
                "RNNStream requires RWKV_NO_JIT=1 (TorchScript ModuleType detected)")
        rnn_cls = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM}[cell]
        self.rnn = rnn_cls(
            input_size=d_model, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = (torch.nn.Linear(hidden, d_model)
                     if hidden != d_model else torch.nn.Identity())
        self.stream_name = stream_name

    def forward(self, in_BTC, time_shift_select_BT, skip_BT):
        # re-flatten after device moves / dtype casts (no-op when already compact;
        # avoids cuDNN re-compacting the weights on every call)
        self.rnn.flatten_parameters()
        B, T, C = in_BTC.shape
        keep_BT = ~skip_BT
        lengths_B = keep_BT.sum(dim=1)
        # non-skipped positions -> dense prefix, original order preserved
        order_BT = torch.argsort(skip_BT.int(), dim=1, stable=True)
        x_comp = time_shift_gather(in_BTC, order_BT.to(torch.int32))
        packed = pack_padded_sequence(
            x_comp, lengths_B.clamp(min=1).cpu(), batch_first=True,
            enforce_sorted=False,
        )
        out_packed, _ = self.rnn(packed)
        out_comp, _ = pad_packed_sequence(
            out_packed, batch_first=True, total_length=T)
        out_comp = self.proj(out_comp)
        # position t reads compact index cumsum(keep)-1: own step if real, else the
        # last real predecessor (probe semantics); zero before the first real token
        cum_BT = keep_BT.to(torch.int32).cumsum(dim=1)
        take_BT = (cum_BT - 1).clamp(min=0)
        out_BTC = time_shift_gather(out_comp, take_BT.to(torch.int32))
        return out_BTC * (cum_BT > 0).unsqueeze(-1).to(out_BTC.dtype)


def baseline_hidden_default(cell: str) -> int:
    # sized so the full 5-stream model lands ~1.5M params at d_model=128 with the
    # champion depths (card2/deck4/note1/preset3/user3 = 13 layers):
    #   gru  h=128: 13 * 99,072  = 1,287,936 (+ trunk/heads ~265k) ~= 1.55M
    #   lstm h=104: ~1,185,600 + 5 projections (13,440 each) + trunk ~= 1.52M
    return {"gru": 128, "lstm": 104}[cell]


def env_baseline_cell() -> str:
    return os.environ.get("RWKV_BASELINE_CELL", "").strip().lower()
