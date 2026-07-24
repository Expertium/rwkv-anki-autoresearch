"""Classic-RNN stream baselines (GRU / LSTM) — Andrew's directive 2026-07-23.

Drop-in replacements for the per-stream RWKV7 stacks: same 5-stream hierarchy, same
input FC, same heads, same pipeline — ONLY the recurrent cell is swapped, to measure
whether RWKV-7's complexity is needed. Env: RWKV_BASELINE_CELL=gru|lstm (consumed in
srs_model.py), RWKV_BASELINE_HIDDEN=<int> (default: gru 128 / lstm 104 — sized so the
full model lands ~1.5M params, matching the track-2 champion scale).

Skip semantics (must match the WKV kernel — skip rows are the pipeline's QUERY rows:
one per non-first review, outcome columns zeroed, elapsed/interval features KEPT,
carrying the labels; ~half of every sequence): at skip positions the recurrent state
must NOT advance, but the output must be an UNCOMMITTED one-step cell evaluation
Cell(x_skip, h_prev) — the prediction conditions on the query's features (elapsed
time!) against the pre-review state, exactly like the WKV kernel's r/k/v-from-x_t
readout of the un-advanced state. v1 of this module returned the bare h_prev at skip
rows — predictions were interval-blind, collapsing imm onto ahead and scoring 0.415
(worse than the featureless blind-RWKV). Implemented vectorized:
  1. stable-argsort each row so non-skipped positions form a dense prefix in original
     order; gather the inputs into that compact layout;
  2. per-layer cuDNN GRU/LSTM over the padded compact rows (NO PackedSequence: the
     2026-07-24 profile showed its per-call lengths.cpu() sync was 94% of the step;
     the recurrence is CAUSAL so pad-tail compute cannot affect any position we read
     back — every read index cumsum(keep)-1 < length sees only its clean prefix, and
     the carried state of already-ended rows is never read. Pad inputs are gathered
     real features, finite, no NaN risk);
  3. scatter back via one gather: position t reads compact output index
     cumsum(~skip)[t]-1 = its own step if real, else its last real predecessor;
     positions before the first real token are zeroed.
The time_shift_select_BT input is accepted for interface parity and ignored — the
token-shift input mix is RWKV machinery; classic cells read only x_t (the point of
the baseline).

v3 (2026-07-24, after the mid-WS sensitivity audit `probe_sensitivity_check.py`):
PRE-NORM PER-LAYER RESIDUALS — x = x + proj(Cell(LN(x))). v2 stacked bare cells
(classic nn.GRU style, no residuals); the trained gates saturated toward retention
(z→1) and the one-step probe signal attenuated ~3-10x PER LAYER, so query features
died before the heads (10.9 at features2card → 1e-6 at the user stream → imm
predictions effectively interval-blind AGAIN, despite the v2 probe fix being
mechanically correct). RWKV never has this failure mode because each layer is a
residual block (x = x + att(ln(x))) — the query row's features ride the residual
stream to the heads at full strength. Keeping the residual skeleton and swapping
only the recurrence operator is also the standard way attention-vs-RNN ablations
are built, so v3 is the fairer "is RWKV needed" baseline. Every layer now reads
the d_model residual stream (input_size=d_model for all layers); per-layer
projs (Identity when hidden==d_model) map cell space back to the stream. Module
names: `.rnn.` / `.rnn_norms.` / `.projs.` — ALL matched by DTYPE_EXCLUDE
(".rnn" and ".proj" are substrings), so the whole stream stays fp32. LayerNorm
weights are 1-D → the optimizer's dim-based grouping auto-assigns wd=0.

Memory (2026-07-24, the 33 GB peak_reserved WDDM-paging incident): fp32 activations +
cuDNN training reserves across 13 layers x ~32k tokens blow past 12 GB — each
(layer, window) segment is gradient-CHECKPOINTED (recompute in backward). Safe with
dropout because the stack is per-layer single-layer cuDNN modules with torch-RNG
F.dropout BETWEEN layers (outside the checkpoints; identical math to
nn.GRU(dropout=)) — cuDNN's internal dropout RNG would NOT survive recompute.

Long sequences: cuDNN rejects T beyond ~65k (CUDNN_STATUS_NOT_SUPPORTED on the
~229k-review mega user at eval) — the recurrence runs in RNN_WINDOW-sized T-windows
with the hidden state carried across windows, mathematically EXACT for GRU/LSTM.

REQUIRES RWKV_NO_JIT=1 (cuDNN RNN + checkpoint under TorchScript is not worth
fighting; the constructor raises otherwise). Runs use RWKV_DETERMINISTIC=0 (cuDNN RNN
backward is nondeterministic) and RWKV_EXIT_HARD=1 (Windows cuDNN-RNN native teardown
crashes with 0xC0000409 after successful runs).
"""

import os

import torch
import torch.utils.checkpoint

from rwkv.model.rwkv_model import ModuleType, time_shift_gather


class RNNStream(ModuleType):
    # cuDNN seq-length ceiling is ~65k; also the checkpoint grain (env-tunable)
    RNN_WINDOW = int(os.environ.get("RWKV_RNN_WINDOW", "32768"))

    def __init__(self, cell: str, d_model: int, hidden: int, n_layers: int,
                 dropout: float, stream_name: str = ""):
        super().__init__()
        if ModuleType is not torch.nn.Module:
            raise RuntimeError(
                "RNNStream requires RWKV_NO_JIT=1 (TorchScript ModuleType detected)")
        rnn_cls = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM}[cell]
        # per-layer single-layer modules (not one multilayer module): inter-layer
        # dropout moves OUTSIDE cuDNN (torch RNG -> checkpoint-safe), and each
        # (layer, window) segment can be checkpointed independently.
        # v3: every layer reads the d_model residual stream (pre-norm residual).
        self.rnn = torch.nn.ModuleList([
            rnn_cls(input_size=d_model, hidden_size=hidden,
                    num_layers=1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.rnn_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.projs = torch.nn.ModuleList([
            (torch.nn.Linear(hidden, d_model)
             if hidden != d_model else torch.nn.Identity())
            for _ in range(n_layers)
        ])
        self.dropout_p = float(dropout)
        self.stream_name = stream_name

    def _apply(self, fn, recurse=True):
        # Keep this stream's params fp32 through ANY dtype cast (selective_cast's
        # module walk casts PARENTS recursively, so the DTYPE_EXCLUDE name list alone
        # can't protect a nested module -- the 03:36 copy_downcast_ assert crash).
        # Device moves pass through; float tensors get re-pinned to fp32.
        def keep_fp32(t):
            out = fn(t)
            if (out is not None and t.is_floating_point()
                    and out.dtype != torch.float32):
                out = out.to(torch.float32)
            return out
        self._flat_ok = False  # param storages replaced -> re-flatten lazily
        return super()._apply(keep_fp32, recurse)

    @staticmethod
    def _layer_call(layer, w, hx):
        return layer(w) if hx is None else layer(w, hx)

    def _run_layer_windowed(self, layer, x):
        T = x.size(1)
        hx = None
        outs = []
        for t0 in range(0, T, self.RNN_WINDOW):
            w = x[:, t0:t0 + self.RNN_WINDOW].contiguous()
            if self.training and torch.is_grad_enabled():
                out_w, hx = torch.utils.checkpoint.checkpoint(
                    self._layer_call, layer, w, hx, use_reentrant=False)
            else:
                out_w, hx = self._layer_call(layer, w, hx)
            outs.append(out_w)
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    def forward(self, in_BTC, time_shift_select_BT, skip_BT):
        # re-flatten only after param storages changed (device move / cast) -- the
        # per-call variant forced a compaction copy every forward
        if not getattr(self, "_flat_ok", False):
            for layer in self.rnn:
                layer.flatten_parameters()
            self._flat_ok = True
        in_dtype = in_BTC.dtype
        B, T, C = in_BTC.shape
        keep_BT = ~skip_BT
        # non-skipped positions -> dense prefix, original order preserved
        order_i32 = torch.argsort(skip_BT.int(), dim=1, stable=True).to(torch.int32)
        cum_BT = keep_BT.to(torch.int32).cumsum(dim=1)
        take_i32 = (cum_BT - 1).clamp(min=0).to(torch.int32)
        alive = (cum_BT > 0).unsqueeze(-1)  # False only at leading skips (zero-state)
        skip_col = skip_BT.unsqueeze(-1)
        # stream weights stay fp32 (DTYPE_EXCLUDE '.rnn'/'.proj'): bf16 nn.GRU/LSTM
        # falls off cuDNN onto a 30x-slower native path -- cast at the boundary
        x = in_BTC.float()
        for i, layer in enumerate(self.rnn):
            xn = self.rnn_norms[i](x)
            # committed pass over the real rows (dense prefix; the skip-tail is
            # processed too but its outputs are never read -- causality)
            x_comp = time_shift_gather(xn, order_i32)
            out_comp = self._run_layer_windowed(layer, x_comp)
            # base: real position -> its own committed h; skip -> predecessor's h
            base = time_shift_gather(out_comp, take_i32)
            # PROBE pass (the v1 bug fix): a skip/query row's output must be the
            # UNCOMMITTED one-step Cell(x_query, h_prev) -- the prediction conditions
            # on the query's features (elapsed time) against the pre-review state,
            # like the WKV kernel's x_t-derived readout of the un-advanced state.
            # Sync-free: probe ALL positions as length-1 sequences in ONE cuDNN call
            # (real positions' probes are garbage and discarded by the where()).
            h_prev = (base * alive.to(base.dtype)).reshape(1, B * T, -1).contiguous()
            probe_in = xn.reshape(B * T, 1, xn.size(-1))
            probe_out, _ = self._layer_call(layer, probe_in, self._probe_hx(h_prev))
            inner = torch.where(skip_col, probe_out.reshape(B, T, -1), base)
            if self.dropout_p > 0:
                inner = torch.nn.functional.dropout(
                    inner, self.dropout_p, self.training)
            # v3 residual: query features (and every layer's input) ride the stream
            # to the heads regardless of gate saturation
            x = x + self.projs[i](inner)
        return x.to(in_dtype)

    def _probe_hx(self, h_prev):
        # GRU hidden = h; LSTM hidden = (h, c). cuDNN does not expose per-step c, so
        # the LSTM probe starts from c=0 ("fresh-cell readout") -- h_prev-conditioned
        # but NOT the exact LSTM probe semantics; documented baseline caveat.
        if isinstance(self.rnn[0], torch.nn.LSTM):
            return (h_prev, torch.zeros_like(h_prev))
        return h_prev


def baseline_hidden_default(cell: str) -> int:
    # sized so the full 5-stream model lands ~1.5M params at d_model=128 with the
    # champion depths (card2/deck4/note1/preset3/user3 = 13 layers). v3 (pre-norm
    # residual): every layer reads d_model, plus per-layer LN (256) and, when
    # hidden != d_model, a per-layer proj:
    #   gru  h=128: 13 * (99,072 + 256)            = 1,291,264 (+ trunk ~265k) ~= 1.56M
    #   lstm h=92:  13 * (81,696 + 11,904 + 256)   = 1,220,128 (+ trunk ~265k) ~= 1.49M
    return {"gru": 128, "lstm": 92}[cell]


def env_baseline_cell() -> str:
    return os.environ.get("RWKV_BASELINE_CELL", "").strip().lower()
