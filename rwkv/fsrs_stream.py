"""The FSRS card core as a drop-in replacement for the card stream's RWKV layers.

RWKV_FSRS_CARD=<n_free>   unset = OFF (default). "0" = on with a pure 3-float FSRS state;
                          a positive value adds that many free state dimensions alongside.

DESIGN, and why it is shaped like this (optimization/HYBRID_100K.md section 9):

  * The card stream is only 13.0% of the champion's parameters, but its WKV state is 2,880
    floats -- and per-card state, not parameters, is the binding deploy budget. Replacing that
    recurrence with FSRS-7's (S_long, S_short, D) takes the card state to 3 + n_free floats.
  * FSRS fits its 34 weights PER USER with an optimizer. A frozen net cannot do that, so here
    the trunk EMITS them per review. That is Andrew's "S depends on other input features", and
    it is the amortized-inference version of what FSRS does by fitting.
  * The two paths keep the existing interface, so the interleave scheduler and the by-name
    state routing need no changes: this module transforms the running residual vector at the
    card slot, exactly as the card RWKV layers did.

THE SCREEN THAT JUSTIFIES THE BET. FSRS has no delta rule, and the record's "+0.208 imm when
`a` is zeroed" reads as fatal -- but that was measured across ALL FIVE streams. Scoped to the
card stream alone it is +0.01137 imm / +0.00966 ahead, i.e. **5.8% of the global cost**
(scratchpad/hybrid100k/card_delta_ablate.py, 3 users, paired). The delta rule earns its keep in
the context streams, which this keeps. Not free, but not fatal.

TWO INITIALIZATION DECISIONS THAT DECIDE WHETHER THIS WORKS AT ALL:

  1. The emitter's WEIGHT is zero and its BIAS is set so that `bounded_w(bias) == INIT_W`. So an
     untrained core is EXACTLY stock FSRS-7 with default parameters, and training moves away
     from that rather than toward it. `smoke_fsrs_stream.py` asserts this.
  2. The writeback is **small-random, NOT zero**. Zero-init is the house style for an ADDITIVE
     lever (iter 48's rcouple, iter 50's level embedding) because it makes the lever inert at
     init and lets gradient decide. It is WRONG here: this is a REPLACEMENT, and with W_out = 0
     the gradient to `g` is W_out^T @ dL/dx_out = 0, so the emitter would receive no gradient
     either and the whole core would stay silent forever. A lever that cannot start is
     indistinguishable from a lever that does not help, which is exactly the null this project
     keeps having to interpret.
"""
import math
import os
from typing import Optional, Tuple

import torch

from rwkv import fsrs_core as fc


def n_free_dims() -> int:
    """-1 = the core is off. Otherwise the number of free state dims (0 = pure FSRS)."""
    raw = os.environ.get("RWKV_FSRS_CARD", "")
    if raw == "":
        return -1
    return int(raw)


def is_on() -> bool:
    return n_free_dims() >= 0


# g = [log(S_long)/10, log(S_short)/10, (D-5)/4.5, r] (+ free dims). The /10 and /4.5 put every
# channel on a comparable scale: S is clamped to 36500 so log S <= 10.5, and D lives in [1,10].
G_CORE = 4


class FsrsCardCore(torch.nn.Module):
    """Weights + math, shared by the training and deploy wrappers.

    Holds no state: both call sites own their state and pass it in, which is what keeps the
    parallel and recurrent paths computing the same function.
    """

    def __init__(self, d_model: int, n_free: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_free = n_free
        self.g_dim = G_CORE + n_free

        self.emit = torch.nn.Linear(d_model, fc.N_PARAMS)
        self.writeback = torch.nn.Linear(self.g_dim, d_model)
        # free-dimension update (a minGRU-style gate); absent when n_free == 0
        self.free_in = torch.nn.Linear(d_model + G_CORE, 2 * n_free) if n_free > 0 else None

        lo = torch.tensor(fc.CLIP_LO, dtype=torch.float32)
        hi = torch.tensor(fc.CLIP_HI, dtype=torch.float32)
        self.register_buffer("clip_lo", lo)
        self.register_buffer("clip_hi", hi)

        with torch.no_grad():
            # emitter: zero weight + a bias that decodes to FSRS-7's own defaults
            self.emit.weight.zero_()
            w0 = torch.tensor(fc.INIT_W, dtype=torch.float32)
            frac = ((w0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
            self.emit.bias.copy_(torch.log(frac / (1 - frac)))
            # writeback: small random, NOT zero -- see the module docstring
            self.writeback.weight.normal_(0.0, 1.0 / math.sqrt(self.g_dim))
            self.writeback.bias.zero_()

    def params_from(self, x: torch.Tensor) -> torch.Tensor:
        return fc.bounded_w(self.emit(x), self.clip_lo, self.clip_hi)

    def encode(self, s: torch.Tensor, ss: torch.Tensor, d: torch.Tensor,
               r: torch.Tensor, free: Optional[torch.Tensor]) -> torch.Tensor:
        g = torch.stack([torch.log(s) / 10.0, torch.log(ss) / 10.0,
                         (d - 5.0) / 4.5, r], dim=-1)
        if free is not None:
            g = torch.cat([g, free], dim=-1)
        return g

    def advance_free(self, x: torch.Tensor, g_core: torch.Tensor,
                     free: torch.Tensor) -> torch.Tensor:
        """minGRU-style gated update: h <- (1-z)*h + z*tanh(c)."""
        assert self.free_in is not None
        zc = self.free_in(torch.cat([x, g_core], dim=-1))
        z, c = zc.chunk(2, dim=-1)
        return (1.0 - torch.sigmoid(z)) * free + torch.sigmoid(z) * torch.tanh(c)

    def review(self, x: torch.Tensor, t: torch.Tensor, rating: torch.Tensor,
               state: torch.Tensor):
        """One review for a batch of cards. `state` is (..., 3 + n_free).

        AN ALL-ZERO STATE MEANS "NO PRIOR STATE", i.e. this card's first review. That is the
        same sentinel srs-benchmark's FSRS7.step uses, adopted deliberately: `None` cannot
        express it, because within one batch some cards are on their first review and others
        are not, and a Python branch on a whole tensor would silently take one path for both.

        It is also a STRUCTURAL test, never a feature test. `scaled_elapsed_days` maps the -1
        first-review sentinel and a genuine same-day gap onto the same value, and bfloat16
        storage smears it further, so no feature can distinguish them.

        Both branches are evaluated for the whole batch and selected with `where`. The
        discarded update-path values stay finite because `fc.step` clamps its incoming state to
        [S_MIN, S_MAX] and [D_MIN, D_MAX] first, so a zero state produces finite garbage rather
        than a NaN that would poison the gradient through `where`.
        """
        w = self.params_from(x)
        s, ss, d, free = unpack(state, self.n_free)
        is_first = ((s == 0) & (ss == 0) & (d == 0))

        i_s, i_ss, i_d = fc.init_state(rating, w)
        i_r = fc.forgetting_curve(t, i_s, i_ss, i_d, w)
        u_r, u_s, u_ss, u_d = fc.step(s, ss, d, t, rating, w)

        r = torch.where(is_first, i_r, u_r)
        s = torch.where(is_first, i_s, u_s)
        ss = torch.where(is_first, i_ss, u_ss)
        d = torch.where(is_first, i_d, u_d)

        g_core = self.encode(s, ss, d, r, None)
        if self.n_free > 0:
            assert free is not None
            free = self.advance_free(x, g_core, free)
            g = torch.cat([g_core, free], dim=-1)
        else:
            g = g_core
        return x + self.writeback(g.to(x.dtype)), r, pack(s, ss, d, free)


def pack(s, ss, d, free: Optional[torch.Tensor]) -> torch.Tensor:
    """State as one tensor (..., 3 + n_free) -- the shape the deploy API stores per card."""
    parts = [s.unsqueeze(-1), ss.unsqueeze(-1), d.unsqueeze(-1)]
    if free is not None:
        parts.append(free)
    return torch.cat(parts, dim=-1)


def unpack(state: torch.Tensor, n_free: int):
    s, ss, d = state[..., 0], state[..., 1], state[..., 2]
    free = state[..., 3:] if n_free > 0 else None
    return s, ss, d, free


def zero_state(shape, n_free: int, dtype, device) -> torch.Tensor:
    """The 'no prior state' value. All zeros, matching srs-benchmark's own sentinel."""
    return torch.zeros(tuple(shape) + (3 + n_free,), dtype=dtype, device=device)


def run_sequence(core: FsrsCardCore, x_BTC: torch.Tensor, t_BT: torch.Tensor,
                 rating_BT: torch.Tensor, skip_BT: Optional[torch.Tensor],
                 state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The PARALLEL-path entry point: a (B, T, C) block of one card's reviews, scanned over T.

    A loop over T is affordable precisely because these are PER-CARD sequences. Measured on
    three VAL users: mean 3.6-7.2 reviews per card, p99 10-32, max 51. The batcher additionally
    buckets sequences by exact length, so T is that bucket's length, never the 65,536 token
    budget.

    `skip_BT` marks rows that must not advance the state (query/probe rows). They still produce
    an output, which is what makes a counterfactual probe possible.
    """
    B, T, _ = x_BTC.shape
    if state is None:
        state = zero_state((B,), core.n_free, x_BTC.dtype, x_BTC.device)
    outs = []
    for i in range(T):
        x_out, _r, new_state = core.review(x_BTC[:, i], t_BT[:, i], rating_BT[:, i], state)
        outs.append(x_out)
        if skip_BT is not None:
            # A skipped row still PRODUCES an output (that is what makes a counterfactual probe
            # possible) but must not advance the state. Selecting per element is required: one
            # bucket mixes probe rows and real rows.
            keep = skip_BT[:, i].to(torch.bool).unsqueeze(-1)
            new_state = torch.where(keep, state, new_state)
        state = new_state
    return torch.stack(outs, dim=1), state
