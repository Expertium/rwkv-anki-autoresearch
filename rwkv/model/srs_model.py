from dataclasses import dataclass
import math

import numpy as np
from rwkv.config import RWKV_SUBMODULES
from rwkv.data_processing import RWKVSample
from rwkv.model.rwkv_model import RWKV7, take_rank1_penalty
from rwkv.model.rwkv_model import _RANK1_REG as _RANK1_REG_LAMBDA
import torch
from typing import NamedTuple, Optional, Tuple

from rwkv.architecture import AnkiRWKVConfig


import os


def __nop(ob):
    return ob


# Match rwkv_model.py: RWKV_NO_JIT=1 (state-QAT) disables torch.jit so the whole model -- incl. the
# quant-aware per-step WKV -- runs as plain Python. Default (JIT on) keeps eval byte-for-byte unchanged.
if os.environ.get("RWKV_NO_JIT"):
    ModuleType = torch.nn.Module
    FunctionType = __nop
else:
    ModuleType = torch.jit.ScriptModule
    FunctionType = torch.jit.script_method


class _PermGather(torch.autograd.Function):
    """index_select whose backward exploits the permutation structure of the stream gathers.

    The hierarchical gather (x -> per-entity rows) references each row of x AT MOST once per
    stream (each review belongs to exactly one card/note/deck/preset/user); -1 entries are
    padding (clamped to row 0 in the forward, matching the original torch.clamp+index_select).
    index_select's stock backward is index_add -- under torch.use_deterministic_algorithms it
    takes the sort-based path that costs ~43% of the whole training step. Here the backward is
    an index_select by the INVERSE permutation (collision-free, deterministic BY CONSTRUCTION):
      grad_x[r] = grad_out[inv[r]]           (r referenced; unique position)
      grad_x[r] = 0                          (r never referenced -- dead/padding rows of x)
      grad_x[0] += sum(grad_out[pads])       (forward clamped -1 -> row 0; pad grads are 0
                                              in practice -- skip rows get no input grad)
    Forward is bit-identical to the original; backward is bit-identical except (at most) the
    row-0 pad-sum order, which only ever adds exact zeros. Validated by a 10-step E2E
    bit-identical loss-trace test vs the index_add path."""

    @staticmethod
    def forward(ctx, x, idx):
        idx_long = torch.clamp(idx, min=0).long()
        ctx.save_for_backward(idx)
        ctx.n_rows = x.size(0)
        return torch.index_select(x, 0, idx_long)

    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        n, m = ctx.n_rows, idx.numel()
        real = idx >= 0
        # inverse permutation via collision-free scatter (unique targets -> deterministic);
        # unreferenced rows keep sentinel m and read the appended zero row.
        inv = torch.full((n,), m, dtype=torch.long, device=grad_out.device)
        pos = torch.arange(m, dtype=torch.long, device=grad_out.device)
        inv.scatter_(0, idx[real].long(), pos[real])
        padded = torch.cat([grad_out, grad_out.new_zeros(1, grad_out.size(1))], dim=0)
        grad_x = padded.index_select(0, inv)
        n_pad = m - int(real.sum())
        if n_pad > 0:
            grad_x[0] = grad_x[0] + grad_out[~real].sum(dim=0)
        return grad_x, None


@torch.jit.ignore
def perm_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return _PermGather.apply(x, idx)


# RWKV_PERM_GATHER=0 restores the stock clamp+index_select (escape hatch; default ON).
_USE_PERM_GATHER = os.environ.get("RWKV_PERM_GATHER", "1") != "0"


def interleave_schedule(depths: list, spread: bool) -> list:
    """Which stream-local layer (if any) each stream runs in each round.

    Returns sched[stream][round] = layer index, or -1 for "this stream sits this round out".
    Rounds = max(depths). Shared by the training path (srs_model) and the deploy RNN mirror
    (srs_model_rnn) so the two schedules cannot drift -- a divergence here is invisible to
    every gate, since each path stays self-consistent (CLAUDE.md SS9).

    FRONT-LOADED (spread=False, iter 41's schedule): layer j runs in round j, so a stream of
    depth d occupies rounds 0..d-1 and then sits out the rest. Bit-identical to the original
    `if r < depth` loop by construction.

    SPREAD (spread=True, iter 44): layers are distributed across ALL rounds with the endpoints
    anchored -- layer 0 in round 0, the LAST layer in the LAST round:
        round(j) = round(j * (R-1) / (d-1))          for d > 1
        round(0) = R-1                                for d == 1
    WHY: under front-loading a shallow stream only ever runs EARLY, so it can never consume the
    cross-scope context that interleaving exists to expose. In the champion (depths
    [2,4,1,3,3]) the note stream has depth 1 -- it runs in round 0 and never again, so it
    contributes to the global context but never reads it. Spreading fixes that (note -> round
    3) and additionally puts every stream's FINAL layer in the last round, so each scope's
    output representation is computed with maximal context. Depth-1 streams go LAST rather than
    to the middle: consuming context is the whole point, and a lone layer cannot do both.
    """
    n_rounds = 0
    for d in depths:
        if d > n_rounds:
            n_rounds = d
    sched = []
    for d in depths:
        rows = [-1] * n_rounds
        if d > 0:
            if not spread:
                for j in range(d):
                    rows[j] = j
            elif d == 1:
                rows[n_rounds - 1] = 0
            else:
                for j in range(d):
                    # round-half-up of j*(R-1)/(d-1); integer arithmetic keeps it exact
                    r = (2 * j * (n_rounds - 1) + (d - 1)) // (2 * (d - 1))
                    rows[r] = j
        sched.append(rows)
    return sched


class SrsRWKVIterStatistics(NamedTuple):
    average_loss: torch.Tensor
    loss_tensor: torch.Tensor
    w_loss_avg: torch.Tensor
    ahead_logits_mag_loss_avg: torch.Tensor
    ahead_logits_diff_loss_avg: torch.Tensor
    ahead_avg: torch.Tensor
    ahead_raw_avg: torch.Tensor
    ahead_n: int
    ahead_equalize_avg: torch.Tensor
    ahead_raw_equalize_avg: torch.Tensor
    ahead_equalize_n: int
    imm_avg: torch.Tensor
    imm_n: int
    imm_binary_equalize_avg: torch.Tensor
    imm_binary_equalize_n: int
    p_curve: torch.Tensor
    p_imm: torch.Tensor
    p_imm_all: torch.Tensor
    w: torch.Tensor
    label_rating: torch.Tensor
    label_elapsed_seconds: torch.Tensor
    label_review_th: torch.Tensor
    is_query: torch.Tensor
    has_label: torch.Tensor
    pava_loss_avg: torch.Tensor
    pava_pool_frac: torch.Tensor


@dataclass
class PreparedBatch:
    num_data: int
    start: torch.Tensor
    sub_gather: list[list[torch.Tensor]]
    sub_gather_lens: list[list[int]]
    time_shift_selects: list[list[torch.Tensor]]
    skips: list[list[torch.Tensor]]
    labels: torch.Tensor
    label_review_th: torch.Tensor
    # iter 23 probe channel (None when probe insertion is off): flat b*global_T+t indices
    probe_rows: "torch.Tensor | None" = None      # (M,4) Again..Easy probe skip rows
    probe_target: "torch.Tensor | None" = None    # (M,) the probed real row
    probe_pressed: "torch.Tensor | None" = None   # (M,) actual rating - 1 in 0..3
    probe_query: "torch.Tensor | None" = None     # (M,) paired imm query row (iter 24 w)
    # iter 46 self-distillation teacher index (RWKV_SELFKD_BETA): (B, global_T) int64 holding, for
    # each row, the flat b*global_T+t index of the query row scoring THAT ROW'S OWN ahead target
    # (-1 = none, incl. every query row). Always built by prepare(); the model ignores it at
    # beta=0, so its presence alone is bit-identical.
    ahead_query: "torch.Tensor | None" = None
    # iter 37 objective alignment (RWKV_USER_WEIGHT): (B,) per-chunk loss weight, already
    # normalized to mean 1. Set by the TRAIN LOOP (which owns the epoch's chunk list), never
    # by the fetch workers -- that keeps the data stream byte-identical, so the KD dump's
    # per-step labels checksum still matches. None => unweighted, bit-identical to pre-iter-37.
    user_weight: "torch.Tensor | None" = None

    def to(self, device):
        start = self.start.to(device)
        sub_gather = [[x.to(device) for x in sub] for sub in self.sub_gather]
        time_shift_selects = [
            [x.to(device) for x in sub] for sub in self.time_shift_selects
        ]
        skips = [[x.to(device) for x in sub] for sub in self.skips]
        labels = self.labels.to(device)
        label_review_th = self.label_review_th.to(device)
        return PreparedBatch(
            num_data=self.num_data,
            start=start,
            sub_gather=sub_gather,
            sub_gather_lens=self.sub_gather_lens,
            time_shift_selects=time_shift_selects,
            skips=skips,
            labels=labels,
            label_review_th=label_review_th,
            probe_rows=None if self.probe_rows is None else self.probe_rows.to(device),
            probe_target=None if self.probe_target is None else self.probe_target.to(device),
            probe_pressed=None if self.probe_pressed is None else self.probe_pressed.to(device),
            probe_query=None if self.probe_query is None else self.probe_query.to(device),
            user_weight=None if self.user_weight is None else self.user_weight.to(device),
            ahead_query=None if self.ahead_query is None else self.ahead_query.to(device),
        )


DTYPE_EXCLUDE = [
    "w_linear",
    "s_linear",
    "d_linear",
    "d_softplus",
    "k_linear",
    "p_linear",
    "ahead_linear",
    "gru_",  # GRU head root Parameters -- fp32 like the linears they replace
    "pava_",  # rectifier junction thetas -- fp32, used in the eager fp32 probe loss
    # RNN baselines (2026-07-24): bf16 nn.GRU/LSTM falls off cuDNN onto a 30x-slower
    # native path (micro-benched 1258 vs 42 ms) -- keep the stream weights fp32;
    # RNNStream.forward casts activations at the boundary. NO trailing dot: the
    # substrings must match MODULE names too ("rwkv_modules.0.rnn" -- selective_cast
    # calls .to() directly on nested modules), not just param names. No RWKV
    # param/module names contain these. RNNStream._apply is the second guard.
    ".rnn",
    ".proj",
]


def is_excluded(name):
    for query in DTYPE_EXCLUDE:
        if query in name:
            return True
    return False


class SrsRWKV(ModuleType):
    def __init__(self, anki_rwkv_config: AnkiRWKVConfig):
        super().__init__()

        self.card_features_dim = 92
        self.use_perm_gather = _USE_PERM_GATHER
        # Research iter 11 (2026-07-13, Andrew's idea): dedicated additive grade embedding.
        # The grade one-hot (cols 9:13 of the 92) already gets an implicit embedding via
        # features2card's first Linear, but there it competes with 88 other dims for the
        # shared fc->d_model squeeze. RWKV_GRADE_EMB=1 adds a 4 x d_model zero-init bypass:
        # x = features2card(f) + onehot @ E. Matmul (not argmax) so ahead-mode query rows
        # (all-zero one-hot) contribute exactly zero. Default unset = module absent =
        # byte-identical, old checkpoints load unchanged.
        self.grade_emb_on = os.environ.get("RWKV_GRADE_EMB", "0") == "1"
        self.prehead_gate_on = os.environ.get("RWKV_PREHEAD_GATE", "0") == "1"
        # Research iter 22 (2026-07-16, MONOTONICITY_PLAN.md stage 2): RWKV_MONO_CURVES=1
        # projects the ahead-logit residual to its running lower envelope (cummin) along the
        # time-point axis, making it non-increasing in t. The fixed-basis mixture is already
        # monotone (0.9^(t/s_i) bases, softmax weights), logit() and the linear point interp
        # preserve monotonicity, so the FINAL curve becomes non-increasing in elapsed time by
        # construction -> "solve P(t)=DR for the interval" is single-crossing/well-defined.
        # cummin (vs a softplus-cumsum generative form): neutral at Linear init (envelope of
        # near-zero noise), exact identity wherever the raw residual is already decreasing,
        # parameter-free (param count unchanged). Fallback if its sparse (argmin-routed)
        # gradients stall training: shifted-softplus-cumsum. Default off = byte-identical.
        self.mono_curve_on = os.environ.get("RWKV_MONO_CURVES", "0") == "1"
        # Directed change (Andrew 2026-07-16, both tracks): RWKV_NO_AHEAD_RESIDUAL=1 disables
        # the piecewise-linear curve correction entirely -- out_ahead_logits becomes constant
        # zeros, so interp() contributes only its fixed 1e-5 affine offset and the curve is
        # EXACTLY the mixture-of-exponentials (monotone in t by construction; supersedes the
        # cummin projection, which is vacuous on a zero residual). The ahead head modules stay
        # constructed (script-compilable, ckpt-compatible) but receive no gradient (zeros are
        # created outside autograd) -> they sit dead at init; ~12.5k params at d=32 / ~131.7k
        # at d=128 are strippable at deploy. The raw-mixture BCE term (AHEAD_RAW_SCALE) already
        # supervises the mixture directly, so training stays well-posed. Default off =
        # byte-identical.
        self.no_ahead_residual = os.environ.get("RWKV_NO_AHEAD_RESIDUAL", "0") == "1"
        # Track-2 A3 (Andrew 2026-07-17): GRU-FAITHFUL curve head (srs-benchmark
        # models/gru.py). RWKV_GRU_HEAD=N (N>=1) replaces the 128-basis fixed-stability
        # mixture with N per-row predicted curves: three tiny linears off the SHARED head_w
        # trunk predict w (softmax), S and d (exp(clamp(.,-25,25)) -> strictly positive), and
        # R(t) = sum_i w_i * (1 + t/(1e-7+S_i))^(-d_i)  -- each curve monotone decreasing in
        # t BY CONSTRUCTION (d_i > 0), so the no-residual monotonicity guarantee carries over
        # (gru forces no_ahead_residual below; the residual path is structurally dead).
        # Param accounting at d=128: drops w_linear (65,664) + the dead ahead head
        # (head_ahead_logits 66,048 + ahead_linear 65,664), adds 3*(w_head_dim*N+N) + 6 dummy
        # params -- ~-194.3k vs A1 at N=2. The replaced modules become 1x1 dummies (NOT
        # absent: scripted head_and_out references them in now-dead branches, and old-style
        # ScriptModule compiles BOTH sides of a runtime-bool if). New learnables are ROOT
        # Parameters accessed via a @torch.jit.ignore F.linear accessor (iter-16 rule:
        # submodule calls from ignored methods crash under JIT); names keep weight/bias + 2D
        # so the optimizer wd-groups classify them like Linear equivalents, and the
        # selective-cast module walk skips them -> they stay fp32 like the DTYPE_EXCLUDE'd
        # heads they replace. Default 0/unset = byte-identical legacy head.
        self.gru_n = int(os.environ.get("RWKV_GRU_HEAD", "0"))
        self.gru_on = self.gru_n > 0
        if self.gru_on:
            self.no_ahead_residual = True
            print(f"[gru] GRU curve head ON: N={self.gru_n} predicted (w, S, d) curves; "
                  f"legacy w_linear/ahead head replaced by 1x1 dummies")
        # Research iter 17 (2026-07-15): direct binary-recall loss term. The benchmark's imm
        # metric IS p_binary_loss (BCE of 1-P(again) vs recall), but the training loss only
        # optimizes it implicitly through the 4-way rating CE. RWKV_PBIN_SCALE=<w> adds
        # w * mean(p_binary_loss over query rows) to the loss ("train what you measure").
        # Instance float: TorchScript can't read env/globals inside scripted methods, but it
        # CAN read instance attributes. Default 0 = term skipped = byte-identical.
        self.pbin_scale = float(os.environ.get("RWKV_PBIN_SCALE", "0"))
        if self.pbin_scale != 0.0:
            print(f"[pbin] direct binary-recall loss term ON, scale={self.pbin_scale}")
        # Research iter 23 (2026-07-17, MONOTONICITY_PLAN.md stage 2, Andrew's design):
        # learnable power-mean PAVA rectifier over the 4 counterfactual button curves,
        # trained on in-sequence probe rows (skip rows inserted at prepare-batch time; see
        # prepare_batch.py + scratchpad/iter23_pava/BUILD_NOTES.md). RWKV_PAVA_LAMBDA=<w>
        # enables the 3 junction-theta params + adds w * BCE(rectified pressed-probe curve,
        # ahead label) to the loss. RWKV_PAVA_PWEIGHT=1 (iter 24) weights the pooling mean
        # by the p-head's button probabilities read at the paired query row. The op runs
        # EAGER inside a @torch.jit.ignore method (rwkv/model/pava.py); probes arrive as an
        # Optional tuple arg (kd precedent). Default 0 = params absent = byte-identical.
        self.pava_lambda = float(os.environ.get("RWKV_PAVA_LAMBDA", "0"))
        self.pava_pweight = os.environ.get("RWKV_PAVA_PWEIGHT", "0") == "1"
        # RWKV_AHEAD_PROBE_ONLY=1 (iter 33, Andrew 2026-07-27): take the `ahead` objective ONLY
        # from the duration-zeroed probe path, dropping probed real rows from the ordinary ahead
        # term. Fixes a train/deploy mismatch worth +0.001451 ahead (mode-2 diagnostic): training
        # scored the real row, which carries the review's own duration, while deploy must predict
        # BEFORE the press and therefore never has it. Pair with RWKV_PROBE_DENSITY=1.0 so every
        # eligible review is covered (measured cost: 2.54x rows). Default 0 = byte-identical.
        self.ahead_probe_only = os.environ.get("RWKV_AHEAD_PROBE_ONLY", "0") == "1"
        # PRIVILEGED SELF-DISTILLATION, imm -> ahead (iter 46, 2026-08-11). RWKV_SELFKD_BETA=<b>
        # softens the probe path's BCE target away from the raw 0/1 ahead label and toward the
        # model's OWN better-informed estimate of THE SAME EVENT. b reallocates only the HARD
        # share, so an active external teacher keeps its tuned weight a exactly:
        #     target = a*d128_teacher + (1-a) * [ b * (1-P(Again))@probe_query.detach()
        #                                         + (1-b) * hard_label@probe_target ]
        # (a = the RWKV_KD_MIX alpha, 0 when KD is off -- e.g. the whole decay phase -- where this
        # reduces to b*teacher + (1-b)*hard. Derivation at the substitution site.)
        # WHY THIS PAIRING IS THE RIGHT ONE, and why it costs no new plumbing: probe_target and
        # probe_query already index the real row supplying the ahead label and *its paired imm
        # query row* -- the same review, scored twice from different information sets. Measured on
        # the iter-41 champion (research_5k_notes.md): identical per-user `size` on all 2500 users,
        # imm better than ahead on 2497 of them, mean gap 0.032411. So the query row is a teacher
        # that is free, online, and already aligned.
        # LUPI, not leakage: the teacher sits at the decision point and sees the intervening
        # reviews and the exact lag; the student (probe row) is the cold-from-history prediction
        # that deploy actually serves. The teacher is used ONLY as a target and is DETACHED, so no
        # gradient reaches the rating head (it must not be dragged toward the weaker head) and
        # nothing enters the student's forward pass. Train/eval/deploy therefore still compute one
        # quantity -- this is a loss-side change only, like the external KD alpha, and the Rust
        # engine is untouched. (Contrast the ranked-#2 coupling variant, which would feed R(t) into
        # the Again logit: same motivation, but a forward-pass edit and a 9th Rust port gap.)
        # ⚠ The gap is an UPPER BOUND: distillation can transfer the variance-reduction part of it
        # (a calibrated soft target beats a 0/1 label -- which is why the external-teacher alpha
        # peaks at 0.9), never the information part. Default 0 = target unchanged = byte-identical.
        self.selfkd_beta = float(os.environ.get("RWKV_SELFKD_BETA", "0"))
        if self.selfkd_beta != 0.0:
            print(
                f"[selfkd] privileged self-distillation ON: ahead target hard-share = "
                f"{self.selfkd_beta} * imm(teacher row, detached) "
                f"+ {1.0 - self.selfkd_beta} * hard label"
            )
        # RECTIFIED EVAL (2026-07-26, Andrew: "Eval should score the rectified model, of
        # course"). Until now the rectifier existed ONLY inside the loss -- curve_probs was
        # returned unrectified -- so every reported `ahead` number from iters 23-30 scored a
        # model that differed from the one intended to ship. RWKV_EVAL_PAVA=1 makes the eval
        # path substitute, at each scored row, the rectified value at the button the user
        # actually pressed (prepare_batch inserts the 4 probes at density 1.0). Affects
        # `ahead` ONLY -- `imm` reads out_p_binary off the rating head, which the rectifier
        # never touches. Models trained without PAVA (e.g. the A18 champion) have no learned
        # theta; they fall back to powers = 1 = classic arithmetic-mean PAVA, which is
        # exactly the theta init, so it is the honest parameter-free default rather than a
        # fudge. Default 0 = byte-identical to every stored result.
        # RWKV_EVAL_PAVA: 0 = off (the scored row's own curve, at its REAL duration)
        #                 1 = substitute the RECTIFIED pressed probe  <- the deploy metric
        #                 2 = substitute the UNRECTIFIED pressed probe <- diagnostic only
        #
        # Mode 2 exists because modes 0 and 1 differ in TWO ways at once, which makes the
        # rect-vs-unrect comparison unable to attribute its own result (Andrew, 2026-07-26).
        # A probe row is the scored row with the grade one-hot swapped AND the current-row
        # duration zeroed, so switching 0 -> 1 changes both the pooling AND the duration the
        # prediction is computed at. Mode 2 moves only the duration, giving an additive split:
        #     mode2 - mode0 = cost of zeroing the current-row duration
        #     mode1 - mode2 = cost of the PAVA pooling itself
        #     mode1 - mode0 = the total, i.e. what a rect-vs-unrect run reports
        # All three land on `ahead` only: `imm` comes from the rating head, and query rows have
        # always carried duration 0, so nothing about it changes.
        #                 3 = insert probes but substitute NOTHING     <- the noise control
        #
        # MODE 3 EXISTS BECAUSE PROBE INSERTION IS NOT NUMERICALLY FREE (measured 2026-07-26).
        # Probes are skip rows and the token shift steps over them (prepare_batch only advances
        # `last` on non-skip rows), so in EXACT arithmetic they change nothing. In bf16 they do:
        # +4 rows per scored review inflates the batch ~30%, which re-buckets sequences by length
        # and reorders the bf16 reductions. Measured on A18 (n=2500), `imm` -- which the rectifier
        # never touches and which therefore isolates the effect -- moved by +0.000280, and the
        # move scales with recurrence length (mean |d| 1.98e-4 at ~4.7k reviews/user -> 3.97e-4 at
        # ~179k) and is one-signed (62% -> 78% of users worse) because LogLoss is CONVEX: zero-mean
        # noise on a prediction raises it.
        # So mode2 - mode0 would charge the duration change for that noise too. Mode 3 inserts the
        # identical probes and then leaves `curve_probs` alone, giving the clean split:
        #     mode3 - mode0 = probe-insertion NOISE (bf16 re-bucketing only)
        #     mode2 - mode3 = cost of zeroing the current-row duration   <- what Andrew asked for
        #     mode1 - mode2 = cost of the PAVA pooling itself
        _eval_pava_mode = os.environ.get("RWKV_EVAL_PAVA", "0")
        self.eval_pava = _eval_pava_mode in ("1", "2", "3")
        self.eval_pava_rectify = _eval_pava_mode != "2"
        # mode 3 keeps the probes but performs no substitution at all
        self.eval_pava_substitute = _eval_pava_mode in ("1", "2")
        if self.eval_pava:
            print("[pava] RECTIFIED EVAL ON: scored ahead predictions come from the "
                  "rectified pressed-button probe"
                  + ("" if self.pava_lambda != 0.0 else " (no trained theta -> classic p=1)"))
        if self.pava_lambda != 0.0:
            print(f"[pava] learnable power-mean rectifier ON, lambda={self.pava_lambda}, "
                  f"p-head weighting={'ON' if self.pava_pweight else 'off'}")
        # Research iter 15 (2026-07-14, Andrew's directive): drop input features by zeroing
        # their columns at the model input. RWKV_ZERO_FEATURES="22" (comma-separated dims of
        # the 92) zeroes those columns in BOTH training and eval, so the column is constant 0
        # = informationally removed (the input FC's bias absorbs it); LMDBs, param count and
        # batch layout stay untouched, and deploy just feeds 0 for the dropped features.
        # Dim 22 = scaled_state (Anki review state Filtered/Review/Learn/Relearn; see
        # data_processing.CARD_FEATURE_COLUMNS). Default unset = all-ones mask, path gated
        # off = byte-identical. Buffer is persistent=False: absent from state_dict, so old
        # and new checkpoints stay interchangeable.
        _zero_feats = [
            int(t) for t in os.environ.get("RWKV_ZERO_FEATURES", "").split(",") if t.strip()
        ]
        assert all(0 <= i < 92 for i in _zero_feats), f"RWKV_ZERO_FEATURES out of range: {_zero_feats}"
        self.input_feat_mask_on = len(_zero_feats) > 0
        _mask = torch.ones(92)
        for _i in _zero_feats:
            _mask[_i] = 0.0
        # Plain attribute, NOT a buffer: ScriptModule forbids persistent=False buffers, and a
        # persistent one would pollute state_dict (breaking ckpt interchange + Rust export).
        # The jit.ignore'd applier below moves it to the right device/dtype per call (92
        # floats, negligible).
        self.input_feat_mask = _mask
        if self.input_feat_mask_on:
            print(f"[feat-mask] zeroing input feature dims {_zero_feats} (train AND eval)")
        self.d_model = anki_rwkv_config.d_model
        self.features_fc_dim = anki_rwkv_config.features_fc_mult * self.d_model
        self.ahead_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.p_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.w_head_dim = anki_rwkv_config.head_fc_mult * self.d_model
        self.num_curves = anki_rwkv_config.num_curves
        if self.gru_on:
            # out_w carries N curves now; the KL-to-uniform w_loss target reads num_curves
            self.num_curves = self.gru_n

        with torch.no_grad():
            self.features2card = torch.nn.Sequential(
                torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
                torch.nn.SiLU(),
                torch.nn.LayerNorm(self.features_fc_dim),
                torch.nn.Linear(self.features_fc_dim, self.d_model),
                torch.nn.SiLU(),
            )
            # stamp each stream's name onto its config (works for every RWKV_ARCH_MODULE
            # without editing the arch files) -- consumed by RWKV_STRIP_CMIX (A6)
            for _name, _cfg in anki_rwkv_config.modules:
                _cfg.stream_name = _name
            # Classic-RNN baselines (Andrew 2026-07-23): RWKV_BASELINE_CELL=gru|lstm
            # swaps ONLY the per-stream recurrent stacks (same hierarchy/depths, same
            # trunk + heads + pipeline). Default unset = RWKV7, byte-identical.
            from rwkv.model.rnn_baseline import (
                RNNStream, baseline_hidden_default, env_baseline_cell,
            )
            _cell = env_baseline_cell()
            if _cell:
                _hidden = int(os.environ.get(
                    "RWKV_BASELINE_HIDDEN", baseline_hidden_default(_cell)))
                print(f"[rnn-baseline] {_cell.upper()} streams ON: hidden={_hidden}, "
                      f"depths={[c.n_layers for _, c in anki_rwkv_config.modules]}")
                self.rwkv_modules = torch.nn.ModuleList(
                    [RNNStream(_cell, config.d_model, _hidden, config.n_layers,
                               config.dropout, stream_name=name)
                     for name, config in anki_rwkv_config.modules]
                )
            else:
                self.rwkv_modules = torch.nn.ModuleList(
                    [RWKV7(config=config) for _, config in anki_rwkv_config.modules]
                )
            # iter 41 (RWKV_INTERLEAVE=1): round-robin the EXISTING layers across scopes --
            # round r runs layer r of every stream that has one, in the hierarchy order the
            # config already defines. Same params, same per-entity states, same per-layer ops;
            # only the execution ORDER changes, so from round 1 on the specific streams see
            # the general streams' round-(r-1) output (the sequential form gives card->...->
            # user exactly one pass, so general context can never reach card-level processing).
            # Default unset = the sequential branch below runs untouched (bit-identical).
            self.interleave_on = os.environ.get("RWKV_INTERLEAVE", "0") == "1"
            self.stream_depths = [config.n_layers for _, config in anki_rwkv_config.modules]
            # iter 44 (RWKV_ILV_SPREAD=1): distribute each stream's layers across ALL rounds
            # (endpoints anchored) instead of front-loading them. Motivated by iter 43's
            # result -- the schedule is the productive lever, the order is not -- and by a
            # concrete deficiency of front-loading: a depth-1 stream runs only in round 0, so
            # it feeds the global context but never reads it. See interleave_schedule().
            self.ilv_spread = os.environ.get("RWKV_ILV_SPREAD", "0") == "1"
            self.ilv_sched = interleave_schedule(self.stream_depths, self.ilv_spread)
            if self.interleave_on:
                assert not env_baseline_cell(), \
                    "RWKV_INTERLEAVE + RWKV_BASELINE_CELL unsupported (no forward_layer on RNNStream)"
                print(f"[interleave] round-robin layer schedule ON: depths={self.stream_depths} "
                      f"-> {max(self.stream_depths)} rounds, {sum(self.stream_depths)} layer-steps "
                      f"(order within each round = the config's hierarchy order)")
                print(f"[interleave] layer placement = "
                      f"{'SPREAD (endpoint-anchored)' if self.ilv_spread else 'front-loaded'}: "
                      f"sched={self.ilv_sched} (per stream, per round; -1 = sits out)")
            else:
                assert not self.ilv_spread, "RWKV_ILV_SPREAD requires RWKV_INTERLEAVE=1"
            self.prehead_norm = torch.nn.LayerNorm(self.d_model)
            self.prehead_dropout = torch.nn.Dropout(p=anki_rwkv_config.dropout)
            if self.gru_on:
                # 1x1 dummies: attributes must EXIST (scripted head_and_out compiles the
                # dead legacy branches), but their params drop out of the model (6 total)
                self.head_ahead_logits = torch.nn.Sequential(
                    torch.nn.Linear(1, 1),
                    torch.nn.ReLU(),
                )
            else:
                self.head_ahead_logits = torch.nn.Sequential(
                    torch.nn.Linear(self.d_model, self.ahead_head_dim),
                    torch.nn.ReLU(),
                )
            self.head_w = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, 1 * self.d_model),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(1 * self.d_model),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(1 * self.d_model, self.w_head_dim),
            )
            self.head_p = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.p_head_dim),
                torch.nn.ReLU(),
            )

            self.max_e = 21
            self.point_spread = 18.5
            self.num_points = anki_rwkv_config.num_points
            if self.gru_on:
                self.ahead_linear = torch.nn.Linear(1, 1)
                self.w_linear = torch.nn.Linear(1, 1)
                # GRU head params (root Parameters, fp32; see the __init__ note). Weights
                # zero-init like the legacy w_linear (input-independent start; W and b get
                # nonzero grads at step 1 so they move immediately). Biases = a sane prior:
                # w uniform, S log-spaced 1 hour .. 1 year, d = 0.5 (moderate FSRS-like decay).
                _N = self.gru_n
                self.gru_w_weight = torch.nn.Parameter(torch.zeros(_N, self.w_head_dim))
                self.gru_w_bias = torch.nn.Parameter(torch.zeros(_N))
                self.gru_s_weight = torch.nn.Parameter(torch.zeros(_N, self.w_head_dim))
                self.gru_s_bias = torch.nn.Parameter(
                    torch.linspace(math.log(3600.0), math.log(31536000.0), _N)
                )
                self.gru_d_weight = torch.nn.Parameter(torch.zeros(_N, self.w_head_dim))
                self.gru_d_bias = torch.nn.Parameter(
                    torch.full((_N,), math.log(0.5))
                )
            else:
                self.ahead_linear = torch.nn.Linear(self.ahead_head_dim, self.num_points)
                torch.nn.init.zeros_(self.ahead_linear.weight)
                torch.nn.init.zeros_(self.ahead_linear.bias)

                self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)
                torch.nn.init.zeros_(self.w_linear.weight)
                torch.nn.init.zeros_(self.w_linear.bias)

            self.s_point_spread = 18.5
            self.s_max = 22

            self.p_linear = torch.nn.Linear(self.p_head_dim, 4)
            torch.nn.init.zeros_(self.p_linear.weight)
            self.p_linear.bias.copy_(torch.tensor([-0.3512, -0.0802, 0.4297, -0.2041]))

            # ---- RETRIEVABILITY-COUPLED RATING HEAD (RWKV_RCOUPLE=1; PROPOSALS.md item #2) -------
            # Feed the curve head's logit R(t) into the 4 rating logits. Two mechanisms, both wanted:
            #   (a) the rating head gains R(t) as an INPUT -- it currently predicts the rating with no
            #       explicit notion of how due the card is, though the curve head computes exactly
            #       that at the SAME t (label_elapsed_seconds on a query row IS the elapsed time of
            #       the review being scored; data_processing.py:301-303 shifts elapsed_seconds by -1);
            #   (b) a GRADIENT PATH from the better-conditioned imm objective back into the curve
            #       head. iter 46 showed the ahead/imm gap is NOT transferable by soft targets
            #       because a same-forward-pass teacher re-expresses what the student computes; the
            #       stated remedy was to change what the ahead path COMPUTES or is FED. This does.
            # 4 coefficients, not 1: softmax is shift-invariant so an Again-only scalar is the
            # special case (g,0,0,0), while the general form can express "more retrievable shifts
            # mass Again->Easy", which is the actual domain structure. 4 params on 558,212.
            # ZERO INIT => byte-identical to the champion at step 0; the coupling is learned or not
            # at all, so a null result cannot be blamed on a bad initialisation.
            # module-level global -> instance attribute, so the scripted forward can see it
            self.rank1_reg_lambda = _RANK1_REG_LAMBDA
            self.rcouple_on = os.environ.get("RWKV_RCOUPLE", "") == "1"
            self.rcouple_detach = os.environ.get("RWKV_RCOUPLE_DETACH", "") == "1"
            # clamp before use: curve_logits = logit(p) can saturate, and an inf here would poison
            # the rating head (which is the DEPLOYED head, retrievability_head = 1 - P(Again)).
            self.rcouple_clip = float(os.environ.get("RWKV_RCOUPLE_CLIP", "8.0"))
            # Gated on the flag (like pava_theta): an unconditional Parameter would add a 422nd key
            # to a 421-key state dict and break strict loading of every existing champion.
            if self.rcouple_on:
                self.rcouple_w = torch.nn.Parameter(torch.zeros(4))
                print(f"[RCOUPLE] retrievability-coupled rating head ON "
                      f"(detach={self.rcouple_detach} clip={self.rcouple_clip})")
                # ⚠ INCOMPATIBLE WITH RWKV_SELFKD_BETA, and silently so. The self-KD block builds
                # its teacher from `out_p_logits` BEFORE the coupling is applied (it runs earlier in
                # forward, since the coupling needs curve_logits), so the teacher would read the
                # UNCOUPLED rating head while the loss uses the coupled one -- two different
                # functions inside one step. Fail loudly instead: iter 46 rejected self-KD, so
                # nothing is lost, and this is the exact mismatch class that cost 8 iterations of
                # trained-but-not-evaluated PAVA.
                assert os.environ.get("RWKV_SELFKD_BETA", "0") in ("0", "0.0", ""), (
                    "RWKV_RCOUPLE + RWKV_SELFKD_BETA compute inconsistent rating logits "
                    "(self-KD reads out_p_logits before the coupling). Enable only one."
                )

            # ⚠ CONDITIONAL LEARNABLES BEHIND jit.ignore MUST BE Parameters, NOT submodules
            # (iter-16 hollow-run lesson, 2026-07-15): calling a SUBMODULE from a
            # @torch.jit.ignore method invoked THROUGH scripted code fails at runtime with
            # "'torch._C.ScriptModule' object is not callable" (the ignored body sees the raw
            # C++ module). Plain tensor/Parameter attribute access works (proven by the
            # iter-15 feat-mask full run) -- so use Parameters + F.linear. Names keep
            # "weight"/2D so train_rwkv's optimizer groups classify them like the Linear
            # equivalents (weight -> decayed, bias -> wd=0).
            if self.grade_emb_on:
                self.grade_emb_weight = torch.nn.Parameter(torch.zeros(self.d_model, 4))

            # Research iter 16 (2026-07-15): prehead OUTPUT GATE. RWKV_PREHEAD_GATE=1 adds
            # x = x * (2 * sigmoid(W x + b)) between prehead norm/dropout and the three heads
            # -- the trunk modulates per-channel how much of the state reaches the readouts.
            # Zero-init W,b -> 2*sigmoid(0) = 1.0 = EXACT identity at init (grade-emb
            # discipline); range (0,2) so it can also amplify. +d*d+d = 1,056 params at d=32.
            if self.prehead_gate_on:
                self.prehead_gate_weight = torch.nn.Parameter(
                    torch.zeros(self.d_model, self.d_model)
                )
                self.prehead_gate_bias = torch.nn.Parameter(torch.zeros(self.d_model))

            if self.pava_lambda != 0.0:
                # 3 junction thetas, p_j = 2*tanh(theta_j), init p = 1 = classic PAVA.
                # 1D name without "weight" -> other_params (wd=0); "pava_" is in
                # DTYPE_EXCLUDE so the root-param cast walk keeps it fp32.
                from rwkv.model.pava import theta_init
                self.pava_theta = torch.nn.Parameter(theta_init())

    @torch.jit.ignore
    def _apply_input_feat_mask(self, batch_start: torch.Tensor) -> torch.Tensor:
        # Eager-Python indirection (same reason as _apply_grade_emb): the mask is a plain
        # tensor attribute, so device/dtype alignment happens here per call.
        return batch_start * self.input_feat_mask.to(batch_start.device, batch_start.dtype)

    @torch.jit.ignore
    def _apply_grade_emb(self, x: torch.Tensor, batch_start: torch.Tensor) -> torch.Tensor:
        # TorchScript-safe indirection: grade_emb only exists when RWKV_GRADE_EMB=1, and the
        # scripted forward_batch must not reference a conditionally-created attribute (the
        # compiler resolves attributes even in dead branches). Ignored body runs in Python --
        # and must use F.linear on a Parameter, NOT a submodule call (see the __init__ note).
        return x + torch.nn.functional.linear(batch_start[:, 9:13], self.grade_emb_weight)

    @torch.jit.ignore
    def _apply_prehead_gate(self, x: torch.Tensor) -> torch.Tensor:
        # TorchScript-safe indirection (same reason as _apply_grade_emb): the gate params
        # only exist when RWKV_PREHEAD_GATE=1. F.linear on Parameters, NOT a submodule call
        # (see the __init__ note -- submodule calls from ignored methods crash under JIT).
        return x * (2.0 * torch.sigmoid(torch.nn.functional.linear(
            x, self.prehead_gate_weight, self.prehead_gate_bias)))

    @torch.jit.ignore
    def _gru_heads(self, x_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TorchScript-safe indirection (gru params only exist under RWKV_GRU_HEAD>0):
        # F.linear on root Parameters, NOT submodule calls (see the __init__ note). x_w is
        # the shared head_w trunk output, already .float()'d by the caller. Returns the RAW
        # (pre-softmax / pre-exp) w-logits, log-S and log-d heads.
        w = torch.nn.functional.linear(x_w, self.gru_w_weight, self.gru_w_bias)
        s = torch.nn.functional.linear(x_w, self.gru_s_weight, self.gru_s_bias)
        d = torch.nn.functional.linear(x_w, self.gru_d_weight, self.gru_d_bias)
        return w, s, d

    @torch.jit.ignore
    def _apply_rcouple(
        self, out_p_logits: torch.Tensor, curve_logits: torch.Tensor
    ) -> torch.Tensor:
        """Retrievability coupling, applied to the 4 rating logits (RWKV_RCOUPLE).

        ⚠ MUST live behind @torch.jit.ignore: `rcouple_w` is a CONDITIONAL Parameter (it exists
        only when the flag is on, so that a 421-key champion checkpoint still loads strictly), and
        TorchScript refuses to compile a forward that touches an attribute which may not exist.
        Same rule as the PAVA thetas. Body uses a ROOT Parameter only -- never a submodule call,
        which crashes when scripted code enters an ignored body (the iter-16 hollow-run lesson).
        ⚠ Root Parameters are invisible to selective_cast's module walk, so `rcouple_w` stays fp32
        while `out_p_logits` is bf16 under autocast; cast explicitly, or the product silently
        promotes the rating logits to fp32 and changes every downstream dtype.
        """
        rc = torch.clamp(curve_logits, -self.rcouple_clip, self.rcouple_clip)
        if self.rcouple_detach:
            rc = rc.detach()
        return out_p_logits + rc.unsqueeze(-1) * self.rcouple_w.to(out_p_logits.dtype)

    @torch.jit.ignore
    def _pava_rectify_eval(
        self,
        curve_probs: torch.Tensor,
        probe_rows: torch.Tensor,
        probe_target: torch.Tensor,
        probe_pressed: torch.Tensor,
    ) -> torch.Tensor:
        """Replace each scored row's ahead prediction with its rectified pressed value.

        Returns a DETACHED copy of curve_probs (eval-only; never on the loss path). The
        four probe rows of a scored review are that review with the grade one-hot swapped
        and the current-row duration zeroed -- identical inputs apart from the button, which
        is what makes the ordering comparison meaningful. Uniform pooling weights: iter 24
        measured p-head weighting as null, so deploy keeps the simpler rectifier.
        """
        from rwkv.model.pava import pava_rectify
        out = curve_probs.detach().clone()
        if not self.eval_pava_substitute:
            # RWKV_EVAL_PAVA=3: probes were inserted (so the batch is re-bucketed exactly as in
            # modes 1/2) but the scored rows keep their OWN prediction. Isolates the bf16 noise.
            return out
        flat = out.reshape(-1)
        v = flat[probe_rows]  # (M,4) Again..Easy
        if self.eval_pava_rectify:
            powers = (2.0 * torch.tanh(self.pava_theta) if self.pava_lambda != 0.0
                      else torch.ones(3, device=v.device, dtype=torch.float32))
            src = pava_rectify(v.float(), torch.ones_like(v, dtype=torch.float32), powers)
        else:
            # RWKV_EVAL_PAVA=2: substitute the probe WITHOUT pooling, so the only thing that
            # differs from mode 0 is the zeroed current-row duration.
            src = v.float()
        pressed = src.gather(1, probe_pressed.unsqueeze(1)).squeeze(1)
        flat[probe_target] = pressed.to(flat.dtype)
        return out

    @torch.jit.ignore
    def _pava_probe_loss(
        self,
        curve_probs: torch.Tensor,
        label_y: torch.Tensor,
        out_p_logits: torch.Tensor,
        probe_rows: torch.Tensor,
        probe_target: torch.Tensor,
        probe_pressed: torch.Tensor,
        probe_query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Eager body (jit.ignore): the intricate mask-simulated PAVA lives in
        # rwkv/model/pava.py; only Parameters + functional ops here (iter-16 rule).
        # curve_probs/label_y are (B,T); probe_rows (M,4) holds b*T+t flat indices of the
        # 4 probe skip rows (Again..Easy) of each probed review; probe_target/probe_query
        # (M,) flat indices of the real row (label source) and its paired imm query row
        # (p-head weight source, iter 24). Counterfactual probes get gradient only through
        # pooling; the pressed probe's rectified value takes the BCE against the real
        # row's ahead label.
        from rwkv.model.pava import pava_rectify
        cp = curve_probs.reshape(-1)
        v = cp[probe_rows]  # (M,4)
        if self.pava_pweight:
            pq = out_p_logits.reshape(-1, 4)[probe_query]  # (M,4) decision-point logits
            w = torch.softmax(pq.float(), dim=-1).clamp(min=1e-4)
        else:
            w = torch.ones_like(v)
        powers = 2.0 * torch.tanh(self.pava_theta)
        rect = pava_rectify(v.float(), w, powers)
        pressed = rect.gather(1, probe_pressed.unsqueeze(1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
        target = label_y.reshape(-1)[probe_target].float()
        loss = torch.nn.functional.binary_cross_entropy(pressed, target)
        pool_frac = (rect != v).any(dim=1).float().mean()
        return loss, pool_frac

    @FunctionType
    def head_and_out(self, input):
        x = self.prehead_dropout(self.prehead_norm(input))
        if self.prehead_gate_on:
            x = self._apply_prehead_gate(x)

        x_w = self.head_w(x).float()
        if self.gru_on:
            out_w_logits, out_s_raw, out_d_raw = self._gru_heads(x_w)
        else:
            out_w_logits = self.w_linear(x_w)
            # dummy placeholders so the return arity/type is branch-independent
            out_s_raw = torch.zeros(1, dtype=torch.float32, device=x.device)
            out_d_raw = torch.zeros(1, dtype=torch.float32, device=x.device)
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        out_w_log_p = torch.nn.functional.log_softmax(out_w_logits, dim=-1)
        if self.no_ahead_residual:
            # piecewise-linear correction disabled (Andrew 2026-07-16): constant-zero
            # residual; mag/diff stats and interp see exact zeros, ahead head gets no grad
            # explicit dims: x.shape concat types differ between TorchScript (List[int])
            # and eager (torch.Size); head_and_out input is always (B, T, C) here
            out_ahead_logits = torch.zeros(
                x.size(0), x.size(1), self.num_points, dtype=torch.float32, device=x.device
            )
        else:
            out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())
            if self.mono_curve_on:
                # running lower envelope over time points -> non-increasing residual (iter 22);
                # projected values feed interp AND the mag/diff stats uniformly
                out_ahead_logits, _ = torch.cummin(out_ahead_logits, dim=-1)

        x_p = self.head_p(x).float()
        return out_ahead_logits, out_w, out_w_log_p, self.p_linear(x_p), out_s_raw, out_d_raw

    @FunctionType
    def forgetting_curve(self, w, label_elapsed_seconds):
        s_space_raw = torch.exp(
            torch.linspace(0, self.s_point_spread, self.num_curves, device=w.device)
        )
        s_space = 0.1 + (s_space_raw - 1) * (np.e ** (self.s_max - self.s_point_spread))
        label_elapsed_seconds = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(
            w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1
        )

    @FunctionType
    def gru_forgetting_curve(self, w, s_raw, d_raw, label_elapsed_seconds):
        # GRU-faithful mixture (srs-benchmark models/gru.py):
        #   R(t) = sum_i w_i * (1 + t/(1e-7+S_i))^(-d_i),  S,d = exp(clamp(., -25, 25))
        # exp => d_i > 0 => every curve strictly decreasing in t (monotone by construction).
        # Power via exp(-d * log1p(t/S)) for stability: t/S <= ~1e16 -> log1p ~ 37, and a
        # huge d only UNDERFLOWS the exp to exact 0 (never inf/NaN). Squash to
        # (1e-5, 1-1e-5) like forgetting_curve so the downstream logit() stays finite.
        s = torch.exp(torch.clamp(s_raw, min=-25.0, max=25.0))
        d = torch.exp(torch.clamp(d_raw, min=-25.0, max=25.0))
        t = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        r = torch.sum(w * torch.exp(-d * torch.log1p(t / (1e-7 + s))), dim=-1)
        return 1e-5 + (1 - 2 * 1e-5) * r

    @FunctionType
    def interp(self, out_ahead_logits, label_elapsed_seconds):
        label_elapsed_seconds = torch.clamp(label_elapsed_seconds.contiguous(), min=1)
        point_space_raw = torch.exp(
            torch.linspace(
                0, self.point_spread, self.num_points, device=out_ahead_logits.device
            )
        )
        point_space = 0.5 + (point_space_raw - 1) * (
            np.e ** (self.max_e - self.point_spread)
        )
        right_idx = torch.searchsorted(point_space, label_elapsed_seconds)
        left_idx = torch.clamp(right_idx - 1, min=0)
        xl, xr = point_space[left_idx], point_space[right_idx]
        yl = torch.gather(out_ahead_logits, dim=-1, index=left_idx)
        yr = torch.gather(out_ahead_logits, dim=-1, index=right_idx)
        res = 1e-5 + (1 - 2 * 1e-5) * (
            yl + (yr - yl) * (label_elapsed_seconds - xl) / (xr - xl)
        )
        return res.squeeze(-1)

    @FunctionType
    def forward_batch(
        self,
        batch_start: torch.Tensor,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
        batch_num_data: int,
    ):
        if self.input_feat_mask_on:
            batch_start = self._apply_input_feat_mask(batch_start)
        x = self.features2card(batch_start)
        if self.grade_emb_on:
            x = self._apply_grade_emb(x, batch_start)

        assert len(batch_sub_gather) == len(self.rwkv_modules)
        if self.interleave_on:
            x = self._interleaved_streams(
                x,
                batch_sub_gather,
                batch_sub_gather_lens,
                batch_time_shift_selects,
                batch_skips,
            )
            x = x.view(batch_num_data, -1, self.d_model)
            return self.head_and_out(x)
        for i, submodule in enumerate(self.rwkv_modules):
            module_splits = batch_sub_gather[i]
            sub_lens = batch_sub_gather_lens[i]
            time_shift_selects = batch_time_shift_selects[i]
            skips = batch_skips[i]
            y = []
            for split_gather, sub_len, time_shift_select, skip in zip(
                module_splits, sub_lens, time_shift_selects, skips
            ):
                if self.use_perm_gather:
                    module_in = perm_gather(x, split_gather).view(
                        -1, sub_len, self.d_model
                    )
                else:
                    module_in = torch.index_select(
                        x, dim=0, index=torch.clamp(split_gather, min=0)
                    ).view(-1, sub_len, self.d_model)
                time_shift_select_BT = time_shift_select.view(-1, sub_len)
                skip_BT = skip.view(-1, sub_len)
                assert module_in.size(0) == time_shift_select_BT.size(
                    0
                ) and module_in.size(0) == skip_BT.size(0)
                module_out = submodule(
                    module_in,
                    time_shift_select_BT=time_shift_select_BT,
                    skip_BT=skip_BT,
                )
                y.append(module_out.view(-1, self.d_model))

            x = torch.cat(y)

        x = x.view(batch_num_data, -1, self.d_model)
        return self.head_and_out(x)

    # iter 41 RWKV_INTERLEAVE -- the round-robin execution of the same 5 stacks.
    #
    # WHY THE GATHER COMPOSITION EXISTS: prepare() builds each stream's gather indices
    # against the PREVIOUS stream's output layout (current_locs_list chains), because the
    # sequential form never returns to an earlier stream. Interleaving does, so every
    # (stream, split) gather is re-anchored to the CANONICAL layout (features2card order --
    # the layout labels/probes index) by composing the chained permutations once per batch;
    # x then lives in canonical order the whole time, each layer-step doing
    # gather -> one layer -> scatter-back. Pad slots (-1) are dropped at scatter, never
    # written. This stays MODEL-side on purpose: prepare() runs in the fetch workers, and
    # touching it would change the batch stream (KD dump identity, byte-identical replays).
    #
    # v0 (the value-residual) is stream-local: each stream's round-0 layer has local
    # layer_id==0 and SETS v0 (the incoming empty tensor is ignored); it is stored per
    # (stream, split) in the STREAM'S layout and re-fed to that stream's later rounds --
    # identical threading to the sequential form, just spread across rounds.
    @FunctionType
    def _interleaved_streams(
        self,
        x,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
    ):
        # -- 1) compose canonical-anchored gathers + their scatter halves, once per batch --
        n_rows = x.size(0)
        cur = torch.arange(n_rows, dtype=torch.long, device=x.device)
        gath: list[list[torch.Tensor]] = []   # per (stream, split): canonical src index, -1 pads
        spos: list[list[torch.Tensor]] = []   # per (stream, split): non-pad positions in the flat split
        stgt: list[list[torch.Tensor]] = []   # per (stream, split): canonical target rows for those
        for i in range(len(batch_sub_gather)):
            gi: list[torch.Tensor] = []
            pi: list[torch.Tensor] = []
            ti: list[torch.Tensor] = []
            parts: list[torch.Tensor] = []
            for p in batch_sub_gather[i]:
                pl = p.long()
                g = torch.where(
                    pl >= 0,
                    torch.index_select(cur, 0, torch.clamp(pl, min=0)),
                    torch.full_like(pl, -1),
                )
                pos = torch.nonzero(g >= 0).squeeze(-1)
                gi.append(g)
                pi.append(pos)
                ti.append(torch.index_select(g, 0, pos))
                parts.append(g)
            gath.append(gi)
            spos.append(pi)
            stgt.append(ti)
            cur = torch.cat(parts)
        # -- 2) per-(stream, split) v0 stores (round 0 sets them; later rounds consume) --
        v0s: list[list[torch.Tensor]] = []
        for i in range(len(batch_sub_gather)):
            vi: list[torch.Tensor] = []
            for _s in range(len(batch_sub_gather[i])):
                vi.append(torch.empty(0))
            v0s.append(vi)
        # -- 3) the rounds --
        max_depth = 0
        for d in self.stream_depths:
            max_depth = max(max_depth, d)
        for r in range(max_depth):
            i = 0
            for submodule in self.rwkv_modules:
                # iter 44: which of THIS stream's layers runs this round (-1 = sits out).
                # With spread off this is exactly `r if r < depth else -1`, so the sequence of
                # forward_layer calls is unchanged and the path stays bit-identical.
                lj = self.ilv_sched[i][r]
                if lj >= 0:
                    sub_lens = batch_sub_gather_lens[i]
                    tss = batch_time_shift_selects[i]
                    sks = batch_skips[i]
                    for s in range(len(gath[i])):
                        g = gath[i][s]
                        sub_len = sub_lens[s]
                        x_in = torch.index_select(x, 0, torch.clamp(g, min=0)).view(
                            -1, sub_len, self.d_model
                        )
                        # ⚠ keyed on the STREAM-LOCAL layer index, not the round: under spread a
                        # stream's layer 0 can land in a later round, and layer 0 is the one that
                        # SETS v0 (the passed tensor is ignored there).
                        if lj == 0:
                            v0_in = torch.empty_like(x_in)
                        else:
                            v0_in = v0s[i][s]
                        x_out, v0_out = submodule.forward_layer(
                            lj,
                            x_in,
                            v0_in,
                            tss[s].view(-1, sub_len),
                            sks[s].view(-1, sub_len),
                        )
                        v0s[i][s] = v0_out
                        flat = x_out.reshape(-1, self.d_model)
                        x = x.index_copy(
                            0, stgt[i][s], torch.index_select(flat, 0, spos[i][s])
                        )
                i += 1
        return x

    @FunctionType
    def nanmin(self, tensor):
        output = tensor.nan_to_num(1e9).min()
        return output

    @FunctionType
    def nanmax(self, tensor):
        output = tensor.nan_to_num(-1e9).max()
        return output

    @FunctionType
    def _get_loss(
        self,
        batch_start: torch.Tensor,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
        batch_num_data: int,
        batch_labels: torch.Tensor,
        batch_label_review_th: torch.Tensor,
        # typed for TorchScript (an untyped kd infers as Tensor and the tuple unpack fails to script)
        kd: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
        kd_mix: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
        # iter 23 probe channel: (probe_rows (M,4), probe_target (M,), probe_pressed (M,),
        # probe_query (M,)) flat b*T+t indices; None = no probes in this batch
        probes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        # iter 37: (B,) per-chunk loss weight (mean 1), or None for the historical flat mean.
        user_weight: Optional[torch.Tensor] = None,
        # iter 46 (RWKV_SELFKD_BETA): (B,T) int64 index of each row's self-distillation teacher
        # row, -1 = none. Ignored at beta=0.
        ahead_query: Optional[torch.Tensor] = None,
    ):
        out_ahead_logits, out_w, out_w_log_p, out_p_logits, out_s_raw, out_d_raw = self.forward_batch(
            batch_start,
            batch_sub_gather,
            batch_sub_gather_lens,
            batch_time_shift_selects,
            batch_skips,
            batch_num_data,
        )
        # NaN probe: with the residual disabled, out_ahead_logits is constant zeros and can
        # never NaN -- probe the (live) rating head instead, so a trunk NaN still returns
        # None (the eval nanskip + train-loop guard both key off this).
        nan_probe = out_p_logits if self.no_ahead_residual else out_ahead_logits
        if torch.isnan(nan_probe).any():
            return None

        global_labels = batch_labels.float()
        (
            label_elapsed_seconds,
            _,
            label_y,
            label_rating,
            has_label,
            label_is_equalize,
            is_query,
        ) = global_labels.unbind(-1)
        has_label = has_label.int()
        label_is_equalize = label_is_equalize.int()
        is_query = is_query.int()

        label_rating = torch.clamp(label_rating - 1, min=0)
        # Warmup-KD target mix (iter 10): kd_mix = (teacher_curve_probs, teacher_p_probs, alpha)
        # from the stored d=128 teacher dump. BCE/CE are linear in the target, so mixing TARGETS
        # (alpha*teacher + (1-alpha)*hard) is exactly the annealed soft-target design. alpha
        # anneals 1 -> 0 across the KD window in train_rwkv; masks/scales untouched. The 4-way
        # rating CE gets its mixed prob target below (after p_loss). None => byte-identical.
        # PRIVILEGED SELF-DISTILLATION (iter 46) -- runs BEFORE the KD mix, and that order is the
        # whole design. It softens the HARD label only, so when an external teacher is also active
        # the composition is
        #     a*d128_teacher + (1-a) * [ beta*imm_teacher + (1-beta)*hard ]
        # i.e. alpha keeps its tuned value exactly (iter 39 swept 0.5/0.75/0.9/1.0; 0.9 won) and
        # beta reallocates only the residual hard share. Softening the POST-KD target instead
        # would drag a from 0.9 to 0.9*(1-beta) -- two changes in one experiment, which is what
        # iters 42/43/44 were spent un-bundling.
        if self.selfkd_beta != 0.0 and ahead_query is not None:
            _aq = ahead_query.reshape(-1)
            # index 0 is Again (see out_p_probs.unbind below), so 1-P(Again) = P(success), the
            # same quantity label_y holds. .detach() is load-bearing: the teacher must not be
            # pulled toward the weaker curve head, or the advantage being distilled is traded away.
            _psucc = (1.0 - torch.softmax(out_p_logits.float(), dim=-1)[:, :, 0]).reshape(-1)
            _teacher = _psucc[_aq.clamp(min=0)].detach()
            _ly = label_y.reshape(-1)
            _soft = self.selfkd_beta * _teacher + (1.0 - self.selfkd_beta) * _ly
            label_y = torch.where(_aq >= 0, _soft, _ly).reshape(label_y.shape)
        # ⚠ label_y is OVERWRITTEN here, and _pava_probe_loss reads the overwritten value -- so
        # under KD the probe path's "ahead label" is already alpha*teacher + (1-alpha)*hard, NOT a
        # hard 0/1.
        if kd_mix is not None:
            _km_curve, _km_p, _km_alpha = kd_mix
            label_y = _km_alpha * _km_curve + (1.0 - _km_alpha) * label_y
        label_elapsed_seconds = label_elapsed_seconds.unsqueeze(-1)
        if self.gru_on:
            curve_probs_raw = self.gru_forgetting_curve(
                out_w, out_s_raw, out_d_raw, label_elapsed_seconds
            )
        else:
            curve_probs_raw = self.forgetting_curve(out_w, label_elapsed_seconds)
        curve_logits_raw = torch.log(
            curve_probs_raw / (1 - curve_probs_raw)
        )  # inverse sigmoid
        ahead_logit_residual = self.interp(out_ahead_logits, label_elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        curve_probs = torch.sigmoid(curve_logits)

        # ---- RETRIEVABILITY COUPLING (see __init__). Placed HERE, after curve_logits and before
        # the rating softmax, so the rating head sees R(t) at the SAME t it is predicting at.
        # NOT gated on is_query: the coupling is a property of the MODEL, not of the loss, so it
        # must apply on every row or train/eval/deploy would compute different functions (the §9
        # three-way-parity rule -- PAVA was trained-but-not-evaluated for 8 iterations exactly
        # because a model-side transform lived inside the loss).
        if self.rcouple_on:
            out_p_logits = self._apply_rcouple(out_p_logits, curve_logits)

        out_p_probs = torch.softmax(out_p_logits, dim=-1)
        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(dim=-1)
        out_p_binary = torch.clamp(1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5)

        if torch.isnan(curve_probs).any():
            raise Exception("nan")
        w_loss = torch.nn.functional.kl_div(
            input=out_w_log_p,
            target=torch.ones_like(out_w) / self.num_curves,
            reduction="none",
        ).mean(dim=-1)
        ahead_mask = (1 - is_query) * has_label
        # iter 33 (Andrew 2026-07-27, "everywhere: duration of the most recent review zeroed out").
        # The ahead loss normally lands on the REAL row, which carries that review's OWN duration --
        # a feature deploy can never have, because Anki must show intervals BEFORE the press. The
        # probe rows are the same review with the duration zeroed, and _pava_probe_loss already
        # scores the pressed probe's RECTIFIED value against the real row's ahead label, which is
        # exactly the quantity deploy serves. So with RWKV_AHEAD_PROBE_ONLY=1 the probed rows drop
        # out of the real-row ahead term and the ahead objective is carried entirely by the probe
        # path -- train, eval and CPU inference then compute one quantity.
        # Measured cost of the duration mismatch this removes: +0.001451 ahead (mode-2 diagnostic).
        # Rows NOT eligible for probes (a card's first in-chunk review) keep the real-row term;
        # they are the honest residual, not an oversight.
        if probes is not None and self.ahead_probe_only:
            _pt = probes[1]  # (M,) flat indices of the probed real rows
            ahead_mask = ahead_mask.clone()
            ahead_mask.reshape(-1)[_pt] = 0
        immediate_mask = is_query * has_label
        assert ahead_mask.shape == label_is_equalize.shape
        ahead_equalize_mask = ahead_mask * label_is_equalize

        immediate_equalize_mask = immediate_mask * label_is_equalize
        # ---- iter 37: OBJECTIVE ALIGNMENT (RWKV_USER_WEIGHT) -------------------------------
        # The gate is the BY-USER mean LogLoss (every user counts once), but this loss is a flat
        # mean over ROWS, so a 360k-review user drives ~720x the gradient of a 500-review one
        # while counting the same at eval. user_weight is (B,) = 1/N_u normalized to mean 1
        # (each batch row is exactly one user's chunk -- see prepare(), which stacks data_list in
        # order), so these become weighted means: same magnitude, hence the tuned LRs still hold.
        # ⚠ WEIGHTED: the OBJECTIVE terms only. The *_equalize_avg metrics and the *_n counts
        # below stay UNWEIGHTED on purpose -- they are what the step trace reports and what the
        # champion's vprune reference is made of, so weighting them would silently make traces
        # non-comparable across runs (and turn a row count into a fractional weight sum).
        # float() on the new path: weights are O(1) but bf16 carries only ~3 digits.
        if user_weight is not None:
            _uw = user_weight.reshape(-1, 1).float()
            ahead_wmask = ahead_mask.float() * _uw
            immediate_wmask = immediate_mask.float() * _uw
        else:
            ahead_wmask = ahead_mask
            immediate_wmask = immediate_mask
        curve_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits, label_y, reduction="none"
        )
        curve_raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits_raw, label_y, reduction="none"
        )
        NUM_LABELS = 4
        B, T = label_rating.shape
        p_loss = torch.nn.functional.cross_entropy(
            out_p_logits.view(-1, NUM_LABELS),
            label_rating.long().view(-1),
            reduction="none",
        ).view(B, T)
        # Warmup-KD (cont.): rating target = alpha*teacher_p_probs + (1-alpha)*one_hot(hard).
        # Soft-target CE via -(p * log_softmax(q)) -- the scripted-proven pattern from the kd block.
        if kd_mix is not None:
            _km2_curve, _km2_p, _km2_alpha = kd_mix
            _km2_target = _km2_alpha * _km2_p + (1.0 - _km2_alpha) * torch.nn.functional.one_hot(
                label_rating.long(), NUM_LABELS
            ).float()
            p_loss = (
                -(
                    _km2_target.view(-1, NUM_LABELS)
                    * torch.log_softmax(out_p_logits.float().view(-1, NUM_LABELS), dim=-1)
                )
                .sum(dim=-1)
                .view(B, T)
            )
        p_binary_loss = torch.nn.functional.binary_cross_entropy(
            out_p_binary, label_y, reduction="none"
        )
        ahead_avg = (curve_loss * ahead_wmask).sum() / (1e-8 + ahead_wmask.sum())
        AHEAD_SCALE = 0.5
        ahead_raw_avg = (curve_raw_loss * ahead_wmask).sum() / (1e-8 + ahead_wmask.sum())
        AHEAD_RAW_SCALE = 0.5
        immediate_avg = (p_loss * immediate_wmask).sum() / (1e-8 + immediate_wmask.sum())
        # Optimization-loop knob (local literal — TorchScript can't use module globals/env):
        # weight on the immediate 4-way rating loss. 1.0 = original. (iter2 tried 2.0 -> imm
        # got WORSE, reverted.)
        IMMEDIATE_SCALE = 1.0
        w_avg = (w_loss * ahead_wmask).sum() / (1e-8 + ahead_wmask.sum())
        W_LOSS_SCALE = 1e-5
        ahead_logits_mag_loss = torch.sqrt(
            1e-16 + out_ahead_logits.square().mean(dim=-1)
        )
        ahead_logits_mag_avg = (ahead_logits_mag_loss * ahead_wmask).sum() / (
            1e-8 + ahead_wmask.sum()
        )
        AHEAD_LOGITS_MAG_LOSS_SCALE = 1e-4
        ahead_logits_diff_loss = torch.sqrt(
            1e-16 + out_ahead_logits.diff().square().mean(dim=-1)
        )
        ahead_logits_diff_avg = (ahead_logits_diff_loss * ahead_wmask).sum() / (
            1e-8 + ahead_wmask.sum()
        )
        AHEAD_LOGITS_DIFF_LOSS_SCALE = 1e-3
        loss_avg = (
            AHEAD_SCALE * ahead_avg
            + IMMEDIATE_SCALE * immediate_avg
            + AHEAD_RAW_SCALE * ahead_raw_avg
            + W_LOSS_SCALE * w_avg
            + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_avg
            + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_avg
        )
        if self.pbin_scale != 0.0:
            # iter 17: the benchmark-imm objective, trained directly (see __init__ note).
            pbin_avg = (p_binary_loss * immediate_wmask).sum() / (1e-8 + immediate_wmask.sum())
            loss_avg = loss_avg + self.pbin_scale * pbin_avg
        # iter 23: learnable power-mean PAVA on the 4 counterfactual probe curves
        pava_loss_avg = ahead_avg.detach() * 0.0
        pava_pool_frac = ahead_avg.detach() * 0.0
        if probes is not None and self.pava_lambda != 0.0:
            probe_rows, probe_target, probe_pressed, probe_query = probes
            pava_loss, pava_frac = self._pava_probe_loss(
                curve_probs, label_y, out_p_logits, probe_rows, probe_target,
                probe_pressed, probe_query,
            )
            loss_avg = loss_avg + self.pava_lambda * pava_loss
            pava_loss_avg = pava_loss.detach()
            pava_pool_frac = pava_frac.detach()
        # RWKV_QAT_RANK1_REG (2026-08-14): rank-1-friendly regularizer. The rank-truncated streams
        # accumulate a per-layer proxy penalty during their forward; drain it here and add it once.
        # Returns None when the lever is off, so the default path adds no term at all.
        # ⚠ self.rank1_reg_lambda, NOT the module-level global: TorchScript cannot resolve a
        # Python global in a scripted forward, and referencing one here made the WHOLE SrsRWKV
        # fail to script -- invisible to every QAT run (QAT forces RWKV_NO_JIT=1) and fatal on the
        # next plain JIT-on run. Found 2026-08-15; see the twin fix in rwkv_model.py.
        _r1 = take_rank1_penalty()
        if _r1 is not None:
            loss_avg = loss_avg + self.rank1_reg_lambda * _r1
        # KD (RWKV_QAT_KD, task22): distill from the un-quantized fp32 champion during QAT. Anchors the
        # base against drift while the net learns quant robustness. kd = (teacher_p_logits,
        # teacher_curve_probs, lambda), computed in train_rwkv under no_grad. Soft-label CE on the 4-way
        # immediate head + soft-label BCE on the retention-curve head, same masks/scales as the data terms.
        if kd is not None:
            t_p_logits, t_curve_probs, kd_lam = kd
            kd_p = -(
                torch.softmax(t_p_logits, dim=-1)
                * torch.log_softmax(out_p_logits.float(), dim=-1)
            ).sum(dim=-1)
            kd_p_avg = (kd_p * immediate_wmask).sum() / (1e-8 + immediate_wmask.sum())
            kd_c = torch.nn.functional.binary_cross_entropy_with_logits(
                curve_logits.float(), t_curve_probs, reduction="none"
            )
            kd_c_avg = (kd_c * ahead_wmask).sum() / (1e-8 + ahead_wmask.sum())
            loss_avg = loss_avg + kd_lam * (
                IMMEDIATE_SCALE * kd_p_avg + AHEAD_SCALE * kd_c_avg
            )
        loss_tensor = (
            AHEAD_SCALE * curve_loss.detach()
            + p_loss.detach()
            + AHEAD_RAW_SCALE * curve_raw_loss.detach()
            + W_LOSS_SCALE * w_loss.detach()
            + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_loss.detach()
            + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_loss.detach()
        )

        ahead_equalize_avg = (curve_loss * ahead_equalize_mask).sum() / (
            1e-8 + ahead_equalize_mask.sum()
        )
        ahead_raw_equalize_avg = (curve_raw_loss * ahead_equalize_mask).sum() / (
            1e-8 + ahead_equalize_mask.sum()
        )
        immediate_binary_equalize_avg = (
            p_binary_loss * immediate_equalize_mask
        ).sum() / (1e-8 + immediate_equalize_mask.sum())

        # Rectified eval: the reported ahead prediction becomes the rectified pressed-button
        # value. Done here, on the OUTPUT only, so losses/gradients are untouched and every
        # downstream consumer (extract_p -> ahead_ps -> get_stats) needs no change.
        p_curve_out = curve_probs.detach()
        if self.eval_pava and probes is not None:
            probe_rows_e, probe_target_e, probe_pressed_e, _pq = probes
            p_curve_out = self._pava_rectify_eval(
                curve_probs, probe_rows_e, probe_target_e, probe_pressed_e
            )

        return SrsRWKVIterStatistics(
            average_loss=loss_avg,
            p_curve=p_curve_out,
            p_imm=out_p_binary.detach(),
            p_imm_all=out_p_probs.detach(),
            loss_tensor=loss_tensor.detach(),
            ahead_avg=ahead_avg.detach(),
            ahead_raw_avg=ahead_raw_avg.detach(),
            ahead_n=int(ahead_mask.sum().detach().item()),
            ahead_equalize_avg=ahead_equalize_avg.detach(),
            ahead_raw_equalize_avg=ahead_raw_equalize_avg.detach(),
            ahead_equalize_n=int(ahead_equalize_mask.sum().detach().item()),
            imm_avg=immediate_avg.detach(),
            imm_n=int(immediate_mask.sum().detach().item()),
            imm_binary_equalize_avg=immediate_binary_equalize_avg.detach(),
            imm_binary_equalize_n=int(immediate_equalize_mask.sum().detach().item()),
            w_loss_avg=w_avg.detach(),
            ahead_logits_mag_loss_avg=ahead_logits_mag_avg.detach(),
            ahead_logits_diff_loss_avg=ahead_logits_diff_avg.detach(),
            w=out_w.detach(),
            label_review_th=batch_label_review_th.detach(),
            label_elapsed_seconds=label_elapsed_seconds.detach(),
            label_rating=label_rating.detach(),
            is_query=is_query.detach(),
            has_label=has_label.detach(),
            pava_loss_avg=pava_loss_avg,
            pava_pool_frac=pava_pool_frac,
        )

    def get_loss(self, batch: PreparedBatch,
                 kd: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
                 kd_mix: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None):
        probes = None
        if batch.probe_rows is not None and batch.probe_rows.numel() > 0:
            probes = (batch.probe_rows, batch.probe_target,
                      batch.probe_pressed, batch.probe_query)
        return self._get_loss(
            batch.start,
            batch.sub_gather,
            batch.sub_gather_lens,
            batch.time_shift_selects,
            batch.skips,
            batch.num_data,
            batch.labels,
            batch.label_review_th,
            kd=kd,
            kd_mix=kd_mix,
            probes=probes,
            user_weight=batch.user_weight,
            ahead_query=batch.ahead_query,
        )

    def copy_downcast_(self, master_model, dtype):
        # Vectorized fp32-master -> (bf16/fp32)-child param copy via torch._foreach_copy_: one fused
        # kernel per dtype group instead of ~440 per-param copy launches (a launch-bound hotspot,
        # ~24 ms/step). copy_ casts, so grouping by target dtype + foreach is BIT-IDENTICAL to the
        # original per-param loop. Arch-agnostic (operates on whatever params exist).
        master_params = dict(master_model.named_parameters())
        groups: dict = {}  # target_dtype -> ([dst...], [src...])
        for name, param in self.named_parameters():
            target_dtype = torch.float32 if is_excluded(name) else dtype
            assert param.dtype == target_dtype
            dst, src = groups.setdefault(target_dtype, ([], []))
            dst.append(param.data)
            src.append(master_params[name].data)
        with torch.no_grad():
            for dst, src in groups.values():
                torch._foreach_copy_(dst, src)

    def selective_cast(self, dtype):
        for name, module in self.named_modules():
            if len(name) == 0:
                # Skip the root module
                continue
            if not is_excluded(name):
                if dtype == torch.bfloat16:
                    module = module.to(dtype)
                elif dtype == torch.half:
                    raise ValueError("not tested.")
                elif dtype == torch.float32:
                    pass
        if dtype == torch.bfloat16:
            # ROOT-LEVEL direct Parameters (prehead_gate_weight/grade_emb_weight -- the
            # jit.ignore-safe Parameter form, iter 16) are invisible to the module walk
            # above (the root is skipped so the excluded fp32 heads survive), so cast them
            # explicitly here; copy_downcast_ asserts child dtype == bf16 for non-excluded
            # names and crashed without this (2026-07-15).
            for pname, p in self.named_parameters(recurse=False):
                if not is_excluded(pname):
                    p.data = p.data.to(dtype)
        return self


@dataclass
class AnkiRWKVDictStatistics:
    ahead_ps: dict[int, float]
    imm_ps: dict[int, float]
    imm_ps_all: dict
    label_ratings: dict[int, float]
    label_elapsed_seconds: dict[int, float]
    w: torch.Tensor


def extract_p(stats: SrsRWKVIterStatistics):
    """Creates a nicer summary"""
    assert stats.label_review_th.size(0) == 1  # Only allow batch sizes of 1
    label_review_ths = stats.label_review_th.squeeze(0).cpu().numpy()
    label_elapsed_seconds_list = stats.label_elapsed_seconds.squeeze(0).cpu().numpy()
    label_ratings_list = stats.label_rating.squeeze(0).cpu().numpy()
    has_labels = stats.has_label.squeeze(0).cpu().numpy()
    is_querys = stats.is_query.squeeze(0).cpu().numpy()
    p_curves = stats.p_curve.squeeze(0).cpu().numpy()
    p_imms = stats.p_imm.squeeze(0).cpu().numpy()
    p_imm_alls = stats.p_imm_all.squeeze(0).cpu().numpy()
    ws = stats.w.squeeze(0).cpu()

    # Vectorized dict builds: same keys/values as the old per-index loop (iterating a
    # 1-D numpy selection yields the identical np scalars in the same order, so later
    # duplicates of a review_th still overwrite earlier ones); the masks mirror the
    # per-element `if has_label` / `if is_query` branches.
    label_mask = has_labels.astype(bool)
    query_mask = label_mask & is_querys.astype(bool)
    ahead_mask = label_mask & ~is_querys.astype(bool)

    label_elapsed_seconds_dict = dict(zip(label_review_ths, label_elapsed_seconds_list))
    label_ratings_dict = dict(zip(label_review_ths[label_mask], label_ratings_list[label_mask]))
    imm_ps_dict = dict(zip(label_review_ths[query_mask], p_imms[query_mask]))
    imm_ps_all_dict = dict(zip(label_review_ths[query_mask], p_imm_alls[query_mask]))
    ahead_ps_dict = dict(zip(label_review_ths[ahead_mask], p_curves[ahead_mask]))

    return AnkiRWKVDictStatistics(
        ahead_ps=ahead_ps_dict,
        imm_ps=imm_ps_dict,
        imm_ps_all=imm_ps_all_dict,
        label_ratings=label_ratings_dict,
        label_elapsed_seconds=label_elapsed_seconds_dict,
        w=ws,
    )


def greedy_splits(
    data_list: list[RWKVSample], factor, allowed_excess_in_one_step=20000
):
    """'factor' puts a limit on the memory complexity.
    'allowed_excess_in_one_step' captures the notion that at some point it is better to just separate the work into sequential calls
    example: if we are given [1, 1e6] then it would be worse to pad the 1 just to fit within the same batch.
    """
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        if submodule == RWKV_SUBMODULES[-1]:
            longest = 0
            for data in data_list:
                module_data = data.modules[submodule]
                longest = max(longest, module_data.split_len.max().item())
            splits_dict[submodule] = [longest]
            continue

        freqs = {}
        for data in data_list:
            module_data = data.modules[submodule]
            for l, b in zip(module_data.split_len, module_data.split_B):
                if l not in freqs:
                    freqs[l] = 0
                freqs[l] += b

        lens = list(reversed(sorted(freqs.keys())))
        splits = []
        l = 0
        while l < len(lens):
            r = l
            used = lens[l] * freqs[lens[l]]
            waste = 0
            while r + 1 < len(lens):
                next_used = used + lens[r + 1] * freqs[lens[r + 1]]
                extra_waste = (lens[l] - lens[r + 1]) * freqs[lens[r + 1]]
                next_waste = waste + extra_waste
                if (
                    factor * next_used >= next_waste
                    and extra_waste <= allowed_excess_in_one_step
                ):
                    used = next_used
                    waste = next_waste
                    r += 1
                else:
                    break

            splits.append(lens[l])
            l = r + 1

        splits.reverse()
        splits_dict[submodule] = splits

    return splits_dict


def naive_splits(data_list: list[RWKVSample]):
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        longest = 0
        for data in data_list:
            module_data = data.modules[submodule]
            longest = max(longest, module_data.split_len.max().item())

        print("longest", submodule, longest)
        if submodule == RWKV_SUBMODULES[-1]:
            splits_dict[submodule] = [longest]
            continue

        splits = []
        while longest > 0:
            splits.append(longest)
            longest = -1 + math.ceil(longest / 1.5)

        splits.reverse()
        splits_dict[submodule] = splits
    return splits_dict


if __name__ == "__main__":
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

    model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    t_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", t_param)
    a_param = sum(p.numel() for p in model.parameters())
    print("Number of parameters", a_param)
