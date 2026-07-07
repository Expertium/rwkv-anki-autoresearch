from dataclasses import dataclass
import math
import os
import torch

from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7, quant_aware_rwkv7

"""
IMPORTANT: the CUDA WKV kernel supports any head dim K = d_model // n_heads that DIVIDES 32 (K-aware
warp reduction added 2026-06-30; K=16 parity-verified). E.g. H=1/K=32 (champion) or H=2/K=16 (2x state).

Sources:
https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/src/model.py#L766
https://github.com/SmerkyG/RWKV_Explained/blob/main/rwkv7.py
"""

torch.manual_seed(2025)


def __nop(ob):
    return ob


# State-QAT (RWKV_NO_JIT=1) disables torch.jit so the quant-aware per-step WKV path runs as plain
# Python (avoids scripting the fake-quant loop). The default (JIT on) keeps the champion/eval path
# byte-for-byte unchanged. state_dict is identical either way, so weights load across both.
if os.environ.get("RWKV_NO_JIT"):
    ModuleType = torch.nn.Module
    FunctionType = __nop
else:
    ModuleType = torch.jit.ScriptModule
    FunctionType = torch.jit.script_method


@dataclass
class RWKV7Config:
    d_model: (
        int  # The model dimension. d_model / n_heads is the dimension for each head.
    )
    n_heads: int
    n_layers: int
    channel_mixer_factor: float

    # For stacking RWKV7 on top of one-another. We allow sending in the total number of layers and a layer offset so that we can achieve better initialization
    layer_offset: int
    total_layers: int

    decay_lora: int
    a_lora: int  # a = in-context learning rate
    v0_mix_amt_lora: int
    gate_lora: int

    dropout: float
    dropout_layer: float

    # State-QAT: per-step int-N round-trip of the WKV recurrent state (inf = off = fp32). Set per
    # stream (card/note get int4/int2 qmax; deck/preset/global stay inf). Only used with RWKV_NO_JIT.
    state_qmax: float = float("inf")
    # Low-rank state-QAT: per-step rank-r truncation of the WKV state (0 = off), factors optionally
    # int-N quantized (inf = fp32 factors). Takes precedence over state_qmax. The QAT analog of the
    # Rust deploy RWKV_STATE_LOWRANK_SCOPE. Only used with RWKV_NO_JIT.
    state_lowrank_rank: int = 0
    state_lowrank_fqmax: float = float("inf")
    # Shift-QAT: per-step int-N round-trip of the token-shift vectors (inf = off). The QAT analog of the
    # deploy RWKV_QUANT_SHIFTS + RWKV_STATE_SHIFT_LEVEL — matches what the engine persists+quantizes per
    # review for a compressed stream (the layernorm'd previous-token input). (The sibling ran this with
    # RWKV_NO_JIT; here it is TorchScript-annotated so the JIT-on path compiles it too.)
    state_shift_qmax: float = float("inf")


def fake_quant_shift(x_BTC: torch.Tensor, qmax: float) -> torch.Tensor:
    """STE per-row symmetric int-N quant of a token-shift tensor (B,T,C), matching the Rust deploy
    `quant_vec_inplace`: scale = max(amax/qmax, 1e-12) per (b,t) vector, q = round(x/scale).clamp(±qmax)*scale.
    forward = quantized shift, backward = identity (the shift is transparent to the gradient)."""
    with torch.no_grad():
        amax = x_BTC.abs().amax(dim=-1, keepdim=True)
        scale = (amax / qmax).clamp_min(1e-12)
        q = (x_BTC / scale).round().clamp(-qmax, qmax) * scale
    return x_BTC + (q - x_BTC).detach()


# ---- Shift-PQ QAT (RWKV_QAT_SHIFT_PQ=<codebook file>) -----------------------------------------------
# Product-quantize the token-shift vectors in the QAT forward, mirroring the Rust deploy RWKV_SHIFT_PQ
# path (PqCodebook::encode_decode, 2 roles: 0 = time-mixer t_xshift, 1 = channel-mixer c_xshift):
# normalize the C-dim vector, replace each of m sub-chunks by its nearest centroid (first strict min,
# matching argmin tie-breaking), rescale by the norm. STE backward for the shift path. Stream gating
# still comes from RWKV_QAT_SHIFT_SCOPE (its int level is IGNORED when the PQ codebook is set).
# RWKV_QAT_SHIFT_PQ_LEARN=1 (Andrew's "learnable parameters to assist QAT"): the codebook becomes a
# trainable f32 Parameter — centroids receive embedding-style gradients through the (frozen) selection,
# i.e. GRADIENT co-training of codebook + weights jointly (unlike the dead post-hoc refit). The learned
# codebook is exported at every save (train_rwkv) in the engine text format → deploy ships it, 0 extra
# per-card bits. Requires RWKV_NO_JIT.
_SHIFT_PQ_PATH = os.environ.get("RWKV_QAT_SHIFT_PQ", "")
_SHIFT_PQ_LEARN = os.environ.get("RWKV_QAT_SHIFT_PQ_LEARN", "") == "1"
_SHIFT_PQ_CB = None  # [2, m, ncent, sub] f32; plain tensor, or Parameter when _SHIFT_PQ_LEARN
_SHIFT_PQ_META = None  # (m, sub, ncent)
# RWKV_QAT_NORM_BITS: model the deploy norm quant (engine RWKV_PQ_NORM_BITS) — shift norms at n bits,
# log2-uniform over the engine's fixed shift range [2.2,2.9] octaves. Matching still uses the TRUE norm;
# only the reconstruction rescale is quantized (exact mirror of PqCodebook::encode_decode). The WKV
# analog is uploaded to the CUDA kernel in rwkv_ops.maybe_upload_pq_codebook (range [-3,0]).
_NORM_BITS = int(os.environ.get("RWKV_QAT_NORM_BITS", "0") or 0)
_SHIFT_NQ_LO, _SHIFT_NQ_HI = 2.2, 2.9


def _nq_quant_norm(norm):
    """Engine norm quant on a (N,1) f32 tensor: log2-uniform, round HALF-AWAY (floor(x+0.5) — clamp to
    [0,levels] makes it identical to Rust f32::round here), exp2 back. Caller masks norm>=1e-20."""
    levels = float((1 << _NORM_BITS) - 1)
    t = (norm.clamp_min(1e-20).log2() - _SHIFT_NQ_LO) / (_SHIFT_NQ_HI - _SHIFT_NQ_LO)
    q = torch.floor(t * levels + 0.5).clamp(0.0, levels)
    return torch.exp2(_SHIFT_NQ_LO + q / levels * (_SHIFT_NQ_HI - _SHIFT_NQ_LO))


# RWKV_QAT_SHIFT_ROT=1: LEARNED PRE-ROTATION for the shift PQ (SpinQuant/QuaRot adapted to product
# quantization). A per-role orthogonal R (Cayley: R = (I-A)(I+A)^-1, A = P - P^T, P learnable, P=0 ->
# R=I at init) rotates the C-dim shift vector BEFORE the chunk split and un-rotates the reconstruction.
# Rationale: product codebooks cannot express cross-chunk correlation; a learned rotation can move it
# across the chunk boundary — the one lever untested against the m4b5 capacity wall. Norms are
# rotation-invariant, so the norm path (incl. _NORM_BITS) is untouched. Deploy: engine RWKV_SHIFT_ROT
# loads the exported matrices and mirrors rotate -> encode_decode -> unrotate.
_SHIFT_ROT_ENV = os.environ.get("RWKV_QAT_SHIFT_ROT", "")
_SHIFT_ROT_LEARN = _SHIFT_ROT_ENV == "1"
_SHIFT_ROT_P = None      # Parameter [2, C, C] (the unconstrained Cayley pre-image), or None
_SHIFT_ROT_FIXED = None  # [2, C, C] orthogonal R loaded from a file (eval: RWKV_QAT_SHIFT_ROT=<path>)

# SPEED FLAGS (2026-07-06 profile: ~36k kernel launches/step, CPU ~1.9 s vs GPU ~0.4 s -> launch-bound).
# Both default OFF: they are numerically equivalent but NOT bit-identical to the old paths (fp summation
# order changes), so a run with a flag on is trajectory-perturbed vs its old-code twin (like a seed
# change). Never flip one mid-A/B.
# RWKV_QAT_ROT_CACHE=1: solve the Cayley transform ONCE per training step per role instead of per
#   fake_pq_shift call (~72 linalg_solve/step -> 2; ~150 ms CPU/step). The cached R is a shared autograd
#   node: grads into the rotation accumulate as J^T(sum u_i) instead of sum(J^T u_i) — equal math, last-
#   ulp different sums. Cache is cleared by shift_rot_cache_clear() (train loop, top of every step) and
#   is only ever filled under grad mode (no_grad validation/export always recompute fresh values).
# RWKV_QAT_FAST_EMB=1: hard-phase codebook lookup as one-hot @ cb instead of advanced indexing. Forward
#   bit-identical (one-hot picks rows exactly); backward turns the DETERMINISTIC-mode index_put_ (33% of
#   hard-phase GPU time, ~170 ms/step) into a deterministic mm with a different accumulation order.
_SHIFT_ROT_CACHE_ON = os.environ.get("RWKV_QAT_ROT_CACHE", "") == "1"
_SHIFT_ROT_CACHE = {}    # role -> R (graph-carrying); valid within one training step only
_FAST_EMB = os.environ.get("RWKV_QAT_FAST_EMB", "") == "1"


def shift_rot_cache_clear():
    """Train-loop hook: call at the TOP of every training step (before the forward). Clearing per step
    makes the cached graph-carrying R live exactly one forward+backward — no freed-graph reuse even on
    skipped (non-finite) steps, no staleness after optimizer.step. No-op when the cache is off/empty."""
    _SHIFT_ROT_CACHE.clear()

# RWKV_QAT_SHIFT_ANNEAL=<tau0>: SOFT-TO-HARD selection annealing for the shift PQ (soft-to-hard vector
# quantization, Agustsson et al. 2017, adapted). For the early fraction of training the hard nearest-
# centroid snap is replaced by a fully differentiable softmax(-d^2/tau) blend over each chunk's
# centroids — gradients reach x, the codebook AND the rotation through the ASSIGNMENT itself, so
# centroids and weights co-adapt without frozen-selection noise. tau decays LINEARLY from tau0 to 0 at
# frac = RWKV_QAT_SHIFT_ANNEAL_END (default 0.5) of total steps; from that point the path is EXACTLY
# the hard deploy quantizer for the entire remainder — no soft/hard gap after training ends (Andrew's
# condition). Driven per-step by set_shift_anneal_progress from train_rwkv; tau stays 0 (hard) in any
# process that never calls it (gpu_eval, parity scripts). Soft path only runs with grad enabled, so
# in-training no_grad validation passes always see the deploy-exact hard quantizer.
_SHIFT_ANNEAL_TAU0 = float(os.environ.get("RWKV_QAT_SHIFT_ANNEAL", "0") or 0)
_SHIFT_ANNEAL_END = float(os.environ.get("RWKV_QAT_SHIFT_ANNEAL_END", "0.5") or 0.5)
_SHIFT_ANNEAL_TAU = 0.0


def set_shift_anneal_progress(frac):
    """Train-loop hook: frac = step/total_steps -> sets the current soft-selection temperature."""
    global _SHIFT_ANNEAL_TAU
    _SHIFT_ANNEAL_TAU = _SHIFT_ANNEAL_TAU0 * max(0.0, 1.0 - frac / _SHIFT_ANNEAL_END)
    return _SHIFT_ANNEAL_TAU


def shift_rot_init(device, c):
    """Create the rotation Parameter (idempotent). Call BEFORE optimizer.add_param_group."""
    global _SHIFT_ROT_P
    if _SHIFT_ROT_LEARN and _SHIFT_ROT_P is None:
        _SHIFT_ROT_P = torch.nn.Parameter(torch.zeros(2, c, c, dtype=torch.float32, device=device))
        print(f"[QAT-SHIFT-ROT] learned shift pre-rotation ON: 2 x {c}x{c} Cayley (init = identity)")
    return _SHIFT_ROT_P


def _shift_rot_load(device):
    """Eval path: RWKV_QAT_SHIFT_ROT=<file> loads the EXPORTED orthogonal matrices directly (the
    trained Parameter lives outside the checkpoint; evals replay the exported engine-format file)."""
    global _SHIFT_ROT_FIXED
    if _SHIFT_ROT_FIXED is None:
        vals = []
        with open(_SHIFT_ROT_ENV) as fh:
            toks = fh.read().split()
        c = int(toks[0])
        vals = [float(x) for x in toks[1:]]
        assert len(vals) == 2 * c * c, f"shift rot file: want {2*c*c} floats, got {len(vals)}"
        _SHIFT_ROT_FIXED = torch.tensor(vals, dtype=torch.float32).view(2, c, c).to(device)
        print(f"[QAT-SHIFT-ROT] loaded FIXED rotation from {_SHIFT_ROT_ENV}: 2 x {c}x{c}")
    return _SHIFT_ROT_FIXED


def _shift_rot_matrix(role):
    """The orthogonal R for `role`, differentiable w.r.t. _SHIFT_ROT_P. None when the lever is off."""
    if _SHIFT_ROT_ENV and not _SHIFT_ROT_LEARN and os.path.isfile(_SHIFT_ROT_ENV):
        return _shift_rot_load("cpu" if _SHIFT_PQ_CB is None else _SHIFT_PQ_CB.device)[role]
    if _SHIFT_ROT_P is None:
        return None
    if _SHIFT_ROT_CACHE_ON and torch.is_grad_enabled():
        hit = _SHIFT_ROT_CACHE.get(role)
        if hit is not None:
            return hit
    p = _SHIFT_ROT_P[role]
    a = p - p.T
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    r = torch.linalg.solve(eye + a, eye - a)  # (I+A)^-1 (I-A), orthogonal for any A skew
    if _SHIFT_ROT_CACHE_ON and torch.is_grad_enabled():
        _SHIFT_ROT_CACHE[role] = r
    return r


def shift_rot_export(path):
    """Write the exact orthogonal matrices (engine text format: 2 role blocks of C rows x C floats)."""
    with torch.no_grad():
        mats = [_shift_rot_matrix(r).detach().float().cpu() for r in (0, 1)]
    c = mats[0].shape[0]
    lines = [f"{c}"]
    for m_ in mats:
        for row in m_:
            lines.append(" ".join(f"{x:.8e}" for x in row.tolist()))
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def shift_pq_init(device):
    """Load the codebook onto `device` (as a trainable Parameter when RWKV_QAT_SHIFT_PQ_LEARN=1).
    Idempotent. Call BEFORE optimizer.add_param_group so the returned Parameter is the stepped object."""
    global _SHIFT_PQ_CB, _SHIFT_PQ_META
    if _SHIFT_PQ_CB is None:
        with open(_SHIFT_PQ_PATH) as fh:
            lines = [ln for ln in fh if ln.strip()]
        m, bits, sub, c, ncent = (int(x) for x in lines[0].split()[:5])
        rows = [[float(x) for x in ln.split()] for ln in lines[1:]]
        assert len(rows) == 2 * m * ncent, f"shift codebook: want {2*m*ncent} rows, got {len(rows)}"
        cb = torch.tensor(rows, dtype=torch.float32).view(2, m, ncent, sub).to(device)
        _SHIFT_PQ_CB = torch.nn.Parameter(cb) if _SHIFT_PQ_LEARN else cb
        _SHIFT_PQ_META = (m, sub, ncent)
        print(f"[QAT-SHIFT-PQ] loaded {_SHIFT_PQ_PATH}: m={m} sub={sub} ncent={ncent} roles=2 "
              f"learnable={_SHIFT_PQ_LEARN}")
    return _SHIFT_PQ_CB


def shift_pq_export(path):
    """Write the (possibly learned) codebook back in the engine text format (RWKV_SHIFT_PQ)."""
    m, sub, ncent = _SHIFT_PQ_META
    bits = ncent.bit_length() - 1
    cb = _SHIFT_PQ_CB.detach().float().cpu().view(2 * m * ncent, sub)
    lines = [f"{m} {bits} {sub} {m * sub} {ncent}"]
    for row in cb:
        lines.append(" ".join(f"{x:.6e}" for x in row.tolist()))
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def fake_pq_shift(x_BTC: torch.Tensor, role: int) -> torch.Tensor:
    """Shift-PQ round-trip of (B,T,C), role 0 = t_xshift / 1 = c_xshift. Compute in f32 like deploy.
    Backward: straight-through to x; embedding-style gradients to the codebook when it is a Parameter
    (selection indices are frozen per step, like a hard-EM assignment)."""
    cb = shift_pq_init(x_BTC.device)
    m, sub, ncent = _SHIFT_PQ_META
    B, T, C = x_BTC.shape
    flat = x_BTC.reshape(-1, C).float()
    rot = _shift_rot_matrix(role)                                      # None unless RWKV_QAT_SHIFT_ROT=1
    tau = _SHIFT_ANNEAL_TAU
    if tau > 0.0 and torch.is_grad_enabled():
        # SOFT phase (RWKV_QAT_SHIFT_ANNEAL): everything differentiable, no STE. Note rot is NOT
        # detached on the encode side here — R gets assignment gradients too.
        work = flat if rot is None else flat @ rot.T
        norm = work.norm(dim=1, keepdim=True)
        ok = (norm.squeeze(1) > 1e-20).detach()
        unit = work / norm.clamp_min(1e-20)
        if _NORM_BITS:
            norm = norm + (_nq_quant_norm(norm) - norm).detach()       # STE on the norm rounding only
        parts = [torch.softmax(-torch.cdist(unit[:, p * sub:(p + 1) * sub], cb[role, p]).square() / tau,
                               dim=1) @ cb[role, p] for p in range(m)]
        q = torch.cat(parts, dim=1) * norm
        if rot is not None:
            q = q @ rot
        q = torch.where(ok.unsqueeze(1), q, flat)
        return q.reshape(B, T, C).to(x_BTC.dtype)
    with torch.no_grad():
        work = flat if rot is None else flat @ rot.T.detach()          # encode in the ROTATED basis
        norm = work.norm(dim=1, keepdim=True)                          # (= ||flat||: R is orthogonal)
        ok = norm.squeeze(1) > 1e-20
        unit = work / norm.clamp_min(1e-20)                            # matching by the TRUE norm
        # Row-chunked encode for BIG catalogs (m2b12: ncent=4096 -> a full cdist matrix is
        # N x 4096 ~ 1 GB fp32 at N~66k, too big a transient on 12 GB). Rows are independent, so
        # slicing N changes NOTHING (bit-identical indices); small catalogs keep the one-shot path.
        def _nearest(u_p, cb_p):
            if u_p.shape[0] * cb_p.shape[0] <= (1 << 27):
                return torch.cdist(u_p, cb_p).argmin(dim=1)
            outs = [torch.cdist(u_p[s:s + 8192], cb_p).argmin(dim=1)
                    for s in range(0, u_p.shape[0], 8192)]
            return torch.cat(outs)
        idxs = [_nearest(unit[:, p * sub:(p + 1) * sub], cb[role, p].detach())
                for p in range(m)]                                     # first strict min, frozen
        if _NORM_BITS:
            norm = _nq_quant_norm(norm)                                # reconstruct with quantized norm
    if _FAST_EMB and ncent <= 256:
        # one-hot @ cb: forward picks the same rows bit-exactly; backward = deterministic mm into the
        # codebook instead of deterministic-mode index_put_ (the hard-phase GPU hotspot, see flag note).
        # Guarded to small catalogs: a one-hot for m2b12 (ncent=4096) would itself be ~1 GB fp32.
        parts = [torch.nn.functional.one_hot(idxs[p], num_classes=ncent).to(cb.dtype) @ cb[role, p]
                 for p in range(m)]
    else:
        parts = [cb[role, p][idxs[p]] for p in range(m)]                # differentiable w.r.t. cb
    q = torch.cat(parts, dim=1) * norm
    if rot is not None:
        q = q @ rot                                                    # un-rotate; real grads into R
    q = torch.where(ok.unsqueeze(1), q, flat.detach()).reshape(B, T, C).to(x_BTC.dtype)
    # forward value = q exactly; backward: identity to x (STE) + real grads into the selected centroids
    return q + (x_BTC - x_BTC.detach())


class RWKV7(ModuleType):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                RWKV7Layer(config, layer_id)
                for layer_id in range(
                    config.layer_offset, config.layer_offset + config.n_layers
                )
            ]
        )

    @FunctionType
    def forward(self, in_BTC, time_shift_select_BT, skip_BT):
        x_BTC, v0_BTC = in_BTC, torch.empty_like(in_BTC)
        for _, block in enumerate(self.blocks):
            x_BTC, v0_BTC = block(
                in_BTC=x_BTC,
                v0_BTC=v0_BTC,
                time_shift_select_BT=time_shift_select_BT,
                skip_BT=skip_BT,
            )
        return x_BTC


class RWKV7Layer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        self.time_mixer = RWKV7TimeMixer(config, layer_id)
        self.channel_mixer = RWKV7ChannelMixer(config, layer_id)
        self.dropout = torch.nn.Dropout(p=config.dropout_layer)

    @FunctionType
    def forward(self, in_BTC, v0_BTC, time_shift_select_BT, skip_BT):
        x_BTC, v0_BTC = self.time_mixer(
            in_BTC=in_BTC,
            v0_BTC=v0_BTC,
            time_shift_select_BT=time_shift_select_BT,
            skip_BT=skip_BT,
        )
        return (
            self.dropout(
                self.channel_mixer(x_BTC, time_shift_select_BT=time_shift_select_BT)
            ),
            v0_BTC,
        )


class RWKV7ChannelMixer(ModuleType):
    # Also the same as for RWKV-5
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        # head dim K = d_model // n_heads. The CUDA WKV kernel now supports any K that DIVIDES 32
        # (K-aware warp reduction, 2026-06-30; parity-verified for K=16). K must divide 32.
        assert 32 % (config.d_model // config.n_heads) == 0
        self.d_model = config.d_model
        self.state_shift_qmax = config.state_shift_qmax  # shift-QAT: inf = off
        with torch.no_grad():
            ratio_1_to_almost_0 = 1.0 - (layer_id / config.total_layers)
            self.layer_norm = torch.nn.LayerNorm(config.d_model)
            self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

            channel_ratio = torch.ones(1, 1, config.d_model)
            for i in range(config.d_model):
                channel_ratio[0, 0, i] = i / config.d_model

            self.lerp_k = torch.nn.Parameter(
                1 - torch.pow(channel_ratio, ratio_1_to_almost_0**4)
            )

            k_dim = int(config.channel_mixer_factor * config.d_model)
            self.W_k = torch.nn.Linear(config.d_model, k_dim, bias=False)
            self.W_v = torch.nn.Linear(k_dim, config.d_model, bias=False)

            self.W_k.weight.data.uniform_(
                -0.5 / (config.d_model**0.5), 0.5 / (config.d_model**0.5)
            )
            self.W_v.weight.data.zero_()

            self.dropout = torch.nn.Dropout(p=config.dropout)

    @FunctionType
    def forward(self, in_BTC, time_shift_select_BT):
        x_BTC = self.layer_norm(in_BTC)
        x_shift_BTC = time_shift_gather(x_BTC, time_shift_select_BT)
        if self.state_shift_qmax != float("inf"):  # shift-QAT: fake-quant the persisted shift (deploy analog)
            if _SHIFT_PQ_PATH:
                x_shift_BTC = fake_pq_shift(x_shift_BTC, 1)  # role 1 = c_xshift
            else:
                x_shift_BTC = fake_quant_shift(x_shift_BTC, self.state_shift_qmax)
        k_BTK = self.W_k(torch.lerp(x_BTC, x_shift_BTC, self.lerp_k))
        o_BTC = self.W_v(torch.square(torch.nn.functional.relu(k_BTK)))
        return in_BTC + self.dropout(o_BTC)


def time_shift_gather(x_BTC: torch.Tensor, sel_BT: torch.Tensor) -> torch.Tensor:
    """torch.gather(x, 1, sel.expand(..,C)) reformulated as a flat ROW index_select.

    Numerically identical forward (same selected values). The win is the backward under
    torch.use_deterministic_algorithms: gather's backward scatter-adds B*T*C individually
    keyed elements through the sort-based deterministic path (~20% of the whole training
    step across the 14 layers), while index_select's backward is a row-wise deterministic
    index_add -- it sorts only B*T keys and accumulates C-wide rows."""
    B, T, C = x_BTC.shape
    offs = torch.arange(B, dtype=torch.long, device=sel_BT.device).unsqueeze(1) * T
    flat = (sel_BT.long() + offs).view(-1)
    return torch.index_select(x_BTC.reshape(B * T, C), 0, flat).view(B, T, C)


def ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            torch.nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                torch.nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            assert False
        return x


class LoraSimple(ModuleType):
    def __init__(self, name, d_model, d_lora, layer_id):
        super().__init__()
        with torch.no_grad():
            # The lambda term can be written out as a linear layer that includes a bias
            self.A = torch.nn.Linear(d_model, d_lora, bias=False)
            torch.nn.init.zeros_(self.A.weight)
            self.B_and_lamb = torch.nn.Linear(d_lora, d_model, bias=True)
            ortho_init(self.B_and_lamb.weight, scale=0.1)
            if name == "v":
                # Bias with ones to let the first layer's value flow directly
                torch.nn.init.ones_(self.B_and_lamb.bias)
            else:
                torch.nn.init.zeros_(self.B_and_lamb.bias)

    @FunctionType
    def forward(self, in_BTC):
        return self.B_and_lamb(self.A(in_BTC))


class LoraMLP(ModuleType):
    def __init__(self, name, config: RWKV7Config, d_lora, out_dim, layer_id):
        super().__init__()
        C = out_dim
        ratio_0_to_1 = layer_id / max(config.total_layers - 1, 1)  # guard 1-layer stream (iter35 card=1)

        with torch.no_grad():
            self.A = torch.nn.Linear(config.d_model, d_lora, bias=False)
            torch.nn.init.zeros_(self.A.weight)
            self.B_and_lamb = torch.nn.Linear(d_lora, out_dim, bias=True)
            ortho_init(self.B_and_lamb.weight, scale=0.1)
            if name == "d":
                decay_speed = torch.ones(C)
                for i in range(C):
                    decay_speed[i] = -7 + 5 * (i / (C - 1)) ** (
                        0.85 + 1.0 * ratio_0_to_1**0.5
                    )
                self.B_and_lamb.bias.copy_(decay_speed + 0.5)
            else:
                torch.nn.init.zeros_(self.B_and_lamb.bias)

    @FunctionType
    def forward(self, in_BTC):
        return self.B_and_lamb(torch.nn.functional.tanh(self.A(in_BTC)))


class RWKV7TimeMixer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.layer_id = layer_id
        C = config.d_model
        self.d_model = C
        self.H = config.n_heads
        self.K = C // config.n_heads
        self.state_qmax = config.state_qmax  # QAT: inf = off (fp32 kernel path)
        self.state_lowrank_rank = config.state_lowrank_rank      # low-rank QAT: 0 = off
        self.state_lowrank_fqmax = config.state_lowrank_fqmax    # int-N factor quant (inf = fp32)
        self.state_shift_qmax = config.state_shift_qmax          # shift-QAT: inf = off

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(config.n_layers - 1, 1)  # guard 1-layer stream (iter35 card=1)
            ratio_1_to_almost_0 = 1.0 - (layer_id / config.n_layers)
            channel_ratio = torch.ones(1, 1, C)
            for i in range(C):
                channel_ratio[0, 0, i] = i / C

            self.layer_norm = torch.nn.LayerNorm(config.d_model)
            self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

            self.rkvdag_lerp = torch.nn.Parameter(torch.empty(8, 1, 1, config.d_model))

            # Overall, the earlier the layer the more that we care about the shifted input.
            self.rkvdag_lerp[0] = 1.0 - torch.pow(
                channel_ratio, 0.2 * ratio_1_to_almost_0
            )  # r
            # The weight for k, v, can become negative and are roughly centered around 0 for the later layers.
            self.rkvdag_lerp[1] = 1.0 - (
                torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0) + 0.4 * ratio_0_to_1
            )  # k
            self.rkvdag_lerp[2] = 1.0 - (
                torch.pow(channel_ratio, 0.2 * ratio_1_to_almost_0) + 0.6 * ratio_0_to_1
            )  # v
            self.rkvdag_lerp[3] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )  # d (aka w)
            self.rkvdag_lerp[4] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )  # a
            self.rkvdag_lerp[5] = 1.0 - torch.pow(
                channel_ratio, 0.2 * ratio_1_to_almost_0
            )  # g
            self.rkvdag_lerp[6] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )
            self.rkvdag_lerp[7] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )

            self.bonus = torch.nn.Parameter(
                torch.zeros(1, 1, config.n_heads, config.d_model // config.n_heads)
            )  # r_k

            self.W_r = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_k = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_v = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_o = torch.nn.Linear(config.d_model, config.d_model, bias=False)

            self.W_r.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.W_k.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.W_v.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.W_o.weight.data.zero_()

            self.k_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
            self.v_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
            torch.nn.init.zeros_(self.k_scale_linear.weight)
            torch.nn.init.zeros_(self.k_scale_linear.bias)
            torch.nn.init.zeros_(self.v_scale_linear.weight)
            torch.nn.init.zeros_(self.v_scale_linear.bias)

            self.v_lora_simple = LoraSimple(
                name="v",
                d_model=config.d_model,
                d_lora=config.v0_mix_amt_lora,
                layer_id=layer_id,
            )
            self.a_lora_simple = LoraSimple(
                name="a",
                d_model=config.d_model,
                d_lora=config.a_lora,
                layer_id=layer_id,
            )
            self.d_lora_mlp = LoraMLP(
                name="d",
                config=config,
                d_lora=config.decay_lora,
                out_dim=config.d_model,
                layer_id=layer_id,
            )

            self.lora_A_g = torch.nn.Linear(
                config.d_model, config.gate_lora, bias=False
            )
            torch.nn.init.zeros_(self.lora_A_g.weight)
            self.lora_B_g = torch.nn.Linear(
                config.gate_lora, config.d_model, bias=False
            )
            ortho_init(self.lora_B_g.weight, 0.1)

            self.out_group_norm = torch.nn.GroupNorm(
                config.n_heads, config.d_model, eps=64e-5
            )
            self.dropout = torch.nn.Dropout(p=config.dropout)

    @FunctionType
    def forward(self, in_BTC, v0_BTC, time_shift_select_BT, skip_BT):
        B, T, C = in_BTC.shape
        H, K = self.H, self.K

        x_BTC = self.layer_norm(in_BTC)
        x_shift_BTC = time_shift_gather(x_BTC, time_shift_select_BT)
        if self.state_shift_qmax != float("inf"):  # shift-QAT: fake-quant the persisted shift (deploy analog)
            if _SHIFT_PQ_PATH:
                x_shift_BTC = fake_pq_shift(x_shift_BTC, 0)  # role 0 = t_xshift
            else:
                x_shift_BTC = fake_quant_shift(x_shift_BTC, self.state_shift_qmax)

        rkvdag_8BTC = torch.lerp(
            x_BTC.unsqueeze(0), x_shift_BTC.unsqueeze(0), self.rkvdag_lerp
        )
        r_BTC, k_BTC, v_BTC, d_BTC, a_BTC, g_BTC, k_scale_BTC, v_scale_BTC = (
            rkvdag_8BTC.unbind(dim=0)
        )
        r_BTC = self.W_r(r_BTC)
        k_BTC = self.W_k(k_BTC)
        k_scale_BTH = torch.nn.functional.sigmoid(self.k_scale_linear(k_scale_BTC))
        v_scale_BTH = torch.nn.functional.sigmoid(self.v_scale_linear(v_scale_BTC))

        if self.layer_id == 0:
            v_BTC = self.W_v(v_BTC)
            v0_BTC = v_BTC
        else:
            v_lerp_BTC = torch.nn.functional.sigmoid(self.v_lora_simple(v_BTC))
            v_BTC = torch.lerp(self.W_v(v_BTC), v0_BTC, v_lerp_BTC)

        a_BTC = torch.nn.functional.sigmoid(self.a_lora_simple(a_BTC))
        g_BTC = self.lora_B_g(torch.nn.functional.sigmoid(self.lora_A_g(g_BTC)))

        _d_BTC = -0.5 - torch.nn.functional.softplus(-self.d_lora_mlp(d_BTC))
        w_BTC = torch.exp(-torch.exp(_d_BTC.float()))

        k_BTHK = k_scale_BTH.unsqueeze(-1) * torch.nn.functional.normalize(
            k_BTC.view(B, T, H, K), dim=-1, p=2.0
        )
        r_BTHK = r_BTC.view(B, T, H, K)
        v_BTHK = v_scale_BTH.unsqueeze(-1) * torch.nn.functional.normalize(
            v_BTC.view(B, T, H, K), dim=-1, p=2.0
        )
        w_BTHK = w_BTC.view(B, T, H, K)
        a_BTHK = a_BTC.view(B, T, H, K)
        k_deformed_BTHK = k_BTHK
        k_BTHK = k_BTHK * a_BTHK

        if self.state_qmax != float("inf") or self.state_lowrank_rank > 0:
            # QAT: simulate per-step deploy state storage for this stream (card/note) -- low-rank
            # truncation if configured, else int-N quant. Runs the quant-aware per-step reference
            # (requires RWKV_NO_JIT). deck/preset/global stay off and keep the fast kernel.
            out_BTHK = quant_aware_rwkv7(
                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT,
                self.state_qmax, self.state_lowrank_rank, self.state_lowrank_fqmax,
            )
        elif r_BTHK.is_cuda:
            out_BTHK = RWKV7_WKV.apply(
                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT
            )
        else:
            out_BTHK = reference_rwkv7(
                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT
            )

        out_BTC = self.out_group_norm(out_BTHK.view(B * T, C)).view(B, T, C)
        bonus_BTC = (
            (r_BTHK * self.bonus * k_BTHK).sum(dim=-1, keepdim=True) * v_BTHK
        ).view(B, T, C)
        out_BTC = self.W_o(g_BTC * (out_BTC + bonus_BTC))
        return in_BTC + self.dropout(out_BTC), v0_BTC
