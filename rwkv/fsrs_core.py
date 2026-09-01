"""FSRS-7's memory-state recurrence, as a per-review-parameterized module (RWKV_FSRS_CARD).

WHAT THIS IS FOR. Andrew's 2026-08-24 proposal: replace the card stream's WKV recurrence with
FSRS-7's structured (S_long, S_short, D) update, and have the trunk EMIT the 34 parameters per
review instead of them being global constants fitted per user. Design + measurements:
optimization/HYBRID_100K.md section 9.

THE ONE DIFFERENCE FROM FSRS-7, AND IT IS THE POINT. In FSRS the 34 weights `w` are a single
per-user vector fitted by an optimizer. Here `w` has shape (..., 34) and varies per review,
produced from the RWKV trunk. Every formula below is therefore written to broadcast over
leading dimensions; nothing may assume `w` is 1-D.

PROVENANCE: ported from srs-benchmark/models/fsrs_v7.py (FSRS7, the finished 34-param
dual-stability model), which is itself a port of fsrs-rs/src/model.rs. That file is READ-ONLY
to this repo. `scratchpad/hybrid100k/smoke_fsrs_port.py` checks this port against it directly
on real review sequences; a port that is not checked against its source is a guess.

WHY THE PARAMETERS ARE BOUNDED, NOT RAW. FSRS clips all 34 weights to per-index ranges and
additionally enforces monotonicity on the initial-stability block. Those ranges are not
cosmetic -- `S^(-w8)`, `(11-D)` and `exp((1-R)*w10)` all blow up outside them, and an emitter
that can output anything will find that out during training. So the trunk emits a REAL number
per parameter and `bounded_w` maps it into FSRS's own range with a sigmoid. The bound is
structural, not a clamp applied after the fact: there is no input that escapes it. (iter 51's
lesson -- a numerical-stability argument must bound the worst case, not the typical one.)
"""
import torch

# --- FSRS-7's own per-index clip ranges (srs-benchmark FSRS7ParameterClipper._CLIP_LO/_HI) ---
# Kept in index order. These ARE the parameterization ranges here, not a post-hoc clamp.
# fmt: off
CLIP_LO = [0.0001, 0.0001, 0.0001, 0.0001, 1.0, 0.001, 0.1, 0.0, 0.0, 0.3, 0.01, 0.1, 0.0,
           0.0, 1.0, 0.0, 0.0, 0.5, 0.001, 0.001, 0.0, 0.0, 1.0, 0.01, 0.01, 0.2, 0.5, 0.01,
           0.1, 0.0, 0.1, 0.0, 0.0, 0.0]
CLIP_HI = [50.0, 100.0, 100.0, 100.0, 10.0, 4.0, 4.0, 4.0, 1.2, 3.0, 1.5, 1.0, 3.5, 1.0, 7.0,
           4.0, 2.0, 6.0, 1.5, 1.0, 5.0, 1.0, 7.0, 0.25, 0.95, 0.85, 0.99, 1.0, 1.0, 0.9, 1.1,
           1.0, 0.6, 0.6]
# srs-benchmark FSRS7.init_w -- the default parameter vector, used to initialize the emitter's
# bias so an untrained emitter reproduces stock FSRS-7 rather than a random model.
# ⚠ VERIFIED AGAINST THE SOURCE, not transcribed: the first version of this list was correct
# for indices 0-3 (the only ones visible in the file excerpt I had read) and INVENTED for the
# other 30. smoke_fsrs_stream.py caught it. The clip arrays above WERE read in full and check
# out exactly; regenerate all three with FSRS7.init_w / FSRS7ParameterClipper._CLIP_LO/_HI.
# ⚠ Three entries sit EXACTLY on a clip bound (8 and 29 at their lo of 0.0, 14 at its lo of
# 1.0). A sigmoid cannot reach its bounds, so bounded_w cannot reproduce those three exactly;
# the residual is (hi-lo)*1e-4, at most 6e-4. That is a property of the parameterization, not
# a bug, and it is why the smoke asserts 1e-3 rather than 1e-6.
INIT_W = [0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0, 1.3826, 0.7024,
          0.5999, 0.8146, 0.6398, 1, 1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162,
          1.0869, 0.1567, 0.0801, 0.2421, 0.9464, 0.1433, 0.7145, 0, 0.5667, 0.3734, 0.5333,
          0.3048]
# fmt: on

# ⚠ THESE ARE PASSED AS DEFAULT ARGUMENTS BELOW, AND THAT IS DELIBERATE. A scripted function
# cannot close over a module-level float -- TorchScript raises "python value of type 'float'
# cannot be used as a value". Measured (scratchpad/tsglobal.py): a bare global FAILS, the
# same value as a DEFAULT ARG scripts fine. `Final[...]` does NOT fix it; it applies to
# module attributes, not module-level globals. So every clamp bound below arrives as a
# defaulted parameter: written once here, captured at def time, callers unaffected.
S_MAX = 36500.0
# srs-benchmark config.py:341-345: s_min is 0.0001 when use_secs_intervals is set, which is our
# configuration (the benchmark runs FSRS-7 with --short --secs, and our pipeline is seconds-based).
S_MIN = 0.0001
D_MIN = 1.0
D_MAX = 10.0
N_PARAMS = 34

# Parameter-block offsets inside `w` (srs-benchmark next_stability's `start`).
LONG = 7    # long-term stability block
SHORT = 15  # short-term stability block


def bounded_w(z: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Map raw emitter outputs (..., 34) into FSRS's per-index ranges, structurally.

    `lo`/`hi` are passed in (registered as buffers by the owning module) rather than rebuilt
    here, so the device/dtype follow the model and TorchScript sees a plain tensor op.

    Monotonicity of the initial-stability block (w1>=w0, w2>=w1, w3>=w2, and w26>=w25) is
    enforced by CUMULATIVE MAXIMUM rather than by FSRS's in-place `torch.maximum` chain: the
    clipper mutates `w.data` between optimizer steps, which has no meaning for a value produced
    fresh by a network on every row. cummax is the differentiable equivalent and gives the same
    ordering guarantee.
    """
    w = lo + (hi - lo) * torch.sigmoid(z)
    init_s = torch.cummax(w[..., 0:4], dim=-1)[0]
    base = torch.cummax(w[..., 25:27], dim=-1)[0]
    return torch.cat([init_s, w[..., 4:25], base, w[..., 27:]], dim=-1)


def short_component_recall(t, s_short, w):
    """r1 -- the short-term recall component (srs-benchmark short_component_recall:174)."""
    t = t.clamp(min=0.0)
    decay1_mag = (w[..., 23] * s_short.pow(w[..., 33] - 0.3)).clamp(0.01, 0.95)
    decay1 = -decay1_mag
    factor1 = (w[..., 25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    return (t / s_short * factor1 + 1.0).pow(decay1)


def forgetting_curve(t, s, s_short, d, w):
    """Dual-stability forgetting curve (srs-benchmark forgetting_curve:187).

    Curve indices: 23 decay1, 24 decay2, 25 base1, 26 base2, 27 base_weight1, 28 base_weight2,
    29 s_weight_power1, 30 s_weight_power2, 31 d_weight, 32 d_decay, 33 s_decay1.
    """
    t = t.clamp(min=0.0)
    r1 = short_component_recall(t, s_short, w)

    decay2 = -w[..., 24].clamp(0.01, 0.95)
    factor2 = w[..., 26].pow(decay2.pow(-1.0)) - 1.0
    d_timescale = ((d - 5.0) * (w[..., 32] - 0.3)).exp()
    r2 = (t / s * factor2 * d_timescale + 1.0).pow(decay2)

    weight1 = w[..., 27] * s_short.pow(-w[..., 29])
    weight2 = w[..., 28] * s.pow(w[..., 30]) * ((d - 5.0) * (w[..., 31] - 0.5)).exp()

    retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
    return retention * (1.0 - 2e-5) + 1e-5


def next_stability(last_s, last_d, r, rating, w, start: int, s_max: float = S_MAX):
    """Stability after a review (srs-benchmark next_stability:216).

    `start` selects the parameter block: 7 for the long-term S, 15 for the short-term S.
    Post-lapse stability is D-INDEPENDENT in the finished model.
    """
    ones = torch.ones_like(last_s)
    hard_penalty = torch.where(rating == 2, w[..., start + 6], ones)
    easy_bonus = torch.where(rating == 4, w[..., start + 7], ones)

    new_s_fail = (
        w[..., start + 3]
        * ((last_s + 1.0).pow(w[..., start + 4]) - 1.0)
        * ((1.0 - r) * w[..., start + 5]).exp()
    )
    pls = torch.minimum(last_s, new_s_fail)

    sinc = (w[..., start] - 1.5).exp() * (11.0 - last_d) * last_s.pow(
        -w[..., start + 1]
    ) * (((1.0 - r) * w[..., start + 2]).exp() - 1.0) * hard_penalty * easy_bonus + 1.0
    new_s_success = torch.maximum(pls, last_s * sinc)
    return torch.where(rating > 1, new_s_success, pls).clamp(max=s_max)


def linear_damping(delta_d, last_d):
    """FSRS-6's damping, inherited unchanged: the difficulty step shrinks as D approaches 10."""
    return delta_d * (10.0 - last_d) / 9.0


def init_d(rating, w):
    """Initial difficulty (srs-benchmark FSRS5.init_d:135): w[4] - exp(w[5]*(rating-1)) + 1.

    ⚠ DELIBERATELY UNCLAMPED, and this cost a failed smoke. FSRS clamps this value at the two
    places that need it (the first-review init path) but NOT inside `next_difficulty`, which
    mean-reverts toward the raw `init_d(4)`. With the default weights `init_d(4)` is about
    -35.2, far below D_MIN, so clamping it here turns a 1% pull toward -35.2 into a 1% pull
    toward 1.0 -- a ~7% error in D that compounds over a card's history. Clamp at the call
    site, exactly where the source does.
    """
    return w[..., 4] - ((rating - 1.0) * w[..., 5]).exp() + 1.0


def next_difficulty(last_d, rating, retention, w, d_min: float = D_MIN, d_max: float = D_MAX):
    """Difficulty update (srs-benchmark next_difficulty:247), with surprise-weighted lapse and
    the fixed 1%/99% mean reversion toward init_d(4)."""
    delta_d = -w[..., 6] * (rating - 3.0)
    delta_d = torch.where(rating == 1, delta_d * (retention + 0.1), delta_d)
    new_d = last_d + linear_damping(delta_d, last_d)
    new_d = 0.01 * init_d(torch.full_like(rating, 4.0), w) + 0.99 * new_d
    return new_d.clamp(d_min, d_max)


def init_state(rating, w, s_min: float = S_MIN, s_max: float = S_MAX,
               d_min: float = D_MIN, d_max: float = D_MAX):
    """Memory state after a card's FIRST review (srs-benchmark FSRS7.step's init path):
    S_long = w[rating-1], **S_short = 0.8 * S_long**, D = init_d(rating).

    ⚠ The 0.8 factor is not decorative and was wrong in this file's first draft: reading the
    source is what caught it. Same for the post-lapse short-term reset in `step`.

    ⚠ "First review" is defined as HAVING NO PRIOR STATE, not by any feature test. The
    `scaled_elapsed_days` column CANNOT distinguish them: the scaler maps the -1 first-review
    sentinel to a pre-standardized 0, and a same-day review with elapsed_days ~ 0 maps to
    log(1+1e-5+0) ~ 0 as well -- so both land on the same value, and bfloat16 storage smears it
    further (srs_model.py:972-977 records the same hazard for RWKV_RGATE). Both paths already
    know the answer structurally: the recurrent path has state=None, the parallel path is at
    t=0 of a per-card sequence, which is where the WKV state also starts fresh.
    """
    idx = (rating.long().clamp(1, 4) - 1)
    s0 = torch.gather(w[..., 0:4], -1, idx.unsqueeze(-1)).squeeze(-1)
    s_long = s0.clamp(s_min, s_max)
    s_short = (0.8 * s0).clamp(s_min, s_max)
    return s_long, s_short, init_d(rating, w).clamp(d_min, d_max)


def step(s, s_short, d, t, rating, w, s_min: float = S_MIN, s_max: float = S_MAX,
         d_min: float = D_MIN, d_max: float = D_MAX,
         long_base: int = LONG, short_base: int = SHORT):
    """One review. Returns (r, next_s, next_s_short, next_d).

    ORDER IS LOAD-BEARING AND MATCHES THE `ahead` SEMANTICS: `r` is computed from the state
    BEFORE this review, at elapsed time `t`. That is exactly "predict cold from history at the
    scheduled gap", which is what the ahead mode scores. The update then consumes that same `r`,
    as FSRS does.

    ⚠ The short-term stability update reads r1 (the short component alone), NOT the mixed
    retention -- srs-benchmark's comment at short_component_recall:176-177. Getting this wrong
    is invisible: both are probabilities in [0,1] and the model still trains.
    """
    s = s.clamp(s_min, s_max)
    s_short = s_short.clamp(s_min, s_max)
    d = d.clamp(d_min, d_max)

    r = forgetting_curve(t, s, s_short, d, w)
    r1 = short_component_recall(t, s_short, w)
    ns = next_stability(s, d, r, rating, w, long_base)
    nss = next_stability(s_short, d, r1, rating, w, short_base)
    # Post-lapse short-term reset: on a lapse, cap s_short at 0.8 * the post-lapse long-term S
    # (srs-benchmark FSRS7.step:300-301). Omitted in this file's first draft.
    nss = torch.where(rating == 1, torch.minimum(nss, 0.8 * ns), nss)
    nd = next_difficulty(d, rating, r, w)
    return (r, ns.clamp(s_min, s_max), nss.clamp(s_min, s_max), nd.clamp(d_min, d_max))
