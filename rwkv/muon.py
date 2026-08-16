"""Hybrid Muon+AdamW optimizer (research iter 29, 2026-07-21).

Muon (Jordan et al., the modded-nanogpt speedrun's backbone optimizer): momentum-SGD
whose update is orthogonalized per 2D weight matrix via a quintic Newton-Schulz
iteration, with an aspect-ratio step scale. Non-matrix params (biases, norms, LoRAs,
scalars) stay on AdamW with numerics identical to torch.optim.AdamW (delegated to the
functional kernel).

Design constraints honored:
- ONE torch.optim.Optimizer subclass: param_groups carry the usual lr/weight_decay keys
  plus `use_muon`, so train_rwkv's LR schedulers (LinearLR/ConstantLR/Cosine multiply
  each group's own base lr), the resume clobber-restore logic, and state_dict save/load
  all work unchanged.
- Weight decay on Muon groups is applied at the SAME absolute per-step rate as the
  champion's AdamW would have applied it (decoupled p *= 1 - lr_adamw_equiv * wd):
  each Muon group carries `wd_lr_scale` = adamw_peak / muon_peak so the schedule-scaled
  Muon lr maps back to the AdamW-equivalent rate. Regularization stays comparable to
  the champion; the optimizer geometry is the only change.
- Default OFF at the call site (RWKV_MUON unset -> plain torch.optim.AdamW, byte-identical).
"""

import os

import torch
from torch.optim.adamw import adamw as _functional_adamw

# RWKV_MUON_BATCHED=1 -> orthogonalize all same-shaped matrices in one batched Newton-Schulz
# and drive momentum with torch._foreach_*. Default OFF, i.e. byte-identical to iter 29-33.
_MUON_BATCHED = os.environ.get("RWKV_MUON_BATCHED", "0") == "1"

# ---- iter 51: RWKV_MUON_POLAR=1 -> a PER-STEP Newton-Schulz coefficient schedule (the Polar
# Express idea, arXiv 2505.16932) instead of one fixed triple for all five steps.
#
# WHY. Newton-Schulz acts on each singular value independently as
#     sigma <- p(sigma) = a*sigma + b*sigma^3 + c*sigma^5,
# so orthogonalization is exactly "map the spectrum onto {1}". The production triple
# (3.4445, -4.7750, 2.0315) is the modded-nanogpt constant, and it is deliberately sloppy:
# a+b+c = 0.7010, so it maps sigma=1 -> 0.70 and lands the whole range in ~[0.70, 1.20]. Muon
# works anyway -- an exact polar factor was never required -- but on OUR momentum buffers that
# leaves a real, measured error.
#
# MEASURED ON THE 69 MOMENTUM MATRICES OF `i50_d_optim_10935.pth`
# (scratchpad/iter51_muon/{ns_error,where_is_the_error,compare_variants}.py):
#   * RMS|sigma-1| over ALL singular values = 0.274, reproducing PROPOSALS.md #5's 0.19-0.31;
#   * it is NOT precision -- bf16 0.289 vs fp32 0.301;
#   * ~half is the near-null tail (median condition number 1.2e4), which no odd polynomial can
#     lift in 5 steps and which we would NOT want lifted (it is noise amplification);
#   * but 0.161 remains on the directions carrying the top 90% of momentum ENERGY, and that is
#     the number this lever attacks. Schedule below: 0.161 -> 0.0251, a 84.4% cut.
#   * CONTROL: merely rescaling the input so sigma_max ~ 1 (the fixed triple is tuned for that)
#     buys only 5.8%, so the win is the per-step schedule, not the input range.
#
# The schedule is FITTED, not quoted: greedy minimax, one step at a time, on [0.0297, 1.0] --
# the lower end being where our 99%-energy cutoff sits, the upper end being GUARANTEED by the
# Frobenius normalisation. (Fitting to the observed median sigma_max of 0.705 instead let step 0
# reach c=66.9 and overflow above it.) Stability over the whole domain: peak intermediate 1.78,
# and sigma in [0.03, 1] maps to 1.0000 to four decimals while sigma=1e-4 stays at 0.019.
#
# ⚠ SIDE EFFECT, small and documented rather than corrected: a more accurate polar factor changes
# the update SIZE as well as its shape -- ||O||_F rises 2.6% (median over the 69 buffers). That is
# far below any learning-rate sensitivity we have resolved (the tuner needed 1.41-2.8x moves), so
# no compensating constant is folded in; a constant fitted to one checkpoint would be arbitrary.
#
# Training-only: no parameter, no state, no deploy path is touched. Default OFF = byte-identical.
_MUON_POLAR = os.environ.get("RWKV_MUON_POLAR", "0") == "1"
_FIXED_NS = (3.4445, -4.7750, 2.0315)
_POLAR_SCHEDULE = [
    (7.372480, -20.782051, 15.190877),
    (2.925176, -2.188391, 0.476201),
    (2.049675, -1.432759, 0.393654),
    (1.877187, -1.253114, 0.375924),
    (2.322228, -2.144455, 0.822228),
]


if _MUON_POLAR:
    print("[muon] POLAR-EXPRESS Newton-Schulz schedule ON: per-step coefficients fitted by greedy "
          "minimax on [0.0297, 1.0]; top-90%-energy RMS|sigma-1| 0.161 -> 0.025 on real buffers")


def _ns_coeffs(steps: int):
    """The (a,b,c) to use per iteration. Falls back to the fixed triple whenever the schedule
    does not cover the requested step count, so RWKV_MUON_NS_STEPS stays usable."""
    if _MUON_POLAR and steps == len(_POLAR_SCHEDULE):
        return _POLAR_SCHEDULE
    return [_FIXED_NS] * steps


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps: int = 5):
    """Quintic Newton-Schulz orthogonalization (modded-nanogpt reference constants)."""
    assert G.ndim == 2
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for a, b, c in _ns_coeffs(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@torch.no_grad()
def zeropower_via_newtonschulz5_batched(G, steps: int = 5):
    """Same iteration, over a STACK of equally-shaped matrices: G is (B, M, N).

    Item-for-item mathematically identical to calling the 2D version B times -- `@` on 3D
    tensors is bmm, which is independent per batch element, and the normalizer is taken
    per item. What changes is that B matrices are orthogonalized in ONE dispatch instead of B.

    Motivation (2026-07-27 profile, optimization/TRAINING_SPEED.md): the per-parameter loop
    issued 2,658 `aten::mm` per step costing 92 ms of CPU dispatch to do 21.6 ms of GPU work,
    on a step that is dispatch-bound overall.

    ⚠ NOT bit-exact vs the 2D path: cuBLAS may choose a different algorithm/reduction order
    for bmm than for mm. Equivalent in exact arithmetic, not in float.
    """
    assert G.ndim == 3
    X = G.to(torch.bfloat16)
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in _ns_coeffs(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups, betas=(0.9, 0.999), eps=1e-8,
                 muon_momentum=0.95, ns_steps=5, cautious_wd=False):
        defaults = dict(lr=1e-3, weight_decay=0.0, betas=betas, eps=eps,
                        use_muon=False, wd_lr_scale=1.0,
                        muon_momentum=muon_momentum, ns_steps=ns_steps,
                        cautious_wd=cautious_wd)
        super().__init__(param_groups, defaults)
        if _MUON_BATCHED:
            # Printed so a benchmark arm can be VERIFIED to have taken the flag, rather than
            # assumed -- a silently-ignored env var makes two arms compare noise to noise.
            print("[muon] BATCHED Newton-Schulz ON (one bmm per shape group, "
                  "torch._foreach_ momentum)")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                wd = group["weight_decay"]
                wd_eff = lr * group["wd_lr_scale"] * wd  # AdamW-equivalent decay rate
                momentum = group["muon_momentum"]
                ns_steps = group["ns_steps"]
                live = [p for p in group["params"] if p.grad is not None]
                # ---- batched path (RWKV_MUON_BATCHED=1): one Newton-Schulz per SHAPE ----
                # The loop below is unchanged; it just consumes a precomputed O per param.
                # See zeropower_via_newtonschulz5_batched for why this is not bit-exact.
                precomputed = {}
                if _MUON_BATCHED and len(live) > 1:
                    by_shape = {}
                    for p in live:
                        g2d = p.grad.reshape(p.grad.size(0), -1)
                        st = self.state[p]
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(g2d)
                        by_shape.setdefault(tuple(g2d.shape), []).append((p, g2d, st))
                    for shp, items in by_shape.items():
                        bufs = [st["momentum_buffer"] for (_, _, st) in items]
                        g2ds = [g2d for (_, g2d, _) in items]
                        # momentum + nesterov for the whole shape group in 3 dispatches
                        torch._foreach_mul_(bufs, momentum)
                        torch._foreach_add_(bufs, g2ds)
                        upds = torch._foreach_add(g2ds, bufs, alpha=momentum)
                        if len(items) == 1:
                            Os = [zeropower_via_newtonschulz5(upds[0], steps=ns_steps)]
                        else:
                            stacked = torch.stack(upds, dim=0)
                            Ob = zeropower_via_newtonschulz5_batched(stacked, steps=ns_steps)
                            Os = list(Ob.unbind(0))
                        for (p, _, _), O in zip(items, Os):
                            precomputed[p] = O

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    assert g.ndim >= 2, "use_muon group must hold matrices"
                    g2d = g.reshape(g.size(0), -1)
                    if p in precomputed:
                        O = precomputed[p]  # momentum/nesterov already applied above
                    else:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g2d)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g2d)
                        upd = g2d.add(buf, alpha=momentum)  # nesterov
                        O = zeropower_via_newtonschulz5(upd, steps=ns_steps)
                    O_full = O.reshape(p.shape)
                    if wd_eff != 0.0:
                        if group["cautious_wd"]:
                            # cautious wd (research iter 30; modded-nanogpt #43/50):
                            # decay ONLY coords whose applied step (-lr*scale*O) agrees
                            # with the weight's sign, i.e. w*O < 0 -- never fight a
                            # component the update is already shrinking toward zero.
                            p.mul_(1.0 - wd_eff * (p * O_full < 0).to(p.dtype))
                        else:
                            p.mul_(1.0 - wd_eff)
                    scale = max(1.0, p.size(0) / p.reshape(p.size(0), -1).size(1)) ** 0.5
                    p.add_(O_full, alpha=-lr * scale)
            else:
                params, grads = [], []
                exp_avgs, exp_avg_sqs, state_steps = [], [], []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    params.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0 or "exp_avg" not in state:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    state_steps.append(state["step"])
                if not params:
                    continue
                beta1, beta2 = group["betas"]
                _functional_adamw(
                    params, grads, exp_avgs, exp_avg_sqs, [], state_steps,
                    amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
                    weight_decay=group["weight_decay"], eps=group["eps"],
                    maximize=False, foreach=None, capturable=False,
                    differentiable=False, fused=None, grad_scale=None, found_inf=None,
                    has_complex=False,
                )
        return loss
