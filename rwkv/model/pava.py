"""Learnable power-mean isotonic rectifier (iter 23, MONOTONICITY_PLAN.md stage 2).

Enforces the button-curve order P_Again <= P_Hard <= P_Good <= P_Easy on a 4-vector of
counterfactual recall probabilities by PAVA pooling-to-tie, where the pooled value of two
adjacent blocks is a LEARNABLE weighted generalized power mean at the JUNCTION's power:

    M_p(a, b; wa, wb) = ((wa*a^p + wb*b^p) / (wa+wb))^(1/p),   p_j = 2*tanh(theta_j)

3 junction powers (Again-Hard, Hard-Good, Good-Easy), init theta = atanh(0.5) -> p = 1 =
classic (weighted arithmetic) PAVA. p -> -2 biases the pooled value toward the lower
curve, p -> +2 toward the higher: the model learns, per button pair, which side to trust
when they conflict. Pooling-to-tie is what guarantees the order (a power mean lies
between min and max, so adjusting one side alone enforces nothing).

Numerics: values are curve probabilities in (1e-5, 1-1e-5) (strictly positive), so the
stable form is exp((1/p)*(logsumexp(p*log v + log w) - log(sum w))), with the geometric
mean exp(sum(w*log v)/sum w) at |p| < 1e-3. Both branches are computed with a where-safe
p (the torch.where NaN-grad trap), then selected.

This module is EAGER-only (called from a @torch.jit.ignore method in srs_model.py); the
same operator is the deploy-time button projection (Rust port when a rectifier model
ships). pava_rectify_scalar is the trivially-correct reference for property tests.
"""

import math

import torch

P_EPS = 1e-3  # |p| below this -> geometric-mean branch


def theta_init() -> torch.Tensor:
    """3 junction thetas at atanh(0.5) -> p = 2*tanh(theta) = 1 (classic PAVA)."""
    return torch.full((3,), math.atanh(0.5))


def power_mean(a, b, wa, wb, p):
    """Weighted generalized power mean of two blocks, elementwise over broadcast tensors.

    All of a, b in (0, 1]; wa, wb > 0; p any real tensor (same shape or broadcastable).
    """
    la, lb = torch.log(a), torch.log(b)
    lwa, lwb = torch.log(wa), torch.log(wb)
    lw_tot = torch.log(wa + wb)
    # geometric branch (p ~ 0)
    geo = torch.exp((wa * la + wb * lb) / (wa + wb))
    # power branch with where-safe p (dead branch must stay finite for grads)
    p_safe = torch.where(p.abs() < P_EPS, torch.ones_like(p), p)
    m = torch.maximum(p_safe * la + lwa, p_safe * lb + lwb)
    lse = m + torch.log(
        torch.exp(p_safe * la + lwa - m) + torch.exp(p_safe * lb + lwb - m)
    )
    pow_ = torch.exp((lse - lw_tot) / p_safe)
    return torch.where(p.abs() < P_EPS, geo, pow_)


def pava_rectify(v: torch.Tensor, w: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
    """Vectorized exact sequential PAVA over (M, 4) with power-mean pooling.

    v: (M, 4) values in (0, 1), slot order Again, Hard, Good, Easy.
    w: (M, 4) positive pooling weights (iter 23: ones -> block sizes; iter 24: p-head
       button probabilities).
    powers: (3,) junction powers p_j (already 2*tanh(theta)).
    Returns (M, 4) non-decreasing along dim 1; gradient flows through v, w, powers.

    Unrolled n=4 left-to-right scan with back-merges, mask-simulated per row: per-slot
    tensors hold each slot's current block value/weight/leftmost index. Junction between
    a block and its left neighbor is (left_ptr - 1), whose power is gathered per row.
    """
    assert v.dim() == 2 and v.size(1) == 4, f"expected (M,4), got {tuple(v.shape)}"
    M = v.size(0)
    if M == 0:
        return v
    dev = v.device
    slots = torch.arange(4, device=dev)
    bv = v.clone()  # (M,4) block value per slot
    bw = w.clone()  # (M,4) block total weight per slot
    lp = slots.expand(M, 4).clone()  # (M,4) leftmost slot of each slot's block

    for k in range(1, 4):
        # after adding slot k, back-merge up to k times
        for _ in range(k):
            lk = lp[:, k]  # (M,) leftmost slot of the block containing k
            j = lk - 1  # junction to the left block; valid iff lk > 0
            j_c = j.clamp(min=0)
            left_v = bv.gather(1, j_c.unsqueeze(1)).squeeze(1)
            left_w = bw.gather(1, j_c.unsqueeze(1)).squeeze(1)
            cur_v = bv[:, k]
            cur_w = bw[:, k]
            merge = (lk > 0) & (left_v > cur_v)  # strict violation; ties are fine
            if not bool(merge.any()):
                break
            p_j = powers[j_c]  # (M,) junction power
            m_val = power_mean(left_v, cur_v, left_w, cur_w, p_j)
            new_l = lp.gather(1, j_c.unsqueeze(1)).squeeze(1)  # left block's leftmost
            # update slots new_l..k on merging rows
            rng = (slots.unsqueeze(0) >= new_l.unsqueeze(1)) & (slots.unsqueeze(0) <= k)
            upd = merge.unsqueeze(1) & rng
            bv = torch.where(upd, m_val.unsqueeze(1), bv)
            bw = torch.where(upd, (left_w + cur_w).unsqueeze(1), bw)
            lp = torch.where(upd, new_l.unsqueeze(1), lp)
    return bv


def pava_rectify_scalar(v, w, powers):
    """Reference implementation: literal stack-based PAVA on one row (python floats).

    v, w: length-4 lists; powers: length-3 list. Returns length-4 list.
    """
    assert len(v) == 4 and len(w) == 4 and len(powers) == 3

    def pmean(a, b, wa, wb, p):
        if abs(p) < P_EPS:
            return math.exp((wa * math.log(a) + wb * math.log(b)) / (wa + wb))
        return ((wa * a**p + wb * b**p) / (wa + wb)) ** (1.0 / p)

    stack = []  # (value, weight, leftmost)
    for k in range(4):
        stack.append((v[k], w[k], k))
        while len(stack) >= 2 and stack[-2][0] > stack[-1][0]:
            bv_, bw_, bl = stack.pop()
            av_, aw_, al = stack.pop()
            p = powers[bl - 1]  # junction between slot bl-1 and bl
            stack.append((pmean(av_, bv_, aw_, bw_, p), aw_ + bw_, al))
    # blocks are contiguous and in slot order on the stack
    out = []
    for i, (val, _, left) in enumerate(stack):
        nxt = stack[i + 1][2] if i + 1 < len(stack) else 4
        out.extend([val] * (nxt - left))
    return out
