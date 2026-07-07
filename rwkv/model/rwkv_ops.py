import os
from typing import Any

import torch
from torch import Tensor


class RWKV7_WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs: Tensor):
        r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT = inputs
        assert all(
            i.is_contiguous()
            for i in [r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT]
        )
        # assert all(not torch.isnan(i).any() for i in [r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT])
        assert w_BTHK.dtype == torch.float32
        assert skip_BT.dtype == torch.bool
        dtype = r_BTHK.dtype
        assert all(
            i.dtype == dtype for i in [r_BTHK, k_BTHK, v_BTHK, a_BTHK, k_deformed_BTHK]
        )
        if r_BTHK.is_cuda:
            if r_BTHK.dtype == torch.bfloat16:
                out, state_checkpoints = (
                    torch.ops.rwkv.rwkv7_wkv_forward_bfloat16.default(
                        r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT
                    )
                )
            elif r_BTHK.dtype == torch.float:
                out, state_checkpoints = torch.ops.rwkv.rwkv7_wkv_forward_float.default(
                    r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT
                )
            elif r_BTHK.dtype == torch.half:
                out, state_checkpoints = torch.ops.rwkv.rwkv7_wkv_forward_half.default(
                    r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT
                )
            else:
                raise ValueError(f"Unsupported dtype: {r_BTHK.dtype}")

            ctx.save_for_backward(
                r_BTHK,
                k_BTHK,
                v_BTHK,
                w_BTHK,
                a_BTHK,
                k_deformed_BTHK,
                skip_BT,
                state_checkpoints,
            )
            return out
        else:
            raise ValueError("Not supported. TODO")
            # return reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        grad_BTHK = grad_outputs[0]
        (
            r_BTHK,
            k_BTHK,
            v_BTHK,
            w_BTHK,
            a_BTHK,
            k_deformed_BTHK,
            skip_BT,
            state_checkpoints,
        ) = ctx.saved_tensors
        if r_BTHK.dtype == torch.bfloat16:
            r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = (
                torch.ops.rwkv.rwkv7_wkv_backward_bfloat16.default(
                    r_BTHK,
                    k_BTHK,
                    v_BTHK,
                    w_BTHK,
                    a_BTHK,
                    k_deformed_BTHK,
                    skip_BT,
                    state_checkpoints,
                    grad_BTHK,
                )
            )
        elif r_BTHK.dtype == torch.float:
            r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = (
                torch.ops.rwkv.rwkv7_wkv_backward_float.default(
                    r_BTHK,
                    k_BTHK,
                    v_BTHK,
                    w_BTHK,
                    a_BTHK,
                    k_deformed_BTHK,
                    skip_BT,
                    state_checkpoints,
                    grad_BTHK,
                )
            )
        elif r_BTHK.dtype == torch.half:
            r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = (
                torch.ops.rwkv.rwkv7_wkv_backward_half.default(
                    r_BTHK,
                    k_BTHK,
                    v_BTHK,
                    w_BTHK,
                    a_BTHK,
                    k_deformed_BTHK,
                    skip_BT,
                    state_checkpoints,
                    grad_BTHK,
                )
            )
        else:
            raise ValueError(f"Unsupported dtype: {r_BTHK.dtype}")
        return r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad, None


class RWKV7_WKV_Stateful(torch.autograd.Function):
    """Stateful (truncated-BPTT) WKV. forward takes an initial state `state0_BHKK` (carried, detached,
    from the previous chunk) and returns `(out_BTHK, final_state_BHKK)`. backward IGNORES the gradient
    of final_state -- the carried state is treated as a constant across the chunk boundary (truncated
    BPTT) -- and returns no gradient for state0 or skip. With state0 = zeros this is mathematically
    identical to RWKV7_WKV; the only difference is the carried state I/O. CUDA-only (forces the
    sequential kernel; the time-parallel path can't take an initial state)."""

    @staticmethod
    def forward(ctx, *inputs: Tensor):
        r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state0_BHKK = inputs
        assert all(
            i.is_contiguous()
            for i in [r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state0_BHKK]
        )
        assert w_BTHK.dtype == torch.float32
        assert skip_BT.dtype == torch.bool
        assert state0_BHKK.dtype == torch.float32
        dtype = r_BTHK.dtype
        assert all(
            i.dtype == dtype for i in [r_BTHK, k_BTHK, v_BTHK, a_BTHK, k_deformed_BTHK]
        )
        if not r_BTHK.is_cuda:
            raise ValueError("Stateful WKV is CUDA-only.")
        if dtype == torch.bfloat16:
            op_f = torch.ops.rwkv.rwkv7_wkv_forward_stateful_bfloat16
        elif dtype == torch.float:
            op_f = torch.ops.rwkv.rwkv7_wkv_forward_stateful_float
        elif dtype == torch.half:
            op_f = torch.ops.rwkv.rwkv7_wkv_forward_stateful_half
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        out, state_checkpoints, final_state = op_f.default(
            r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state0_BHKK
        )
        ctx.save_for_backward(
            r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints
        )
        return out, final_state

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        grad_BTHK = grad_outputs[0].contiguous()
        # grad_outputs[1] (the grad of final_state) is intentionally ignored: truncated BPTT.
        (
            r_BTHK,
            k_BTHK,
            v_BTHK,
            w_BTHK,
            a_BTHK,
            k_deformed_BTHK,
            skip_BT,
            state_checkpoints,
        ) = ctx.saved_tensors
        dtype = r_BTHK.dtype
        if dtype == torch.bfloat16:
            op_b = torch.ops.rwkv.rwkv7_wkv_backward_stateful_bfloat16
        elif dtype == torch.float:
            op_b = torch.ops.rwkv.rwkv7_wkv_backward_stateful_float
        elif dtype == torch.half:
            op_b = torch.ops.rwkv.rwkv7_wkv_backward_stateful_half
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = op_b.default(
            r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints, grad_BTHK
        )
        # No gradient for skip_BT or state0_BHKK (the latter detached -> truncated BPTT).
        return r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad, None, None


# Unused reference code for backpropagation for RWKV-7 wkv.
def reference_backward(
    r_BTHK: Tensor,
    k_BTHK: Tensor,
    v_BTHK: Tensor,
    w_BTHK: Tensor,
    a_BTHK: Tensor,
    k_deformed_BTHK: Tensor,
    state_checkpoints,
    grad_BTHK: Tensor,
):
    B, T, H, K = r_BTHK.shape

    with torch.no_grad():
        # compute all the states. For this example we don't need to use the checkpoints since we don't care about memory usage.
        states_BTHKK = torch.zeros(
            B, T, H, K, K, dtype=r_BTHK.dtype, device=r_BTHK.device
        )
        state_BHKK = torch.zeros(B, H, K, K, dtype=r_BTHK.dtype, device=r_BTHK.device)
        for t in range(T):
            _, state_BHKK = single_timestep(
                r_BTHK[:, t],
                k_BTHK[:, t],
                v_BTHK[:, t],
                w_BTHK[:, t],
                a_BTHK[:, t],
                k_deformed_BTHK[:, t],
                state_BHKK,
            )
            states_BTHKK[:, t] = state_BHKK.detach()

        # r_grad_BTHK = torch.empty(B, T, H, K, dtype=r_BTHK.dtype, device=r_BTHK.device)
        grad_BTHK1 = grad_BTHK.unsqueeze(-1)
        r_BTHK1 = r_BTHK.unsqueeze(-1)
        v_BTHK1 = v_BTHK.unsqueeze(-1)
        k_BTHK1 = k_BTHK.unsqueeze(-1)
        # w_BTHK1 = w_BTHK.unsqueeze(-1)
        w_diag_BTHKK = w_BTHK.diag_embed()
        a_BTHK1 = a_BTHK.unsqueeze(-1)
        k_deformed_BTHK1 = k_deformed_BTHK.unsqueeze(-1)
        # r_grad_BTHK1 = k_BTHK1 @ v_BTHK1.mT @ grad_BTHK1
        r_grad_BTHK1 = states_BTHKK.mT @ grad_BTHK1
        r_grad_BTHK = r_grad_BTHK1.squeeze(-1)

        dS_BTHKK = grad_BTHK1 @ r_BTHK1.mT
        scale_BTHKK = w_diag_BTHKK - k_deformed_BTHK1 @ (a_BTHK1 * k_deformed_BTHK1).mT
        v_grad_BTHK = torch.empty_like(r_grad_BTHK)
        k_grad_BTHK = torch.empty_like(r_grad_BTHK)
        w_grad_BTHK = torch.empty_like(r_grad_BTHK)
        a_grad_BTHK = torch.empty_like(r_grad_BTHK)
        k_deformed_grad_BTHK = torch.empty_like(r_grad_BTHK)
        for t in reversed(range(T)):
            v_grad_BTHK[:, t] = (dS_BTHKK[:, t] @ k_BTHK1[:, t]).squeeze(-1)
            k_grad_BTHK[:, t] = (dS_BTHKK[:, t].mT @ v_BTHK1[:, t]).squeeze(-1)

            # derivative wrt diag(w) - k_def a k_def^T
            if t > 0:
                # We can avoid a full matrix multiply by going back to the definition
                grad_decay_remove_BHKK = states_BTHKK[:, t - 1].mT @ dS_BTHKK[:, t]

                a_grad_BTHK[:, t] = -(
                    (grad_decay_remove_BHKK.mT @ k_deformed_BTHK1[:, t])
                    * k_deformed_BTHK1[:, t]
                ).squeeze(-1)
                k_deformed_grad_BTHK[:, t] = -(
                    grad_decay_remove_BHKK @ (a_BTHK1[:, t] * k_deformed_BTHK1[:, t])
                ).squeeze(-1)
                # for the dot product, do it directly with grad_decay (broadcast into rows)
                k_deformed_grad_BTHK[:, t] -= (
                    a_BTHK1[:, t] * (grad_decay_remove_BHKK.mT @ k_deformed_BTHK1[:, t])
                ).squeeze(-1)
                w_grad_BTHK[:, t] = grad_decay_remove_BHKK.diagonal(dim1=-2, dim2=-1)
            else:
                w_grad_BTHK[:, t] = 0
                a_grad_BTHK[:, t] = 0
                k_deformed_grad_BTHK[:, t] = 0

            # find the contribution to the t-1's S gradient
            dS_t_BHKK = dS_BTHKK[:, t]
            if t > 0:
                bonus_dS_BHKK = dS_t_BHKK @ scale_BTHKK[:, t].mT
                dS_BTHKK[:, t - 1] += bonus_dS_BHKK

    return (
        r_grad_BTHK,
        k_grad_BTHK,
        v_grad_BTHK,
        w_grad_BTHK,
        a_grad_BTHK,
        k_deformed_grad_BTHK,
    )


def single_timestep(
    r_BHK: Tensor,
    k_BHK: Tensor,
    v_BHK: Tensor,
    w_BHK: Tensor,
    a_BHK: Tensor,
    k_deformed_BHK: Tensor,
    state_BHKK: Tensor,
):
    r_BHK1 = r_BHK.unsqueeze(-1)
    k_BHK1 = k_BHK.unsqueeze(-1)
    v_BHK1 = v_BHK.unsqueeze(-1)
    w_BHK1 = w_BHK.unsqueeze(-1)
    a_BHK1 = a_BHK.unsqueeze(-1)
    k_deformed_BHK1 = k_deformed_BHK.unsqueeze(-1)

    # Uses broadcasting. Remember that each column in vk_skate gets its own decay.
    state_BHKK = (
        state_BHKK * w_BHK1.mT
        - state_BHKK @ k_deformed_BHK1 @ (a_BHK1 * k_deformed_BHK1).mT
    )
    state_BHKK = state_BHKK + (v_BHK1 @ k_BHK1.mT)

    # Now we have a new updated S. We evaluate it at r and return the output.
    out_BHK1 = state_BHKK @ r_BHK1
    return out_BHK1.squeeze(-1), state_BHKK


# NaN/inf SAFEGUARD (QAT): coarse int2 fake-quant can make the recurrent state run away to inf/NaN on a
# fragile card; fed back every step it then poisons the WHOLE batch's loss (all cards skipped). Replacing
# non-finite entries with a bounded value BEFORE quant isolates the blow-up to that one card (its state is
# capped, its neighbours' losses stay finite and train) and stops NaN from propagating. No-op on finite
# states (torch.nan_to_num leaves finite values untouched), so healthy batches are numerically unchanged.
_STATE_NAN_CAP = 1e4
def _sanitize_state(s: Tensor) -> Tensor:
    return torch.nan_to_num(s, nan=0.0, posinf=_STATE_NAN_CAP, neginf=-_STATE_NAN_CAP)


def fake_quant_state(s_BHKK: Tensor, qmax: float) -> Tensor:
    """Symmetric per-(B) per-tensor int-N round-trip of the WKV state with a straight-through
    gradient (forward = quantized, backward = identity). amax is taken over (H,K,K) per batch
    element, matching the Rust inference `quant_roundtrip_batched`. qmax: int8=127, int4=7, int2=1.
    qmax=inf disables (returns input). This is the QAT analog of the deploy-time state quant."""
    if qmax == float("inf"):
        return s_BHKK
    s_BHKK = _sanitize_state(s_BHKK)  # NaN/inf safeguard (see note above)
    amax = torch.amax(s_BHKK.abs(), dim=[1, 2, 3], keepdim=True)  # list dim = TorchScript-safe
    scale = (amax / qmax).clamp_min(1e-12)
    q = torch.round(s_BHKK / scale).clamp(-qmax, qmax) * scale
    return s_BHKK + (q - s_BHKK).detach()


def _fake_quant_factor(f: Tensor, qmax: float) -> Tensor:
    """Symmetric per-matrix int-N round-trip of a low-rank factor (amax over its last two dims),
    matching the Rust `quant_factor_inplace`. qmax=inf returns input."""
    if qmax == float("inf"):
        return f
    amax = torch.amax(f.abs(), dim=[-2, -1], keepdim=True)
    scale = (amax / qmax).clamp_min(1e-12)
    return torch.round(f / scale).clamp(-qmax, qmax) * scale


def fake_lowrank_state(s_BHKK: Tensor, rank: int, factor_qmax: float) -> Tensor:
    """STE rank-r truncation of the WKV state (optionally with int-N quantized factors), the QAT
    analog of the Rust deploy `lowrank_roundtrip`. forward = rank-r reconstruction A_r =
    (U_r sqrt S)(V_r sqrt S)^T (factors optionally quantized), backward = identity. The rank-r
    reconstruction is sign-convention-invariant, so it matches the Rust nalgebra SVD."""
    if rank <= 0:
        return s_BHKK
    s_BHKK = _sanitize_state(s_BHKK)  # NaN/inf safeguard: keep SVD input finite (non-finite -> NaN factors)
    B, H, K, _ = s_BHKK.shape
    with torch.no_grad():
        s = s_BHKK.reshape(B * H, K, K).float()
        u, sv, vh = torch.linalg.svd(s, full_matrices=False)  # u(BH,K,K) sv(BH,K) vh(BH,K,K)
        sq = sv[:, :rank].clamp_min(0).sqrt()                  # (BH,r)
        uf = u[:, :, :rank] * sq.unsqueeze(1)                  # (BH,K,r)
        vf = vh[:, :rank, :] * sq.unsqueeze(-1)                # (BH,r,K)
        uf = _fake_quant_factor(uf, factor_qmax)
        vf = _fake_quant_factor(vf, factor_qmax)
        recon = (uf @ vf).reshape(B, H, K, K).to(s_BHKK.dtype)  # (BH,K,r)@(BH,r,K)=(BH,K,K)
    return s_BHKK + (recon - s_BHKK).detach()


class RWKV7_WKV_QAT(torch.autograd.Function):
    """Fused per-step WKV + full-matrix int-N state quant (STE), the CUDA analog of the Python
    `quant_aware_rwkv7` int-N path. forward = the quant-aware output; the recurrent state is round-
    tripped through symmetric per-batch int-N quant each step (matching `fake_quant_state`). STE =>
    the quant is transparent to the gradient, so backward = plain-WKV backward over the quantized
    state trajectory (the CUDA backward re-applies the per-step quant during its checkpoint recompute).
    Runs in fp32 (inputs cast by the caller) to match the fp32 Python reference exactly. CUDA-only."""

    @staticmethod
    def forward(ctx, *inputs):
        r, k, v, w, a, kd, skip, qmax = inputs
        assert all(t.is_contiguous() for t in (r, k, v, w, a, kd, skip))
        assert r.dtype == torch.float32 and w.dtype == torch.float32
        assert skip.dtype == torch.bool
        out, ckpt, scale = torch.ops.rwkv.rwkv7_wkv_qat_forward_float.default(
            r, k, v, w, a, kd, skip, qmax
        )
        ctx.save_for_backward(r, k, v, w, a, kd, skip, ckpt, scale)
        ctx.qmax = qmax
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0].contiguous()
        r, k, v, w, a, kd, skip, ckpt, scale = ctx.saved_tensors
        rg, kg, vg, wg, ag, kdg = torch.ops.rwkv.rwkv7_wkv_qat_backward_float.default(
            r, k, v, w, a, kd, skip, ckpt, scale, grad, ctx.qmax
        )
        return rg, kg, vg, wg, ag, kdg, None, None


class RWKV7_WKV_QAT_LR(torch.autograd.Function):
    """Fused per-step WKV + RANK-1 int-N low-rank truncation (STE), matching the DEPLOY rank-1 compression
    (engine compress_wkv_state r==1: power-iterate the top singular vector of the max-normalized state,
    split-sqrt factors, per-column int-N quant, HALF-AWAY rounding). This is the train==deploy analog of
    the low-rank card/note deploy scheme -- unlike full-matrix int QAT, it teaches rank-1-truncation
    robustness. STE => backward = plain-WKV backward over the truncated trajectory. fp32, CUDA-only."""

    @staticmethod
    def forward(ctx, *inputs):
        r, k, v, w, a, kd, skip, qmax = inputs
        assert all(t.is_contiguous() for t in (r, k, v, w, a, kd, skip))
        assert r.dtype == torch.float32 and w.dtype == torch.float32 and skip.dtype == torch.bool
        out, ckpt = torch.ops.rwkv.rwkv7_wkv_qat_lr_forward_float.default(r, k, v, w, a, kd, skip, qmax)
        ctx.save_for_backward(r, k, v, w, a, kd, skip, ckpt)
        ctx.qmax = qmax
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0].contiguous()
        r, k, v, w, a, kd, skip, ckpt = ctx.saved_tensors
        rg, kg, vg, wg, ag, kdg = torch.ops.rwkv.rwkv7_wkv_qat_lr_backward_float.default(
            r, k, v, w, a, kd, skip, ckpt, grad, ctx.qmax
        )
        return rg, kg, vg, wg, ag, kdg, None, None


_PQ_UPLOADED = False
_PQ_LEARN = os.environ.get("RWKV_QAT_PQ_LEARN", "") == "1"
_PQ_CB_PARAM = None   # f32 Parameter [2*m*ncent*sub] (roles 0,1) when _PQ_LEARN; [ncent*sub] when joint
_PQ_META = None       # (m, sub, ncent, header_line, tail_rows, joint) — tail = roles 2,3 rows for export


def _set_pq_cb(cb, m, sub, ncent, joint):
    """Upload shim: a pre-joint RWKV_CUDA .pyd exposes the 4-arg schema (it can be file-locked by a
    still-running role-mode job while the new build waits in build/). Role-mode uploads (joint=0) are
    semantically identical on either build; joint=1 REQUIRES the new one."""
    try:
        torch.ops.rwkv.rwkv7_set_pq_codebook(cb, m, sub, ncent, joint)
    except (RuntimeError, TypeError):
        assert not joint, "joint-uv codebook needs the rebuilt RWKV_CUDA extension (5-arg set_pq_codebook)"
        torch.ops.rwkv.rwkv7_set_pq_codebook(cb, m, sub, ncent)
def maybe_upload_pq_codebook():
    """One-time upload of the rank-1 PQ codebook (roles 0=u, 1=v) to the CUDA device globals when
    RWKV_QAT_PQ=<codebook file> is set. After this, the fused rank-1 low-rank QAT kernel codebook-encodes
    the factor directions (via qat_lr_rank1's PQ branch) INSTEAD of int-N quant -- the train==deploy analog
    of the engine RWKV_LOWRANK_PQ path. Codebook file format (scratchpad/pq_train.py): line1
    `m bits sub_dim k ncent`, then 4*m blocks (role-major, then pos) of ncent centroid rows; we take the
    first 2*m blocks (roles 0,1) in layout ((role*m+pos)*ncent+c)*sub+j. No-op if RWKV_QAT_PQ unset.
    RWKV_QAT_PQ_LEARN=1 (Andrew's learnable-QAT-params doctrine, WKV analog of the shift-cb lever): the
    codebook becomes a trainable Parameter; the lr backward kernel accumulates dL/dcentroid into a device
    buffer (fetched per step by train_rwkv via wkv_pq_grad_fetch, re-uploaded via wkv_pq_reupload)."""
    global _PQ_UPLOADED, _PQ_CB_PARAM, _PQ_META
    if _PQ_UPLOADED:
        return
    path = os.environ.get("RWKV_QAT_PQ", "").strip()
    if not path:
        _PQ_UPLOADED = True
        return
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, k, ncent = (int(x) for x in lines[0].split()[:5])
    # joint-uv (task23): header sub == 2*k with m == 1 -> single catalog of concat(u,v) entries,
    # ONE code per head selects both directions. File = 1 centroid block (no roles).
    joint = 1 if (sub == 2 * k and m == 1) else 0
    vals = []
    for ln in lines[1:]:
        vals.extend(float(x) for x in ln.split())
    need = ncent * sub if joint else 2 * m * ncent * sub  # role mode: roles 0,1 only (rank-1)
    assert len(vals) >= need, f"PQ codebook {path}: {len(vals)} floats < {need}"
    cb = torch.tensor(vals[:need], dtype=torch.float32, device="cuda")
    _set_pq_cb(cb, m, sub, ncent, joint)
    print(f"[QAT-PQ] uploaded rank-1 codebook {path}: m={m} sub={sub} ncent={ncent} joint={joint} ({need} floats)")
    n_rows = ncent if joint else 2 * m * ncent
    _PQ_META = (m, sub, ncent, lines[0].strip(), [ln.strip() for ln in lines[1 + n_rows:]], joint)
    if _PQ_LEARN:
        _PQ_CB_PARAM = torch.nn.Parameter(cb.clone())
        torch.ops.rwkv.rwkv7_set_pq_learn(cb, 1)
        print(f"[QAT-PQ] WKV codebook LEARNABLE: kernel centroid-grad accumulation ON")
    # RWKV_QAT_NORM_BITS=<n>: model the deploy norm quant (engine RWKV_PQ_NORM_BITS) in the QAT forward —
    # WKV factor norms at n bits, log2-uniform over the engine's fixed WKV range [-3,0] octaves. The
    # shift-path analog lives in rwkv_model.fake_pq_shift (range [2.2,2.9]).
    nq_bits = int(os.environ.get("RWKV_QAT_NORM_BITS", "0") or 0)
    if nq_bits > 0:
        torch.ops.rwkv.rwkv7_set_norm_quant(cb, nq_bits, -3.0, 0.0)
        print(f"[QAT-PQ] norm quant ON: int{nq_bits} log2-uniform over [-3,0] octaves (WKV factor norms)")
    _PQ_UPLOADED = True


def wkv_pq_cb_param():
    """The learnable WKV codebook Parameter (None unless RWKV_QAT_PQ_LEARN=1). Call after
    maybe_upload_pq_codebook (train_rwkv calls this via wkv_pq_init before adding the optim group)."""
    maybe_upload_pq_codebook()
    return _PQ_CB_PARAM


def wkv_pq_grad_zero():
    """Zero the kernel's centroid-grad accumulator. Once per optimizer step, BEFORE backward."""
    torch.ops.rwkv.rwkv7_pq_cb_grad_zero(_PQ_CB_PARAM)


def wkv_pq_grad_fetch():
    """Fetch the accumulated centroid grads into the Parameter's .grad (after backward, before clip)."""
    g = torch.ops.rwkv.rwkv7_pq_cb_grad_get(_PQ_CB_PARAM)
    _PQ_CB_PARAM.grad = g.view_as(_PQ_CB_PARAM)


def wkv_pq_reupload():
    """Push the stepped Parameter values back to the kernel's codebook globals (after optimizer.step)."""
    m, sub, ncent, _, _, joint = _PQ_META
    _set_pq_cb(_PQ_CB_PARAM.detach(), m, sub, ncent, joint)


def wkv_pq_export(path):
    """Write the (learned) codebook back in the engine text format: learned roles 0,1 + the ORIGINAL
    roles 2,3 rows (rank-1 never touches them; keeps the 4-role file format the engine expects).
    Joint mode: the single learned block (the engine loader detects joint from the header)."""
    m, sub, ncent, header, tail, joint = _PQ_META
    cb = _PQ_CB_PARAM.detach().float().cpu().view(ncent if joint else 2 * m * ncent, sub)
    out_lines = [header]
    for row in cb:
        out_lines.append(" ".join(f"{x:.6e}" for x in row.tolist()))
    out_lines.extend(tail)
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(out_lines) + "\n")


@torch.jit.ignore  # never scripted: the QAT per-step loop (+ torch.linalg.svd) isn't TorchScript-able,
# and this path only runs under RWKV_NO_JIT (eager). Marking it ignore lets the JIT scripter compile
# RWKV7TimeMixer.forward's hot kernel path again (JIT was silently broken by adding this call).
def quant_aware_rwkv7(
    r_BTHK: Tensor,
    k_BTHK: Tensor,
    v_BTHK: Tensor,
    w_BTHK: Tensor,
    a_BTHK: Tensor,
    k_deformed_BTHK: Tensor,
    skip_BT: Tensor,
    state_qmax: float,
    lowrank_rank: int = 0,
    lowrank_fqmax: float = float("inf"),
) -> Tensor:
    """Per-step reference WKV with the recurrent state round-tripped each step (quant-aware training).
    If lowrank_rank>0 the state is rank-r truncated (+ optional int-N factor quant) instead of full
    int-N quant -- the QAT analog of the deploy low-rank card/note state. Identical to `reference_rwkv7`
    when state_qmax=inf and lowrank_rank=0. Used ONLY for short-recurrence card/note streams in QAT."""
    out_dtype = k_BTHK.dtype
    # Fused CUDA kernel for the int-N full-matrix path (the F12 recipe): ~2 orders of magnitude faster
    # than the Python loop below, bit-parity via fp32. Falls back to the loop for the low-rank/SVD path,
    # on CPU, or when RWKV_QAT_FUSED=0 (A/B parity switch).
    _fused = r_BTHK.is_cuda and os.environ.get("RWKV_QAT_FUSED", "1") != "0"
    if _fused and lowrank_rank <= 0 and state_qmax != float("inf"):
        args = [t.float().contiguous() for t in (r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)]
        out_BTHK = RWKV7_WKV_QAT.apply(*args, skip_BT.contiguous(), float(state_qmax))
        return out_BTHK.to(out_dtype)
    # Fused RANK-1 int-N low-rank path (matches the deploy rank-1 compression). Only rank==1 is fused;
    # higher ranks (needing deflation/SVD) fall through to the Python loop below.
    if _fused and lowrank_rank == 1 and lowrank_fqmax != float("inf"):
        maybe_upload_pq_codebook()  # if RWKV_QAT_PQ set, switches qat_lr_rank1 to codebook-encode the factors
        args = [t.float().contiguous() for t in (r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)]
        out_BTHK = RWKV7_WKV_QAT_LR.apply(*args, skip_BT.contiguous(), float(lowrank_fqmax))
        return out_BTHK.to(out_dtype)
    r_BTHK = r_BTHK.float()
    k_BTHK = k_BTHK.float()
    v_BTHK = v_BTHK.float()
    w_BTHK = w_BTHK.float()
    a_BTHK = a_BTHK.float()
    k_deformed_BTHK = k_deformed_BTHK.float()
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    out_BTHK = torch.empty(B, T, H, K, dtype=torch.float32, device=r_BTHK.device)
    state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32, device=r_BTHK.device)
    for t in range(T):
        out_BTHK[:, t], next_state_BHKK = single_timestep(
            r_BTHK[:, t],
            k_BTHK[:, t],
            v_BTHK[:, t],
            w_BTHK[:, t],
            a_BTHK[:, t],
            k_deformed_BTHK[:, t],
            state_BHKK,
        )
        if lowrank_rank > 0:  # rank-r truncation (+ factor quant) -- the low-rank deploy analog
            next_state_BHKK = fake_lowrank_state(next_state_BHKK, lowrank_rank, lowrank_fqmax)
        else:
            next_state_BHKK = fake_quant_state(next_state_BHKK, state_qmax)  # quant before next step
        skip_B111 = skip_BT111[:, t]
        state_BHKK = torch.where(skip_B111, state_BHKK, next_state_BHKK)
    return out_BTHK.to(out_dtype)


def reference_rwkv7(
    r_BTHK: Tensor,
    k_BTHK: Tensor,
    v_BTHK: Tensor,
    w_BTHK: Tensor,
    a_BTHK: Tensor,
    k_deformed_BTHK: Tensor,
    skip_BT: Tensor,
):
    out_dtype = k_BTHK.dtype
    r_BTHK = r_BTHK.float()
    k_BTHK = k_BTHK.float()
    v_BTHK = v_BTHK.float()
    w_BTHK = w_BTHK.float()
    a_BTHK = a_BTHK.float()
    k_deformed_BTHK = k_deformed_BTHK.float()
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    out_BTHK = torch.empty(B, T, H, K, dtype=torch.float32, device=r_BTHK.device)
    state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32, device=r_BTHK.device)
    for t in range(T):
        out_BTHK[:, t], next_state_BHKK = single_timestep(
            r_BTHK[:, t],
            k_BTHK[:, t],
            v_BTHK[:, t],
            w_BTHK[:, t],
            a_BTHK[:, t],
            k_deformed_BTHK[:, t],
            state_BHKK,
        )
        skip_B111 = skip_BT111[:, t]
        state_BHKK = torch.where(skip_B111, state_BHKK, next_state_BHKK)
    return out_BTHK.to(out_dtype)


def reference_rwkv7_stateful(
    r_BTHK: Tensor,
    k_BTHK: Tensor,
    v_BTHK: Tensor,
    w_BTHK: Tensor,
    a_BTHK: Tensor,
    k_deformed_BTHK: Tensor,
    skip_BT: Tensor,
    state0_BHKK: Tensor = None,
):
    """Pure-PyTorch differentiable WKV with an optional initial state, returning
    `(out_BTHK, final_state_BHKK)`. Mathematically identical to `reference_rwkv7` when
    state0 is None/zeros. Used to parity-test the CUDA stateful kernel: the forward should
    match exactly, and autograd through this with a *detached* carried state0 gives the
    truncated-BPTT gradient reference the stateful CUDA backward must match."""
    out_dtype = k_BTHK.dtype
    r_BTHK = r_BTHK.float()
    k_BTHK = k_BTHK.float()
    v_BTHK = v_BTHK.float()
    w_BTHK = w_BTHK.float()
    a_BTHK = a_BTHK.float()
    k_deformed_BTHK = k_deformed_BTHK.float()
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    if state0_BHKK is None:
        state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32, device=r_BTHK.device)
    else:
        state_BHKK = state0_BHKK.float()
    outs = []
    for t in range(T):
        out_t, next_state_BHKK = single_timestep(
            r_BTHK[:, t],
            k_BTHK[:, t],
            v_BTHK[:, t],
            w_BTHK[:, t],
            a_BTHK[:, t],
            k_deformed_BTHK[:, t],
            state_BHKK,
        )
        outs.append(out_t)
        skip_B111 = skip_BT111[:, t]
        state_BHKK = torch.where(skip_B111, state_BHKK, next_state_BHKK)
    out_BTHK = torch.stack(outs, dim=1)
    return out_BTHK.to(out_dtype), state_BHKK
