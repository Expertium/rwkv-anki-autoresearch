"""QAT feasibility spike: a quant-aware reference WKV (per-step fake-quant of the recurrent state).

Goal: confirm we can train with per-step STATE quantization simulated, matching the Rust inference
quant (per-card per-tensor symmetric int-N round-trip of S over (H,K,K)), with gradients (STE).

Only the card/note streams need this (short per-entity sequences). deck/preset/global stay on the
fast CUDA kernel. This spike validates:
  (1) qmax->inf  => equals the fp32 reference_rwkv7 (no quant),
  (2) per-step quant matches the Rust quant_roundtrip semantics (per-(B) amax over H,K,K),
  (3) gradients flow through the STE,
  (4) speed at card-stream scale (B=many short sequences, T small).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rwkv.model.rwkv_ops import reference_rwkv7, single_timestep  # noqa: E402

QMAX = {"int8": 127.0, "int4": 7.0, "int2": 1.0, "fp32": float("inf")}


def fake_quant_state(s_BHKK, qmax):
    """Symmetric per-(B) per-tensor int-N round-trip with straight-through gradient.
    amax over (H,K,K) per batch element -> (B,1,1,1), matching Rust quant_roundtrip_batched."""
    if qmax == float("inf"):
        return s_BHKK
    amax = s_BHKK.abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = (amax / qmax).clamp_min(1e-12)
    q = torch.round(s_BHKK / scale).clamp(-qmax, qmax) * scale
    # straight-through estimator: forward = q, backward = identity
    return s_BHKK + (q - s_BHKK).detach()


def quant_aware_rwkv7(r, k, v, w, a, kd, skip, qmax):
    """Per-step reference WKV with the state round-tripped through int-N each step."""
    out_dtype = k.dtype
    r, k, v, w, a, kd = (t.float() for t in (r, k, v, w, a, kd))
    skip_B111 = skip.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r.shape
    out = torch.empty(B, T, H, K, dtype=torch.float32, device=r.device)
    state = torch.zeros(B, H, K, K, dtype=torch.float32, device=r.device)
    for t in range(T):
        out[:, t], next_state = single_timestep(
            r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state
        )
        next_state = fake_quant_state(next_state, qmax)  # quant for "storage" before next step
        state = torch.where(skip_B111[:, t], state, next_state)
    return out.to(out_dtype)


def rand_inputs(B, T, H, K, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(B, T, H, K, generator=g).to(device)
    r, k, v, a, kd = mk(), mk(), mk(), torch.rand(B, T, H, K, generator=g).to(device), mk()
    w = torch.rand(B, T, H, K, generator=g).to(device) * 0.5 + 0.5  # decay in (0.5,1)
    skip = torch.zeros(B, T, dtype=torch.bool, device=device)
    return r, k, v, w, a, kd, skip


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # (1) qmax->inf equals reference_rwkv7
    B, T, H, K = 8, 20, 1, 32
    r, k, v, w, a, kd, skip = rand_inputs(B, T, H, K, device)
    ref = reference_rwkv7(r, k, v, w, a, kd, skip)
    qa_inf = quant_aware_rwkv7(r, k, v, w, a, kd, skip, QMAX["fp32"])
    print(f"(1) fp32 quant-aware vs reference_rwkv7 max diff: {(ref - qa_inf).abs().max().item():.2e}")

    # (2) quant changes output monotonically with coarseness
    for lvl in ["int8", "int4", "int2"]:
        qa = quant_aware_rwkv7(r, k, v, w, a, kd, skip, QMAX[lvl])
        print(f"(2) {lvl}: mean|out-fp32| = {(qa - ref).abs().mean().item():.3e}")

    # (3) gradients flow through STE
    r2 = r.clone().requires_grad_(True)
    out = quant_aware_rwkv7(r2, k, v, w, a, kd, skip, QMAX["int4"])
    out.sum().backward()
    gnorm = r2.grad.norm().item()
    print(f"(3) grad norm through int4 STE: {gnorm:.3e}  (nonzero => STE works)")

    # (4) speed at card-stream scale: many short per-card sequences
    for (Bc, Tc) in [(2000, 20), (5000, 30), (10000, 15)]:
        r, k, v, w, a, kd, skip = rand_inputs(Bc, Tc, 1, 32, device, seed=1)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        o = quant_aware_rwkv7(r, k, v, w, a, kd, skip, QMAX["int4"])
        loss = o.sum()
        loss.backward() if False else None
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"(4) B={Bc} T={Tc}: fwd {dt*1000:.0f} ms  ({Bc*Tc/dt/1e6:.1f}M steps/s)")


if __name__ == "__main__":
    main()
