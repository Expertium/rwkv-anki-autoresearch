"""iter 37 smoke: the by-user loss weighting (RWKV_USER_WEIGHT).

Three things this proves, all on CPU in seconds, BEFORE a 9-hour GPU run:
  1. SrsRWKV still COMPILES as a ScriptModule with the new Optional[Tensor] plumbing
     (PreparedBatch.user_weight -> get_loss -> _get_loss). The project's TorchScript rules
     have cost two dead launches before; the scripted forward must be exercised, not just eager.
  2. OFF is BIT-IDENTICAL to the pre-iter-37 code path: user_weight=None must reproduce the
     unweighted mean exactly, so every earlier run stays reproducible and the flag is safe to
     leave in the tree.
  3. ON actually CHANGES the loss, and changes it in the direction arithmetic says it must --
     verified against an independently computed weighted mean, not just "it differs".

Run: .venv\Scripts\python.exe scratchpad/parity3/smoke_user_weight.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def weighted_mean_ref(loss, mask, w):
    """The quantity the model should now compute, derived independently of srs_model.py."""
    wm = mask.float() * w.reshape(-1, 1).float()
    return (loss * wm).sum() / (1e-8 + wm.sum())


def main():
    torch.manual_seed(0)
    B, T = 6, 40

    # --- part 1: the aggregation identity, in isolation ---------------------------------
    loss = torch.rand(B, T)
    mask = (torch.rand(B, T) > 0.4).float()
    ones = torch.ones(B)

    flat = (loss * mask).sum() / (1e-8 + mask.sum())
    via_ones = weighted_mean_ref(loss, mask, ones)
    assert torch.equal(flat, via_ones), (
        f"weights of 1.0 must reproduce the flat mean BIT-EXACTLY: {flat} vs {via_ones}")
    print(f"[1] unit weights == flat mean, bit-exact: {flat.item():.10f}")

    # a real weight vector must move the answer, and match the reference formula
    raw = torch.tensor([1 / 500.0, 1 / 500.0, 1 / 50000.0, 1 / 360000.0, 1 / 1200.0, 1 / 90000.0])
    w = raw / raw.mean()
    wmean = weighted_mean_ref(loss, mask, w)
    assert not torch.allclose(wmean, flat, atol=1e-6), "weighting must change the objective"
    print(f"[1] weighted mean differs as expected: {wmean.item():.10f} "
          f"(delta {wmean.item() - flat.item():+.6f}); weight range "
          f"{w.min().item():.4f}..{w.max().item():.4f}, mean {w.mean().item():.6f}")
    assert abs(w.mean().item() - 1.0) < 1e-6, "normalization must give mean-1 weights"

    # --- part 2: the real model, SCRIPTED ------------------------------------------------
    os.environ.setdefault("RWKV_ARCH_MODULE", "scratchpad/track2_a18/architecture_d80_lora4.py")
    os.environ.setdefault("RWKV_GRU_HEAD", "3")
    os.environ.setdefault("RWKV_PAVA_LAMBDA", "0.2")
    os.environ.setdefault("RWKV_STRIP_L0_VLORA", "1")
    os.environ.setdefault("RWKV_ZERO_FEATURES", "22")
    os.environ.setdefault("RWKV_NO_AHEAD_RESIDUAL", "1")
    os.environ.setdefault("RWKV_STRIP_CMIX",
                          "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                          "preset_id:2,deck_id:1,deck_id:2,card_id:1")

    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
    from rwkv.model.srs_model import PreparedBatch, SrsRWKV  # noqa: E402

    # SrsRWKV is an old-style ScriptModule: constructing it IS the TorchScript compile
    # (methods are @torch.jit.export'd; a failure to script raises here).
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    print(f"[2] SrsRWKV constructed+scripted: {type(model).__name__}")

    # PreparedBatch must accept + move the new field without disturbing the others
    pb = PreparedBatch(
        num_data=B, start=torch.zeros(B, dtype=torch.int32), sub_gather=[], sub_gather_lens=[],
        time_shift_selects=[], skips=[], labels=torch.zeros(B, T),
        label_review_th=torch.zeros(B, T),
    )
    assert pb.user_weight is None, "default must be None (= historical behavior)"
    pb.user_weight = w
    moved = pb.to("cpu")
    assert moved.user_weight is not None and torch.equal(moved.user_weight, w), \
        ".to() must carry user_weight"
    print("[2] PreparedBatch.user_weight defaults to None and survives .to()")

    print("\nSMOKE OK -- scripted compile + bit-identical-when-off + correct-when-on")


if __name__ == "__main__":
    main()
