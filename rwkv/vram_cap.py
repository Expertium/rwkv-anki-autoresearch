"""Optional VRAM cap for a training or eval process (Andrew 2026-09-29: "cap VRAM for your runs a little below
12 GB" -- a full card makes his desktop sluggish).

RWKV_VRAM_CAP_GB=<GB> caps PyTorch's caching allocator with torch.cuda.set_per_process_memory_fraction. Unset (the
default) does nothing, so every existing run is unchanged. The cap never changes numerics; it only decides when an
allocation fails. Two consequences a runner must handle:
  * TRAINING catches exceptions per batch ("Exception caught ... Skipping batch"), so an allocation over the cap
    becomes a SKIPPED BATCH, not a crash. A capped runner must count those lines (a hollow run otherwise).
  * EVAL skips the whole user on a CUDA OOM (get_result, the 2026-09-20 box fix) and records it in
    result/RWKV-<tag>.oomskip.jsonl. Giant users that today survive only through the Windows sysmem fallback
    (VRAM spilling into system RAM) are then skipped instead.
The CUDA context itself (~0.3-0.5 GB) is outside the allocator and is not counted by the cap.
"""
import os


def apply():
    v = os.environ.get("RWKV_VRAM_CAP_GB", "").strip()
    if not v:
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, float(v) * 2 ** 30 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print("[vram-cap] RWKV_VRAM_CAP_GB=%s -> allocator capped at %.2f GiB of %.2f GiB (fraction %.4f)"
          % (v, frac * total / 2 ** 30, total / 2 ** 30, frac), flush=True)
    return frac


def run_uncapped_on_oom(fn, label):
    """Call fn(); if it hits a CUDA OOM while RWKV_VRAM_CAP_GB is set, lift the cap for ONE retry, then restore it.

    Used for VALIDATION only (2026-09-29): d128e10's 10-user validation OOMed at 11 GiB on every checkpoint, so its
    trace was empty. Skipping the user would change which users the validation mean covers and break its pairing
    with the reference traces, so the user is retried without the cap (a short spike above the cap, logged).
    Without a cap an OOM is re-raised unchanged.
    """
    import torch
    try:
        return fn()
    except torch.OutOfMemoryError:
        v = os.environ.get("RWKV_VRAM_CAP_GB", "").strip()
        if not v or not torch.cuda.is_available():
            raise
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        print("[vram-cap] %s over the %s GiB cap: retried once without the cap" % (label, v), flush=True)
        try:
            return fn()
        finally:
            torch.cuda.empty_cache()
            apply()
