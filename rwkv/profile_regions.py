"""Named profiling regions — so the kernel profile can attribute time to CODE, not kernels.

Why this exists: `RWKV_PROFILE_STEP`'s summary buckets by CUDA **kernel name**
(`train_rwkv.py::_print_kernel_profile`). That works for the WKV kernels and gemms, which have
distinctive names, but PAVA / the GRU head / Muon / the state clamp emit only generic
elementwise / reduce / gather / where kernels, so every one of them lands in the catch-all
"other (elementwise/reduce/copy/optim)" bucket — which was already 78% of the step at the d=32
profile. In other words the profile could not answer "is PAVA slow?" at all.

Wrapping a region in `region("pava.rectify")` emits a `record_function` scope; the profiler then
reports that scope's total device time (its own kernels plus its children), which IS attributable
to code.

⚠ **Device time is not the whole story for these regions.** They are suspected to be
LAUNCH- and SYNC-bound, and neither cost appears in GPU kernel time:
  * a few hundred tiny kernels cost ~5-10 us of launch overhead each on the CPU side while the
    GPU sits idle;
  * a `bool(t.any())` / `.item()` / `if tensor:` forces a device->host sync that drains the
    pipeline (`pava_rectify` does one per back-merge iteration, up to 6 per call).
So always read `region_report()`'s CPU column next to the device column: **cpu >> device means
the region is overhead-bound, and shrinking its kernels will not help — removing launches and
syncs will.**

Off by default (`RWKV_PROFILE_REGIONS=1` to enable) and a no-op singleton context manager when
off, so annotated code stays byte-identical in normal training.
"""
import os
from contextlib import nullcontext

ENABLED = os.environ.get("RWKV_PROFILE_REGIONS", "0") == "1"

_NULL = nullcontext()


def region(name: str):
    """Context manager naming a profiling region. Free (a shared nullcontext) when disabled."""
    if not ENABLED:
        return _NULL
    import torch

    return torch.profiler.record_function("rgn::" + name)


def region_report(profiler, n_steps: int) -> None:
    """Print per-region CPU and DEVICE time from a finished profiler. Plain ASCII."""
    rows = []
    for e in profiler.key_averages():
        if not e.key.startswith("rgn::"):
            continue
        dev = 0.0
        for attr in ("device_time_total", "cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                dev = v
                break
        cpu = getattr(e, "cpu_time_total", 0.0) or 0.0
        rows.append((e.key[5:], cpu, dev, e.count))
    if not rows:
        print("\n(no rgn:: regions recorded -- set RWKV_PROFILE_REGIONS=1 to enable them)")
        return
    rows.sort(key=lambda r: -r[1])
    # If NOTHING recorded device time, this is a CPU-only profile -- the device column is
    # meaningless, so say so once rather than flagging every region as "no device work".
    have_device = any(r[2] > 0 for r in rows)
    print(f"\n===== NAMED REGIONS ({n_steps} steps) =====")
    if not have_device:
        print("  (CPU-only profile -- no CUDA activity recorded, so the device column and the"
              " overhead-bound flags are NOT meaningful. Profile with"
              " ProfilerActivity.CUDA to get them.)")
    print(f"{'region':38s} {'cpu ms/step':>12s} {'dev ms/step':>12s} {'calls/step':>11s}  note")
    for name, cpu, dev, cnt in rows:
        c, d = cpu / 1e3 / n_steps, dev / 1e3 / n_steps
        note = ""
        if have_device and c > 0.5:
            if d == 0.0:
                note = "<-- NO device work at all (pure CPU / sync stall)"
            elif c / d > 3.0:
                note = "<-- OVERHEAD-BOUND (launches/syncs, not kernel work)"
        print(f"{name:38s} {c:12.3f} {d:12.3f} {cnt / n_steps:11.1f}  {note}")
    print("REGIONS_DONE")
