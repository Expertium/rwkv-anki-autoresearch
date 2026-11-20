"""Probe candidate .pth checkpoints for geometry compatibility with this arm's harness.

Every model in the d=32 / 92-input lineage is 192,800 params with 444 tensors and 92 features.
Anything that prints a different shape cannot be scored on the adaptive92 VAL database or with the
state-quantization env of this arm, so it is not a candidate anchor.
"""
import torch

PATHS = [
    "pretrain/rwkv/opt_iter36/rwkv_iter36_124.pth",
    "pretrain/rwkv/opt_iter36/rwkv_iter36_558.pth",
    "pretrain/rwkv/opt_qat44/rwkv_iter44_248.pth",
    "pretrain/rwkv/opt_qat45/rwkv_iter45_496.pth",
    "pretrain/rwkv/opt_qat46/rwkv_iter46_248.pth",
    "scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/attempt_03_epoch_03_wd_0.01/adaptive_130062.pth",
    "scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/attempt_04_epoch_04_wd_0.01/adaptive_173416.pth",
]

for p in PATHS:
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                                    # noqa: BLE001
        print(f"{p}: LOAD ERROR {type(e).__name__}: {e}")
        continue
    sd = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not isinstance(sd, dict):
        print(f"{p}: unexpected object {type(obj).__name__}")
        continue
    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    wide = sd["features2card.0.weight"].shape if "features2card.0.weight" in sd else None
    streams = sorted({k.split(".")[0] for k in sd})
    extra = [k for k in obj.keys() if k != "model"] if (isinstance(obj, dict) and "model" in obj) else []
    print(f"RESULT {p} | tensors={len(sd)} params={n} features2card={wide} | "
          f"streams={len(streams)} | wrapper_keys={extra}")
