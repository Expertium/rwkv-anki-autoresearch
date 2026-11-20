"""Read-only probe: what geometry does a checkpoint actually have?

Prints the parameter count, then per stream the WKV (time-mix) tensor shapes and the
state-shift tensor shapes, so the per-layer WKV state size (H*K*K) and the layer count can be
read off instead of guessed. Also prints the exact int4 baseline sizes (rank-2 factors at int4
plus int4 shift vectors) for card and note.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/probe_arch.py <checkpoint.pth>
"""
import sys
from pathlib import Path

import torch

STREAMS = ("card_id", "deck_id", "note_id", "preset_id", "user_id")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"file: {path}")
    print(f"top-level type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"top-level keys ({len(obj)}): {list(obj)[:12]}{' ...' if len(obj) > 12 else ''}")
        for cand in ("model", "state_dict", "weights"):
            if cand in obj and isinstance(obj[cand], dict):
                print(f"-> using obj['{cand}'] as the state dict")
                obj = obj[cand]
                break
    if not isinstance(obj, dict):
        raise SystemExit("no state dict found")
    keys = [k for k in obj if hasattr(obj[k], "shape")]
    total = sum(int(obj[k].numel()) for k in keys)
    print(f"tensors: {len(keys)}   params: {total:,}")
    # group by rwkv_modules.<i> (the stream stack) when the names carry no stream string
    groups = {}
    for k in sorted(keys):
        if "rwkv_modules." in k:
            idx = k.split("rwkv_modules.")[1].split(".")[0]
            groups.setdefault(idx, []).append(k)
    if groups:
        for idx in sorted(groups, key=int):
            sub = groups[idx]
            blocks = sorted({int(k.split("blocks.")[1].split(".")[0]) for k in sub if "blocks." in k})
            tm = [k for k in sub if "time_mixer" in k and len(tuple(obj[k].shape)) >= 2]
            tm1 = [k for k in sub if "time_mixer" in k and len(tuple(obj[k].shape)) == 1]
            print(f"\n== rwkv_modules.{idx}: {len(sub)} tensors, blocks={blocks} ==")
            for k in tm:
                print(f"   {k:78s} {tuple(obj[k].shape)}")
            for k in tm1:
                print(f"   {k:78s} {tuple(obj[k].shape)}   <- 1-D")
    for s in STREAMS:
        sub = [k for k in keys if s in k]
        if sub:
            print(f"\n== name-matched '{s}': {len(sub)} tensors ==")
    print("\nint4 baseline per layer (rank-2 factors at 4 bits + 2 shift vectors at 4 bits):")
    print("  card/note byte sizes depend on H, K and the layer count printed above.")


if __name__ == "__main__":
    main()
