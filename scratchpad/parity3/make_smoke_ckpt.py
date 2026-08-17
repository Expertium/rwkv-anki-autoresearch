"""Write a randomly-initialised checkpoint for the CURRENT env, so the scripted-eval guard can run
BEFORE training instead of after it.

WHY IT IS NEEDED. `smoke_scripted_eval.sh` runs the real `get_result` path on one user, which is the
only way to exercise TorchScript-ignored bodies at RUNTIME (compiling proves nothing about them --
iter 48 lost a 6.5 h eval to an `@torch.jit.ignore` whose return type only misbehaved once a tensor
flowed through). But it needs a checkpoint, and `load_state_dict` is STRICT: an arch change means no
existing checkpoint has the right keys, so the guard could only ever run AFTER the new arch had
already been trained -- i.e. after the spend it exists to protect.

A random checkpoint is enough, because the guard is about CODE PATHS, not weights. The logloss it
produces is meaningless and is thrown away; what matters is that the scripted model loads, runs, and
returns without an INTERNAL ASSERT.

Usage:  python scratchpad/parity3/make_smoke_ckpt.py <out_path.pth>
The caller must already have the run's arch env exported, and RWKV_NO_JIT must NOT be set (the
checkpoint's keys must match what the JIT-on eval will construct).
"""
import os
import sys

import torch

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    torch.manual_seed(0)
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    n = sum(p.numel() for p in model.parameters())
    sd = model.state_dict()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(sd, out)
    print(f"[smoke-ckpt] wrote {out}: {len(sd)} keys, {n:,} params")
    return 0


if __name__ == "__main__":
    sys.exit(main())
