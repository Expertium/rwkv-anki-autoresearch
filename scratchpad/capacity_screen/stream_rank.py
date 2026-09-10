"""Are the COARSE streams (deck / preset / user) capacity-starved, and did 10x the budget change
that? CPU only, minutes, zero GPU.

WHY THIS EXISTS. Arm 1 measured that the 4.95x parameter reduction is free on imm and costs ~0.003
on ahead once the budget is large enough to expose it (Q2). Andrew: growing deck/preset/global
"would be nice, but 38 h is a pretty steep price" -- 38 h because a capacity change alters the
architecture, so it cannot branch off ws10's shared WS checkpoint. This screen is the rung BEFORE
that purchase: if a stream is not using the width it already has, widening it cannot help, and the
question dies for the cost of a coffee.

Precedent, and it is the same instrument: `scratchpad/arch_2026-09-07/headrank_screen.py` measured
the curve head's hidden at an effective 2.4 dimensions and killed a richer ahead head outright.

WHAT IS MEASURED. One deploy-RNN pass per checkpoint over real rows of real train users, hooking
the LAST block of every stream (the stream's final representation for that row; hooks fire because
`forward_layer` calls `self.blocks[i](...)` as a module). Per stream: participation ratio, dims to
90% / 99% of variance, and dead dims.

THE COMPARISON IS THE POINT, not the level. A single checkpoint's participation ratio has no
absolute meaning -- it is compared BETWEEN two budgets:
  * utilization RISES from realcyc (1.25 ep) to arm 1 (12 ep)  -> the model is pressing against a
    ceiling it was not pressing against at the budget where every capacity reject was measured;
    the coarse-stream ladder is worth its GPU.
  * utilization FLAT or LOW at both -> the stream is not using the width it has, and widening is
    the wrong lever. Look for the ahead gap somewhere else.

⚠ WHAT IT CANNOT SAY. Effective rank of activations is a PROXY for capacity pressure, not a
measurement of it: a stream could be information-limited rather than width-limited and would look
identical. This screen redirects a purchase; it never settles a gate.

Usage:  python scratchpad/capacity_screen/stream_rank.py <ckpt.pth> [--limit N] [users...]
"""
import os
import sys
import time

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, os.getcwd())

_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
                        "deck_id:1,deck_id:2,card_id:1"),
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
}
for k, v in _ENV.items():
    os.environ[k] = v

from pathlib import Path                                        # noqa: E402
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
torch.set_num_threads(2)                                        # beside a GPU run; be a good guest
import rwkv.run_as_rnn as rnn_mod                               # noqa: E402
from rwkv.data_processing import get_rwkv_data                  # noqa: E402

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")

args = [a for a in sys.argv[1:]]
if not args:
    raise SystemExit(__doc__)
ckpt = args.pop(0)
limit = 4000
if "--limit" in args:
    i = args.index("--limit")
    limit = int(args[i + 1])
    del args[i:i + 2]
users = [int(u) for u in args] or [107, 136, 156]


def report(name, rows, depth):
    A = np.stack(rows)
    A = A - A.mean(0)
    s = np.linalg.svd(A, compute_uv=False)
    e = s ** 2 / (s ** 2).sum()
    pr = 1.0 / (e ** 2).sum()
    c = np.cumsum(e)
    k90 = int(np.searchsorted(c, 0.90) + 1)
    k99 = int(np.searchsorted(c, 0.99) + 1)
    dead = int((A.std(0) < 1e-6).sum())
    print(f"  {name:<10} depth {depth}  dim {A.shape[1]:>3}  n {A.shape[0]:>6}  "
          f"PR {pr:>6.1f}  90% {k90:>3}  99% {k99:>3}  dead {dead:>3}  "
          f"PR/dim {pr / A.shape[1]:.3f}")
    return pr / A.shape[1]


t0 = time.time()
print(f"checkpoint {ckpt}")
proc = rnn_mod.RNNProcess(path=ckpt, device=torch.device("cpu"), dtype=torch.float32)
rnn = proc.rnn
names = list(rnn.stream_names)
depths = list(rnn.stream_depths)
buf = {n: [] for n in names}
handles = []
for i, nm in enumerate(names):
    blocks = rnn.rwkv_modules[i].blocks
    last = blocks[len(blocks) - 1]

    def make(nm):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            buf[nm].append(o.detach().float().reshape(-1).numpy().copy())
        return hook
    handles.append(last.register_forward_hook(make(nm)))

for uid in users:
    torch.manual_seed(uid)
    df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
    recs = df.to_dict("records")[:limit]
    with torch.inference_mode():
        for row in recs:
            proc.run(row, skip=False)
    print(f"  user {uid}: {len(recs)} rows  [{time.time() - t0:.0f}s]", flush=True)
for h in handles:
    h.remove()

print(f"\nper-stream representation rank ({sum(len(v) for v in buf.values())} hooked vectors)")
ratios = {}
for i, nm in enumerate(names):
    if buf[nm]:
        ratios[nm] = report(nm, buf[nm], depths[i])
print("\nPR/dim is the number to compare BETWEEN checkpoints; the level alone means nothing.")
print("SCREEN_JSON " + repr({k: round(v, 4) for k, v in ratios.items()}))
