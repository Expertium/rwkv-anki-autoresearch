"""Patch rwkv_ops.py: make the E1 tail penalty memory-viable by subsampling steps per stream.

WHY: `accum_tail_penalty` ran a differentiable `torch.linalg.svdvals` once per timestep per
rank-truncated layer. Every one of those SVD graphs stays alive until backward, so the penalty's
memory grows with the WINDOW length and both lambda smokes (l2i8s4e1a lambda=1.0, l2i8s4e1b
lambda=10.0) raised CUDA OOM at 10.34/11.00 GiB before the first training step, while the control
`l2i8s4pfsmoke` (same layout, penalty off) trained 6 steps in 21.4 s at the same 10.34 GiB.

THE FIX: accumulate on ONE step in RWKV_QAT_TAIL_STRIDE per stream (default 8). The mean over the
penalized steps is an unbiased estimate of the mean tail over the window, at 1/stride of the graph
memory. stride<=1 restores the old every-step behaviour.

SAFETY: the lever is read once at import from RWKV_QAT_TAIL_PENALTY (default "0"), so every
candidate that does not set it is byte-identical to before -- including the chain-17 run that is
training right now in a process that imported this module earlier.

Run: .venv\\Scripts\\python.exe scratchpad\\_patch_e1_stride.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "rwkv" / "model" / "rwkv_ops.py"

BEFORE_1 = '''_TAIL_REG = float(os.environ.get("RWKV_QAT_TAIL_PENALTY", "0") or 0)
_TAIL_ACC: dict = {}
'''

AFTER_1 = '''_TAIL_REG = float(os.environ.get("RWKV_QAT_TAIL_PENALTY", "0") or 0)
# MEMORY GUARD (2026-11-23). The penalty needs gradient, so it runs a differentiable svdvals per
# timestep per rank-truncated layer; every one of those graphs stays alive until backward and the
# memory therefore grows with the training window, which OOM'd an 11 GiB cap at 10.34 GiB (both
# lambda smokes, 2026-11-23). Accumulate on ONE step in RWKV_QAT_TAIL_STRIDE per stream instead:
# the mean over the sampled steps is an unbiased estimate of the mean tail over the window at
# 1/stride of the memory. stride <= 1 = every step (the pre-guard behaviour, still OOMs).
_TAIL_STRIDE = int(os.environ.get("RWKV_QAT_TAIL_STRIDE", "8") or 8)
_TAIL_ACC: dict = {}
_TAIL_STEP: dict = {}
'''

BEFORE_2 = '''    `tag` is the stream name (card/note/...) and doubles as the training flag: the call site passes an
    empty tag outside training, so eval passes accumulate nothing and add no loss term.
    """
    s = s_BHKK.reshape(-1, s_BHKK.shape[-2], s_BHKK.shape[-1]).float()
'''

AFTER_2 = '''    `tag` is the stream name (card/note/...) and doubles as the training flag: the call site passes an
    empty tag outside training, so eval passes accumulate nothing and add no loss term.

    Only one step in RWKV_QAT_TAIL_STRIDE is sampled, per `tag`, to bound the graph memory (see the
    note on _TAIL_STRIDE above). The counter is per tag, so a 3-layer stream is not sampled three
    times as often as a 1-layer stream.
    """
    if _TAIL_STRIDE > 1:
        _n = _TAIL_STEP.get(tag, 0) + 1
        _TAIL_STEP[tag] = _n
        if _n % _TAIL_STRIDE:
            return
    s = s_BHKK.reshape(-1, s_BHKK.shape[-2], s_BHKK.shape[-1]).float()
'''

TAIL_LINE = '''    print(f"[tail-penalty] lambda={_TAIL_REG} " + " ".join(shown), flush=True)
'''
TAIL_LINE_NEW = '''    print(f"[tail-penalty] lambda={_TAIL_REG} stride={_TAIL_STRIDE} " + " ".join(shown), flush=True)
'''


def main() -> int:
    raw = P.read_text(encoding="utf-8")
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    print(f"line endings before: CRLF={crlf} LF={lf}")
    for name, before in (("env+counter", BEFORE_1), ("accum guard", BEFORE_2), ("log line", TAIL_LINE)):
        n = raw.count(before)
        print(f"anchor {name}: {n} occurrence(s)")
        if n != 1:
            print(f"FAIL: anchor {name} is not unique")
            return 1
    out = raw.replace(BEFORE_1, AFTER_1).replace(BEFORE_2, AFTER_2).replace(TAIL_LINE, TAIL_LINE_NEW)
    assert out != raw
    n_stride = out.count("_TAIL_STRIDE")
    n_step = out.count("_TAIL_STEP")
    print(f"after: _TAIL_STRIDE x{n_stride} _TAIL_STEP x{n_step}")
    if n_stride < 7 or n_step != 3:
        print("FAIL: replacement looks wrong, file NOT written")
        return 1
    P.write_text(out, encoding="utf-8")
    chk = P.read_text(encoding="utf-8")
    if chk != out:
        print("FAIL: round-trip mismatch, file NOT written")
        return 1
    print("patched ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
