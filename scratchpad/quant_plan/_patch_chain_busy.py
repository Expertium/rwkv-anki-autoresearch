"""Patch the two waiting QAT chains so their GPU-idle test also sees the OTHER chain's tail work.

Bug: `it36qat496_chain.py` and `it36qatl1i4_chain.py` decide "the GPU is free" from
`WAIT_FOR.exists() and 0 <busy> processes`, where <busy> only matched `quant_sweep.py|rwkv.train_rwkv`.
But the chain they wait for (`it36qatl1i8_chain.py`) still has an EXPORT step and a `rust_speed.py`
step after its last result file, and the Rust engine (`rwkv-infer.exe`) is a GPU workload that the
'nothing is running' test would not see. The waiter would therefore start training on top of the
speed benchmark.

Fix: widen the busy pattern to include the other chain's script, `rust_speed.py`, `rust_grid.py` and
`rwkv-infer`, and widen the process-name filter from `python` to `python|rwkv-infer` (the engine is
not a python process).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"

OLD_NAME = "$_.Name -match 'python' -and "
NEW_NAME = "$_.Name -match 'python|rwkv-infer' -and "

PATCHES = {
    "it36qat496_chain.py": (
        'SWEEP_PAT = "quant_sweep.py|rwkv.train_rwkv"',
        'BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatl1i8_chain.py|rust_speed.py|'
        'rust_grid.py|rwkv-infer")'),
    "it36qatl1i4_chain.py": (
        'BUSY_PAT = "quant_sweep.py|rwkv.train_rwkv"',
        'BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qat496_chain.py|rust_speed.py|'
        'rust_grid.py|rwkv-infer")'),
}

for fname, (old, new) in PATCHES.items():
    p = RUN / fname
    t = p.read_text(encoding="utf-8")
    assert t.count(old) == 1, (fname, "pattern count", t.count(old))
    assert t.count(OLD_NAME) == 1, (fname, "name-filter count", t.count(OLD_NAME))
    t = t.replace(old, new).replace(OLD_NAME, NEW_NAME)
    # `running(BUSY_PAT)` argument names in the wait loops
    t = t.replace("busy = running(SWEEP_PAT)", "busy = running(BUSY_PAT)")
    assert "SWEEP_PAT" not in t, fname
    p.write_text(t, encoding="utf-8")
    print(f"{fname}: patched")
