#!/usr/bin/env python
"""Assert the model built under THIS PROCESS'S env has exactly N parameters. Exit 44 if not.

WHY A PARAM COUNT AND NOT A BANNER. `RWKV_STRIP_CMIX` prints nothing, so there is no log line to
grep -- but more importantly, the QAT-inert bug (2026-08-12) proved a banner is the wrong check even
when one exists: `[QAT-LOWRANK] set:` was truthful about what had been PARSED while the object it
mutated was discarded one line later. **Inspect the CONSUMED state.** The parameter count is exactly
that for a capacity lever: it can only be right if the strip list actually reached the layers that
build the channel mixers.

A typo'd stream name, a stale env, or an arch module that overwrites the config all show up here as
a wrong number, before any GPU is spent.

Usage:  python scratchpad/parity3/assert_param_count.py <expected_params>
        (run with the launch's full env exported; phase 0 of the runner, ~10 s CPU)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("RWKV_NO_JIT", "1")          # CPU construction only; keep it cheap
want = int(sys.argv[1])
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as C   # noqa: E402
from rwkv.model.srs_model import SrsRWKV                       # noqa: E402
got = sum(p.numel() for p in SrsRWKV(C).parameters())
print(f"[assert-params] built model has {got:,} params; expected {want:,}")
print(f"[assert-params] RWKV_STRIP_CMIX={os.environ.get('RWKV_STRIP_CMIX','')!r}")
if got != want:
    print(f"[assert-params] MISMATCH -- the env did not produce the intended model. ABORTING.")
    sys.exit(44)
print("[assert-params] OK")
