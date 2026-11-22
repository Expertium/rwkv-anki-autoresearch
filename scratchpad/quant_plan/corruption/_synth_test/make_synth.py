"""Throwaway: synthesize a rank-2-ish state set so analyze_state_subspace.py can be smoke-tested."""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
K, N = 32, 400
bu = rng.normal(size=(K, 3))
bv = rng.normal(size=(K, 3))
X = np.zeros((N, K, K), dtype=np.float32)
for i in range(N):
    a = rng.normal(size=3) * np.array([1.0, 0.4, 0.2])
    b = rng.normal(size=3) * np.array([1.0, 0.4, 0.2])
    X[i] = bu @ np.diag(a) @ bv.T + 0.05 * rng.normal(size=(K, K))
d = Path("scratchpad/quant_plan/corruption/_synth_test")
d.mkdir(exist_ok=True)
np.savez_compressed(d / "states_synth.npz", a0=X)
(d / "states_synth.meta.json").write_text(json.dumps({
    "tag": "synth", "every": 1, "max": 4000, "keys": ["x|r2|L0"],
    "meta": {"x|r2|L0": {"tag": "", "rank": 2, "layer": 0}},
    "kept": {"x|r2|L0": N}, "calls": {"x|r2|L0": N}, "total_states": N}), encoding="utf-8")
print("synth written", X.shape)
