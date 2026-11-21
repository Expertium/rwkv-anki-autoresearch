from pathlib import Path

s = Path("scratchpad/quant_plan/RESULTS.md").read_text(encoding="utf-8")
probes = [
    "**Rows 15-18 have no deploy speed yet",
    "They change the CARD's",
    "rank, and the engine reconstructs the full 32x32 matrix every step for rank >= 2 instead of using a",
    "fused readout, so the engine's rank-2 measurement is 0.629x rev/s / 0.448x states/s - below the 0.75x",
    "floor. Their loss numbers are still the right first measurement (the torch eval path mirrors the",
    "engine's rounding), but they cannot be deployed until the engine gets a fused rank-r readout.",
]
for p in probes:
    print(s.count(p), repr(p[:60]))
