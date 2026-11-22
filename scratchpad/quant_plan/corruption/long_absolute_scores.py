"""Show the ABSOLUTE per-user scores per config, to judge whether one user dominates a 3-user mean.

The 3-user paired means are suspicious: user 5052 swings by about -0.006 under several UNRELATED
configs (VLEVEL int3, and both error-feedback configs), while the other two users barely move. That
pattern says 5052 is volatile under ANY state perturbation rather than that a config is better. With
only 3 users, a swing that large on one of them is 1/3 of the mean, so it decides the conclusion.

This prints each config's raw imm/ahead per user plus the mean, and the per-user spread, so the
volatility can be judged directly instead of being hidden inside a mean.
"""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
doc = json.loads((RUN / "rust_grid_long_paired.json").read_text(encoding="utf-8"))

users = ["7250", "5052", "5939"]
print(f"{'config':12s} {'head':6s} " + " ".join(f"{u:>10s}" for u in users)
      + f" {'mean':>10s} {'spread':>9s}")
for name, per in doc.items():
    for head in ("imm", "ahead"):
        vals = [per[u][head] for u in users]
        mean = sum(vals) / len(vals)
        print(f"{name:12s} {head:6s} " + " ".join(f"{v:10.6f}" for v in vals)
              + f" {mean:10.6f} {max(vals) - min(vals):9.6f}")
    print()