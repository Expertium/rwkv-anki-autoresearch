"""Per-user paired deltas for the long-history factor-precision probe (no engine runs).

The grid JSONs only store per-config MEANS, which cannot say whether a +0.0003 mean is systematic
across histories or driven by one user. The prediction directories are still on disk, so the scores
can be recomputed per user and differenced pairwise - the same users, the same trace, one config
changed, which is what `rust_score.py` does for the 750-user gate.

Compares:
    l2i8s4   (row 17, 1,280/2,304)   from rust_grid_preds/fast_wide3
    l2i6s4   (1,024/1,920)           from rust_grid_preds/int6long
    l2i5s4   (  896/1,728)           from rust_grid_preds/int6long
    v4/v3/v2 (VLEVEL)                from rust_grid_preds/vlevel3
all on the same 3 longest-history corrupt50 users, same weights, same traces.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
sys.path.insert(0, str(RUN))
import rust_score  # noqa: E402

TRACE_DIR = RUN / "trace_corrupt50"
PREDS = RUN / "rust_grid_preds"
USERS = [7250, 5052, 5939]

GROUPS = {
    "l2i8s4": "fast_wide3",
    "l2i6s4": "int6long",
    "l2i5s4": "int6long",
    "l2i8s4v4": "vlevel3",
    "l2i8s4v3": "vlevel3",
    "l2i8s4v2": "vlevel3",
    "l2i8s4ef": "ef3",
    "l2i4s4n4ef": "ef3",
    "plain": "fast_wide3",
}

scores = {}
for name, tag in GROUPS.items():
    d = PREDS / tag / name
    if not d.exists():
        print(f"missing preds: {d}")
        continue
    scores[name] = {u: rust_score.score_user(TRACE_DIR, d, u) for u in USERS}
    if not all(scores[name].values()):
        print(f"{name}: incomplete scores")

base = "l2i8s4"
print(f"\nper-user paired delta vs {base} (negative = candidate is BETTER)\n")
hdr = f"{'config':10s} " + " ".join(f"{u:>10d}" for u in USERS) + f" {'mean':>10s} {'std':>9s}"
print(hdr)
for name, sc in scores.items():
    if name == base:
        continue
    for head in ("imm", "ahead"):
        d = [sc[u][head] - scores[base][u][head] for u in USERS]
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / max(1, len(d) - 1)
        print(f"{name:10s} " + " ".join(f"{x:+10.6f}" for x in d)
              + f" {mean:+10.6f} {var ** 0.5:9.6f}   [{head}]")
    print()

out = {name: {str(u): sc[u] for u in USERS} for name, sc in scores.items()}
(RUN / "rust_grid_long_paired.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"written to {RUN / 'rust_grid_long_paired.json'}")
