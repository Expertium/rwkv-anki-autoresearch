"""Aggregate deploy-side screens over MANY users and MANY batch tags (per-user paired deltas).

WHY. Every deploy-side probe in this project so far ran on 2-3 users, and that is not enough: the
per-user imm baseline on three long traces spreads 0.170 / 0.265 / 0.544, so a 3-user mean is
effectively one hard user plus two others, and a config that happens to suit the hard user looks like
a win (Amendment 17). The fix is many users, and the natural source is `trace_corrupt50` as the export
lands it.

`rust_grid.py` has no resume, and a 26-user x 4-config run does not fit a GPU window, so the screen is
run in BATCHES with one tag per batch (`ef_b1`, `ef_b2`, ...). Each batch contains `plain` and
`l2i8s4` as its reference, so every user's deltas are computed from the same weights, the same trace
and the same process tree. This script unions the batches, per user, and reports:

    * the mean paired delta versus `l2i8s4` for every config, over all users,
    * the SIGN COUNT, which is what this project actually needs (a mean can be carried by one outlier;
      a config that is better on 40 of 47 users is a different claim from one that is better on 1),
    * a bootstrapped 95% interval over users, so "0.0013 mean with a 0.001 std" is not mistaken for a
      result.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/wide_screen.py --tags ef_b1,ef_b2
  ... --configs l2i8s4ef,l2i4s4n4ef --base l2i8s4
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                      # scratchpad/quant_plan
ROOT = HERE.parents[2]
sys.path.insert(0, str(RUN))
import rust_score  # noqa: E402

PREDS = RUN / "rust_grid_preds"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--configs", default="l2i8s4ef,l2i4s4n4ef")
    ap.add_argument("--base", default="l2i8s4")
    ap.add_argument("--trace-dir", default="scratchpad/quant_plan/trace_corrupt50")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    tags = args.tags.split(",")
    tdir = ROOT / args.trace_dir
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    # gather (user, config) -> scores, one dict per user so the pairing is exact
    rows = {}
    for tag in tags:
        root = PREDS / tag
        if not root.exists():
            print(f"[skip] no preds dir for tag {tag}")
            continue
        for name in configs + [args.base]:
            d = root / name
            if not d.exists():
                print(f"[skip] {tag}/{name}")
                continue
            for f in sorted(d.glob("rust_pred_*.json")):
                u = int(f.stem.split("_")[-1])
                sc = rust_score.score_user(tdir, d, u)
                if not sc:
                    continue
                slot = rows.setdefault(u, {})
                if name in slot:
                    continue          # first tag wins; batches must not overlap in users
                slot[name] = sc
    if not rows:
        print("no scored users -- check --tags / --trace-dir")
        return 1

    print(f"users scored: {len(rows)}   tags: {tags}")
    out = {"tags": tags, "base": args.base, "n_users": len(rows), "configs": {}}

    for name in configs:
        paired = [(u, r[name], r[args.base]) for u, r in sorted(rows.items())
                  if name in r and args.base in r]
        if not paired:
            print(f"\n{name}: no paired users")
            continue
        entry = {"n": len(paired)}
        for head in ("ahead", "imm"):
            d = [c[head] - b[head] for _, c, b in paired]
            mean = sum(d) / len(d)
            better = sum(1 for x in d if x < 0)
            rng = random.Random(20261119)
            boots = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(args.boot))
            lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
            entry[head] = {"mean": mean, "better": better, "n": len(d),
                           "boot_lo": lo, "boot_hi": hi,
                           "min": min(d), "max": max(d)}
            print(f"{name:12s} {head:5s} n={len(d):3d}  mean {mean:+.6f}  "
                  f"95% [{lo:+.6f}, {hi:+.6f}]  better on {better}/{len(d)} users  "
                  f"range [{min(d):+.6f}, {max(d):+.6f}]")
        out["configs"][name] = entry
        print()

    p = RUN / f"rust_wide_screen_{'_'.join(tags)}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"written to {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())