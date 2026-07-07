"""Promote a 5k candidate to CHAMPION: store its per-step WS logloss trace + final eval logloss as the
canonical prune reference (optimization/champion_5k.json). Run this as part of ACCEPTING a champion --
it is the "stored logloss values update automatically" step (Andrew 2026-07-02): every future candidate
run passes RWKV_PRUNE_REF=optimization/champion_5k.json and is Wilcoxon-early-pruned against THIS data.

Usage:
  python optimization/promote_champion_5k.py --name h2k16_5k --trace scratchpad/xxx/ws_trace.jsonl \
      --final-ahead 0.3097 --final-imm 0.2766

The previous champion's metadata (not the full trace) is appended to champion_5k_history.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "optimization" / "champion_5k.json"
HISTORY = ROOT / "optimization" / "champion_5k_history.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="champion run name")
    ap.add_argument("--trace", required=True, help="candidate's WS step-trace jsonl (RWKV_STEP_TRACE output)")
    ap.add_argument("--final-ahead", type=float, required=True, help="final EVAL by-user-mean ahead logloss")
    ap.add_argument("--final-imm", type=float, required=True, help="final EVAL by-user-mean imm logloss")
    # Learnable-codebook runs (2026-07-08): a champion is (weights + ITS learned codebooks) -- evals,
    # deploys and Rust-parity checks of this champion must use these files, not the reference q72u cbs.
    ap.add_argument("--ckpt", default="", help="champion checkpoint .pth path")
    ap.add_argument("--cb-wkv", default="", help="champion's final learned WKV codebook (txt export)")
    ap.add_argument("--cb-shift", default="", help="champion's final learned shift codebook (txt export)")
    args = ap.parse_args()

    steps, aheads, imms = [], [], []
    seen = set()
    for line in open(args.trace):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["step"] in seen:  # resume overlap: keep the first occurrence
            continue
        seen.add(r["step"])
        steps.append(int(r["step"]))
        aheads.append(float(r["ahead"]))
        imms.append(float(r["imm"]))
    if not steps:
        raise SystemExit(f"no steps parsed from {args.trace}")

    # archive the outgoing champion's metadata (finals + provenance; trace omitted to keep history small)
    if OUT.exists():
        old = json.loads(OUT.read_text())
        with open(HISTORY, "a") as f:
            f.write(json.dumps({k: old.get(k) for k in
                                ("name", "date", "final_ahead", "final_imm", "n_trace_steps",
                                 "ckpt", "cb_wkv", "cb_shift")}) + "\n")

    OUT.write_text(json.dumps({
        "name": args.name,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_ahead": args.final_ahead,
        "final_imm": args.final_imm,
        "ckpt": args.ckpt,
        "cb_wkv": args.cb_wkv,
        "cb_shift": args.cb_shift,
        "n_trace_steps": len(steps),
        "trace_step": steps,
        "trace_ahead": aheads,
        "trace_imm": imms,
    }))
    print(f"PROMOTED '{args.name}' -> {OUT}  ({len(steps)} WS steps, "
          f"final ahead {args.final_ahead:.6f} / imm {args.final_imm:.6f})")


if __name__ == "__main__":
    main()
