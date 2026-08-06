"""Extract a step trace + val trace from a train_rwkv console log.

The seed-pair arms (2026-08-06) deliberately ran WITHOUT RWKV_STEP_TRACE (nothing to pair
against at seed 4321), but promote_champion_5k.py requires a trace and the champion json is
the vprune reference for future candidates. The WS log carries the same quantities the trace
would have -- per-step lines `0 <step> <step+1>, all: X, ahead: Y (Y), imm: Z` and val lines
`Mean ahead validation loss: A (A), imm: I, ...` -- at 4-decimal precision instead of the
trace's full float precision. That is plenty for both downstream uses (Wilcoxon train-loss
pruning pairs steps; vprune thresholds are 0.004/0.006). Provenance: values are EXTRACTED
from the log, not re-measured; each val line is attributed to the most recent step line
printed before it.

Usage: python extract_trace_from_log.py <ws_log> <out_trace.jsonl>
       (also writes <out_trace.jsonl>.val.jsonl)
"""
import json
import re
import sys

STEP_RE = re.compile(r"^0 (\d+) \d+, all: ([\d.]+), ahead: ([\d.]+) \(([\d.]+)\), imm: ([\d.]+)")
VAL_RE = re.compile(r"^Mean ahead validation loss: ([\d.]+) \(([\d.]+)\), imm: ([\d.]+)")


def main():
    log_path, out_path = sys.argv[1], sys.argv[2]
    steps, vals = [], []
    last_step = 0
    seen = set()
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = STEP_RE.match(line)
        if m:
            step = int(m.group(1))
            if step in seen:  # resume overlap: keep first occurrence, like promote does
                continue
            seen.add(step)
            last_step = step
            steps.append({"step": step, "ahead": float(m.group(3)), "imm": float(m.group(5))})
            continue
        m = VAL_RE.match(line)
        if m:
            vals.append({"step": last_step, "val_ahead": float(m.group(1)), "val_imm": float(m.group(3))})
    if not steps:
        raise SystemExit(f"no step lines parsed from {log_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in steps:
            f.write(json.dumps(r) + "\n")
    with open(out_path + ".val.jsonl", "w", encoding="utf-8") as f:
        for r in vals:
            f.write(json.dumps(r) + "\n")
    print(f"steps: {len(steps)} (first {steps[0]['step']}, last {steps[-1]['step']})  "
          f"val points: {len(vals)} (steps {[v['step'] for v in vals[:3]]}...)")


if __name__ == "__main__":
    main()
