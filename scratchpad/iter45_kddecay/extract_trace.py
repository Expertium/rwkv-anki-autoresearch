"""Rebuild a WS step-trace (+ val trace) from a training log, for promote_champion_5k.py.

The runners do not set RWKV_STEP_TRACE (it costs a file per step), so a promoted champion's trace
is reconstructed from its WS log -- the same thing that was done for iter 41
(`i41_ws_trace_fromlog.jsonl`). Formats, matching what promote_champion_5k --trace/--val-trace read:

    train: {"step": <int>, "ahead": <float>, "imm": <float>}
    val  : {"step": <int>, "val_ahead": <float>, "val_imm": <float>}

Log lines:
    `0 9093 9094, all: 0.867900, ahead: 0.2976 (0.2976), imm: 0.512`
    `Mean ahead validation loss: 0.6102 (0.6102), imm: 0.3920, validation n: 595795`
The val line carries no step, so it takes the step of the most recent train line (the same
convention iter 41's val trace shows: first entry at step 49, i.e. VALIDATE_EVERY-1).

⚠ ITER 45's WS PHASE IS THE ITER-41 CHAMPION'S, UNCHANGED -- the lever is decay-only. So this
trace MUST match iter 41's line for line; `--compare` asserts that, which is a real check that the
two runs shared a recipe and seed rather than an assumption.

Usage:
    python extract_trace.py <ws_log> <out.jsonl> [--compare <other_trace.jsonl>]
"""
import io
import json
import re
import sys

TRAIN = re.compile(r"^0 (\d+) \d+, all: [\d.]+, ahead: ([\d.]+) \([\d.]+\), imm: ([\d.]+)")
VAL = re.compile(r"^Mean ahead validation loss: ([\d.]+) \([\d.]+\), imm: ([\d.]+)")


def main():
    log, out = sys.argv[1], sys.argv[2]
    compare = None
    if "--compare" in sys.argv:
        compare = sys.argv[sys.argv.index("--compare") + 1]

    train, val, last_step = [], [], 0
    with io.open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN.match(line)
            if m:
                last_step = int(m.group(1))
                train.append({"step": last_step, "ahead": float(m.group(2)),
                              "imm": float(m.group(3))})
                continue
            m = VAL.match(line)
            if m:
                val.append({"step": last_step, "val_ahead": float(m.group(1)),
                            "val_imm": float(m.group(2))})

    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with io.open(out + ".val.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(train)} train steps -> {out}")
    print(f"wrote {len(val)} val points  -> {out}.val.jsonl")

    if compare:
        other = [json.loads(l) for l in io.open(compare, encoding="utf-8") if l.strip()]
        n = min(len(other), len(train))
        diffs = [i for i in range(n) if train[i] != other[i]]
        print(f"\ncompare vs {compare}: {len(train)} vs {len(other)} steps, "
              f"{len(diffs)} of the first {n} differ")
        if diffs[:1]:
            i = diffs[0]
            print(f"  first difference at index {i}: ours={train[i]} theirs={other[i]}")
        print("VERDICT:", "IDENTICAL - the two runs shared WS recipe and seed"
              if not diffs and len(train) == len(other) else "DIFFERS - investigate")
        return 0 if (not diffs and len(train) == len(other)) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
