"""Build a FAKE champion_5k-style json from traceA with all per-step losses shifted 0.05 LOWER
(champion strictly better) -> a rerun of the same deterministic config must show diffs == +0.05 at
every paired step and Wilcoxon-prune at the first check. Finals chosen so the expected estimate is
exact: est = final + 0.05 (ahead 0.31 -> 0.36, imm 0.28 -> 0.33)."""
import json
import sys

trace_in, out = sys.argv[1], sys.argv[2]
steps, aheads, imms = [], [], []
for line in open(trace_in):
    r = json.loads(line)
    steps.append(r["step"])
    aheads.append(r["ahead"] - 0.05)
    imms.append(r["imm"] - 0.05)
json.dump({"name": "fake_champion_test", "final_ahead": 0.31, "final_imm": 0.28,
           "n_trace_steps": len(steps),
           "trace_step": steps, "trace_ahead": aheads, "trace_imm": imms}, open(out, "w"))
print(f"fake champion written: {len(steps)} steps -> {out}")
