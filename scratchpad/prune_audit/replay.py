"""Replay the Wilcoxon prune test for hp5k_peak_lr_0p0014 from the surviving traces.
At each 300-step checkpoint print: p(worse), p(better), sign structure, and windowed
mean diffs -- to see exactly why the test flipped from 'better @ 1e-51' to 'worse @ 1e-4'.
"""
import json

import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
champ = json.load(open(f"{ROOT}/optimization/champion_5k.json"))
champ_a = {s: a for s, a in zip(champ["trace_step"], champ["trace_ahead"])}
champ_i = {s: v for s, v in zip(champ["trace_step"], champ["trace_imm"])}

cand = [json.loads(l) for l in open(
    f"{ROOT}/scratchpad/tuner5k/hp5k_peak_lr_0p0014/hp5k_peak_lr_0p0014_ws_trace.jsonl")]
steps = [r["step"] for r in cand if r["step"] in champ_a]
da = np.array([r["ahead"] - champ_a[r["step"]] for r in cand if r["step"] in champ_a])
di = np.array([r["imm"] - champ_i[r["step"]] for r in cand if r["step"] in champ_a])
steps = np.array(steps)
print(f"paired steps: {len(steps)} (cand trace {len(cand)})")
print(f"{'step':>6} {'n':>6} | {'p_worse_a':>10} {'p_worse_i':>10} | {'p_bett_a':>10} {'p_bett_i':>10} | "
      f"{'%worse_a':>8} {'%worse_i':>8} | {'d_a_last300':>11} {'d_i_last300':>11}")
for cp in list(range(300, len(steps) + 1, 300)) + ([len(steps)] if len(steps) % 300 else []):
    a, i = da[:cp], di[:cp]
    pw_a = float(wilcoxon(a, alternative="greater").pvalue)
    pw_i = float(wilcoxon(i, alternative="greater").pvalue)
    pb_a = float(wilcoxon(a, alternative="less").pvalue)
    pb_i = float(wilcoxon(i, alternative="less").pvalue)
    print(f"{steps[cp-1]:>6} {cp:>6} | {pw_a:>10.2e} {pw_i:>10.2e} | {pb_a:>10.2e} {pb_i:>10.2e} | "
          f"{(a > 0).mean():>8.1%} {(i > 0).mean():>8.1%} | {a[-300:].mean():>+11.4f} {i[-300:].mean():>+11.4f}")

# magnitude structure: are the late 'worse' diffs bigger in |d| than the early 'better' ones?
half = len(steps) // 2
for nm, d in (("ahead", da), ("imm", di)):
    print(f"\n{nm}: early half  mean {d[:half].mean():+.5f}  median {np.median(d[:half]):+.5f}  "
          f"mean|d| {np.abs(d[:half]).mean():.5f}")
    print(f"{nm}: late  half  mean {d[half:].mean():+.5f}  median {np.median(d[half:]):+.5f}  "
          f"mean|d| {np.abs(d[half:]).mean():.5f}")
