"""Replay the NEW last-1500-window prune test on the 0p0014 traces: fire step + no false fires."""
import json

import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
champ = json.load(open(f"{ROOT}/optimization/champion_5k.json"))
champ_a = {s: a for s, a in zip(champ["trace_step"], champ["trace_ahead"])}
champ_i = {s: v for s, v in zip(champ["trace_step"], champ["trace_imm"])}
cand = [json.loads(l) for l in open(
    f"{ROOT}/scratchpad/tuner5k/hp5k_peak_lr_0p0014/hp5k_peak_lr_0p0014_ws_trace.jsonl")]
da = np.array([r["ahead"] - champ_a[r["step"]] for r in cand if r["step"] in champ_a])
di = np.array([r["imm"] - champ_i[r["step"]] for r in cand if r["step"] in champ_a])

W, ALPHA, MIN_STEP = 1500, 1e-4, 400
fired = None
for cp in range(300, len(da) + 1, 300):
    if cp < max(MIN_STEP, 300):
        continue
    a, i = da[:cp][-W:], di[:cp][-W:]
    p_a = float(wilcoxon(a, alternative="greater").pvalue)
    p_i = float(wilcoxon(i, alternative="greater").pvalue)
    both = p_a < ALPHA and p_i < ALPHA
    print(f"step {cp:>5}  n={len(a):>4}  p_worse_a={p_a:.2e}  p_worse_i={p_i:.2e}  "
          f"est_a={champ['final_ahead'] + da[:cp][-300:].mean():+.6f} "
          f"est_i={champ['final_imm'] + di[:cp][-300:].mean():+.6f}  {'<<< FIRE' if both else ''}")
    if both and fired is None:
        fired = cp
print(f"\nNEW window=1500 fires at step {fired} (old full-window fired at 6300)")
