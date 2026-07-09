"""NULL CONTROL for the windowed prune test: champ5k_r1 epoch-1 vs champ5k_b1 (identical config,
identical seed/data order, same frozen env -- differ only by run-to-run compile noise).

Replays the exact trial prune test (W=1500, alpha=1e-4 BOTH modes, checkpoints every 300 from
MIN_STEP=400) with r1 as "candidate" and the champion ref (= b1's trace) as champion. Any FIRE
here = false positive under the null => the 5/5 trial prunes are suspect. No fire + p never
near alpha => the prune machinery is sound and the prunes were real HP effects.
"""
import json

import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
champ = json.load(open(f"{ROOT}/optimization/champion_5k.json"))
champ_a = {s: a for s, a in zip(champ["trace_step"], champ["trace_ahead"])}
champ_i = {s: v for s, v in zip(champ["trace_step"], champ["trace_imm"])}
cand = [json.loads(l) for l in open(f"{ROOT}/scratchpad/champ5k_r1/champ5k_r1_ws_trace.jsonl")]
cand = [r for r in cand if r["step"] in champ_a]  # r1 epoch 1 only (steps 1..6554)

da = np.array([r["ahead"] - champ_a[r["step"]] for r in cand])
di = np.array([r["imm"] - champ_i[r["step"]] for r in cand])
print(f"paired steps: {len(da)}")
print(f"first-5 |diff| ahead: {np.abs(da[:5])}")  # ~0 until compile noise kicks in => pairing is right
print(f"overall mean diff: ahead {da.mean():+.2e}  imm {di.mean():+.2e} (should be ~0 under null)")

W, ALPHA, MIN_STEP = 1500, 1e-4, 400
fired, min_pa, min_pi = None, 1.0, 1.0
print(f"\n{'step':>5} {'n':>5} {'p_worse_a':>10} {'p_worse_i':>10}")
for cp in range(300, len(da) + 1, 300):
    if cp < max(MIN_STEP, 300):
        continue
    a, i = da[:cp][-W:], di[:cp][-W:]
    p_a = float(wilcoxon(a, alternative="greater").pvalue)
    p_i = float(wilcoxon(i, alternative="greater").pvalue)
    min_pa, min_pi = min(min_pa, p_a), min(min_pi, p_i)
    both = p_a < ALPHA and p_i < ALPHA
    print(f"{cp:>5} {len(a):>5} {p_a:>10.2e} {p_i:>10.2e}  {'<<< FALSE FIRE' if both else ''}")
    if both and fired is None:
        fired = cp

print(f"\nNULL RESULT: fired={fired}  min_p ahead={min_pa:.2e}  imm={min_pi:.2e}  (alpha={ALPHA})")
