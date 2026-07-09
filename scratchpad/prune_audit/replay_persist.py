"""Replay the windowed prune test with the persist=2 rule on this era's pruned-trial traces:
would each prune still have fired, and at what step? (Traces end AT the original prune step, so
a trial that fired on its FIRST below-alpha checkpoint shows persist-fire = None here -- meaning
it would have run >=300 more steps before the (near-certain) confirming second strike.)
Trace files may be append-polluted (re-run configs) -> keep only the last monotonic segment.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
champ = json.load(open(f"{ROOT}/optimization/champion_5k.json"))
champ_a = {s: a for s, a in zip(champ["trace_step"], champ["trace_ahead"])}
champ_i = {s: v for s, v in zip(champ["trace_step"], champ["trace_imm"])}

W, ALPHA = 1500, 1e-4

for path in sorted(glob.glob(f"{ROOT}/scratchpad/tuner5k/hp5k_*/hp5k_*_ws_trace.jsonl")):
    name = os.path.basename(path).replace("_ws_trace.jsonl", "")
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    steps = np.array([r["step"] for r in rows])
    resets = np.where(np.diff(steps) < 0)[0]
    if len(resets):
        rows = rows[resets[-1] + 1:]
    rows = [r for r in rows if r["step"] in champ_a]
    if not rows:
        continue
    da = np.array([r["ahead"] - champ_a[r["step"]] for r in rows])
    di = np.array([r["imm"] - champ_i[r["step"]] for r in rows])
    sidecar = json.load(open(f"{os.path.dirname(path)}/{name}.json"))
    min_step = 2 * int(sidecar["config"]["warmup_steps"])
    strikes, first_fire, persist_fire, history = 0, None, None, []
    for cp in range(300, len(da) + 1, 300):
        if cp < max(min_step, 300):
            continue
        a, i = da[:cp][-W:], di[:cp][-W:]
        p_a = float(wilcoxon(a, alternative="greater").pvalue)
        p_i = float(wilcoxon(i, alternative="greater").pvalue)
        hit = p_a < ALPHA and p_i < ALPHA
        strikes = strikes + 1 if hit else 0
        history.append(f"{cp}:{p_a:.1e}/{p_i:.1e}{'*' if hit else ''}")
        if hit and first_fire is None:
            first_fire = cp
        if strikes >= 2 and persist_fire is None:
            persist_fire = cp
    print(f"{name}: steps={len(da)}  first_joint_hit={first_fire}  persist2_fire={persist_fire}")
    print("   " + "  ".join(history[-6:]))
