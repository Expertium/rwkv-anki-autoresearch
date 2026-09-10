"""10-user VALIDITY gate for a decomposition arm, run before its 300-user eval.

Removing a VALUE quantizer from a model trained through it should change the loss a little, usually
downward. A model that gets much WORSE without a quantizer depends on it -- the cell-3 signature,
where the QAT checkpoint at full precision scored +0.105 ahead / +0.28 imm -- and then deploy minus
arm is not that quantizer's cost. The line is 0.005 in either mode: ~50x the accept bar, far above
10-user noise for a same-model paired difference, far below the cell-3 failure.

Usage: python probe_gate.py <tag> <arm>      (reads result/RWKV{,-P}-<tag>p.jsonl and <tag>d<arm>p)
exit 0 = valid, 48 = off distribution or unreadable
"""
import json
import os
import math
import sys

tag, arm = sys.argv[1], sys.argv[2]
LIMIT = 0.005
RES = os.environ.get("DECOMP_RESULT_DIR", "result")   # tests point this at a scratch dir


def load(p):
    return {r["user"]: r["metrics"]["LogLoss"] for r in
            (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}


ok = True
for mode, pre in (("ahead", "RWKV-"), ("imm", "RWKV-P-")):
    try:
        dep, a = load(f"{RES}/{pre}{tag}p.jsonl"), load(f"{RES}/{pre}{tag}d{arm}p.jsonl")
    except OSError as e:
        print(f"[probe-gate] {arm} {mode}: cannot read ({e})")
        sys.exit(48)
    users = sorted(set(dep) & set(a))
    if len(users) != len(dep) or len(users) < 8:
        print(f"[probe-gate] {arm} {mode}: users deploy {len(dep)} arm {len(a)} common {len(users)}")
        ok = False
        continue
    if any(not math.isfinite(dep[u]) or not math.isfinite(a[u]) for u in users):
        print(f"[probe-gate] {arm} {mode}: non-finite LogLoss")
        ok = False
        continue
    d = sum(a[u] - dep[u] for u in users) / len(users)     # positive = the arm is WORSE than deploy
    flag = d > LIMIT
    ok &= not flag
    print(f"[probe-gate] {arm} {mode}: arm - deploy = {d:+.6f} over {len(users)} users"
          f"{'  OFF DISTRIBUTION' if flag else ''}")
print(f"[probe-gate] {arm}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 48)
