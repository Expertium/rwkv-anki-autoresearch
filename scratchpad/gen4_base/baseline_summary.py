"""gen4base = the features-lineage BASELINE (gen-4 dbs: Bug C fixed, e2s-selected equalize set,
user 6701 EXCLUDED -> 2,499 VAL users). Prints the numbers the record needs and an INFORMATIONAL
comparison vs featB (gen 3). That comparison is a TWO-variable bundle (Bug C fix + the label
filter swap), so it is context, never a gate; the size-matched subset is reported for that reason.

Usage: baseline_summary.py [tag=gen4base] [ref=featB]
"""
import json
import math
import os
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "gen4base"
ref = sys.argv[2] if len(sys.argv) > 2 else "featB"


def load(t, mode):
    f = "result/RWKV-%s.jsonl" % t if mode == "ahead" else "result/RWKV-P-%s.jsonl" % t
    out = {}
    for line in open(f):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["user"]] = r
    return out


def nanskip(t):
    n = 0
    for f in ("result/RWKV-%s.nanskip.jsonl" % t, "result/RWKV-P-%s.nanskip.jsonl" % t):
        if os.path.exists(f):
            n = max(n, sum(1 for l in open(f) if l.strip()))
    return n


def mean(xs):
    return sum(xs) / len(xs)


res = {}
for mode in ("ahead", "imm"):
    d = load(tag, mode)
    users = sorted(d)
    ll = [d[u]["metrics"]["LogLoss"] for u in users]
    bad = [u for u in users if not math.isfinite(d[u]["metrics"]["LogLoss"])]
    assert not bad, "non-finite LogLoss for users %s" % bad[:5]
    res[mode] = dict(n=len(users), mean=mean(ll), users=users, d=d)
    print("%-6s n=%d  mean LogLoss %.6f  RMSE(bins) %.6f  min user %d max user %d" % (
        mode, len(users), mean(ll), mean([d[u]["metrics"]["RMSE(bins)"] for u in users]),
        users[0], users[-1]))
assert res["ahead"]["users"] == res["imm"]["users"], "ahead/imm user sets differ"
assert 6701 not in res["ahead"]["d"], "6701 must be excluded from this lineage"
print("nan_users (nanskip files): %d" % nanskip(tag))
print("total size: %d" % sum(res["ahead"]["d"][u]["size"] for u in res["ahead"]["users"]))

# informational vs ref (different label filter => sizes differ; not a gate)
for mode in ("ahead", "imm"):
    r = load(ref, mode)
    common = [u for u in res[mode]["users"] if u in r]
    d = res[mode]["d"]
    same = [u for u in common if d[u]["size"] == r[u]["size"]]
    delta = mean([r[u]["metrics"]["LogLoss"] - d[u]["metrics"]["LogLoss"] for u in common])
    delta_s = mean([r[u]["metrics"]["LogLoss"] - d[u]["metrics"]["LogLoss"] for u in same]) if same else float("nan")
    print("%-6s vs %s: n=%d  %s-better-by %+.6f  | size-identical n=%d  %+.6f  | %d users differ in size" % (
        mode, ref, len(common), tag, delta, len(same), delta_s, len(common) - len(same)))
