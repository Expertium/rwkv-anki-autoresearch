"""realcyc's verdict, written BEFORE realcyc reported (2026-09-04 01:55; eval running). Tests the
predictions in realcyc/PREREG.md against the control gen4base. Do not edit once numbers exist.

Sign convention: positive delta = candidate BETTER (lower LogLoss) than control.

Usage: realcyc_verdict.py [cand=realcyc] [ctrl=gen4base]
  1. size gate  (optimization/size_baseline.py check id_e2s)
  2. means + deltas on the intersection, both modes
  3. the BOTH-modes gate: raw >= 1e-4 AND paired one-sided Wilcoxon p < 1e-4 (paired_pvalue.py)
  4. P2: per-user delta vs history span (days, from the -id parquet; cached in span_days.json)
  5. P3 / abort line
"""
import json
import math
import os
import subprocess
import sys

cand = sys.argv[1] if len(sys.argv) > 1 else "realcyc"
ctrl = sys.argv[2] if len(sys.argv) > 2 else "gen4base"
os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
PY = r".venv\Scripts\python.exe"
SPAN_CACHE = "scratchpad/realcyc/span_days.json"
FLOOR = 7.5e-5


def load(tag, mode):
    f = "result/RWKV-%s.jsonl" % tag if mode == "ahead" else "result/RWKV-P-%s.jsonl" % tag
    out = {}
    for line in open(f):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["user"]] = r
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def span_days(users):
    cache = {}
    if os.path.exists(SPAN_CACHE):
        cache = {int(k): v for k, v in json.load(open(SPAN_CACHE)).items()}
    missing = [u for u in users if u not in cache]
    if missing:
        import pyarrow.parquet as pq
        for u in missing:
            d = r"C:\Users\Andrew\anki-revlogs-10k-id\revlogs\user_id=%d" % u
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet")]
            lo, hi = None, None
            for f in files:
                t = pq.read_table(f, columns=["review_time"]).column("review_time")
                a, b = int(min(t.to_pylist())), int(max(t.to_pylist()))
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
            cache[u] = (hi - lo) / 86400000.0
        json.dump({str(k): v for k, v in cache.items()}, open(SPAN_CACHE, "w"))
    return {u: cache[u] for u in users}


print("=== 1. size gate vs the id_e2s baseline")
r = subprocess.run([PY, "optimization/size_baseline.py", "check", "id_e2s", "result/RWKV-%s.jsonl" % cand],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:])
size_ok = r.returncode == 0

print("=== 2. means on the intersection")
deltas = {}
for mode in ("ahead", "imm"):
    c, k = load(cand, mode), load(ctrl, mode)
    common = sorted(u for u in c if u in k)
    assert 6701 not in common
    mismatch = [u for u in common if c[u]["size"] != k[u]["size"]]
    dc = [k[u]["metrics"]["LogLoss"] - c[u]["metrics"]["LogLoss"] for u in common]
    deltas[mode] = dict(zip(common, dc))
    print("%-6s n=%d  %s %.6f  %s %.6f  delta %+.6f  size mismatches %d" % (
        mode, len(common), cand, mean([c[u]["metrics"]["LogLoss"] for u in common]),
        ctrl, mean([k[u]["metrics"]["LogLoss"] for u in common]), mean(dc), len(mismatch)))

print("=== 3. BOTH-modes gate (paired_pvalue.py --intersect)")
r = subprocess.run([PY, "optimization/paired_pvalue.py",
                    "--cand-ahead", "result/RWKV-%s.jsonl" % cand, "--cand-imm", "result/RWKV-P-%s.jsonl" % cand,
                    "--champ-ahead", "result/RWKV-%s.jsonl" % ctrl, "--champ-imm", "result/RWKV-P-%s.jsonl" % ctrl,
                    "--intersect"], capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
print(r.stdout.strip())
p_ok = r.returncode == 0
raw_ok = all(mean(deltas[m].values()) >= 1e-4 for m in deltas)
print("gate: size %s  raw>=1e-4 both %s  p<1e-4 both %s  =>  %s" % (
    "PASS" if size_ok else "FAIL", "PASS" if raw_ok else "FAIL", "PASS" if p_ok else "FAIL",
    "ACCEPT" if (size_ok and raw_ok and p_ok) else "REJECT"))

print("=== P1: bands (ahead +0.0001..+0.0006, imm +0.0000..+0.0004; ahead > imm relatively)")
da, di = mean(deltas["ahead"].values()), mean(deltas["imm"].values())
print("ahead %+.6f in band: %s   imm %+.6f in band: %s   rel ahead/imm: %.2f / %.2f = %s" % (
    da, 1e-4 <= da <= 6e-4, di, 0.0 <= di <= 4e-4, da / 0.298089, di / 0.263548,
    "ahead larger" if da / 0.298089 > di / 0.263548 else "imm larger"))

print("=== P2: per-user ahead delta vs history span (days)")
users = sorted(deltas["ahead"])
sp = span_days(users)
xs = [sp[u] for u in users]
ys = [deltas["ahead"][u] for u in users]
rho = spearman(xs, ys)
order = sorted(users, key=lambda u: sp[u])
n = len(order)
q1, q4 = order[: n // 4], order[3 * n // 4:]
m1, m4 = mean([deltas["ahead"][u] for u in q1]), mean([deltas["ahead"][u] for u in q4])
ratio = (m4 / m1) if m1 > 0 else float("inf") if m4 > 0 else float("nan")
print("rho(span, ahead delta) = %+.4f (predicted > +0.10)   bottom-span quartile %+.6f  top %+.6f  ratio %s (predicted > 1.5)" % (
    rho, m1, m4, ("%.2f" % ratio) if math.isfinite(ratio) else str(ratio)))
rho_i = spearman(xs, [deltas["imm"][u] for u in users])
print("rho(span, imm delta)   = %+.4f" % rho_i)

print("=== P3 / abort")
if abs(da) <= FLOOR and abs(di) <= FLOOR:
    print("both inside the +/-7.5e-5 floor: P3 null -- calendar phase adds nothing beyond the clock columns;"
          " the 28-dim/row-11 drop is a free simplification, not an accuracy accept")
if da < -2e-4 or di < -2e-4:
    print("ABORT LINE CROSSED: a mode is worse by > 0.0002 -- suspect the decade/century pairs as per-user identifiers")
