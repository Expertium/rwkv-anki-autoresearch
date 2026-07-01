"""Print by-user-mean ahead/imm LogLoss for a result pair, with deltas vs the WS-15 tuned champion
and the d=128 baseline-to-beat. Usage: python scratchpad/score_jsonl.py <FILE_AHEAD> <FILE_IMM>
(file stems under result/, e.g. RWKV-decay15 RWKV-P-decay15)."""
import json
import sys


def mean(f):
    t = n = 0
    for line in open(f):
        r = json.loads(line)
        t += r["metrics"]["LogLoss"]
        n += 1
    return t / n, n


ahead, na = mean(f"result/{sys.argv[1]}.jsonl")
imm, ni = mean(f"result/{sys.argv[2]}.jsonl")
# champion to gate against (default = 1500u-data champion t1500d, fp32 101-200); override via argv[3]/[4]
ca = float(sys.argv[3]) if len(sys.argv) > 3 else 0.309706
ci = float(sys.argv[4]) if len(sys.argv) > 4 else 0.276357
print(f"ahead {ahead:.6f}  imm {imm:.6f}  (users {na}/{ni})")
da, di = ca - ahead, ci - imm
gate = "ACCEPT" if (da >= 0.0003 and di >= 0.0003) else "REJECT"
print(f"vs champion:      ahead {da:+.6f}  imm {di:+.6f}  -> {gate} (need BOTH >= +0.0003)")
ba, bi = 0.320295, 0.281913  # d=128 baseline-to-beat (fp32)
print(f"vs d128 baseline: ahead {ba - ahead:+.6f}  imm {bi - imm:+.6f}  (positive = our model is better)")
