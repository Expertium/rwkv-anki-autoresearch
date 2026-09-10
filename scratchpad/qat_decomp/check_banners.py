"""Did a decomposition arm's eval process EXECUTE the arm's quantizers? Read its own log's banners.

A banner proves construction, not execution -- but these are the ones printed when a quantizer is
actually built or uploaded (the WKV codebook prints only at its first CUDA call), and assert_arm.py
has already proven the config. The failure this catches is an eval that loaded something else.
Banner strings are copied from a real quant-aware eval log (scratchpad/qat_tax/np_norm1_30320.log).

Usage: python check_banners.py <arm> <log> <wkvcb> <shiftcb>      exit 0 = right, 47 = wrong
"""
import sys

arm, log, wkvcb, shiftcb = sys.argv[1:5]
text = open(log, encoding="utf-8", errors="replace").read()
LR4 = "[QAT-LOWRANK] set: card_id=rank1/fq7.0, note_id=rank1/fq7.0"
LR8 = "[QAT-LOWRANK] set: card_id=rank1/fq127.0, note_id=rank1/fq127.0"
WCB = f"[QAT-PQ] uploaded rank-1 codebook {wkvcb}:"
WNQ = "[QAT-PQ] norm quant ON: int1"
SCB = f"[QAT-SHIFT-PQ] loaded {shiftcb}:"
SSC = "[QAT-SHIFT] state_shift_qmax set: card_id=3.0, note_id=3.0"
RULES = {
    "noshift": ([LR4, WCB, WNQ], [LR8, "[QAT-SHIFT-PQ] loaded", "[QAT-SHIFT] state_shift_qmax set"]),
    "nowcb": ([LR8, SCB, SSC, "learnable=False"], [LR4, "[QAT-PQ] uploaded", "[QAT-PQ] norm quant ON"]),
}
need, forbid = RULES[arm]
bad = [f"MISSING  {s}" for s in need if s not in text] + [f"PRESENT  {s}" for s in forbid if s in text]
if "learnable=True" in text:
    bad.append("PRESENT  learnable=True (a catalog is still training in an eval)")
if "Traceback" in text:
    bad.append("PRESENT  Traceback")
for b in bad:
    print(f"[banners] {arm}: {b}")
print(f"[banners] {arm}: {'PASS' if not bad else 'FAIL'} ({log})")
sys.exit(47 if bad else 0)
