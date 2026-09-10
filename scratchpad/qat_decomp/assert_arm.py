"""Assert that a QAT-tax DECOMPOSITION arm's env reached the FINAL arch config -- exactly, both ways.

The decomposition (scratchpad/qat_decomp/PREREG.md) evaluates the SAME QAT checkpoint three ways:
  deploy   every quantizer on, with the LEARNED catalogs (the shipped model)
  noshift  deploy MINUS shift quantization (shift scope and shift catalog cleared)
  nowcb    deploy MINUS the WKV codebook + WKV norm quant (rank-1 kept, factors at int8, near exact)
Each arm differs from deploy by ONE component, so deploy - arm IS that component's cost.

assert_qat_live.py only checks that NAMED streams are quantized. A decomposition arm also needs the
converse (nothing else quantized) and the exact levels, because the failure that matters here is an
arm that silently equals deploy -- it would print a component cost of zero, which reads as a finding.

Usage (under the arm's env): python scratchpad/qat_decomp/assert_arm.py <arm> <steps>
Exit 0 = the config and the env are exactly the arm's; 46 = anything else.
"""
import math
import os
import sys

INF = math.inf
EXPECT = {                      # card/note: (lowrank rank, factor qmax, shift qmax, WKV codebook on)
    "deploy": (1, 7.0, 3.0, True),
    "noshift": (1, 7.0, INF, True),
    "nowcb": (1, 127.0, 3.0, False),
}
QUANT_STREAMS = ("card_id", "note_id")


def main():
    arm, steps = sys.argv[1], sys.argv[2]
    rank, fq, shq, cb_on = EXPECT[arm]
    problems = []

    def env(k):
        return os.environ.get(k, "").strip()

    for k in ("RWKV_QAT_PQ_LEARN", "RWKV_QAT_SHIFT_PQ_LEARN", "RWKV_QAT_SCOPE", "RWKV_QAT_KD"):
        if env(k):
            problems.append(f"{k}={env(k)!r} must be empty in an eval arm")
    if env("RWKV_QAT_NORM_BITS") != "1":
        problems.append(f"RWKV_QAT_NORM_BITS={env('RWKV_QAT_NORM_BITS')!r}, every arm keeps 1 (shift norms)")
    pq, spq = env("RWKV_QAT_PQ"), env("RWKV_QAT_SHIFT_PQ")
    if cb_on:
        if not pq.endswith(f"_wkvcb_{steps}.txt") or not os.path.exists(pq):
            problems.append(f"RWKV_QAT_PQ={pq!r} is not an existing LEARNED step-{steps} WKV catalog")
    elif pq:
        problems.append(f"RWKV_QAT_PQ={pq!r} must be empty in arm {arm}")
    if shq != INF:
        if not spq.endswith(f"_shiftcb_{steps}.txt") or not os.path.exists(spq):
            problems.append(f"RWKV_QAT_SHIFT_PQ={spq!r} is not an existing LEARNED step-{steps} shift catalog")
    elif spq:
        problems.append(f"RWKV_QAT_SHIFT_PQ={spq!r} must be empty in arm {arm}")

    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as cfg
    seen = set()
    for name, lc in cfg.modules:
        got = (getattr(lc, "state_lowrank_rank", 0), getattr(lc, "state_lowrank_fqmax", INF),
               getattr(lc, "state_shift_qmax", INF), getattr(lc, "state_qmax", INF))
        want = (rank, fq, shq, INF) if name.split("@")[0] in QUANT_STREAMS else (0, INF, INF, INF)
        print(f"[arm-assert] {arm:8s} {name:10s} rank={got[0]} fq={got[1]} shift={got[2]} qmax={got[3]}",
              flush=True)
        seen.add(name.split("@")[0])
        if got[0] != want[0] or (want[0] and got[1] != want[1]) or got[2] != want[2] or got[3] != want[3]:
            problems.append(f"{name}: got rank/fq/shift/qmax {got}, want {want}")
    for s in QUANT_STREAMS:
        if s not in seen:
            problems.append(f"stream {s} absent from the arch -- the arm quantizes nothing")
    if problems:
        print(f"[arm-assert] FAIL {arm}:", flush=True)
        for p in problems:
            print("    " + p, flush=True)
        return 46
    print(f"[arm-assert] PASS {arm}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
