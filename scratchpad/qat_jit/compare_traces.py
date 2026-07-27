"""Are the NO_JIT and JIT quant-aware training paths bit-identical?

Compares the two per-step traces written by run_qat_jit.cmd. The step losses are a scalar
summary of the whole forward+backward, so any divergence in the QAT kernels (or in what the
scripted graph dispatches to) shows up here.

⚠ Deliberately strict: exact float equality, not a tolerance. The claim being tested is
"JIT does not change the numerics", and the CPU smoke (smoke_qat_jit.py) already showed
bit-identical eager-vs-scripted checksums -- so anything other than 0 ULP is a finding.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    rows = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["step"]] = r
    return rows


def main():
    pa = os.path.join(HERE, "trace_nojit.jsonl")
    pb = os.path.join(HERE, "trace_jit.jsonl")
    for p in (pa, pb):
        if not os.path.exists(p):
            print(f"QATJIT_FAIL missing trace: {p}")
            return
    a, b = load(pa), load(pb)
    common = sorted(set(a) & set(b))
    print(f"steps: nojit={len(a)} jit={len(b)} common={len(common)}")
    if not common:
        print("QATJIT_FAIL no common steps")
        return

    bad = []
    for s in common:
        for k in ("ahead", "imm"):
            va, vb = a[s].get(k), b[s].get(k)
            if va is None or vb is None:
                continue
            if va != vb:
                bad.append((s, k, va, vb, abs(va - vb)))

    if not bad:
        print(f"QATJIT_BITEXACT OK -- all {len(common)} steps identical in both modes (0 ULP)")
    else:
        worst = max(bad, key=lambda r: r[4])
        print(f"QATJIT_DIVERGES on {len(bad)} (step,metric) pairs of {2*len(common)}")
        print(f"  first: step {bad[0][0]} {bad[0][1]} nojit={bad[0][2]!r} jit={bad[0][3]!r}")
        print(f"  worst: step {worst[0]} {worst[1]} |d|={worst[4]:.3e}")
        print("  => JIT is NOT a pure speedup for QAT; it changes the numerics. Investigate "
              "before adopting (a bf16 reduction-order change is the likely cause, but the "
              "no-JIT path is the one every existing QAT number was measured on).")


if __name__ == "__main__":
    main()
