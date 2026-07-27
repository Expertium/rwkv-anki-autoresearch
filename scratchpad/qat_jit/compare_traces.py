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


def diff(a, b):
    """(n_mismatches, n_compared, first, worst) over the shared steps."""
    common = sorted(set(a) & set(b))
    bad = []
    for s in common:
        for k in ("ahead", "imm"):
            va, vb = a[s].get(k), b[s].get(k)
            if va is None or vb is None:
                continue
            if va != vb:
                bad.append((s, k, va, vb, abs(va - vb)))
    worst = max(bad, key=lambda r: r[4]) if bad else None
    return len(bad), 2 * len(common), (bad[0] if bad else None), worst


def main():
    paths = {
        "A  nojit": os.path.join(HERE, "trace_nojit.jsonl"),
        "A2 nojit (null control)": os.path.join(HERE, "trace_nojit2.jsonl"),
        "B  jit": os.path.join(HERE, "trace_jit.jsonl"),
    }
    tr = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"QATJIT_FAIL missing trace for {name}: {p}")
            return
        tr[name] = load(p)
        print(f"{name}: {len(tr[name])} steps")

    # The null control FIRST: if two identical-flag runs already differ, a JIT-vs-NO_JIT
    # difference proves nothing about JIT.
    n_null, tot_null, first_null, worst_null = diff(tr["A  nojit"], tr["A2 nojit (null control)"])
    n_jit, tot_jit, first_jit, worst_jit = diff(tr["A  nojit"], tr["B  jit"])

    print(f"\nNULL CONTROL  (nojit vs nojit): {n_null}/{tot_null} mismatches"
          + (f", worst |d|={worst_null[4]:.3e}" if worst_null else ""))
    print(f"TREATMENT     (nojit vs jit)  : {n_jit}/{tot_jit} mismatches"
          + (f", worst |d|={worst_jit[4]:.3e}" if worst_jit else ""))

    if n_null:
        print("\nQATJIT_INCONCLUSIVE -- the run is not reproducible under identical flags, so "
              "this harness cannot attribute any difference to JIT. Fix determinism first "
              "(RWKV_DETERMINISTIC / augment seed / codebook init) before reading the treatment.")
        return

    if not n_jit:
        print(f"\nQATJIT_BITEXACT OK -- null control clean AND all {tot_jit // 2} steps identical "
              "with JIT on (0 ULP). RWKV_NO_JIT=1 is not required for numerics.")
    else:
        print(f"\nQATJIT_DIVERGES -- null control is clean, so this IS attributable to JIT.")
        print(f"  first: step {first_jit[0]} {first_jit[1]} nojit={first_jit[2]!r} jit={first_jit[3]!r}")
        print(f"  worst: step {worst_jit[0]} {worst_jit[1]} |d|={worst_jit[4]:.3e}")
        print("  => JIT is not a free speedup for QAT; it changes the numerics. Every existing "
              "QAT number was measured on the no-JIT path, so adopting it re-bases them.")


if __name__ == "__main__":
    main()
