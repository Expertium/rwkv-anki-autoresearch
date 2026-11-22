"""One-shot: record the iteration-count finding and remove the artifacts of the buggy driver run.

Finding: `RWKV_LOWRANK_FAST_ITERS=4` is WORSE than the default 2 on the deploy-relevant metric
(scored imm tax +0.000221 vs +0.000051; max per-review |dp| 5.8e-2 vs 8.1e-2 on the immediate head
but a worse mean 4.99e-4 vs 4.74e-4, and a worse relative-Frobenius histogram). So 2 stays the
default. Also removes `rust_fastrank_iters.json` and its dumps: that batch ran with a driver version
whose "ref" mode did NOT clear the fast flag, so its reference row was the fast path itself.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]


def raw_patch(rel, pairs):
    p = ROOT / rel
    text = p.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
    for i, (old, new) in enumerate(pairs, 1):
        cands = [(old, new)]
        if nl != "\n":
            cands.append((old.replace("\n", nl), new.replace("\n", nl)))
        hit = None
        for o, n in cands:
            if text.count(o) == 1:
                hit = (o, n)
                break
        if hit is None:
            print(f"ABORT [{rel} {i}]: anchor found {text.count(cands[0][0])} (raw) / "
                  f"{text.count(cands[-1][0])} (converted) times")
            sys.exit(1)
        text = text.replace(*hit)
        print(f"ok [{rel} {i}]")
    p.open(encoding="utf-8", newline="", mode="w").write(text)
    print(f"wrote {rel}")


ITER_TEXT = (
    "`RWKV_LOWRANK_FAST_ITERS` is the subspace-iteration count and the default 2 was CHOSEN, not assumed: "
    "`4` was measured on the same config and is WORSE (scored imm tax +0.000221 vs +0.000051, mean "
    "per-review |dp| 4.99e-4 vs 4.74e-4, and a worse relative-Frobenius histogram - 173 calls above 1e-1 "
    "vs 57), because a longer iteration drives the block further towards the top-3 invariant subspace of "
    "the PREVIOUS state, which is a worse warm start for the current one (`rust_fastrank_iters2.json`).")

raw_patch("scratchpad/quant_plan/RESULTS.md", [
    ("A finiteness run with the flag on still PASSES (`rust_finite_fastrank.json`: 0 non-finite of\n"
     "65,671,168 card/note state observations and 29,347 p(recall) values, state_maxabs 0.62).",
     "A finiteness run with the flag on still PASSES (`rust_finite_fastrank.json`: 0 non-finite of\n"
     "65,671,168 card/note state observations and 29,347 p(recall) values, state_maxabs 0.62). "
     + ITER_TEXT),
])

raw_patch("scratchpad/quant_plan/rust_grid.py", [
    ("# rust_score.json, rust_speed_fastrank_{on,off}.json). Do NOT set it for a Python<->Rust PARITY run:",
     "# rust_score.json, rust_speed_fastrank_{on,off}.json). ITERS defaults to 2 and 4 was measured WORSE\n"
     "    # (rust_fastrank_iters2.json), so the default is the deploy choice. NOTE the flag is IN this dict:\n"
     "    # to measure the REFERENCE solver, remove it (an inherited RWKV_LOWRANK_FAST cannot override it).\n"
     "    # Do NOT set it for a Python<->Rust PARITY run:"),
])

for junk in ("scratchpad/quant_plan/rust_fastrank_iters.json",
             "scratchpad/quant_plan/rust_fastrank_preds_live_iters",
             "scratchpad/quant_plan/rust_fastrank_preds_iters"):
    p = ROOT / junk
    if p.is_dir():
        shutil.rmtree(p)
        print(f"removed dir {junk}")
    elif p.exists():
        p.unlink()
        print(f"removed {junk}")
