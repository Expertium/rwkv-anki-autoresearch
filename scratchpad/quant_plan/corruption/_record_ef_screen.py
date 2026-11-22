"""Record the error-feedback screen (Amendment 17 + a NOTES bullet).

The finding is deliberately recorded as INCONCLUSIVE, with the reason and the blocker, because the
3-user mean on these traces is dominated by one hard user (5052: imm 0.533-0.545 versus 0.170 for
7250 and 0.265 for 5939) and swings by ~0.006 under several unrelated configs. Recording it as a win
would be wrong, and recording it as a failure would be wrong too: on 5052 coarser state reads up to
0.012 better, and whether that generalizes to the 750-user roster is exactly what is not known.

Numbers (row-17 deploy weights, users 7250/5052/5939, `rust_grid_ef3.json`,
`rust_grid_long_paired.json`, `rust_grid_long_absolute.json`):

    l2i8s4     1,280/2,304 = 3,584    imm 0.324755   (reference)
    l2i8s4ef   1,536/3,072 = 4,608    imm 0.322530   delta -0.002225  [-0.005988 on 5052 alone]
    l2i4s4n4ef 1,024/2,304 = 3,328    imm 0.323445   delta -0.001310  [-0.005933 on 5052 alone]

per-user imm deltas vs l2i8s4: l2i8s4ef +0.000187 / -0.005988 / -0.000874, and
l2i4s4n4ef +0.000774 / -0.005933 / +0.001229.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

AMENDMENT = """
## Amendment 17 (2026-11-22 17:55): error feedback is implemented, was never measured, and the 3-user screen cannot settle it

**`RWKV_LOWRANK_EF` (error feedback / noise shaping) has no record anywhere in this lineage.** The
deploy engine implements it and documents it as "carry a per-card quant-error buffer, add it back to
the fresh state BEFORE re-compressing each step, then update e = compensated - reconstructed. Cancels
the compounding DC bias of the low-rank factor quantization", with budget knobs `RWKV_EF_ERANK` (rank
of the carried buffer), `RWKV_EF_ELEVEL` (its factor width) and `RWKV_EF_PQ`. Searching RESULTS.md,
NOTES.md, this board and every other .md/.json under `optimization/` and `scratchpad/quant_plan` for
LOWRANK_EF / error-feedback / noise-shaping / EF_RANK / EF_LEVEL / EF_PQ returns NOTHING, while the
same search finds the neighbouring HADAMARD and per-column findings - so it is an unmeasured lever,
not a re-run. It is well motivated by today's other result: the cost of cutting factor precision
GROWS with history length, which is what a compounding bias looks like, and EF is the implemented
mechanism for cancelling one.

**First screen (row-17 deploy weights, the 3 longest corrupt50 users, rank-1 int4 error buffer):**

| config | bits card/note | total | imm | vs row 17 | per-user imm deltas (7250/5052/5939) |
| --- | --- | --- | --- | --- | --- |
| `l2i8s4` (row 17) | 1,280/2,304 | 3,584 | 0.324755 | - | - |
| `l2i8s4ef` | 1,536/3,072 | 4,608 | 0.322530 | -0.002225 | +0.000187 / -0.005988 / -0.000874 |
| `l2i4s4n4ef` | 1,024/2,304 | 3,328 | 0.323445 | -0.001310 | +0.000774 / -0.005933 / +0.001229 |

On paper `l2i4s4n4ef` is attractive: BOTH streams at int4 plus a cheap error buffer, 3,328 bits, i.e.
256 BELOW row 17 while reading 0.0013 better on imm. **It is not established, and the reason matters
for every future deploy-side probe:** the 3-user mean is dominated by user 5052, whose imm is
0.533-0.545 against 0.170 (7250) and 0.265 (5939). 5052 moves by about -0.006 under several UNRELATED
configs (VLEVEL int3 -0.006228, both EF configs -0.00599), i.e. on that one hard trace coarse state
reads up to 0.012 better than fp32, while the other two users barely move and EF is slightly negative
on them. With three users, one such user is a third of the mean. So the EF delta is a one-user effect,
not a demonstrated improvement.

**Blocker to resolve before EF can become a recipe: the 750-user gate cannot score it.**
`RWKV_LOWRANK_EF` exists only in the Rust deploy engine; the Python eval path
(`rwkv_ops.fake_lowrank_state`, which `quant_sweep.py` uses) has no EF, and the formal gate is an
eval-path measurement (fp32 twin versus QAT+quantized on the 750-user roster). So EF is measurable
only on the Rust trace path, which is a ranking instrument and not the gate. Making EF a candidate
therefore needs a decision: either implement EF in the Python eval path, or extend the accepted gate
to a Rust-path measurement. **That is Andrew's call, not a change to make quietly** - it touches how
results are accepted.

**Next step that needs no decision:** widen the screen rather than conclude from three users. The
`trace_corrupt50` export (26/50 complete at 17:45) gives up to 50 long users for free, and the
exporter is CPU-only (`device=torch.device("cpu")`) and matches none of the chains' BUSY_PAT, so a
GENUINE wide screen can be built without touching the GPU queue. If EF holds up across tens of users,
the eval-path question above becomes worth Andrew's time; if it does not, the idea is closed for the
cost of a few CPU hours.
"""

BULLET = (
    "* 2026-11-22 17:55 - **Error feedback (`RWKV_LOWRANK_EF`) is implemented in the deploy engine and "
    "has NO record anywhere in this lineage** (RESULTS.md, NOTES.md, the board and every other "
    ".md/.json under `optimization/` and `scratchpad/quant_plan` return nothing for LOWRANK_EF / "
    "error-feedback / noise-shaping / EF_RANK / EF_LEVEL / EF_PQ). The engine documents it as "
    "cancelling the compounding DC bias of factor quantization, which is exactly the shape of "
    "today's other finding (the factor-precision cost grows with history), so it was screened. Result: "
    "**inconclusive, and recordable only as such.** On the 3 longest users a rank-1 int4 error buffer "
    "looks good - `l2i8s4ef` 1,536/3,072 = 4,608 bits reads imm -0.002225 vs row 17, and "
    "`l2i4s4n4ef` (both streams int4 + EF) 1,024/2,304 = 3,328 bits reads -0.001310, i.e. 256 bits "
    "BELOW row 17 and still better. But the mean is dominated by user 5052 (imm 0.533-0.545 versus "
    "0.170/0.265 for the others), which swings ~-0.006 under several unrelated configs (VLEVEL int3 "
    "too); the per-user deltas are +0.000187/-0.005988/-0.000874 for `l2i8s4ef`. So it is a one-user "
    "effect. **Blocker:** `RWKV_LOWRANK_EF` exists ONLY in the Rust engine, and the 750-user gate is "
    "an eval-path measurement whose Python quantization has no EF - so an EF recipe cannot be scored "
    "by the current gate at all. Turning EF into a candidate needs a decision from Andrew (implement "
    "EF in the eval path, or accept a Rust-path gate); it is NOT a change to make quietly because it "
    "touches how results are accepted. Next step needing no decision: widen the screen over the "
    "`trace_corrupt50` set (26/50 at 17:45) - the exporter is CPU-only and matches no BUSY_PAT, so "
    "this costs no GPU time.\n"
    "\n"
    "* 2026-11-22 17:50 - **Method lesson about every deploy-side probe in this project.** All of them "
    "run on 2-3 users, and the per-user spread is enormous: on the same three long traces the imm "
    "baseline ranges 0.170 / 0.265 / 0.544. A mean over three such users is a mean over one easy, one "
    "medium and one hard trace, so any config that happens to suit the hard user looks like a win (and "
    "the hard user is the one whose behaviour is most sensitive to state precision). Absolute taxes on "
    "long traces are also unusable, because every quantized config reads ~0.0036 BETTER than `plain` "
    "(the documented trace-path offset). What survives this is the PAIRED per-user delta with a "
    "consistent sign: that is why `int6` (+0.000067/+0.000226/+0.000742, all positive) is credible and "
    "EF / VLEVEL are not. Treat any single-sign-free deploy-side delta below ~0.002 as unmeasured.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def main() -> int:
    notes = NOTES.read_bytes()
    notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                "NOTES bullets")
    NOTES.write_bytes(notes)

    board = BOARD.read_bytes()
    if b"## Amendment 17" in board:
        print("  BOARD: Amendment 17 already present")
    else:
        if not board.endswith(b"\n"):
            board += b"\n"
        BOARD.write_bytes(board + AMENDMENT.encode("utf-8"))
        print("  BOARD: Amendment 17 appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())