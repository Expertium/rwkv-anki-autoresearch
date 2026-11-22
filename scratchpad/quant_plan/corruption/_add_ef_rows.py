"""Add the ERROR-FEEDBACK probe (RWKV_LOWRANK_EF) to `rust_grid` (raw bytes).

WHY, and why NOW. `RWKV_LOWRANK_EF` is fully implemented in the deploy engine and is documented there as
"carry a per-card quant-error buffer, add it back to the fresh state BEFORE re-compressing each step,
then update e = compensated - reconstructed. Cancels the compounding DC bias of the low-rank factor
quantization." The budget knobs (`RWKV_EF_ERANK` = low-rank truncation of `e`, `RWKV_EF_ELEVEL` =
quantize `e`'s factors) make the carried buffer deploy-honest rather than the over-budget proof of
concept.

It has NO record anywhere in this lineage: grepping RESULTS.md, NOTES.md, GPU_RUN_BOARD.md, every other
.md/.json under optimization/ and scratchpad/quant_plan for LOWRANK_EF / error-feedback / noise-shaping
/ EF_RANK / EF_LEVEL / EF_PQ returns nothing, while the same search finds the neighbouring HADAMARD and
per-column findings. So this is an unmeasured lever rather than a re-run.

It is also WELL MOTIVATED by today's measurement: the cost of cutting factor precision grows with
history length (int6 imm deltas +0.000067 / +0.000226 / +0.000742 on 28-29k-review users, positive on
every user, against +0.000003 on short traces). A cost that compounds with history is exactly what a DC
bias looks like, and EF is the implemented mechanism for cancelling one.

TWO EDITS, both additive:
  1. `bits_for` learns EF, so the error buffer is counted. It is real state: per stream per layer,
     a rank-`ef_erank` pair at `ef_elevel`, i.e. n * 2*erank*32*ebits. Not counting it would make EF look
     free.
  2. two configs, and NO no-EF twin because all three users already have `l2i8s4` measured
     (`rust_grid_fast_wide3.json`), so the comparison is against existing numbers:
       l2i8s4ef   row 17 + EF (erank 1, e at int4) -> 1,536/3,072 = 4,608 bits
       l2i4s4n4ef both streams at int4 + EF        -> 1,024/2,304 = 3,328 bits (256 BELOW row 17)

The second one is the axis-1 point: if EF cancels the bias, `int4` (which fails badly on its own) might
match row 17's accuracy at 256 fewer bits. The first one isolates EF itself: if EF helps nothing at row
17's own config, the mechanism is not paying for its buffer and the idea is dropped.

PTQ only - the Python QAT path has no EF, so this is a SCREEN on the int8-adapted checkpoint, exactly
like the int6/VLEVEL probes. It can rank configs against each other but cannot settle the gate.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RG = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR_TMPL = (
    '    "l2i8s4v2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",\n'
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int2"},\n'
)

BODY = [
    "    # ERROR FEEDBACK (2026-11-22): `e` = the previous step's compression error is added back",
    "    # BEFORE re-compressing. The engine documents it as cancelling the compounding DC bias of",
    "    # factor quantization, and it has never been measured in this lineage. The buffer is REAL",
    "    # state (per stream per layer, rank-1 pair at int4 = 256 b/layer), which bits_for now counts.",
    '    "l2i8s4ef": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",',
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_EF": "1",',
    '                 "RWKV_EF_ERANK": "1", "RWKV_EF_ELEVEL": "int4"},',
    '    "l2i4s4n4ef": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",',
    '                   "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_EF": "1",',
    '                   "RWKV_EF_ERANK": "1", "RWKV_EF_ELEVEL": "int4"},',
]

EF_OLD = (
    '    vbits = QBITS.get(str(env.get("RWKV_LOWRANK_VLEVEL", "")).strip())\n'
)
EF_NEW = (
    '    vbits = QBITS.get(str(env.get("RWKV_LOWRANK_VLEVEL", "")).strip())\n'
    '    # RWKV_LOWRANK_EF carries a real per-entity error buffer; counting it is what keeps EF honest.\n'
    '    _ef_on = str(env.get("RWKV_LOWRANK_EF", "")) in ("1", "true")\n'
    '    _ef_rank = int(env.get("RWKV_EF_ERANK", "1") or "1") if _ef_on else 0\n'
    '    _ef_bits = QBITS.get(str(env.get("RWKV_EF_ELEVEL", "int4")).strip(), 4) if _ef_on else 0\n'
    '    ef_bits = 2 * _ef_rank * 32 * _ef_bits\n'
)


def main() -> int:
    raw = RG.read_bytes()
    if b'"l2i8s4ef"' in raw:
        print("EF configs already present; nothing to do")
    else:
        for label, tmpl, new_text in (
            ("configs", ANCHOR_TMPL, None),
            ("ef bits", EF_OLD, EF_NEW),
        ):
            done = False
            for nl in ("\r\n", "\n"):
                ob = tmpl.replace("\n", nl).encode("utf-8")
                if raw.count(ob) != 1:
                    continue
                if new_text is None:
                    ins = ob + nl.join(BODY).encode("utf-8") + nl.encode("utf-8")
                else:
                    ins = new_text.replace("\n", nl).encode("utf-8")
                print(f"  {label}: patched (newline={nl!r})")
                raw = raw.replace(ob, ins)
                done = True
                break
            if not done:
                raise SystemExit(f"ABORT [{label}]: no single-newline match")
        # count the buffer where the per-stream bits are summed
        old_ret = "    return out"
        add = ("        if ef_bits and s in low:\n"
               "            out[s] += n * ef_bits\n"
               "    return out")
        for nl in ("\r\n", "\n"):
            ob = old_ret.replace("\n", nl).encode("utf-8")
            if raw.count(ob) == 1:
                raw = raw.replace(ob, add.replace("\n", nl).encode("utf-8"))
                print(f"  ef sum: patched (newline={nl!r})")
                break
        else:
            raise SystemExit("ABORT [ef sum]: no single-newline match")
        RG.write_bytes(raw)

    import importlib.util
    spec = importlib.util.spec_from_file_location("rust_grid", RG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("l2i8s4", "l2i8s4ef", "l2i4s4n4ef"):
        b = mod.bits_for(mod.CONFIGS[name])
        print(f"  {name}: card={b['card']}b note={b['note']}b total={sum(b.values())}")
    for name, want in (("l2i8s4", (1280, 2304)), ("l2i6s4", (1024, 1920)), ("l2i4s4", (768, 2304)),
                       ("l2i8s4v4", (1024, 1920)), ("l2i8s4n4", (1280, 1536))):
        b = mod.bits_for(mod.CONFIGS[name])
        got = (b["card"], b["note"])
        print(f"  regress {name}: {got} {'OK' if got == want else 'MISMATCH want ' + str(want)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
