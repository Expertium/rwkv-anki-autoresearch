"""E1: the tail-energy (rank-truncation) penalty for QAT -- the advisor's first recommendation.

The deploy tax of a rank-truncated card/note state is RANK TRUNCATION, not precision (`cn8`,
full-matrix int8 with no truncation, is 4x over the bit bars and scores +0.000033/+0.000022, while
every rank-r row carries the whole tax). So the state's SPECTRUM is the binding constraint, and the
model can be trained to put its energy in the directions that survive the truncation:

    tail = sum_{i > r} sigma_i^2 / sum_i sigma_i^2     per (B,H), from the state's own spectrum

This costs ZERO state bits and ZERO deploy speed: it changes only what the weights learn. The spectrum
is already computed by `fake_lowrank_state` (torch.linalg.svd), so the quantity is free there; the
penalty uses a separate `torch.linalg.svdvals` OUTSIDE the `no_grad` block that builds the factors,
because the penalty must carry gradient to the weights while the reconstruction must stay a pure STE.
A run with the lever on therefore produces the same forward numbers as one with it off -- only the loss
has an extra term.

Coverage: only streams with lowrank_rank > 0 accumulate, and only while training (the call site passes
an empty stream tag outside training, so a validation pass adds no term). The rank-1 FUSED kernel
truncates inside the kernel where the pre-truncation state is invisible, so `quant_aware_rwkv7` REFUSES
that branch with the lever on and names RWKV_QAT_FUSED=0 (the documented bit-parity Python twin) --
otherwise the penalty would silently skip exactly the streams it exists to fix.

Files: rwkv/model/rwkv_ops.py (the penalty + the guard), rwkv/model/rwkv_model.py (the stream tag and
the call-site plumbing), rwkv/model/srs_model.py (the loss term), rwkv/ta3.py (the TA3 lever guard),
scratchpad/quant_plan/qat_decay.py (let a candidate request the Python loop).

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_add_tail_penalty.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PENALTY_BLOCK = '''# ---- TAIL-ENERGY PENALTY (RWKV_QAT_TAIL_PENALTY=<lambda>, 2026-11-22) ---------------------------
# WHY. The deploy tax of a rank-truncated card/note state is RANK TRUNCATION, not precision: `cn8`
# (full-matrix int8, NO truncation) is 4x over the bit bars and scores +0.000033/+0.000022, while the
# rank-r rows carry the whole tax. So the state's SPECTRUM, not its precision, is the binding
# constraint, and the model can be trained to put its energy in the directions that survive the
# truncation. This term is exactly the quantity the truncation throws away:
#     tail = sum_{i > r} sigma_i^2 / sum_i sigma_i^2        (per (B,H), the state's own spectrum)
# It costs ZERO state bits and ZERO deploy speed: it changes only what the weights learn.
#
# WHERE THE SPECTRUM COMES FROM. `fake_lowrank_state` already computes the full SVD of the state to
# build the rank-r factors, so the tail energy is free there. It is computed with a SEPARATE
# `torch.linalg.svdvals` OUTSIDE the `no_grad` block that builds the factors: the penalty must carry
# gradient back to the weights, while the reconstruction must stay a pure STE (its forward value is
# unchanged, so a run with the lever on produces the same NUMBERS as one with it off -- only the loss
# has an extra term).
#
# COVERAGE. Only streams that are actually rank-truncated (lowrank_rank > 0) accumulate, and only
# while training: the call site passes an empty stream tag in eval, so a validation pass adds no term.
# The rank-1 FUSED kernel truncates INSIDE the kernel, where the pre-truncation state is not visible,
# so `quant_aware_rwkv7` REFUSES that branch with the lever on and says to set RWKV_QAT_FUSED=0 (the
# documented bit-parity Python twin) -- otherwise the penalty would silently skip exactly the streams
# it exists to fix. RWKV_QAT_FUSED=0 is bit-parity with the fused kernels, so an E1 row stays
# comparable with the rows trained fused.
# Default unset = 0.0 = completely inert (no accumulation, no loss term, byte-identical).
_TAIL_REG = float(os.environ.get("RWKV_QAT_TAIL_PENALTY", "0") or 0)
_TAIL_ACC: dict = {}


@torch.jit.ignore
def accum_tail_penalty(s_BHKK: Tensor, rank: int, tag: str) -> None:
    """Accumulate one stream's truncated tail energy per (B,H).

    `tag` is the stream name (card/note/...) and doubles as the training flag: the call site passes an
    empty tag outside training, so eval passes accumulate nothing and add no loss term.
    """
    s = s_BHKK.reshape(-1, s_BHKK.shape[-2], s_BHKK.shape[-1]).float()
    sv = torch.linalg.svdvals(s)  # differentiable; only the singular values are needed
    e = sv.square()  # (BH, K)
    tot = e.sum(dim=-1)
    tail = e[:, rank:].sum(dim=-1)
    _TAIL_ACC.setdefault(tag, []).append((tail / (tot + 1e-12)).mean())


@torch.jit.ignore
def take_tail_penalty() -> Optional[Tensor]:
    """Mean of the accumulated per-stream tail energies, and CLEAR them.

    Returns None when the lever is off or nothing accumulated, so the caller adds no term at all (the
    default path stays byte-identical).

    \u26a0\ufe0f THE RETURN ANNOTATION IS LOAD-BEARING -- do not remove it. An unannotated
    `@torch.jit.ignore` function is typed by TorchScript as returning a plain `Tensor`; returning None
    then hands the scripted caller an UNDEFINED tensor and `is not None` refines to True. See the twin
    note on `take_rank1_penalty` for the exact failure it caused on 2026-08-15.
    """
    if not _TAIL_ACC:
        return None
    parts = []
    shown = []
    for tag in sorted(_TAIL_ACC):
        vals = _TAIL_ACC[tag]
        m = torch.stack(vals).mean()
        parts.append(m)
        shown.append(f"{tag}={float(m.detach()):.6f}(n={len(vals)})")
    _TAIL_ACC.clear()
    print(f"[tail-penalty] lambda={_TAIL_REG} " + " ".join(shown), flush=True)
    return torch.stack(parts).mean()


'''


def load(p: Path):
    text = p.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
    return text, nl


def apply(path: Path, pairs, label: str):
    text, nl = load(path)
    for i, (old, new) in enumerate(pairs):
        o, n = old.replace("\n", nl), new.replace("\n", nl)
        c = text.count(o)
        if c != 1:
            raise SystemExit(f"ABORT [{label} {i + 1}]: anchor found {c} times, expected 1:\n{o[:200]}")
        text = text.replace(o, n, 1)
        print(f"  ok [{label} {i + 1}]")
    path.write_text(text, encoding="utf-8", newline="")
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    ops = ROOT / "rwkv" / "model" / "rwkv_ops.py"
    model = ROOT / "rwkv" / "model" / "rwkv_model.py"
    srs = ROOT / "rwkv" / "model" / "srs_model.py"
    ta3 = ROOT / "rwkv" / "ta3.py"
    decay = ROOT / "scratchpad" / "quant_plan" / "qat_decay.py"

    # ---- 1. rwkv_ops.py: Optional import, the penalty block, the hook, the guard -------------
    apply(ops, [
        ("import os\nfrom typing import Any\n", "import os\nfrom typing import Any, Optional\n"),
        # The block goes immediately before the function it hooks into.
        ("def fake_lowrank_state(s_BHKK: Tensor, rank: int, factor_qmax: float) -> Tensor:",
         PENALTY_BLOCK + "def fake_lowrank_state(s_BHKK: Tensor, rank: int, factor_qmax: float,\n"
                         "                      tag: str = \"\") -> Tensor:"),
        ("    qmax=inf disables (returns input). This is the QAT analog of the deploy-time state quant.\"\"\"",
         "    qmax=inf disables (returns input). This is the QAT analog of the deploy-time state quant.\"\"\""),
        # docstring + the accumulation hook (outside no_grad: the penalty needs gradient)
        ("    reconstruction is sign-convention-invariant, so it matches the Rust nalgebra SVD.\"\"\"\n"
         "    if rank <= 0:\n"
         "        return s_BHKK\n"
         "    s_BHKK = _sanitize_state(s_BHKK)  # NaN/inf safeguard: keep SVD input finite (non-finite -> NaN factors)\n",
         "    reconstruction is sign-convention-invariant, so it matches the Rust nalgebra SVD.\n"
         "\n"
         "    `tag` names the stream (card/note/...) and doubles as the training flag for the\n"
         "    tail-energy penalty (RWKV_QAT_TAIL_PENALTY): an empty tag means eval, so nothing is\n"
         "    accumulated and no loss term is added. The penalty is computed OUTSIDE the no_grad block\n"
         "    below, because it has to carry gradient; the reconstruction stays a pure STE.\"\"\"\n"
         "    if rank <= 0:\n"
         "        return s_BHKK\n"
         "    s_BHKK = _sanitize_state(s_BHKK)  # NaN/inf safeguard: keep SVD input finite (non-finite -> NaN factors)\n"
         "    if _TAIL_REG > 0.0 and tag:\n"
         "        accum_tail_penalty(s_BHKK, rank, tag)\n"),
    ], "rwkv_ops.py")

    # the QAT entry point: the new kwarg + the fused-rank-1 refusal
    apply(ops, [
        ("    state_qmax: float,\n"
         "    lowrank_rank: int = 0,\n"
         "    lowrank_fqmax: float = float(\"inf\"),\n"
         ") -> Tensor:",
         "    state_qmax: float,\n"
         "    lowrank_rank: int = 0,\n"
         "    lowrank_fqmax: float = float(\"inf\"),\n"
         "    tail_tag: str = \"\",\n"
         ") -> Tensor:"),
        ("    if _fused and lowrank_rank == 1 and lowrank_fqmax != float(\"inf\"):\n"
         "        maybe_upload_pq_codebook()",
         "    if _fused and lowrank_rank == 1 and lowrank_fqmax != float(\"inf\"):\n"
         "        if _TAIL_REG > 0.0 and tail_tag:\n"
         "            raise RuntimeError(\n"
         "                f\"RWKV_QAT_TAIL_PENALTY={_TAIL_REG} is set but the {tail_tag!r} stream's rank-1 \"\n"
         "                \"truncation would run in the FUSED kernel, where the pre-truncation state is \"\n"
         "                \"not visible, so the tail penalty would silently skip it. Set \"\n"
         "                \"RWKV_QAT_FUSED=0 for tail-penalty runs: that is the documented bit-parity \"\n"
         "                \"Python twin of the fused kernels, so the numbers stay comparable.\"\n"
         "            )\n"
         "        maybe_upload_pq_codebook()"),
        ("        if lowrank_rank > 0:  # rank-r truncation (+ factor quant) -- the low-rank deploy analog\n"
         "            next_state_BHKK = fake_lowrank_state(next_state_BHKK, lowrank_rank, lowrank_fqmax)",
         "        if lowrank_rank > 0:  # rank-r truncation (+ factor quant) -- the low-rank deploy analog\n"
         "            next_state_BHKK = fake_lowrank_state(\n"
         "                next_state_BHKK, lowrank_rank, lowrank_fqmax, tail_tag)"),
    ], "rwkv_ops.py (QAT entry)")

    # ---- 2. rwkv_model.py: import the lever, keep the stream tag, pass it --------------------
    apply(model, [
        ("from rwkv.model.rwkv_ops import RWKV7_WKV, RWKV7_WKV_Stateful, reference_rwkv7, quant_aware_rwkv7",
         "from rwkv.model.rwkv_ops import (\n"
         "    RWKV7_WKV, RWKV7_WKV_Stateful, reference_rwkv7, quant_aware_rwkv7,\n"
         "    _TAIL_REG,\n"
         ")"),
        ("        self.rank1_reg = _RANK1_REG",
         "        self.rank1_reg = _RANK1_REG\n"
         "        # Tail-energy penalty (RWKV_QAT_TAIL_PENALTY, 2026-11-22): the QAT state path needs the\n"
         "        # STREAM name so the penalty can be attributed per stream (card vs note), and an empty tag\n"
         "        # means \"not training\", so an eval pass accumulates nothing and adds no loss term.\n"
         "        _tail_stream = config.stream_name\n"
         "        if _tail_stream.endswith(\"_id\"):\n"
         "            _tail_stream = _tail_stream[:-3]\n"
         "        self.tail_tag = _tail_stream\n"
         "        self.tail_reg = _TAIL_REG"),
        ("                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT,\n"
         "                self.state_qmax, self.state_lowrank_rank, self.state_lowrank_fqmax,\n"
         "            )",
         "                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT,\n"
         "                self.state_qmax, self.state_lowrank_rank, self.state_lowrank_fqmax,\n"
         "                tail_tag=(self.tail_tag if (self.training and self.tail_reg > 0.0) else \"\"),\n"
         "            )"),
    ], "rwkv_model.py")

    # ---- 3. srs_model.py: the loss term ------------------------------------------------------
    apply(srs, [
        ("from rwkv.model.rwkv_model import RWKV7, take_rank1_penalty",
         "from rwkv.model.rwkv_model import RWKV7, take_rank1_penalty\n"
         "from rwkv.model.rwkv_ops import take_tail_penalty\n"
         "from rwkv.model.rwkv_ops import _TAIL_REG as _TAIL_REG_LAMBDA"),
        ("            self.rank1_reg_lambda = _RANK1_REG_LAMBDA",
         "            self.rank1_reg_lambda = _RANK1_REG_LAMBDA\n"
         "            # module-level global -> instance attribute, same TorchScript reason as above\n"
         "            self.tail_reg_lambda = _TAIL_REG_LAMBDA"),
        ("        _r1 = take_rank1_penalty()\n"
         "        if _r1 is not None:\n"
         "            loss_avg = loss_avg + self.rank1_reg_lambda * _r1",
         "        _r1 = take_rank1_penalty()\n"
         "        if _r1 is not None:\n"
         "            loss_avg = loss_avg + self.rank1_reg_lambda * _r1\n"
         "        # RWKV_QAT_TAIL_PENALTY (2026-11-22): the truncated tail energy of the rank-truncated\n"
         "        # streams' own states, accumulated per stream during their forward. Drains to None when\n"
         "        # the lever is off or nothing accumulated, so the default path adds no term.\n"
         "        _tl = take_tail_penalty()\n"
         "        if _tl is not None:\n"
         "            loss_avg = loss_avg + self.tail_reg_lambda * _tl"),
    ], "srs_model.py")

    # ---- 4. ta3.py: TA3's exact merge must refuse the new lever too --------------------------
    apply(ta3, [
        ("    if float(getattr(model, \"rank1_reg_lambda\", 0.0) or 0.0) != 0.0:\n"
         "        problems.append(\"RWKV_QAT_RANK1_REG\")",
         "    if float(getattr(model, \"rank1_reg_lambda\", 0.0) or 0.0) != 0.0:\n"
         "        problems.append(\"RWKV_QAT_RANK1_REG\")\n"
         "    if float(getattr(model, \"tail_reg_lambda\", 0.0) or 0.0) != 0.0:\n"
         "        problems.append(\"RWKV_QAT_TAIL_PENALTY\")"),
    ], "ta3.py")

    # ---- 5. qat_decay.py: a candidate may request the Python per-step loop -------------------
    apply(decay, [
        ("    if not args.no_quant:\n        env[\"RWKV_QAT_FUSED\"] = \"1\"",
         "    if not args.no_quant:\n"
         "        # The fused kernel is the default for QAT runs. A CANDIDATE may ask for the Python per-step\n"
         "        # loop instead (RWKV_QAT_FUSED=0), which is bit-parity with the fused kernels: the\n"
         "        # tail-energy penalty needs it, because the fused rank-1 kernel truncates inside the kernel\n"
         "        # where the pre-truncation state is invisible. Only a candidate can do this -- a stray value\n"
         "        # in the parent shell must not silently slow every run down.\n"
         "        env[\"RWKV_QAT_FUSED\"] = str(env_extra.get(\"RWKV_QAT_FUSED\", \"1\"))\n"
         "        if env[\"RWKV_QAT_FUSED\"] != \"1\":\n"
         "            print(f\"[qat_decay] RWKV_QAT_FUSED={env['RWKV_QAT_FUSED']} requested by candidate \"\n"
         "                  f\"{args.candidate}: the Python per-step QAT loop (bit-parity twin)\")"),
    ], "qat_decay.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
