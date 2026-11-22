"""Patch the eval/QAT source for FIXED-BASIS state coding (approach 1, Andrew 2026-11-22 20:51).

Adds an env-gated, INERT-BY-DEFAULT branch to `rwkv/model/rwkv_ops.py` plus two kwargs at the
`quant_aware_rwkv7` call site in `rwkv/model/rwkv_model.py`:

  RWKV_FIXEDBASIS_PATH=<file.npz>   -> the state round-trip becomes P (quant(P^T W Q)) Q^T with the
                                       fixed K x m bases P, Q stored per (stream, layer) in that file
                                       (keys `P__<stream>__L<layer>` / `Q__<stream>__L<layer>`, with a
                                       `__ALL` per-stream fallback). The `rank` slot of the QAT scope
                                       carries m, and its int-N level quantizes the coefficient block.
  unset                             -> byte-identical to before (no new imports, no new work).

Why the source and not a monkey-patch: every other state lever (QAT scopes, EF, compand, ALS, PQ, the
tail penalty) lives here for the same reason - the deploy engine and the eval path must be changed
together, and a scratchpad patch cannot reach the eval that `quant_sweep.py` spawns.

Line endings matter: `rwkv_ops.py` is CRLF-only and `rwkv_model.py` is LF-only, so this patcher works
on raw bytes with per-file newline constants and a uniqueness assert on every anchor.

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_add_fixedbasis.py [--revert]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "rwkv" / "model" / "rwkv_ops.py"
MODEL = ROOT / "rwkv" / "model" / "rwkv_model.py"

# ---------------------------------------------------------------- rwkv_ops.py (CRLF)
NL = "\r\n"
A_FQF = (b"    return torch.round(f / scale).clamp(-qmax, qmax) * scale\r\n\r\n\r\n"
         b"# ---- TAIL-ENERGY PENALTY")
A_SIG = b'    tail_tag: str = "",\r\n) -> Tensor:\r\n    """Per-step reference WKV'
A_FUSED1 = b'if _fused and lowrank_rank == 1 and lowrank_fqmax != float("inf"):\r\n'
A_CALL = (b'        if lowrank_rank > 0:  # rank-r truncation (+ factor quant) -- the low-rank '
          b'deploy analog\r\n'
          b'            next_state_BHKK = fake_lowrank_state(\r\n'
          b'                next_state_BHKK, lowrank_rank, lowrank_fqmax, tail_tag)\r\n')

BLOCK = """# ---- FIXED-BASIS STATE CODING (RWKV_FIXEDBASIS_PATH=<file.npz>, 2026-11-22) -------------------
# WHY. The rank-r scheme stores U_r and V_r, i.e. 2*r*K numbers per layer, because it re-describes a
# BASIS at every step. The state is a decayed sum of the model's OWN key/value projections, so its
# dominant directions are shared across users and time, and a FIXED basis carries them in far fewer
# numbers: store the m x m coefficient block C = P^T W Q (P, Q fixed K x m) and reconstruct P C Q^T.
# That is m*m numbers instead of 2*r*K, and the deploy cost is two small matmuls, so it is FASTER than
# the Gram/power-iteration compress as well.
#
# MEASURED FIRST (corruption/state_probe.py + analyze_state_subspace.py, users 7213/7044/6960, row
# 17's own checkpoint, 1,347 states per layer): at m=8 the fixed basis captures 0.9998-0.9999 of
# E||W||^2 on every stream for 512 bits per layer, where the shipped note rank-1 (768 bits) captures
# 0.877-0.976 and the card rank-2 (1,280 bits) captures 0.9998. At m=4 it still captures 0.994-0.998
# for 128 bits per layer. Andrew approved this direction 2026-11-22 20:51.
#
# THE BASIS IS A MODEL-SIDE OBJECT, NOT STATE: a fixed linear transform (a codec basis), not a
# codebook - no discrete entries, no nearest-neighbour lookup, no per-user vectors - stored once per
# (stream, layer) in the named file and costing no state bits. Inert unless the path is set.
_FB_PATH = os.environ.get("RWKV_FIXEDBASIS_PATH", "")
_FB_CACHE: dict = {}


def fixedbasis_loaded() -> bool:
    return bool(_FB_PATH)


def _fb_lookup(stream: str, layer: int, m: int):
    \"\"\"(P, Q) for this (stream, layer) at m modes per side, or None when the file has no entry.\"\"\"
    if not _FB_PATH or not stream:
        return None
    if not _FB_CACHE:
        import numpy as _np
        with _np.load(_FB_PATH) as z:
            for k in z.files:
                _FB_CACHE[k] = torch.from_numpy(z[k].astype("float32"))
    for key in (f"P__{stream}__L{layer}", f"P__{stream}__ALL"):
        p = _FB_CACHE.get(key)
        q = _FB_CACHE.get(key.replace("P__", "Q__", 1))
        if p is not None and q is not None and p.shape[1] >= m:
            return p[:, :m], q[:, :m]
    return None


def fake_fixedbasis_state(s_BHKK: Tensor, m: int, qmax: float, stream: str = "",
                          layer: int = -1) -> Tensor:
    \"\"\"STE fixed-basis round-trip W -> P (quant(P^T W Q)) Q^T, the deploy analog of the rank-r path.

    `m` is the modes per side (the QAT scope's rank slot) and `qmax` quantizes the m x m coefficient
    block with the SAME symmetric per-matrix int-N round-trip the factors use. Falls back to
    `fake_lowrank_state` when the basis file has no entry for this (stream, layer), so a stream whose
    basis was never fitted keeps the old behavior instead of silently changing.
    \"\"\"
    if m <= 0:
        return s_BHKK
    basis = _fb_lookup(stream, layer, m)
    if basis is None:
        return fake_lowrank_state(s_BHKK, m, qmax, "")
    p, q = basis
    s_BHKK = _sanitize_state(s_BHKK)  # NaN/inf safeguard: keep the coefficients finite
    B, H, K, _ = s_BHKK.shape
    p = p.to(device=s_BHKK.device, dtype=torch.float32)
    q = q.to(device=s_BHKK.device, dtype=torch.float32)
    with torch.no_grad():
        s = s_BHKK.reshape(B * H, K, K).float()
        c = torch.einsum("ik,bij,jl->bkl", p, s, q)          # (BH, m, m) coefficients
        c = _fake_quant_factor(c, qmax)                      # symmetric int-N, per (BH) matrix
        recon = torch.einsum("ik,bkl,jl->bij", p, c, q).reshape(B, H, K, K).to(s_BHKK.dtype)
    return s_BHKK + (recon - s_BHKK).detach()


""".replace("\n", NL)

NEW_SIG = ('    tail_tag: str = "",\r\n'
           '    stream: str = "",\r\n'
           '    layer: int = -1,\r\n'
           ') -> Tensor:\r\n    """Per-step reference WKV')

NEW_FUSED1 = ('if _fused and lowrank_rank == 1 and lowrank_fqmax != float("inf") '
              'and not _FB_PATH:\r\n')

NEW_CALL = ('        if lowrank_rank > 0:  # rank-r truncation, or the fixed-basis round-trip\r\n'
            '            if _FB_PATH:  # RWKV_FIXEDBASIS_PATH: fixed P, Q instead of a fresh SVD\r\n'
            '                next_state_BHKK = fake_fixedbasis_state(\r\n'
            '                    next_state_BHKK, lowrank_rank, lowrank_fqmax, stream, layer)\r\n'
            '            else:\r\n'
            '                next_state_BHKK = fake_lowrank_state(\r\n'
            '                    next_state_BHKK, lowrank_rank, lowrank_fqmax, tail_tag)\r\n')

# ---------------------------------------------------------------- rwkv_model.py (LF)
A_CALLSITE = (b'                tail_tag=(self.tail_tag if (self.training and self.tail_reg > 0.0) '
              b'else ""),\n')
NEW_CALLSITE = (b'                tail_tag=(self.tail_tag if (self.training and self.tail_reg > 0.0) '
                b'else ""),\n'
                b'                # fixed-basis state coding (RWKV_FIXEDBASIS_PATH) needs the stream\n'
                b'                # and layer to pick its fixed P, Q; both are inert when unset.\n'
                b'                stream=self.tail_tag, layer=self.layer_id,\n')


def patch(raw: bytes, old: bytes, new: bytes, what: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{what}]: anchor found {n} times, expected 1")
    return raw.replace(old, new, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    ops = OPS.read_bytes()
    model = MODEL.read_bytes()

    if args.revert:
        if b"_FB_PATH" in ops:
            raise SystemExit("revert by hand: git checkout -- rwkv/model/rwkv_ops.py "
                             "rwkv/model/rwkv_model.py")
        print("nothing to revert (no RWKV_FIXEDBASIS code found)")
        return 0

    if b"_FB_PATH" in ops:
        print("rwkv_ops.py already patched")
    else:
        ops = patch(ops, A_FQF, A_FQF.replace(b"# ---- TAIL-ENERGY PENALTY",
                                             BLOCK.encode("utf-8") + b"# ---- TAIL-ENERGY PENALTY"),
                    "fixed-basis block")
        ops = patch(ops, A_SIG, NEW_SIG.encode("utf-8"), "quant_aware_rwkv7 signature")
        ops = patch(ops, A_FUSED1, NEW_FUSED1.encode("utf-8"), "fused rank-1 guard")
        ops = patch(ops, A_CALL, NEW_CALL.encode("utf-8"), "lowrank call site")
        # document the new kwargs in the function docstring
        old_doc = (b"    If lowrank_rank>0 the state is rank-r truncated (+ optional int-N factor "
                   b"quant) instead of full\r\n")
        new_doc = (b"    If lowrank_rank>0 the state is rank-r truncated (+ optional int-N factor "
                   b"quant) instead of full\r\n"
                   b"    int-N quant. With RWKV_FIXEDBASIS_PATH set, the same slot instead carries "
                   b"the modes-per-side\r\n"
                   b"    m of a FIXED basis (see fake_fixedbasis_state) and `stream`/`layer` select "
                   b"its P, Q.\r\n")
        ops = patch(ops, old_doc, new_doc, "docstring note")
        OPS.write_bytes(ops)
        print(f"patched {OPS.relative_to(ROOT)} ({len(ops)} bytes)")

    if b"stream=self.tail_tag" in model:
        print("rwkv_model.py already patched")
    else:
        model = patch(model, A_CALLSITE, NEW_CALLSITE, "quant_aware_rwkv7 call site")
        MODEL.write_bytes(model)
        print(f"patched {MODEL.relative_to(ROOT)} ({len(model)} bytes)")

    # compile check
    import ast
    ast.parse(OPS.read_text(encoding="utf-8"))
    ast.parse(MODEL.read_text(encoding="utf-8"))
    print("both files parse OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
