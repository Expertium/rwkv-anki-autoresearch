"""Give the Python fixed-basis path a PER-STREAM scope (RWKV_FIXEDBASIS_SCOPE).

Why: the first fb candidates put BOTH card and note on the fixed basis. The measured leave-one-user-out
capture (corruption/fixedbasis_l2i8s4all.json) says the card is the wrong place for it - row 17's card
is EXACT at rank 2 for 1,280 bits, while the best fixed basis on the card needs m=16 (2,304 bits) to
reach only 0.96 held out. The note is the interesting side: the shipped note rank-1 captures
0.904-0.951 and PASSES the gate, so a codec that reaches ~0.88-0.92 for 512 bits/layer is worth a real
log-loss measurement. That candidate keeps the card on the rank-r path and switches only the note, so
the path needs a per-stream switch - the same shape the Rust engine now has
(RWKV_STATE_FIXEDBASIS_SCOPE).

Empty scope = every stream with a basis entry (the previous behaviour, so the existing fb candidates
are unchanged).
"""
from pathlib import Path

OPS = Path(__file__).resolve().parents[3] / "rwkv" / "model" / "rwkv_ops.py"
EOL = b"\r\n"

OLD_DECL = b'''_FB_PATH = os.environ.get("RWKV_FIXEDBASIS_PATH", "")
_FB_CACHE: dict = {}
_FB_MISSED: set = set()


def fixedbasis_loaded() -> bool:
    return bool(_FB_PATH)
'''

NEW_DECL = b'''_FB_PATH = os.environ.get("RWKV_FIXEDBASIS_PATH", "")
# WHICH STREAMS use it. Empty = every stream that has a basis entry (the both-stream screens). A comma
# list ("note") keeps the other streams on the rank-r path - the shape a "card keeps rank 2, note uses
# the fixed basis" candidate needs. Mirrors the Rust RWKV_STATE_FIXEDBASIS_SCOPE.
_FB_SCOPE = {s.strip() for s in os.environ.get("RWKV_FIXEDBASIS_SCOPE", "").split(",") if s.strip()}
_FB_CACHE: dict = {}
_FB_MISSED: set = set()


def fixedbasis_loaded() -> bool:
    return bool(_FB_PATH)


def _fb_wanted(stream: str) -> bool:
    """Is THIS stream on the fixed-basis path? False = it keeps the rank-r round-trip."""
    return bool(_FB_PATH) and (not _FB_SCOPE or stream in _FB_SCOPE)
'''

OLD_LOOKUP = b'''    if not _FB_PATH or not stream:
        return None
'''
NEW_LOOKUP = b'''    if not _fb_wanted(stream):
        return None
'''

OLD_CALL = b'''            if _FB_PATH:  # RWKV_FIXEDBASIS_PATH: fixed P, Q instead of a fresh SVD
'''
NEW_CALL = b'''            if _fb_wanted(stream):  # RWKV_FIXEDBASIS_PATH: fixed P, Q instead of a fresh SVD
'''

OLD_FUSED = b'if _fused and lowrank_rank == 1 and lowrank_fqmax != float("inf") and not _FB_PATH:'
NEW_FUSED = b'if _fused and lowrank_rank == 1 and lowrank_fqmax != float("inf") and not _fb_wanted(stream):'

raw = OPS.read_bytes()
for label, old, new in (("decl", OLD_DECL, NEW_DECL), ("lookup", OLD_LOOKUP, NEW_LOOKUP),
                        ("call", OLD_CALL, NEW_CALL), ("fused", OLD_FUSED, NEW_FUSED)):
    ob, nb = old.replace(b"\n", EOL), new.replace(b"\n", EOL)
    n = raw.count(ob)
    if n != 1:
        raise SystemExit(f"{label}: anchor matched {n} times (need 1)")
    raw = raw.replace(ob, nb, 1)
OPS.write_bytes(raw)
print(f"patched {OPS} ({len(raw)} bytes)")
