"""One-shot: make a fixed-basis lookup MISS loud instead of silent.

`fake_fixedbasis_state` falls back to `fake_lowrank_state` when the basis file has no entry for this
(stream, layer). That is the right behavior for a stream whose basis was never fitted, but it would
also silently invalidate a screen if the key were wrong (e.g. the stream name or the layer id did not
match the fit). So a miss prints ONCE per (stream, layer), and a `--debug` env prints every hit.

CRLF: rwkv_ops.py is CRLF-only, so this works on raw bytes.

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_patch_fb_warn.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "rwkv" / "model" / "rwkv_ops.py"

OLD = (b'    for key in (f"P__{stream}__L{layer}", f"P__{stream}__ALL"):\r\n'
       b'        p = _FB_CACHE.get(key)\r\n'
       b'        q = _FB_CACHE.get(key.replace("P__", "Q__", 1))\r\n'
       b'        if p is not None and q is not None and p.shape[1] >= m:\r\n'
       b'            return p[:, :m], q[:, :m]\r\n'
       b'    return None\r\n')

NEW = (b'    for key in (f"P__{stream}__L{layer}", f"P__{stream}__ALL"):\r\n'
       b'        p = _FB_CACHE.get(key)\r\n'
       b'        q = _FB_CACHE.get(key.replace("P__", "Q__", 1))\r\n'
       b'        if p is not None and q is not None and p.shape[1] >= m:\r\n'
       b'            if os.environ.get("RWKV_FIXEDBASIS_DEBUG", "0") == "1":\r\n'
       b'                print(f"[fixedbasis] {stream}/L{layer} <- {key} (m={m})", flush=True)\r\n'
       b'            return p[:, :m], q[:, :m]\r\n'
       b'    # A miss silently falls back to the rank-r path, which would invalidate a screen: say so\r\n'
       b'    # once per (stream, layer) instead. A basis file that simply does not cover a stream is\r\n'
       b'    # fine - but then that stream is NOT running the fixed-basis round-trip.\r\n'
       b'    if key not in _FB_MISSED:\r\n'
       b'        _FB_MISSED.add(key)\r\n'
       b'        print(f"[fixedbasis] NO BASIS for {stream}/L{layer} in {_FB_PATH} -- falling back to '
       b'the rank-r path for this stream", flush=True)\r\n'
       b'    return None\r\n')


def main() -> int:
    raw = OPS.read_bytes()
    if b"_FB_MISSED" in raw:
        print("already patched")
        return 0
    n = raw.count(OLD)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times, expected 1")
    raw = raw.replace(OLD, NEW, 1)
    old_decl = b"_FB_CACHE: dict = {}\r\n"
    if raw.count(old_decl) != 1:
        raise SystemExit("ABORT: _FB_CACHE declaration not found exactly once")
    raw = raw.replace(old_decl, old_decl + b"_FB_MISSED: set = set()\r\n", 1)
    OPS.write_bytes(raw)
    import ast
    ast.parse(OPS.read_text(encoding="utf-8"))
    print("patched rwkv_ops.py: fixed-basis misses are now loud (RWKV_FIXEDBASIS_DEBUG=1 for hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
