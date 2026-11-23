from pathlib import Path

P = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\corruption\_record_row42.py")
raw = P.read_bytes()


def rep(old: bytes, new: bytes, label: str) -> bytes:
    global raw
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    raw = raw.replace(old, new, 1)
    print(f"{label}: patched")
    return raw


# `l2i8s4` is not in rust_grid_it36int6shift.json (that file holds only `plain` and
# `l2i6s4n1i8s3`); its absolutes come from the fixed-basis wide grid, which was run on the
# same 7 held-out long histories.
rep(b'    grid = json.loads(GRID.read_text(encoding="utf-8"))',
    b'    grid = json.loads(GRID.read_text(encoding="utf-8"))\r\n'
    b'    gridb = json.loads((RUN / "rust_grid_it36fbwide_b1.json").read_text(encoding="utf-8"))',
    "load gridb")
rep(b"grid['l2i8s4']['means']['ahead']", b"gridb['l2i8s4']['means']['ahead']", "l2i8s4 ahead")
rep(b"grid['l2i8s4']['means']['imm']", b"gridb['l2i8s4']['means']['imm']", "l2i8s4 imm")

P.write_bytes(raw)
print("done")
