import json
import pathlib
import struct
import sys

paths = sys.argv[1:]
root = pathlib.Path(".")
for p in paths:
    f = root / p
    if not f.exists():
        print(f"{p} MISSING")
        continue
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    shapes = {k: (v["dtype"], v["shape"]) for k, v in hdr.items() if k != "__metadata__"}
    print(f"{p}  meta={hdr.get('__metadata__')}")
    for k, v in sorted(shapes.items()):
        print(f"    {k} {v}")
