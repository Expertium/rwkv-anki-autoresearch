import pathlib
P = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\RESULTS.md")
t = P.read_bytes().decode("utf-8").split("\n")
bad = 0
for n in range(64, 91):
    line = t[n - 1]
    cells = line.split("|")
    if len(cells) != 14:
        print("LINE", n, "cells", len(cells) - 2, "::", line[:80])
        bad += 1
print("checked lines 64-90 (hdr, sep, rows 0..23), malformed:", bad)
print("last row text:", t[89][:40])
