import pathlib, re
base = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
for f in ("it36rankprec_chain.py", "it36longdecay_chain.py", "it36shrink_chain.py", "it36corrupt_chain.py"):
    t = (base / f).read_text(encoding="utf-8")
    i = t.find("POINTS")
    print("==", f)
    print(t[i:i + 900])
    print()
