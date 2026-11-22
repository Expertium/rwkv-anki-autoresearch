"""One-shot: fix the deadlocking BUSY_PAT in the three waiting it36 chains (2026-11-21 22:30).

Diagnosis: `it36ladder_lib.wait_for_prev` requires BOTH the predecessor's end marker AND
`running(BUSY_PAT) == 0`. Each 09:15 chain's BUSY_PAT listed its two *sibling* chains, and each sibling
is a live waiting process, so the three-way wait never resolved (GPU idle since 12:25).

Fix: BUSY_PAT matches real GPU work only. Sequencing between chains is already carried by the end
markers, and `IDLE_PROCEED_MIN` (12 consecutive idle checks) remains the leftover-run safety net.
`qat_decay.py` and `eval_sharded.py` are added because they own the GPU during setup/eval phases.

The venv `python.exe` is a shim, so one chain shows two processes; that is why a chain must never
count its own script, and now never counts a waiting sibling either.
"""
import pathlib
import re

NEW = ('BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|'
       'rust_speed.py|rust_grid.py|rwkv-infer")')

FILES = ["it36rankprec_chain.py", "it36longdecay_chain.py", "it36shrink_chain.py"]
RUN = pathlib.Path("scratchpad/quant_plan")

for name in FILES:
    p = RUN / name
    with p.open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    old = re.search(r"BUSY_PAT = \(.*?\)", text, re.S)
    if not old:
        raise SystemExit(f"{name}: no BUSY_PAT found")
    if "it36" in old.group(0) is False and "chain.py" not in old.group(0):
        print(f"{name}: already fixed, skipping")
        continue
    text = text[:old.start()] + NEW + text[old.end():]
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"{name}: BUSY_PAT -> {NEW}")
    # prove the file still parses and carries no sibling chain name
    src = p.read_text(encoding="utf-8")
    assert "chain.py" not in src, name
    compile(src, str(p), "exec")
print("all three files rewritten and compiled clean")
