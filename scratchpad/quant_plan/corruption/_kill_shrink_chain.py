"""Kill the running it36shrink chain and its whole tree before the reorder relaunch (11:30)."""
import psutil

NAME = "it36shrink_chain.py"
roots = []
for p in psutil.process_iter(["pid", "cmdline"]):
    cl = " ".join(p.info["cmdline"] or [])
    if NAME in cl and "cmd.exe" not in cl:
        roots.append(p)
for p in roots:
    kids = p.children(recursive=True)
    for k in kids:
        try:
            print(f"kill child {k.pid}: {' '.join(k.cmdline() or [])[:100]}")
            k.kill()
        except Exception as exc:
            print(f"  child {k.pid}: {exc}")
    try:
        print(f"kill root {p.info['pid']}: {' '.join(p.cmdline() or [])[:100]}")
        p.kill()
    except Exception as exc:
        print(f"  root {p.info['pid']}: {exc}")

lock = psutil.__dict__  # noqa: F841  (keep the import obviously used)
import pathlib
lk = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\it36shrink_chain.lock")
if lk.exists():
    lk.unlink()
    print("deleted it36shrink_chain.lock")
print("done")