"""Kill the wedged it36shrink l3i4s4n4 training and the five waiting chain processes, delete their
locks, so all five can be relaunched with the 2026-11-22 wedge watchdog. (11:20)

Nothing is lost: the wedged point has no result (its sweep JSON was never written), so the relaunched
chain reruns it from scratch. Every finished point of the shrink chain, and every earlier chain, keeps
its result and is skipped by the chain.
"""
import pathlib
import sys

import psutil

RUN = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
CHAINS = ("it36shrink_chain.py", "it36corrupt_chain.py", "it36equalbits_chain.py",
          "it36combo_chain.py", "it36corrupt2_chain.py")
WORK = ("qat_decay.py", "rwkv.train_rwkv", "quant_sweep.py", "eval_sharded.py")


def cmdof(p):
    try:
        return " ".join(p.info["cmdline"] or [])
    except Exception:
        return ""


print("=== killing the wedged training tree (l3i4s4n4) ===")
victims = []
for p in psutil.process_iter(["pid", "cmdline"]):
    cl = cmdof(p)
    if "it36l3i4s4n4" in cl and any(w in cl for w in WORK):
        victims.append(p)
    elif "multiprocessing" in cl:                      # the fetch worker of that trainer
        try:
            par = psutil.Process(p.info["pid"]).parent()
            if par is not None and "qat_it36l3i4s4n4" in " ".join(par.cmdline() or []):
                victims.append(p)
        except Exception:
            pass
for p in victims:
    cl = cmdof(p)
    try:
        p.kill()
        print(f"  killed pid {p.info['pid']}: {cl[:120]}")
    except Exception as exc:
        print(f"  pid {p.info['pid']} kill failed ({exc}): {cl[:100]}")

print("\n=== killing the five waiting chain processes ===")
for name in CHAINS:
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = cmdof(p)
        if name in cl:
            try:
                p.kill()
                print(f"  killed pid {p.info['pid']}: {cl[:120]}")
            except Exception as exc:
                print(f"  pid {p.info['pid']} kill failed ({exc})")

print("\n=== leftover processes matching the five chains or l3i4s4n4 ===")
psutil.wait_procs([], timeout=0)
for p in psutil.process_iter(["pid", "cmdline"]):
    cl = cmdof(p)
    if any(n in cl for n in CHAINS) or "it36l3i4s4n4" in cl:
        print(f"  STILL ALIVE pid {p.info['pid']}: {cl[:120]}")

print("\n=== deleting stale lock files ===")
for name in CHAINS:
    lock = RUN / name.replace("_chain.py", "_chain.lock")
    if lock.exists():
        lock.unlink()
        print(f"  deleted {lock.name}")
    else:
        print(f"  (no {lock.name})")
print("done")