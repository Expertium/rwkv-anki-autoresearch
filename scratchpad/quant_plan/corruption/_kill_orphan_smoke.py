"""Kill the orphaned l2i8s4n3 smoke tree left by the chain kill (its parent chain died first), 11:22."""
import psutil

ROOTS = (28528, 27416)          # the qat_decay shim + real process of the orphaned smoke
for pid in ROOTS:
    try:
        p = psutil.Process(pid)
    except Exception as exc:
        print(f"pid {pid} gone ({exc})")
        continue
    kids = p.children(recursive=True)
    for k in kids:
        try:
            print(f"killing child {k.pid}: {' '.join(k.cmdline() or [])[:110]}")
            k.kill()
        except Exception as exc:
            print(f"  child {k.pid} kill failed: {exc}")
    try:
        print(f"killing root {pid}: {' '.join(p.cmdline() or [])[:110]}")
        p.kill()
    except Exception as exc:
        print(f"  root {pid} kill failed: {exc}")

psutil.wait_procs([], timeout=0)
print("\n=== remaining work processes ===")
for p in psutil.process_iter(["pid", "cmdline"]):
    cl = " ".join(p.info["cmdline"] or [])
    if any(k in cl for k in ("qat_decay.py", "rwkv.train_rwkv", "quant_sweep.py", "eval_sharded.py",
                             "it36shrink_chain", "it36corrupt_chain", "it36equalbits_chain",
                             "it36combo_chain", "it36corrupt2_chain")):
        print(f"  pid {p.info['pid']}: {cl[:130]}")
print("done")