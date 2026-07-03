"""Launcher: run rwkv.train_rwkv's main in a thread with a 64 MB stack.

torch.compile (Dynamo) tracing recurses deeply in native code; Windows' default 1 MB
main-thread stack overflows (0xC00000FD) where Linux's 8 MB survives. threading.stack_size
only applies to NEW threads, so main runs in one.

NOTE: imports rwkv.train_rwkv as a normal module (NOT runpy run_name="__main__") so the
multiprocessing data-fetcher workers can pickle their targets by real module path; the
__main__ guard below keeps spawn's re-import of this script side-effect-free.

Usage: python scratchpad/train_bigstack.py --config <toml>   (same args as -m rwkv.train_rwkv)
"""
import sys
import threading

if __name__ == "__main__":
    threading.stack_size(64 * 1024 * 1024)   # Windows rejects >=256MB; 64MB = 8x Linux default
    sys.setrecursionlimit(100000)

    from rwkv import train_rwkv
    from rwkv.parse_toml import parse_toml

    config = parse_toml()
    rc = []

    def run():
        try:
            train_rwkv.main(config)
            rc.append(0)
        except SystemExit as e:
            rc.append(int(e.code or 0))

    t = threading.Thread(target=run)
    t.start()
    t.join()
    sys.exit(rc[0] if rc else 1)
