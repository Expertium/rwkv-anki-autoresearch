"""Measure per-step overheads the on-device profiler (profile_train.py) OMITTED but the REAL
train loop pays: (1) torch.cuda.empty_cache() EVERY step for the first 1000 steps (train_rwkv.py
~498-499) -- for a <=1000-step run that is EVERY step; (2) the input batch .to(device) H2D.

profile_fetch.py already showed get()~5ms and .to()~5ms (fetching is hidden). This isolates whether
the empty_cache-per-step is a big, cheap-to-recover chunk of the real 1087 ms/step (vs 588 ceiling).

GPU-only (caches batches on-device), single process -> won't disturb the CPU-bound build.
Run: PYTHONPATH=. RWKV_NO_JIT=1 .venv\\Scripts\\python.exe scratchpad/profile_emptycache.py --config scratchpad/champ_off1_ws.toml --db train_db_sc8k
"""
import argparse
import time
from argparse import Namespace

import tomli
import torch
import lmdb

from rwkv.prepare_batch import prepare, get_data
from rwkv.model.srs_model import SrsRWKV
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.train_rwkv import get_optimizer, get_groups, transfer_child_grad_to_master


def load_config(path):
    with open(path, "rb") as f:
        a = tomli.load(f)
    a["DTYPE"] = torch.bfloat16 if a.get("DTYPE") == "bfloat16" else torch.float32
    a["DEVICE"] = torch.device(a.get("DEVICE", "cuda"))
    return Namespace(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--db", default=None)
    ap.add_argument("--nbatch", type=int, default=8)
    ap.add_argument("--measure", type=int, default=40)
    args = ap.parse_args()

    config = load_config(args.config)
    db = args.db or config.TRAIN_DATASET_LMDB_PATH
    size = getattr(config, "TRAIN_DATASET_LMDB_SIZE", 80_000_000_000)
    device, dtype = config.DEVICE, config.DTYPE
    MAX = config.MAX_TRAIN_GLOBAL_LEN
    print(f"db={db} MAX={MAX} dtype={dtype}", flush=True)

    master = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(device)
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).selective_cast(dtype).to(device)
    optimizer = get_optimizer(config, master)
    model.copy_downcast_(master, dtype=dtype)

    users = list(range(config.TRAIN_USERS_START, config.TRAIN_USERS_END + 1))
    groups = get_groups(db, size, MAX, users=users)
    env = lmdb.open(db, map_size=size)
    nb = min(args.nbatch, len(groups))
    cpu_batches, gpu_batches = [], []
    with env.begin(write=False) as txn:
        for i in range(nb):
            pb = prepare([get_data(txn, k, device="cpu") for k in groups[i]], target_len=MAX, seed=1234)
            cpu_batches.append(pb)
            gpu_batches.append(pb.to(device))
    env.close()
    print(f"cached {nb} batches (B={[len(groups[i]) for i in range(nb)]})", flush=True)

    def step(b):
        model.copy_downcast_(master, dtype=dtype)
        model.train()
        stats = model.get_loss(b)
        stats.average_loss.backward()
        transfer_child_grad_to_master(master=master, child=model)
        torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
        optimizer.step(); optimizer.zero_grad()

    M = args.measure
    # warmup
    for i in range(6):
        step(gpu_batches[i % nb])
    torch.cuda.synchronize()

    def bench(label, empty_cache=False, from_cpu=False):
        torch.cuda.synchronize(); t = time.perf_counter()
        for i in range(M):
            if empty_cache:
                torch.cuda.empty_cache()
            b = cpu_batches[i % nb].to(device) if from_cpu else gpu_batches[i % nb]
            step(b)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        print(f"[{label:52s}] {1000*dt/M:8.2f} ms/step  {M/dt:6.2f} steps/s", flush=True)
        return 1000 * dt / M

    base = bench("on-device batch, NO empty_cache (profile_train ceiling)")
    ec = bench("on-device batch, empty_cache() EVERY step", empty_cache=True)
    cpu = bench("CPU batch -> .to() each step", from_cpu=True)
    cpu_ec = bench("CPU batch -> .to() + empty_cache EVERY step (REAL first-1000)", empty_cache=True, from_cpu=True)
    print(f"\n  empty_cache per-step cost          ~{ec - base:7.2f} ms", flush=True)
    print(f"  input .to() per-step cost          ~{cpu - base:7.2f} ms", flush=True)
    print(f"  REAL-loop (first 1000 steps) total  {cpu_ec:7.2f} ms/step vs {base:.2f} ceiling", flush=True)


if __name__ == "__main__":
    main()
