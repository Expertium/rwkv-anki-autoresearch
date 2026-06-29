"""Profile one GPU training step of SrsRWKV to find the launch-bound bottleneck.

Mirrors train_rwkv.main_loop's per-step body (master fp32 + bf16 child) but fetches batches
SYNCHRONOUSLY (get_data + prepare, no multiprocessing -- Windows spawn hangs in a scratch script).
Caches a handful of real batches on-device, then times:
  (1) per-section breakdown (copy / fwd / bwd / transfer / grad_norm / clip / opt), cuda-synchronized;
  (2) end-to-end steps/sec for the CURRENT body (per-step get_grad_norm + .item() prints) vs an
      OPTIMIZED body (logging-only syncs removed + foreach copy/transfer) -- quantifies the win.

Run: setx-style env then
  .venv\\Scripts\\python.exe scratchpad/profile_train.py --config rwkv/train_rwkv_config_iter46_qat_decay.toml
(LOAD_MODEL is ignored -- random init; timing is value-independent. Set RWKV_NO_JIT=1: JIT is broken
 in torch 2.12.1+cu130, so real training already runs this path.)
"""
import argparse
import time
import tomli
from argparse import Namespace

import torch
import lmdb

from rwkv.prepare_batch import prepare, get_data
from rwkv.model.srs_model import SrsRWKV
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.train_rwkv import get_optimizer, get_groups, transfer_child_grad_to_master, get_grad_norm


def load_config(path):
    with open(path, "rb") as f:
        a = tomli.load(f)
    a["DTYPE"] = torch.bfloat16 if a.get("DTYPE") == "bfloat16" else torch.float32
    a["DEVICE"] = torch.device(a.get("DEVICE", "cuda"))
    return Namespace(**a)


# ---- OLD per-param Python loops (baseline) ----
def old_copy_downcast_(child: SrsRWKV, master: SrsRWKV, dtype):
    from rwkv.model.srs_model import is_excluded
    mp = dict(master.named_parameters())
    with torch.no_grad():
        for name, p in child.named_parameters():
            target = torch.float32 if is_excluded(name) else dtype
            p.data.copy_(mp[name].to(target))


def old_transfer_grad(master: SrsRWKV, child: SrsRWKV):
    mp = dict(master.named_parameters())
    for name, p in child.named_parameters():
        m = mp[name]
        if p.grad is not None:
            with torch.no_grad():
                if m.grad is None:
                    m.grad = torch.zeros_like(m, requires_grad=True)
                m.grad.add_(p.grad.to(torch.float32))
            p.grad.zero_()


# ---- foreach-vectorized variants (the optimization; grouped by dtype for the fast path) ----
def copy_downcast_foreach(child: SrsRWKV, master: SrsRWKV, dtype):
    from rwkv.model.srs_model import is_excluded
    mp = dict(master.named_parameters())
    groups = {}
    with torch.no_grad():
        for name, p in child.named_parameters():
            target = torch.float32 if is_excluded(name) else dtype
            d, s = groups.setdefault(target, ([], []))
            d.append(p.data); s.append(mp[name].data)
        for d, s in groups.values():
            torch._foreach_copy_(d, s)  # copy_ casts fp32->target per dst dtype


def transfer_grad_foreach(master: SrsRWKV, child: SrsRWKV):
    mp = dict(master.named_parameters())
    groups = {}
    with torch.no_grad():
        for name, p in child.named_parameters():
            if p.grad is None:
                continue
            m = mp[name]
            if m.grad is None:
                m.grad = torch.zeros_like(m, requires_grad=True)
            g = groups.setdefault((m.grad.dtype, p.grad.dtype), ([], []))
            g[0].append(m.grad); g[1].append(p.grad)
        for mg, cg in groups.values():
            torch._foreach_add_(mg, cg)   # fp32 += bf16 (casts)
            torch._foreach_zero_(cg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nbatch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--measure", type=int, default=40)
    args = ap.parse_args()

    config = load_config(args.config)
    device, dtype = config.DEVICE, config.DTYPE
    print(f"device={device} dtype={dtype} MAX_TRAIN_GLOBAL_LEN={config.MAX_TRAIN_GLOBAL_LEN}", flush=True)

    master = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(device)
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).selective_cast(dtype).to(device)
    optimizer = get_optimizer(config, master)
    model.copy_downcast_(master, dtype=dtype)
    n_pt = sum(1 for _ in master.parameters())
    print(f"params={sum(p.numel() for p in master.parameters())} param_tensors={n_pt} "
          f"(get_grad_norm = ~{n_pt} .item() syncs/step)", flush=True)

    users = list(range(config.TRAIN_USERS_START, config.TRAIN_USERS_END + 1))
    groups = get_groups(config.TRAIN_DATASET_LMDB_PATH, config.TRAIN_DATASET_LMDB_SIZE,
                        config.MAX_TRAIN_GLOBAL_LEN, users=users)

    # --- synchronous fetch + cache N batches on-device ---
    env = lmdb.open(config.TRAIN_DATASET_LMDB_PATH, map_size=config.TRAIN_DATASET_LMDB_SIZE)
    batches = []
    nb = min(args.nbatch, len(groups))
    with env.begin(write=False) as txn:
        for i in range(nb):
            samples = [get_data(txn, key, device="cpu") for key in groups[i]]
            pb = prepare(samples, target_len=config.MAX_TRAIN_GLOBAL_LEN).to(device)
            batches.append(pb)
            print(f"  cached batch {i+1}/{nb} (B={groups[i] and len(groups[i])})", flush=True)
    env.close()

    def fwd_bwd(b, foreach=False):
        (copy_downcast_foreach(model, master, dtype) if foreach
         else old_copy_downcast_(model, master, dtype))
        model.train()
        stats = model.get_loss(b)
        stats.average_loss.backward()
        (transfer_grad_foreach(master, model) if foreach
         else old_transfer_grad(master, model))
        return stats

    for i in range(args.warmup):
        fwd_bwd(batches[i % nb])
        torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
        optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()
    print("warmup done", flush=True)

    # ---- section breakdown ----
    sec = {k: 0.0 for k in ["copy", "fwd", "bwd", "transfer", "grad_norm", "clip", "opt"]}
    M = args.measure
    for i in range(M):
        b = batches[i % nb]
        torch.cuda.synchronize(); t = time.perf_counter()
        old_copy_downcast_(model, master, dtype)
        torch.cuda.synchronize(); sec["copy"] += time.perf_counter()-t; t = time.perf_counter()
        model.train(); stats = model.get_loss(b)
        torch.cuda.synchronize(); sec["fwd"] += time.perf_counter()-t; t = time.perf_counter()
        stats.average_loss.backward()
        torch.cuda.synchronize(); sec["bwd"] += time.perf_counter()-t; t = time.perf_counter()
        old_transfer_grad(master, model)
        torch.cuda.synchronize(); sec["transfer"] += time.perf_counter()-t; t = time.perf_counter()
        get_grad_norm(master)
        torch.cuda.synchronize(); sec["grad_norm"] += time.perf_counter()-t; t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
        torch.cuda.synchronize(); sec["clip"] += time.perf_counter()-t; t = time.perf_counter()
        optimizer.step(); optimizer.zero_grad()
        torch.cuda.synchronize(); sec["opt"] += time.perf_counter()-t
    print("\n--- per-section ms/step (synchronized) ---", flush=True)
    tot = 0.0
    for k, v in sec.items():
        ms = 1000*v/M; tot += ms
        print(f"  {k:10s} {ms:8.2f} ms")
    print(f"  {'TOTAL':10s} {tot:8.2f} ms   ({1000/tot:.2f} steps/s synchronized)", flush=True)

    # ---- end-to-end bodies ----
    def body_current(b):
        stats = fwd_bwd(b, foreach=False)
        _ = (stats.average_loss.item(), stats.ahead_avg.item(),
             stats.ahead_raw_avg.item(), stats.imm_avg.item())   # per-step print syncs
        _ = get_grad_norm(master)                                # logging-only (wandb off)
        torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
        optimizer.step(); optimizer.zero_grad()

    def body_no_sync(b):  # remove logging-only syncs only
        fwd_bwd(b, foreach=False)
        torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
        optimizer.step(); optimizer.zero_grad()

    def body_full_opt(b):  # logging syncs removed + foreach copy/transfer
        fwd_bwd(b, foreach=True)
        torch.nn.utils.clip_grad_norm_(master.parameters(), 0.5)
        optimizer.step(); optimizer.zero_grad()

    for name, body in [("CURRENT (grad_norm + .item prints)", body_current),
                       ("no-sync (drop logging syncs)", body_no_sync),
                       ("full-opt (no-sync + foreach copy/transfer)", body_full_opt)]:
        for i in range(2):
            body(batches[i % nb])
        torch.cuda.synchronize(); t = time.perf_counter()
        for i in range(M):
            body(batches[i % nb])
        torch.cuda.synchronize(); dt = time.perf_counter()-t
        print(f"[{name:46s}]  {1000*dt/M:7.2f} ms/step   {M/dt:6.2f} steps/s", flush=True)


if __name__ == "__main__":
    main()
