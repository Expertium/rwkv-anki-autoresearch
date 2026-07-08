import os

# Must precede `import torch` / first cuBLAS call: required for deterministic cuBLAS matmuls when
# RWKV_DETERMINISTIC is on (see _maybe_enable_determinism). Harmless otherwise.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from rwkv.data_fetcher import DataFetcher
import lmdb
import re
import random
import torch
import wandb

from rwkv.parse_toml import parse_toml
from rwkv.prepare_batch import prepare_data_train_test
from rwkv.model import rwkv_model as _rwkv_model_rc
from rwkv.model.srs_model import SrsRWKV
from rwkv.architecture import *
from rwkv.utils import (
    KeyValueAverage,
    get_number_of_trainable_parameters,
)

random.seed(12345)


def _maybe_enable_determinism():
    """RWKV_DETERMINISTIC=1 (default): pin the TRAINING process's RNG + cuBLAS/cuDNN algorithm
    selection so run-to-run training is reproducible APART from the intentional per-batch data
    augmentation (which lives in the fetch child processes and is deliberately left stochastic --
    Andrew 2026-06-29). The custom WKV CUDA kernel has no atomics, so it is already deterministic.
    warn_only=True so an op lacking a deterministic impl warns instead of crashing. Call in main()
    (training process only) -- NOT at module level, so the fetch children keep stochastic augmentation."""
    if os.environ.get("RWKV_DETERMINISTIC", "1") != "1":
        return
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)
    np.random.seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # RWKV_QAT_NO_MEMFILL=1 (default off): deterministic mode also NaN-fills EVERY freshly
    # allocated tensor (torch.utils.deterministic.fill_uninitialized_memory, a debug aid to
    # surface uninitialized reads — NOT part of the algorithm-determinism guarantee). Trace
    # attribution (2026-07-06): that fill is ~73k of the 92k fill_ launches per 8 steps ≈ 9k
    # launches/step ≈ 25% of the launch-bound step's kernel storm. Disabling changes NOTHING
    # for correct code (empty buffers are always overwritten before use); flag-gated anyway.
    if os.environ.get("RWKV_QAT_NO_MEMFILL", "") == "1":
        torch.utils.deterministic.fill_uninitialized_memory = False
        print("[determinism] fill_uninitialized_memory OFF (RWKV_QAT_NO_MEMFILL=1)")
    print("[determinism] training-process RNG + cuBLAS/cuDNN pinned (augmentation seed set separately)")


FINAL_LR = 0

ADAMW_BETAS = (0.90, 0.999)
ADAMW_EPS = 1e-18
# HP-tuner env overrides (Andrew 2026-06-30): default == current champion values, so an UNSET env var
# leaves behavior byte-identical. The greedy coordinate-descent tuner sweeps these without source edits.
WEIGHT_DECAY = float(os.environ.get("RWKV_WEIGHT_DECAY") or "0.01")
WEIGHT_DECAY_CHANNEL_MIXER = float(os.environ.get("RWKV_WEIGHT_DECAY") or "0.01")
WEIGHT_DECAY_HEAD = float(os.environ.get("RWKV_WEIGHT_DECAY") or "0.01")
CLIP = float(os.environ.get("RWKV_CLIP") or "0.5")
FETCH_AHEAD = 10  # prefetch depth = MAX concurrent fetch workers (the main loop keeps this many batches
# in flight). Raised 5->10 (Andrew 2026-06-30) so NUM_FETCH_PROCESSES=10 is actually usable -- with
# FETCH_AHEAD=5 only ~5 workers were ever busy no matter the process count. Buffer-only: it changes WHEN a
# batch is prepared, not its order/content -> results are bit-identical. (Costs ~10x21MB CPU RAM of buffer.)


def extract_numbers(name):
    match = re.findall(r"(\d+)_([\d]+)-([\d]+)_([\d]+)", name)
    if match:
        return tuple(map(int, match[0]))
    return None


def maybe_compile_mixers(model, label=""):
    """RWKV_QAT_COMPILE=1 (default off): torch.compile(dynamic=True) the time/channel-mixer
    forwards — fuses the elementwise soup (mul/add/sigmoid/pow/lerp chains) between the custom WKV
    kernels, which graph-break cleanly. Needs triton (triton-windows 3.7.1 present, Andrew OK'd
    2026-07-06; small trajectory perturbation acceptable). Rebinds the bound forward instead of
    wrapping the Module so parameter names stay intact (copy_downcast_ / master-child grad matching
    depend on them). First few steps pay compile latency per new shape family; dynamic=True keeps
    recompiles bounded across the variable-length buckets."""
    # RWKV_QAT_COMPILE=1/all -> compile student AND teacher; =student -> student only (round-3 A/B:
    # the compiled no_grad TEACHER got 177 ms/step SLOWER — dynamo guard overhead without a backward
    # to amortize it — while the student won 231 ms across fwd+bwd).
    _mode = os.environ.get("RWKV_QAT_COMPILE", "")
    if _mode not in ("1", "all", "student"):
        return model
    if _mode == "student" and "teacher" in label:
        print(f"[compile] skipping {label} (RWKV_QAT_COMPILE=student)")
        return model
    from rwkv.model import rwkv_model as _rm
    n = 0
    for m in model.modules():
        if isinstance(m, (_rm.RWKV7TimeMixer, _rm.RWKV7ChannelMixer)):
            m.forward = torch.compile(m.forward, dynamic=True)
            n += 1
    print(f"[compile] torch.compile(dynamic=True) on {n} mixer forwards {label}")
    return model


def get_optimizer(config, model):
    encode_params = []
    decay_params = []
    channel_mixer_params = []
    decay_head_params = []
    other_params = []
    head_targets = [
        "head",
        "p_linear",
        "s_linear",
        "d_linear",
        "w_linear",
        "ahead_linear",
        "head_ahead_logit",
        "head_w",
        "head_s",
        "head_d",
        "head_p",
    ]
    for name, param in model.named_parameters():
        # Param constraint is to exclude layer/group norm weights
        if (
            "weight" in name
            and "lora" not in name
            and "scale" not in name
            and len(param.squeeze().shape) >= 2
        ):
            is_head_param = False
            for head_target in head_targets:
                if head_target in name:
                    is_head_param = True
            if is_head_param:
                decay_head_params.append(param)
            elif "features2card" in name:
                encode_params.append(param)
            elif "channel_mixer" in name:
                channel_mixer_params.append(param)
            else:
                decay_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {
                "params": channel_mixer_params,
                "weight_decay": WEIGHT_DECAY_CHANNEL_MIXER,
                "lr": config.PEAK_LR,
            },
            {
                "params": decay_head_params,
                "weight_decay": WEIGHT_DECAY_HEAD,
                "lr": config.PEAK_LR,
            },
            {"params": encode_params, "weight_decay": 1e-2, "lr": config.PEAK_LR},
            {"params": other_params, "weight_decay": 0.0, "lr": config.PEAK_LR},
        ],
        eps=ADAMW_EPS,
        betas=ADAMW_BETAS,
    )


def log_model(log, model: SrsRWKV):
    for name, param in model.named_parameters():
        log[f"{name}.data.mean"] = param.mean().item()
        log[f"{name}.data.std"] = param.std().item()
        log[f"{name}.data.min"] = param.min().item()
        log[f"{name}.data.max"] = param.max().item()
        log[f"{name}.data.25th"] = torch.quantile(param, 0.25).item()
        log[f"{name}.data.50th"] = torch.quantile(param, 0.50).item()
        log[f"{name}.data.75th"] = torch.quantile(param, 0.75).item()
        if param.grad is not None:
            log[f"{name}.grad.mean"] = param.grad.mean().item()
            log[f"{name}.grad.std"] = param.grad.std().item()
            log[f"{name}.grad.min"] = param.grad.min().item()
            log[f"{name}.grad.max"] = param.grad.max().item()
            log[f"{name}.grad.25th"] = torch.quantile(param.grad, 0.25).item()
            log[f"{name}.grad.50th"] = torch.quantile(param.grad, 0.50).item()
            log[f"{name}.grad.75th"] = torch.quantile(param.grad, 0.75).item()


def get_groups(db_path, db_size, max_train_global_len, users):
    lmdb_env = lmdb.open(db_path, map_size=db_size)
    with lmdb_env.begin(write=False) as txn:
        keys = []
        for user_id in users:
            user_batches_raw = txn.get(f"{user_id}_batches".encode())
            if user_batches_raw is None:
                print("No data found for user", {user_id})
                continue

            batches = json.loads(user_batches_raw)
            for batch in batches:
                keys.append((user_id, batch[0], batch[1], batch[2]))

        random.shuffle(keys)
        keys.sort(key=lambda x: x[3], reverse=True)  # stable sort
        groups = []
        l = 0
        while l < len(keys):
            _, _, _, size = keys[l]
            max_batch = math.floor(max_train_global_len / size - 1e-6)
            if max_batch == 0:
                l += 1
                continue

            r = l - 1
            while r + 1 < len(keys) and r + 1 - l + 1 <= max_batch:
                r += 1

            if l <= r:
                groups.append(keys[l : (r + 1)])

            l = r + 1

        print("Number of groups:", len(groups))
        random.shuffle(groups)

    lmdb_env.close()
    return groups


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def _clear_device_cache(device):
    """Clear CUDA cache only when CUDA is actually available."""
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_on_user(user_id, batch, model: SrsRWKV):
    model.eval()
    with torch.no_grad():
        stats = model.get_loss(batch)
        if stats is None:
            raise Exception("Stats is none.")
        print(
            f"{user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f} ({stats.ahead_raw_equalize_avg.item():.3f}), imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}"
        )
    return (
        stats.ahead_equalize_avg * stats.ahead_equalize_n,
        stats.ahead_equalize_n,
        stats.ahead_raw_equalize_avg * stats.ahead_equalize_n,
        stats.imm_binary_equalize_avg * stats.imm_binary_equalize_n,
        stats.imm_binary_equalize_n,
    )


def validate(model, data_fetcher, all_db_keys, users, device):
    _clear_device_cache(device)
    tot_ahead_loss = 0
    tot_ahead_raw_loss = 0
    tot_ahead_n = 0
    tot_imm_loss = 0
    tot_imm_n = 0

    for i in range(min(FETCH_AHEAD, len(users))):
        user_id = users[i]
        data_fetcher.enqueue((f"validate-{user_id}", [all_db_keys[user_id]]))

    try:
        for i, user_id in enumerate(users):
            batch = data_fetcher.get(f"validate-{user_id}")
            batch = batch.to(device)
            if i + FETCH_AHEAD < len(users):
                fetch_ahead_user_id = users[i + FETCH_AHEAD]
                data_fetcher.enqueue(
                    (
                        f"validate-{fetch_ahead_user_id}",
                        [all_db_keys[fetch_ahead_user_id]],
                    )
                )

            (
                user_ahead_loss,
                user_ahead_n,
                user_ahead_raw_loss,
                user_imm_loss,
                user_imm_n,
            ) = evaluate_on_user(user_id, batch, model)
            assert user_ahead_n == user_imm_n
            tot_ahead_loss += user_ahead_loss
            tot_ahead_raw_loss += user_ahead_raw_loss
            tot_ahead_n += user_ahead_n
            tot_imm_loss += user_imm_loss
            tot_imm_n += user_imm_n

        print(
            f"Mean ahead validation loss: {tot_ahead_loss / tot_ahead_n:.4f} ({tot_ahead_raw_loss / tot_ahead_n:.4f}), imm: {tot_imm_loss / tot_imm_n:.4f}, validation n: {tot_ahead_n}"
        )
        return tot_ahead_loss / tot_ahead_n, tot_imm_loss / tot_imm_n
    except Exception as e:
        print("Exception in validate. RWKV-7 nan?")
        print(e)
        return None


def transfer_child_grad_to_master(master, child):
    # Vectorized child(bf16)-grad -> master(fp32)-grad accumulation via torch._foreach_add_: one
    # fused kernel per dtype group instead of ~440 per-param add+zero launches (a launch-bound
    # hotspot, ~43 ms/step). add_ upcasts the operand, so grouping + foreach is BIT-IDENTICAL to the
    # original per-param loop. Arch-agnostic. (None grads -- first few iters -- are skipped as before.)
    master_params = dict(master.named_parameters())
    groups = {}  # (master_grad_dtype, child_grad_dtype) -> ([master_grad...], [child_grad...])
    with torch.no_grad():
        for name, param in child.named_parameters():
            if param.grad is None:
                continue
            master_param = master_params[name]
            if master_param.grad is None:
                master_param.grad = torch.zeros_like(master_param, requires_grad=True)
            key = (master_param.grad.dtype, param.grad.dtype)
            mg, cg = groups.setdefault(key, ([], []))
            mg.append(master_param.grad)
            cg.append(param.grad)
        for mg, cg in groups.values():
            torch._foreach_add_(mg, cg)  # fp32 += child grad (casts)
            torch._foreach_zero_(cg)


def get_test_keys(dataset_path, dataset_size, users):
    dataset = lmdb.open(dataset_path, map_size=dataset_size)
    keys = {}
    with dataset.begin(write=False) as txn:
        for user_id in users:
            user_batches_raw = txn.get(f"{user_id}_batches".encode())
            if user_batches_raw is None:
                print("No data found for user", {user_id})
                continue

            batches = json.loads(user_batches_raw)
            assert len(batches) == 1
            for batch in batches:
                keys[user_id] = (user_id, batch[0], batch[1], batch[2])
    return keys


class KeyValueStatistics:
    def __init__(self):
        self.ahead_average = KeyValueAverage()
        self.ahead_raw_average = KeyValueAverage()
        self.ahead_raw_diff_average = KeyValueAverage()
        self.imm_average = KeyValueAverage()
        self.ahead_equalize_average = KeyValueAverage()
        self.imm_binary_equalize_average = KeyValueAverage()

    def add(self, keys, stats):
        self.ahead_average.add_value(
            key=keys, avg=stats.ahead_avg.detach(), weight=stats.ahead_n
        )
        self.ahead_raw_average.add_value(
            key=keys, avg=stats.ahead_raw_avg.detach(), weight=stats.ahead_n
        )
        self.ahead_raw_diff_average.add_value(
            key=keys,
            avg=stats.ahead_avg.detach() - stats.ahead_raw_avg.detach(),
            weight=stats.ahead_n,
        )
        self.ahead_equalize_average.add_value(
            key=keys,
            avg=stats.ahead_equalize_avg.detach(),
            weight=stats.ahead_equalize_n,
        )
        self.imm_average.add_value(
            key=keys, avg=stats.imm_avg.detach(), weight=stats.imm_n
        )
        self.imm_binary_equalize_average.add_value(
            key=keys,
            avg=stats.imm_binary_equalize_avg.detach(),
            weight=stats.imm_binary_equalize_n,
        )

    def add_log(self, log):
        log["ahead_avg"] = self.ahead_average.get_value()
        log["ahead_raw_avg"] = self.ahead_raw_average.get_value()
        log["ahead_raw_diff_avg"] = self.ahead_raw_diff_average.get_value()
        log["ahead_equalize_avg"] = self.ahead_equalize_average.get_value()
        log["imm_avg"] = self.imm_average.get_value()
        log["imm_binary_equalize_avg"] = self.imm_binary_equalize_average.get_value()


def _dump_kernel_profile(profiler, n_steps):
    """Bucketed self-GPU-time summary for RWKV_PROFILE_STEP mode (plain ASCII)."""
    def cuda_us(e):  # self GPU time in us, robust across torch versions
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return v
        return 0.0

    def bucket(name):
        n = name.lower()
        if "qat" in n:
            return "wkv QAT (card/note, per-timestep)"
        if "scan_kernel" in n or "add_kernel" in n:
            return "wkv scan (matmul)"
        if ("time_parallel" in n or "wkv_forward_kernel" in n or "wkv_backward_kernel" in n
                or "lr_trunc" in n):
            return "wkv plain recurrence (chunked-rewrite target)"
        if "gemm" in n or "cutlass" in n or "cublas" in n or "dot_kernel" in n:
            return "gemm (linear layers)"
        return "other (elementwise/reduce/copy/optim)"

    tot, per_kernel = {}, {}
    for e in profiler.key_averages():
        t = cuda_us(e)
        if t <= 0:
            continue
        b = bucket(e.key)
        tot[b] = tot.get(b, 0.0) + t
        per_kernel[e.key] = per_kernel.get(e.key, 0.0) + t
    grand = sum(tot.values())
    print(f"\n===== KERNEL PROFILE ({n_steps} steps) =====")
    print(f"total GPU kernel time: {grand / 1e3:.2f} ms  ({grand / 1e3 / n_steps:.2f} ms/step)")
    for b in sorted(tot, key=lambda x: -tot[x]):
        print(f"  {tot[b] / grand * 100:6.2f}%  {tot[b] / 1e3 / n_steps:8.2f} ms/step  {b}")
    print("  --- top 15 kernels ---")
    for name in sorted(per_kernel, key=lambda x: -per_kernel[x])[:15]:
        short = name if len(name) < 90 else name[:87] + "..."
        print(f"    {per_kernel[name] / grand * 100:6.2f}%  {short}")
    print("PROFILE_DONE")


def main_loop(config, task_queue, batch_queue):
    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = (
        SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        .selective_cast(config.DTYPE)
        .to(config.DEVICE)
    )
    maybe_compile_mixers(model, "(student)")
    optimizer = get_optimizer(config, master_model)

    # Learnable codebooks (shift cb / shift rotation / WKV cb): create the Parameters up front so
    # that when a resumed optimizer state ALREADY CONTAINS their groups (any LEARN=1 run's own
    # checkpoint -- the WS->decay seam of a champion run, or a mid-run crash resume), the groups
    # can be registered BEFORE optimizer.load_state_dict and the group counts match (2026-07-08:
    # the first 5k champion run died here with "different number of parameter groups"). A state
    # saved WITHOUT them (warm-start from a pre-LEARN champion optim) still loads first and the
    # groups are registered after, exactly as before. wd=0 -- centroids are a codebook, not
    # weights to shrink. Registration order MUST stay shift, rot, wkv (== save order).
    _extra_cb_groups = []
    _shift_cb_param = None
    _shift_rot_param = None
    _wkv_cb_param = None
    if os.environ.get("RWKV_QAT_SHIFT_PQ", "") and os.environ.get("RWKV_QAT_SHIFT_PQ_LEARN", "") == "1":
        from rwkv.model import rwkv_model as _rwkv_model_mod
        _shift_cb_param = _rwkv_model_mod.shift_pq_init(config.DEVICE)
        _extra_cb_groups.append((_shift_cb_param, "[shift-pq] codebook LEARNABLE"))
    if os.environ.get("RWKV_QAT_SHIFT_PQ", "") and os.environ.get("RWKV_QAT_SHIFT_ROT", "") == "1":
        from rwkv.model import rwkv_model as _rwkv_model_mod
        _c = int(os.environ.get("RWKV_N_HEADS", "2")) * int(os.environ.get("RWKV_HEAD_DIM", "16"))
        _shift_rot_param = _rwkv_model_mod.shift_rot_init(config.DEVICE, _c)
        _extra_cb_groups.append((_shift_rot_param, "[shift-rot] learned pre-rotation"))
    if os.environ.get("RWKV_QAT_PQ", "") and os.environ.get("RWKV_QAT_PQ_LEARN", "") == "1":
        from rwkv.model import rwkv_ops as _rwkv_ops_mod
        _wkv_cb_param = _rwkv_ops_mod.wkv_pq_cb_param()
        _extra_cb_groups.append((_wkv_cb_param, "[wkv-pq] codebook LEARNABLE"))
    _cb_groups_added = False

    def _register_cb_groups():
        for _p, _tag in _extra_cb_groups:
            optimizer.add_param_group(
                {"params": [_p], "lr": config.PEAK_LR, "weight_decay": 0.0}
            )
            print(f"{_tag}: {tuple(_p.shape)} added as optim group (wd=0)")

    if config.LOAD_MODEL:
        model_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}.pth"
        optim_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}_optim.pth"
        print("Loading model:", model_path)
        master_model.load_state_dict(torch.load(model_path, weights_only=True))
        # weights_only=False for the optim: some champion optim files (e.g. the decay champion
        # h2k16d_optim_904) hold a numpy scalar in their state which weights_only=True rejects.
        # These are trusted local checkpoints; this is a deserialization-security flag, NOT a
        # numerics flag -- the loaded optimizer state is byte-identical either way.
        # Capture the INTENDED per-group weight_decay before the load: load_state_dict restores the
        # SAVED param_group hyperparams (same clobber class as the lr bug below), which silently
        # overrode RWKV_WEIGHT_DECAY (discovered 2026-07-02 in the sibling quant loop: a WD=0 run
        # came out hash-identical to its WD=0.01 twin). Per-group: the groups carry different WDs.
        _optim_state = torch.load(optim_path, weights_only=False)
        if _extra_cb_groups and len(_optim_state["param_groups"]) == len(
            optimizer.param_groups
        ) + len(_extra_cb_groups):
            # state saved by a LEARN=1 run (own-checkpoint resume / WS->decay seam):
            # register the cb groups first so counts match AND the cbs' Adam moments resume
            _register_cb_groups()
            _cb_groups_added = True
            print(f"[optim] resumed state includes {len(_extra_cb_groups)} learnable-cb "
                  f"group(s); registered before load (moments resume)")
        _intended_wd = [g["weight_decay"] for g in optimizer.param_groups]
        optimizer.load_state_dict(_optim_state)
        for _g, _wd in zip(optimizer.param_groups, _intended_wd):
            _g["weight_decay"] = _wd
        print(f"[wd] reset optimizer per-group weight_decay to intended {_intended_wd} after loading champion optim")
        # The champion optim was saved under a LambdaLR, so its param_groups carry initial_lr = the
        # champion's PEAK_LR (1e-3) and lr = 0 (end of decay). load_state_dict restores BOTH; LambdaLR then
        # reuses initial_lr as its base_lr, SILENTLY OVERRIDING config.PEAK_LR. Reset lr AND initial_lr to
        # config.PEAK_LR so the configured LR actually controls the fine-tune tail (warm moments are kept).
        for _g in optimizer.param_groups:
            _g["lr"] = config.PEAK_LR
            _g["initial_lr"] = config.PEAK_LR
        print(f"[lr] reset optimizer lr/initial_lr to config.PEAK_LR = {config.PEAK_LR} after loading champion optim")
    else:
        print("No model loaded.")
    model.copy_downcast_(master_model, dtype=config.DTYPE)

    # KD teacher (RWKV_QAT_KD=<lambda>, task22): a frozen, UN-quantized copy of the run's starting
    # champion. QAT gating lives in per-module fields copied from the arch config at build time, so
    # resetting those fields on this instance strips every fake-quant hook (WKV low-rank/PQ, shift
    # PQ/rotation, norm quant all nest inside these guards) while `model` keeps them.
    _kd_lam = float(os.environ.get("RWKV_QAT_KD", "0") or 0)
    _teacher = None
    if _kd_lam > 0:
        assert config.LOAD_MODEL, "KD needs a champion checkpoint to distill from"
        _teacher = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        _teacher.load_state_dict(torch.load(model_path, weights_only=True))
        _teacher = _teacher.selective_cast(config.DTYPE).to(config.DEVICE)
        _n_stripped = 0
        for _m in _teacher.modules():
            if getattr(_m, "state_shift_qmax", float("inf")) != float("inf"):
                _m.state_shift_qmax = float("inf"); _n_stripped += 1
            if getattr(_m, "state_qmax", float("inf")) != float("inf"):
                _m.state_qmax = float("inf"); _n_stripped += 1
            if getattr(_m, "state_lowrank_rank", 0) > 0:
                _m.state_lowrank_rank = 0; _n_stripped += 1
        _teacher.eval()
        for _p in _teacher.parameters():
            _p.requires_grad_(False)
        maybe_compile_mixers(_teacher, "(teacher)")
        print(f"[KD] teacher = {model_path}, {_n_stripped} QAT hooks stripped, lambda = {_kd_lam}")

    # Learnable-codebook groups: created up front (pre-load, above); register here when the
    # loaded (or absent) optim state did not already carry them -- AFTER the champion-optim
    # restore and BEFORE the scheduler is built (so the LambdaLR covers the new groups).
    # Gradients arrive from `model`'s forward directly on the shared global Parameters.
    if not _cb_groups_added:
        _register_cb_groups()
        _cb_groups_added = True
    # Soft-to-hard selection annealing (RWKV_QAT_SHIFT_ANNEAL=<tau0>): per-step temperature schedule,
    # linear tau0 -> 0 at RWKV_QAT_SHIFT_ANNEAL_END of training, exactly-hard thereafter (no end gap).
    _rm_anneal = None
    if os.environ.get("RWKV_QAT_SHIFT_PQ", "") and float(os.environ.get("RWKV_QAT_SHIFT_ANNEAL", "0") or 0) > 0:
        from rwkv.model import rwkv_model as _rm_anneal
        print(f"[shift-anneal] soft-to-hard selection annealing ON: tau0={_rm_anneal._SHIFT_ANNEAL_TAU0} "
              f"-> fully HARD from {_rm_anneal._SHIFT_ANNEAL_END:.0%} of training")
    # (WKV-PQ learnable codebook: created up front pre-load with the other cb Parameters; its
    # grads do NOT arrive via autograd — the lr backward kernel accumulates them in a device
    # buffer; the loop below zeroes it before backward and fetches it into .grad after, then
    # re-uploads the stepped centroids to the kernel globals.)

    # Dead-centroid resurrection (RWKV_QAT_CB_RESURRECT=1, task22): with 16-32-entry catalogs, a
    # centroid nothing selects receives ~zero gradient and is wasted capacity exactly where capacity
    # binds. Grads for both learnable codebooks already exist every step, so track a per-centroid
    # grad-norm EMA and periodically re-seed dead entries next to the busiest centroid of their block.
    _resurrect = os.environ.get("RWKV_QAT_CB_RESURRECT", "") == "1"
    _res_ema = {}
    _RES_EVERY, _RES_WARMUP, _RES_REL, _RES_DECAY = 250, 500, 0.02, 0.99

    def _resurrect_step(step_no):
        specs = []
        if _shift_cb_param is not None and _shift_cb_param.grad is not None:
            _r2, _m, _nc, _sd = _shift_cb_param.shape
            specs.append(("shiftcb", _shift_cb_param, (_r2 * _m, _nc, _sd), None))
        if _wkv_cb_param is not None and _wkv_cb_param.grad is not None:
            from rwkv.model import rwkv_ops as _ro
            _m, _sd, _nc = _ro._PQ_META[0], _ro._PQ_META[1], _ro._PQ_META[2]
            _jt = _ro._PQ_META[5] if len(_ro._PQ_META) > 5 else 0  # joint-uv: 1 block (whole catalog)
            specs.append(("wkvcb", _wkv_cb_param, ((1 if _jt else 2 * _m), _nc, _sd), _ro))
        for name, p, (nb, nc, sd), ro in specs:
            gn = p.grad.detach().float().view(nb, nc, sd).norm(dim=-1)  # [blocks, centroids]
            ema = _res_ema.get(name)
            _res_ema[name] = gn.clone() if ema is None else _RES_DECAY * ema + (1 - _RES_DECAY) * gn
            if step_no < _RES_WARMUP or step_no % _RES_EVERY != 0:
                continue
            ema = _res_ema[name]
            data = p.data.view(nb, nc, sd)
            n_res = 0
            with torch.no_grad():
                for b in range(nb):
                    med = ema[b].median()
                    if med <= 0:
                        continue
                    dead = (ema[b] < _RES_REL * med).nonzero().flatten()
                    if dead.numel() == 0:
                        continue
                    busy = int(ema[b].argmax())
                    noise = 0.05 * data[b].std()
                    for c in dead.tolist():
                        data[b, c] = data[b, busy] + noise * torch.randn(sd, device=data.device)
                        ema[b, c] = ema[b, busy]  # grace period before it can be re-killed
                        n_res += 1
            if n_res:
                print(f"[resurrect] step {step_no}: {name} re-seeded {n_res} dead centroid(s)")
                if ro is not None:
                    ro.wkv_pq_reupload()  # push the edited centroids to the kernel globals

    num_trainable_parameters = get_number_of_trainable_parameters(model)
    print(f"Trainable parameters: {num_trainable_parameters}")

    TRAIN_USERS = list(range(config.TRAIN_USERS_START, config.TRAIN_USERS_END + 1))
    groups = get_groups(
        config.TRAIN_DATASET_LMDB_PATH,
        config.TRAIN_DATASET_LMDB_SIZE,
        config.MAX_TRAIN_GLOBAL_LEN,
        users=TRAIN_USERS,
    )
    VALIDATION_USERS = list(
        range(config.VALIDATE_USERS_START, config.VALIDATE_USERS_END + 1)
    )
    all_db_keys = get_test_keys(
        config.VALIDATE_DATASET_LMDB_PATH,
        config.VALIDATE_DATASET_LMDB_SIZE,
        users=VALIDATION_USERS,
    )

    if config.USE_WANDB:
        wandb_config = {
            "epochs": config.EPOCHS,
            "peak_lr": config.PEAK_LR,
            "final_lr": FINAL_LR,
            "adamw_betas": ADAMW_BETAS,
            "adamw_eps": ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "weight_decay_channel_mixer": WEIGHT_DECAY_CHANNEL_MIXER,
            "weight_decay_head": WEIGHT_DECAY_HEAD,
            "dropout": DROPOUT,
            "dropout_long": DROPOUT_LONG,
            "dropout_layer": DROPOUT_LAYER,
            "clip": CLIP,
            "anki_rwkv_config": DEFAULT_ANKI_RWKV_CONFIG,
            "trainable parameters": num_trainable_parameters,
        }
        if config.WANDB_RESUME:
            wandb.init(
                project=config.WANDB_PROJECT_NAME,
                id=config.WANDB_RESUME_ID,
                resume="must",
                config=wandb_config,
            )
        else:
            wandb.init(project=config.WANDB_PROJECT_NAME, config=wandb_config)

    total_steps = int(config.EPOCHS * len(groups))

    # Benchmark mode (batch-size/throughput sweep): RWKV_MAX_STEPS caps the run to N steps and, after
    # RWKV_BENCH_WARMUP steps (excluded: first-batch fetch + CUDA init), times steady-state steps/s +
    # peak reserved VRAM. Off (0) by default -> normal training, byte-identical.
    bench_max_steps = int(os.environ.get("RWKV_MAX_STEPS") or "0")
    bench_warmup = int(os.environ.get("RWKV_BENCH_WARMUP") or "30")
    bench_t0 = None
    bench_work = 0
    if bench_max_steps > 0:
        total_steps = min(total_steps, bench_max_steps)
        print(f"[bench] cap {total_steps} steps, warmup {bench_warmup} excluded from timing")

    # Kernel-profile mode: RWKV_PROFILE_STEP=N wraps steps [N, N+RWKV_PROFILE_COUNT) in
    # torch.profiler (CUDA activity only), prints a bucketed self-GPU-time summary + top kernels,
    # then exits. GPU self-times are valid even when the CPU is contended (fetch stalls inflate
    # wall time, not kernel time). Off (0) by default -> byte-identical training.
    prof_start = int(os.environ.get("RWKV_PROFILE_STEP") or "0")
    prof_count = int(os.environ.get("RWKV_PROFILE_COUNT") or "5")
    profiler = None
    if prof_start > 0:
        print(f"[profile] will profile steps {prof_start}..{prof_start + prof_count - 1} then exit")

    # RWKV_COMPILE=1: torch.compile (inductor/triton-windows) on the training-step method.
    # EXPERIMENTAL; requires RWKV_NO_JIT=1 (Dynamo cannot trace ScriptModules). dynamic=True
    # because every step has different per-split shapes. Custom torch.ops.rwkv.* calls graph-break
    # (expected); the win, if any, is fusing the elementwise mass between the breaks.
    if os.environ.get("RWKV_COMPILE") == "1":
        if not os.environ.get("RWKV_NO_JIT"):
            raise RuntimeError("RWKV_COMPILE=1 requires RWKV_NO_JIT=1 (Dynamo can't trace ScriptModules)")
        # Run-to-run determinism: inductor's runtime autotune picks kernel configs by TIMING,
        # which varies across runs and can change reduction order -> two identical runs diverge
        # (observed 2026-07-03). Disable every timing-dependent selection path.
        import torch._inductor.config as inductor_config
        for knob, val in [("deterministic", True), ("max_autotune", False),
                          ("coordinate_descent_tuning", False), ("benchmark_kernel", False),
                          ("benchmark_epilogue_fusion", False)]:
            if hasattr(inductor_config, knob):
                setattr(inductor_config, knob, val)
                print(f"[compile] inductor_config.{knob} = {val}")
        # Compile the MIXER forwards only (not get_loss): whole-graph Dynamo traces blow Python
        # 3.12's FIXED per-thread C-recursion cap (RecursionError on most steps, silently eaten by
        # the NaN-safety except -> hollow steps). The mixers hold the fusable norm/lerp/LoRA
        # elementwise chains anyway; the glue stays eager.
        n_wrapped = 0
        for m in model.modules():
            if m.__class__.__name__ in ("RWKV7TimeMixer", "RWKV7ChannelMixer"):
                m.forward = torch.compile(m.forward, dynamic=True)
                n_wrapped += 1
        print(f"[compile] wrapped {n_wrapped} mixer forwards with torch.compile(dynamic=True)")

    # Per-step WS logloss trace + Wilcoxon early-prune (Andrew 2026-07-02, 5k methodology pt 9).
    # RWKV_STEP_TRACE=path -> append {"step","ahead","imm"} per successful step (WS PHASE ONLY -- the
    # decay phase has a variable step count, so traces are not comparable there). Valid pairing relies
    # on the seeded epoch shuffle: same db + MAX + seeds => same batch at the same step in every run.
    # RWKV_PRUNE_REF=champion_5k.json -> every RWKV_PRUNE_EVERY (300) steps, one-sided Wilcoxon
    # signed-rank on per-step (candidate - champion) over ALL common steps so far; if BOTH ahead and imm
    # are worse at p < RWKV_PRUNE_ALPHA (1e-4), write a .pruned.json marker (incl. Andrew's estimated
    # final logloss: champ_final + cand@s - champ@s) and exit(42). RWKV_PRUNE_MIN_STEP delays the first
    # check (e.g. past a longer warmup, whose early losses are worse by construction). Default off.
    step_trace_path = os.environ.get("RWKV_STEP_TRACE") or ""
    prune_ref_path = os.environ.get("RWKV_PRUNE_REF") or ""
    prune_every = int(os.environ.get("RWKV_PRUNE_EVERY") or "300")
    prune_alpha = float(os.environ.get("RWKV_PRUNE_ALPHA") or "1e-4")
    prune_min_step = int(os.environ.get("RWKV_PRUNE_MIN_STEP") or "0")
    if config.TRAIN_MODE != "WS":
        step_trace_path, prune_ref_path = "", ""
    trace_file = None
    trace_steps = {}  # step -> (ahead, imm), this run
    if step_trace_path:
        Path(step_trace_path).parent.mkdir(parents=True, exist_ok=True)
        trace_file = open(step_trace_path, "a", buffering=1)  # line-buffered: survives crashes
        print(f"[trace] per-step WS logloss -> {step_trace_path}")
    champ_meta, champ_steps = None, {}
    if prune_ref_path:
        with open(prune_ref_path) as _f:
            champ_meta = json.load(_f)
        champ_steps = {int(s): (a, i) for s, a, i in zip(
            champ_meta["trace_step"], champ_meta["trace_ahead"], champ_meta["trace_imm"])}
        print(f"[prune] vs champion '{champ_meta.get('name')}' ({len(champ_steps)} steps), "
              f"every {prune_every}, alpha {prune_alpha:g}, min_step {prune_min_step}")

    if config.TRAIN_MODE == "WS":
        warmup_steps = config.WARMUP_STEPS
        print("Warmup steps:", warmup_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    elif config.TRAIN_MODE == "D":

        def cosine_down(step, total_steps):
            return 1 + np.cos(0.5 * np.pi * (1 + step / total_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: cosine_down(t, total_steps)
        )
    else:
        raise ValueError(f"Invalid train mode: {config.TRAIN_MODE}")

    key_value_stats = KeyValueStatistics()
    train_start = time.time()
    group_start = time.time()

    assert FETCH_AHEAD <= len(groups)

    # EMA weight averaging (research lever, 2026-06-30): RWKV_EMA_DECAY=0.999 -> maintain an exponential
    # moving average of the fp32 master weights and ALSO save it (as {prefix}_ema_{step}.pth) so eval can
    # use the flatter-minimum averaged model. Off by default (decay<=0) => byte-identical to before. EMA
    # starts after warmup (init'd to the post-warmup weights so the random init isn't averaged in).
    ema_decay = float(os.environ.get("RWKV_EMA_DECAY") or "0")
    ema_start = config.WARMUP_STEPS if config.TRAIN_MODE == "WS" else 0
    ema_state = None
    # RWKV_QAT_EMA_FOREACH=1 (default off): vectorize the per-step EMA update via _foreach_mul_/_foreach_add_
    # -- ~880 tiny kernel launches -> 4. Same in-place ops on the same tensors element-wise, so unlike the
    # other speed flags this one IS bit-identical; kept flag-gated anyway so in-flight runs stay untouched.
    ema_foreach = os.environ.get("RWKV_QAT_EMA_FOREACH", "") == "1"
    ema_lists = None  # cached ([ema tensors...], [master tensors...]) -- objects are stable across steps
    if ema_decay > 0:
        print(f"[ema] weight averaging ON, decay={ema_decay}, start after step {ema_start}"
              + (", foreach" if ema_foreach else ""))

    # The early-step torch.cuda.empty_cache() (next 1000 steps) guards against allocator
    # fragmentation OOM under the variable-seq-length workload, but it COSTS ~150 ms/step
    # (measured, scratchpad/profile_emptycache.py) -- and short research runs (~960-2400 steps)
    # pay it on EVERY step. RWKV_EMPTY_CACHE_EVERY (default 1 == byte-identical to before) lets a
    # run clear less often (e.g. 50) or never (0) once it's known not to OOM -> ~1.2x for short
    # runs. Arch-agnostic. Read once here, not per-step.
    empty_cache_every = int(os.environ.get("RWKV_EMPTY_CACHE_EVERY") or "1")
    if empty_cache_every != 1:
        print(f"[empty_cache] clearing device cache every {empty_cache_every} steps "
              f"(first 1000), 0=never (default 1)")

    checkpoint_step_count = 0
    checkpoint_loss_n = 0

    step = config.STEP_OFFSET - 1
    for epoch_i in range(0, int(1e9)):
        if step > total_steps:
            break

        random.shuffle(groups)
        for i in range(FETCH_AHEAD):
            data_fetcher.enqueue((f"train-{i}", groups[i]))

        for group_i in range(len(groups)):
            step += 1
            if step > total_steps:
                break

            if prof_start > 0 and profiler is None and step == prof_start:
                torch.cuda.synchronize()
                from torch.profiler import profile as _tprof, ProfilerActivity as _PA
                profiler = _tprof(activities=[_PA.CUDA])
                profiler.__enter__()
            elif profiler is not None and step == prof_start + prof_count:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                _dump_kernel_profile(profiler, prof_count)
                raise SystemExit(0)

            if bench_max_steps > 0 and step == config.STEP_OFFSET + bench_warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                bench_t0 = time.time()

            if (
                empty_cache_every > 0
                and step < config.STEP_OFFSET + 1000
                and (step - config.STEP_OFFSET) % empty_cache_every == 0
            ):
                _clear_device_cache(config.DEVICE)

            # VALIDATE_EVERY (default 500) controls validation/checkpoint cadence by global
            # step — small datasets have too few groups for the original (group_i+1)%500.
            validate_every = getattr(config, "VALIDATE_EVERY", 500)
            validate_iter = (
                step == 50 or step % validate_every == 0 or step == total_steps
            )
            log = {}
            log["step"] = step
            log["lr"] = optimizer.param_groups[0]["lr"]
            if _rm_anneal is not None:
                _rm_anneal.set_shift_anneal_progress(step / total_steps)

            # Rotation-cache hygiene (RWKV_QAT_ROT_CACHE): drop the previous step's graph-carrying
            # cached Cayley R before this step's forward. No-op when the cache flag is off.
            _rwkv_model_rc.shift_rot_cache_clear()

            keys = str(groups[group_i])
            print(f"\n{keys}")
            time_fetch = time.time()
            prepared_batch = data_fetcher.get(f"train-{group_i}")
            print(f"Got: {time.time() - time_fetch:.4f}s")
            prepared_batch = prepared_batch.to(config.DEVICE)
            fetch_ahead_group_i = group_i + FETCH_AHEAD
            if fetch_ahead_group_i < len(groups):
                data_fetcher.enqueue(
                    (f"train-{fetch_ahead_group_i}", groups[fetch_ahead_group_i])
                )

            model.copy_downcast_(master_model, dtype=config.DTYPE)
            model.train()
            try:
                kd_args = None
                if _teacher is not None:
                    with torch.no_grad():
                        t_ahead, t_w, _t_wlp, t_p = _teacher.forward_batch(
                            prepared_batch.start,
                            prepared_batch.sub_gather,
                            prepared_batch.sub_gather_lens,
                            prepared_batch.time_shift_selects,
                            prepared_batch.skips,
                            prepared_batch.num_data,
                        )
                        _les = prepared_batch.labels.float()[..., 0].unsqueeze(-1)
                        _tcr = _teacher.forgetting_curve(t_w, _les).clamp(1e-5, 1 - 1e-5)
                        _tcl = torch.log(_tcr / (1 - _tcr)) + _teacher.interp(t_ahead, _les)
                        kd_args = (
                            t_p.float(),
                            torch.sigmoid(_tcl).clamp(1e-5, 1 - 1e-5).float(),
                            _kd_lam,
                        )
                stats = model.get_loss(prepared_batch, kd=kd_args)
                if stats is None:
                    raise Exception("Stats is none.")
                if not torch.isfinite(stats.average_loss):  # NaN/inf safeguard: skip, don't backprop garbage
                    raise Exception("non-finite training loss")

                print(
                    f"{epoch_i} {group_i} {step}, all: {stats.average_loss.item():3f}, ahead: {stats.ahead_avg.item():.4f} ({stats.ahead_raw_avg.item():.4f}), imm: {stats.imm_avg.item():.3f}"
                )
                log["train_nan"] = 0
                if _wkv_cb_param is not None:
                    from rwkv.model import rwkv_ops as _rwkv_ops_mod
                    _rwkv_ops_mod.wkv_pq_grad_zero()
                stats.average_loss.backward()
                transfer_child_grad_to_master(master=master_model, child=model)
                if _wkv_cb_param is not None:
                    _rwkv_ops_mod.wkv_pq_grad_fetch()

                if validate_iter and config.USE_WANDB:
                    log_model(log, master_model)
                log["loss"] = stats.average_loss.detach()
                log["w_divergence"] = stats.w_loss_avg.detach()
                log["ahead_logits_mag_loss"] = stats.ahead_logits_mag_loss_avg.detach()
                log["ahead_logits_diff_loss"] = (
                    stats.ahead_logits_diff_loss_avg.detach()
                )
                # get_grad_norm does ~440 per-param .item() syncs/step (~28 ms) and is consumed
                # ONLY by wandb -- skip it entirely when wandb is off (every iter config is off).
                if config.USE_WANDB:
                    log["norm"] = get_grad_norm(master_model)
                key_value_stats.add(keys, stats)
                key_value_stats.add_log(log)

                checkpoint_step_count += 1
                checkpoint_loss_n += stats.ahead_n
                if bench_t0 is not None:
                    bench_work += int(stats.ahead_n)
                if trace_file is not None or champ_meta is not None:
                    _ta, _ti = float(stats.ahead_avg.item()), float(stats.imm_avg.item())
                    trace_steps[step] = (_ta, _ti)
                    if trace_file is not None:
                        trace_file.write(json.dumps({"step": step, "ahead": _ta, "imm": _ti}) + "\n")

                # NaN/inf safeguard: clip_grad_norm_ returns the total grad norm; if it's non-finite, a NaN
                # grad slipped through -> DO NOT step (that would write NaN into the weights and kill the model).
                # The learnable QAT codebooks/rotation (if any) join the clip so a NaN centroid grad also blocks the step.
                _clip_params = list(master_model.parameters())
                if _shift_cb_param is not None:
                    _clip_params.append(_shift_cb_param)
                if _shift_rot_param is not None:
                    _clip_params.append(_shift_rot_param)
                if _wkv_cb_param is not None:
                    _clip_params.append(_wkv_cb_param)
                total_norm = torch.nn.utils.clip_grad_norm_(_clip_params, CLIP)
                if torch.isfinite(total_norm):
                    optimizer.step()
                    if _wkv_cb_param is not None:  # push stepped centroids to the kernel globals
                        _rwkv_ops_mod.wkv_pq_reupload()
                    if _resurrect:
                        _resurrect_step(step)
                else:
                    print("Non-finite grad norm; skipping optimizer step (weights protected).")
                    log["train_nan"] = 1
                optimizer.zero_grad()
                if ema_decay > 0 and step >= ema_start:
                    msd = master_model.state_dict()
                    if ema_state is None:  # init EMA to the post-warmup weights
                        ema_state = {k: v.detach().float().clone()
                                     for k, v in msd.items() if v.is_floating_point()}
                    elif ema_foreach:
                        if ema_lists is None:
                            ema_lists = ([ema_state[k] for k in msd if k in ema_state],
                                         [msd[k].detach() for k in msd if k in ema_state])
                        torch._foreach_mul_(ema_lists[0], ema_decay)
                        torch._foreach_add_(ema_lists[0], ema_lists[1], alpha=1 - ema_decay)
                    else:
                        for k, v in msd.items():
                            if k in ema_state:
                                ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1 - ema_decay)
            except Exception as e:
                print("Exception caught. Nan from RWKV-7? Skipping batch.")
                print(e)
                log["train_nan"] = 1

            # Wilcoxon early-prune: growing 300n window over all common steps so far. Both modes must be
            # worse at p<alpha (no false positives: only abysmally-bad runs die). NaN-skipped steps are
            # simply absent from a trace; pairing uses the intersection.
            if (champ_meta is not None and step % prune_every == 0
                    and step >= max(prune_min_step, prune_every)):
                common = sorted(s for s in trace_steps if s in champ_steps)
                if len(common) >= 50:
                    from scipy.stats import wilcoxon
                    diff_a = [trace_steps[s][0] - champ_steps[s][0] for s in common]
                    diff_i = [trace_steps[s][1] - champ_steps[s][1] for s in common]
                    try:
                        p_a = float(wilcoxon(diff_a, alternative="greater").pvalue)
                        p_i = float(wilcoxon(diff_i, alternative="greater").pvalue)
                    except ValueError:  # all-zero diffs (identical traces), older scipy
                        p_a = p_i = 1.0
                    if math.isnan(p_a) or math.isnan(p_i):  # newer scipy: zero-diffs -> NaN
                        p_a, p_i = 1.0, 1.0
                    print(f"[prune] step {step}: n={len(common)} p_ahead={p_a:.3e} p_imm={p_i:.3e}")
                    if p_a < prune_alpha and p_i < prune_alpha:
                        s0 = common[-1]  # last paired step (== `step` unless it was NaN-skipped)
                        est_a = champ_meta["final_ahead"] + (trace_steps[s0][0] - champ_steps[s0][0])
                        est_i = champ_meta["final_imm"] + (trace_steps[s0][1] - champ_steps[s0][1])
                        marker = {
                            "pruned_at_step": step, "paired_steps": len(common),
                            "p_ahead": p_a, "p_imm": p_i, "alpha": prune_alpha,
                            "champion": champ_meta.get("name"),
                            "estimated_ahead": est_a, "estimated_imm": est_i,
                            "estimate_formula": "champ_final + (cand@s - champ@s), s=last paired step",
                            "estimated_ahead_meandiff": champ_meta["final_ahead"] + float(np.mean(diff_a)),
                            "estimated_imm_meandiff": champ_meta["final_imm"] + float(np.mean(diff_i)),
                        }
                        marker_path = (step_trace_path + ".pruned.json") if step_trace_path \
                            else "ws_prune_marker.json"
                        with open(marker_path, "w") as _f:
                            json.dump(marker, _f, indent=1)
                        print(f"PRUNED step={step} p_ahead={p_a:.3e} p_imm={p_i:.3e} "
                              f"est_ahead={est_a:.6f} est_imm={est_i:.6f} -> {marker_path}")
                        if trace_file is not None:
                            trace_file.close()
                        sys.exit(42)

            scheduler.step()

            if validate_iter:
                save_model_path = (
                    f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}.pth"
                )
                save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_optim_{step}.pth"
                Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
                torch.save(master_model.state_dict(), save_model_path)
                torch.save(optimizer.state_dict(), save_optim_path)
                if ema_state is not None:  # save the averaged weights for eval ({prefix}_ema_{step}.pth)
                    ema_full = {k: ema_state.get(k, v) for k, v in master_model.state_dict().items()}
                    torch.save(ema_full, f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_ema_{step}.pth")
                if _shift_cb_param is not None:  # export the LEARNED shift codebook (engine text format)
                    from rwkv.model import rwkv_model as _rwkv_model_mod
                    _rwkv_model_mod.shift_pq_export(
                        f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_shiftcb_{step}.txt"
                    )
                if _wkv_cb_param is not None:  # export the LEARNED WKV codebook (engine text format)
                    from rwkv.model import rwkv_ops as _rwkv_ops_mod2
                    _rwkv_ops_mod2.wkv_pq_export(
                        f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_wkvcb_{step}.txt"
                    )
                if _shift_rot_param is not None:  # export the LEARNED shift rotation (engine format)
                    from rwkv.model import rwkv_model as _rwkv_model_mod
                    _rwkv_model_mod.shift_rot_export(
                        f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_shiftrot_{step}.txt"
                    )
                print("MODEL SAVED.")
                elapsed = time.time() - group_start
                log["elapsed"] = elapsed
                log["steps per second"] = checkpoint_step_count / elapsed
                log["loss_n per second"] = checkpoint_loss_n / elapsed
                log["train_elapsed_min"] = (time.time() - train_start) / 60
                print("Elapsed:", elapsed)
                print("Steps per second:", checkpoint_step_count / elapsed)
                print("loss_n per second:", checkpoint_loss_n / elapsed)
                checkpoint_step_count = 0
                checkpoint_loss_n = 0
                group_start = time.time()
                model.copy_downcast_(master_model, dtype=config.DTYPE)
                validation_out = validate(
                    model, data_fetcher, all_db_keys, VALIDATION_USERS, config.DEVICE
                )
                if validation_out is not None:
                    log["validation_ahead_loss"], log["validation_imm_loss"] = (
                        validation_out
                    )
                    log["validation_nan"] = 0
                else:
                    log["validation_nan"] = 1

            if config.USE_WANDB:
                wandb.log(log, step=step)

    if trace_file is not None:
        trace_file.close()

    if bench_max_steps > 0 and bench_t0 is not None:
        torch.cuda.synchronize()
        bench_elapsed = time.time() - bench_t0
        measured = total_steps - (config.STEP_OFFSET + bench_warmup) + 1
        sps = measured / bench_elapsed if bench_elapsed > 0 else 0.0
        rps = bench_work / bench_elapsed if bench_elapsed > 0 else 0.0
        peak_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"BENCH_RESULT max_len={config.MAX_TRAIN_GLOBAL_LEN} steps_per_sec={sps:.4f} "
              f"reviews_per_sec={rps:.1f} peak_reserved_gb={peak_gb:.3f} "
              f"measured_steps={measured} elapsed_s={bench_elapsed:.2f}")


def main(config):
    _maybe_enable_determinism()
    # AUGMENTATION TOGGLE (Andrew 2026-06-29): the per-batch input-feature randomization in prepare()
    # (random ID-encoding vectors + random time-of-day baselines, drawn fresh each batch in the unseeded
    # fetch children) adds ~0.0024 run-to-run logloss variance, which SWAMPS the 0.0003 research-phase
    # acceptance gate. DISABLED by default = a FIXED augmentation seed -> deterministic objective
    # (variance ~0). Set env RWKV_AUGMENT_SEED=none to re-enable stochastic augmentation later.
    _aug = os.environ.get("RWKV_AUGMENT_SEED", "1234")
    augment_seed = None if _aug.strip().lower() in ("none", "off", "", "-1") else int(_aug)
    print(f"[augmentation] training fetch seed = {augment_seed} "
          f"({'STOCHASTIC (on)' if augment_seed is None else 'FIXED (disabled, run-to-run variance ~0)'})")
    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()

        prepare_processes = []
        for _ in range(config.NUM_FETCH_PROCESSES):
            process = multiprocessing.Process(
                target=prepare_data_train_test,
                args=(
                    config.TRAIN_DATASET_LMDB_PATH,
                    config.TRAIN_DATASET_LMDB_SIZE,
                    config.VALIDATE_DATASET_LMDB_PATH,
                    config.VALIDATE_DATASET_LMDB_SIZE,
                    task_queue,
                    batch_queue,
                    config.MAX_TRAIN_GLOBAL_LEN,
                    augment_seed,
                ),
            )
            process.start()
            prepare_processes.append(process)

        try:
            main_loop(config=config, task_queue=task_queue, batch_queue=batch_queue)
        except Exception:
            traceback.print_exc()
        finally:
            for process in prepare_processes:
                process.terminate()
            print("Killed processes.")


if __name__ == "__main__":
    config = parse_toml()
    if config.DEVICE.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE is set to CUDA, but this PyTorch build lacks CUDA support. "
                "Install a CUDA-enabled build or set DEVICE to 'cpu'."
            )
    else:
        print(
            f"Running on {config.DEVICE}. Training without CUDA is supported but will be significantly slower."
        )
    main(config)
