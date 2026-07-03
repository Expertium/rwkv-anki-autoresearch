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


def main_loop(config, task_queue, batch_queue):
    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = (
        SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        .selective_cast(config.DTYPE)
        .to(config.DEVICE)
    )
    optimizer = get_optimizer(config, master_model)

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
        _intended_wd = [g["weight_decay"] for g in optimizer.param_groups]
        optimizer.load_state_dict(
            torch.load(
                optim_path,
                weights_only=False,
            )
        )
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
    if ema_decay > 0:
        print(f"[ema] weight averaging ON, decay={ema_decay}, start after step {ema_start}")

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
                stats = model.get_loss(prepared_batch)
                if stats is None:
                    raise Exception("Stats is none.")
                if not torch.isfinite(stats.average_loss):  # NaN/inf safeguard: skip, don't backprop garbage
                    raise Exception("non-finite training loss")

                print(
                    f"{epoch_i} {group_i} {step}, all: {stats.average_loss.item():3f}, ahead: {stats.ahead_avg.item():.4f} ({stats.ahead_raw_avg.item():.4f}), imm: {stats.imm_avg.item():.3f}"
                )
                log["train_nan"] = 0
                stats.average_loss.backward()
                transfer_child_grad_to_master(master=master_model, child=model)

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
                total_norm = torch.nn.utils.clip_grad_norm_(master_model.parameters(), CLIP)
                if torch.isfinite(total_norm):
                    optimizer.step()
                else:
                    print("Non-finite grad norm; skipping optimizer step (weights protected).")
                    log["train_nan"] = 1
                optimizer.zero_grad()
                if ema_decay > 0 and step >= ema_start:
                    msd = master_model.state_dict()
                    if ema_state is None:  # init EMA to the post-warmup weights
                        ema_state = {k: v.detach().float().clone()
                                     for k, v in msd.items() if v.is_floating_point()}
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
