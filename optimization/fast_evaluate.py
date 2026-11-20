"""Experimental accelerated held-out evaluator with exact-parity checks against get_result.

Batches complete original sections within their owning user, preserving per-user section order,
reference seed, and skip scope. The regular evaluator remains the source of truth until the
full-VAL validation gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import datetime as dt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _user_keys(config, users):
    import lmdb

    env = lmdb.open(config.DATASET_LMDB_PATH, map_size=config.DATASET_LMDB_SIZE,
                    readonly=True, lock=False, readahead=False, max_readers=1024)
    result = {}
    try:
        with env.begin(write=False) as txn:
            for user in users:
                raw = txn.get(f"{user}_batches".encode())
                if raw is not None:
                    result[user] = [(user, *entry) for entry in json.loads(raw)]
    finally:
        env.close()
    return result


def _batch_groups(keys, capacity):
    """Pack complete LMDB sections in order, never splitting one section."""
    if capacity <= 0:
        raise ValueError("group capacity must be positive")
    group, work = [], 0
    for key in keys:
        size = int(key[3])
        if size < 0:
            raise ValueError(f"negative section row count in LMDB key: {key}")
        if group and work + size > capacity:
            yield group
            group, work = [], 0
        group.append(key)
        work += size
    if group:
        yield group


def _group_sections(user_keys, capacity):
    """Pack sections within each user and keep each group owner's ID list."""
    groups = []
    for user, sections in user_keys.items():
        groups.extend(([user], group) for group in _batch_groups(sections, capacity))
    return groups


def _prepare_group_job(args):
    from rwkv.prepare_batch import get_data
    import lmdb

    db_path, db_size, keys = args
    started = time.perf_counter()
    env = lmdb.open(db_path, map_size=db_size, readonly=True, lock=False,
                    readahead=False, max_readers=1024)
    try:
        with env.begin(write=False) as txn:
            samples = [get_data(txn, key, device="cpu") for key in keys]
    finally:
        env.close()
    return samples, [int(key[0]) for key in keys], len(keys), time.perf_counter() - started


def _apply_model_environment(config):
    """Apply model architecture and evaluation flags before importing RWKV modules."""
    names = {
        "ARCH_MODULE": "RWKV_ARCH_MODULE", "GRU_HEAD": "RWKV_GRU_HEAD",
        "INTERLEAVE": "RWKV_INTERLEAVE", "PAVA_LAMBDA": "RWKV_PAVA_LAMBDA",
        "MUON": "RWKV_MUON", "MUON_LR": "RWKV_MUON_LR",
        "WEIGHT_DECAY": "RWKV_WEIGHT_DECAY", "WEIGHT_DECAY_HEAD": "RWKV_WEIGHT_DECAY_HEAD",
        "NO_AHEAD_RESIDUAL": "RWKV_NO_AHEAD_RESIDUAL",
        "STRIP_L0_VLORA": "RWKV_STRIP_L0_VLORA", "STRIP_CMIX": "RWKV_STRIP_CMIX",
        "DROPOUT_SCALE": "RWKV_DROPOUT_SCALE", "ID_FEATURES": "RWKV_ID_FEATURES",
        "ZERO_FEATURES": "RWKV_ZERO_FEATURES",
        "REAL_CYCLES": "RWKV_REAL_CYCLES", "FINAL_LAYOUT": "RWKV_FINAL_LAYOUT",
        "CLOCK_AT_PREV_ANSWER": "RWKV_CLOCK_AT_PREV_ANSWER",
        "STATE_CLAMP_TAU": "RWKV_STATE_CLAMP_TAU",
        "STATE_CLAMP_WINDOW": "RWKV_STATE_CLAMP_WINDOW", "PROBE_DUR": "RWKV_PROBE_DUR",
        "EVAL_PAVA": "RWKV_EVAL_PAVA", "QAT_COMPILE": "RWKV_QAT_COMPILE",
        "EVAL_CAST_FP32": "RWKV_EVAL_CAST_FP32", "DETERMINISTIC": "RWKV_DETERMINISTIC",
        "AUGMENT_SEED": "RWKV_AUGMENT_SEED", "PROBE_DENSITY": "RWKV_PROBE_DENSITY",
    }
    for field, variable in names.items():
        if hasattr(config, field):
            value = getattr(config, field)
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = str(value)


def _load_model(config):
    _apply_model_environment(config)
    import torch
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.model.srs_model import SrsRWKV

    master = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    print("Loading:", config.MODEL_PATH, flush=True)
    state = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    master.load_state_dict(state, strict=True)
    del state
    master = master.to(config.DEVICE)
    model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).selective_cast(config.DTYPE).to(config.DEVICE)
    model.copy_downcast_(master, config.DTYPE)
    model.eval()
    del master
    return model


def _extract_group(stats, sample_users):
    """Extract the same labeled dictionaries that extract_p builds per user."""
    review_ths = stats.label_review_th.detach().cpu().numpy()
    elapsed = stats.label_elapsed_seconds.detach().cpu().numpy()
    ratings = stats.label_rating.detach().cpu().numpy()
    has_label = stats.has_label.detach().cpu().numpy().astype(bool)
    is_query = stats.is_query.detach().cpu().numpy().astype(bool)
    curves = stats.p_curve.detach().cpu().numpy()
    immediates = stats.p_imm.detach().cpu().numpy()
    records = []
    for i, user in enumerate(sample_users):
        valid = review_ths[i] >= 0
        keys = review_ths[i, valid]
        labeled = has_label[i, valid]
        query = is_query[i, valid]
        records.append({
            "user": user,
            "ahead": dict(zip(keys[labeled & ~query], curves[i, valid][labeled & ~query])),
            "imm": dict(zip(keys[labeled & query], immediates[i, valid][labeled & query])),
            "rating": dict(zip(keys[labeled], ratings[i, valid][labeled])),
            "elapsed": dict(zip(keys, elapsed[i, valid])),
        })
    return records


def _forward_group(model, samples, device):
    import torch
    from rwkv.prepare_batch import prepare
    import rwkv.prepare_batch as prepare_batch

    if len(samples) != 1:
        raise RuntimeError("one-user groups may contain only one original LMDB section")
    batch = prepare(
        samples, target_len=800000, seed=1234,
        probe_density=1.0 if prepare_batch._EVAL_PAVA else 0.0,
        probe_equalize_only=True,
    ).to(device)
    if os.environ.get("RWKV_EVAL_CAST_FP32", "0") == "1":
        batch.start = batch.start.float()
        if batch.labels.dtype == torch.bfloat16:
            batch.labels = batch.labels.float()
    with torch.inference_mode():
        stats = model.get_loss(batch)
    if stats is None:
        return None
    return _extract_group(stats, [sample.user_id for sample in samples])


def evaluate(config, users, group_capacity=50000, prep_workers=2, phase_label="evaluation"):
    if group_capacity <= 0:
        raise ValueError("group_capacity must be positive")
    if prep_workers <= 0:
        raise ValueError("prep_workers must be positive")
    if not phase_label or Path(phase_label).name != phase_label or any(ch in phase_label for ch in "\\/:*?\"<>|\r\n"):
        raise ValueError("phase_label must be a non-empty simple filename stem")
    if not users or len(set(users)) != len(users):
        raise ValueError("users must be a non-empty sequence of unique user IDs")
    if not getattr(config, "MODEL_PATH", None):
        raise ValueError("config must define MODEL_PATH")
    if not getattr(config, "DATASET_LMDB_PATH", None) or not getattr(config, "LABEL_FILTER_LMDB_PATH", None):
        raise ValueError("config must define dataset and label-filter LMDB paths")
    if not all(isinstance(user, int) and user >= 0 for user in users):
        raise ValueError("users must contain nonnegative integer IDs")
    os.environ.setdefault("RWKV_VRAM_CAP_GB", "11")
    if os.environ.get("RWKV_VRAM_CAP_GB") != "11":
        raise RuntimeError("fast evaluation requires RWKV_VRAM_CAP_GB=11")
    os.environ.setdefault("RWKV_EMPTY_CACHE_EVERY", "1")
    if os.environ.get("RWKV_EMPTY_CACHE_EVERY") != "1":
        raise RuntimeError("fast evaluation requires RWKV_EMPTY_CACHE_EVERY=1")
    if getattr(config, "RAW", False):
        raise ValueError("fast evaluation does not support RAW prediction exports")

    _apply_model_environment(config)
    os.environ["RWKV_EVAL_EMPTY_CACHE_EVERY"] = "1"
    os.environ.setdefault("RWKV_EVAL_PAVA", "0")
    os.environ.setdefault("RWKV_EVAL_CAST_FP32", "0")
    os.environ.setdefault("RWKV_DETERMINISTIC", "1")
    os.environ.setdefault("RWKV_AUGMENT_SEED", "4321")

    import lmdb
    import torch
    from rwkv import vram_cap
    from rwkv.get_result import get_stats
    from rwkv.utils import load_tensor

    os.environ.setdefault("RWKV_VRAM_CAP_GB", "11")
    os.environ.setdefault("RWKV_EMPTY_CACHE_EVERY", "1")
    os.environ.setdefault("RWKV_EMPTY_CACHE_WINDOW", "0")
    os.environ.setdefault("RWKV_NO_JIT", "1")
    vram_cap.apply()

    user_keys = _user_keys(config, users)
    empty = sorted(user for user, sections in user_keys.items() if not sections)
    if empty:
        raise RuntimeError(f"LMDB has empty section lists for requested users: {empty[:20]}")
    absent = sorted(set(users) - set(user_keys))
    if absent:
        raise RuntimeError(f"LMDB is missing requested users: {absent[:20]}")
    groups = _group_sections(user_keys, group_capacity)
    if not groups:
        raise RuntimeError("No LMDB samples found for requested users")
    group_lengths = [sum(int(key[3]) for key in group) for _, group in groups]
    print(f"Prepared {sum(map(len, user_keys.values()))} LMDB samples into {len(groups)} groups; "
          f"max group rows={max(group_lengths):,}", flush=True)
    oversized_sections = [key for sections in user_keys.values() for key in sections
                          if int(key[3]) > group_capacity]
    print(f"Sections larger than pack capacity={len(oversized_sections)}; each is processed alone", flush=True)
    import rwkv.prepare_batch as prepare_batch
    prepare_batch._EVAL_PAVA = os.environ.get("RWKV_EVAL_PAVA", "0") in ("1", "2", "3")
    model = _load_model(config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    predictions = defaultdict(lambda: {"ahead": {}, "imm": {}, "rating": {}})
    remaining = {user: len(user_keys[user]) for user in users}
    failed, completed = set(), set()
    outputs = {"ahead": [], "imm": []}
    skipped = {"oom": [], "nan": [], "incomplete": []}
    eval_started = time.perf_counter()
    eval_started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    timings = {"users": len(users), "groups": 0, "samples": 0,
               "eval_started_at": eval_started_at,
               "prepare_seconds": 0.0, "prepare_worker_seconds": 0.0,
               "prepare_queue_wait_seconds": 0.0,
               "forward_seconds": 0.0, "fallback_seconds": 0.0, "score_seconds": 0.0}
    filter_env = lmdb.open(config.LABEL_FILTER_LMDB_PATH,
                           map_size=config.LABEL_FILTER_LMDB_SIZE, readonly=True,
                           lock=False, max_readers=1024)
    fetch_limit = max(1, int(os.environ.get("RWKV_EVAL_FETCH_DEPTH", "4")))

    def record_sample(user, record):
        if user not in failed:
            predictions[user]["ahead"].update(record["ahead"])
            predictions[user]["imm"].update(record["imm"])
            predictions[user]["rating"].update(record["rating"])

    def score_user(user):
        if user in failed or user in completed:
            return
        with filter_env.begin(write=False) as txn:
            eq_key, bin_key = f"{user}_review_ths", f"{user}_rmse_bins"
            if txn.get(eq_key.encode()) is None or txn.get(bin_key.encode()) is None:
                skipped["incomplete"].append({"user": user, "reason": "missing label-filter keys"})
                failed.add(user)
                predictions.pop(user, None)
                return
            eq = load_tensor(txn, eq_key, device="cpu").tolist()
            bins = load_tensor(txn, bin_key, device="cpu").tolist()
        if not eq:
            skipped["incomplete"].append({"user": user, "reason": "empty equalized review set"})
            failed.add(user)
            predictions.pop(user, None)
            return
        if len(eq) != len(bins) or len(set(eq)) != len(eq):
            skipped["incomplete"].append({"user": user, "reason": "invalid equalized review/bin vectors"})
            failed.add(user)
            predictions.pop(user, None)
            return
        bins_by_th = dict(zip(eq, bins))
        started = time.perf_counter()
        try:
            user_stats = {}
            user_started = time.perf_counter()
            for mode in ("ahead", "imm"):
                user_stats[mode], _ = get_stats(user, eq, bins_by_th,
                                                predictions[user][mode], predictions[user]["rating"])
            timings["score_seconds"] += time.perf_counter() - user_started
            outputs["ahead"].append(user_stats["ahead"])
            outputs["imm"].append(user_stats["imm"])
        except (AssertionError, KeyError) as exc:
            skipped["incomplete"].append({"user": user, "reason": str(exc)})
            failed.add(user)
            predictions.pop(user, None)
        else:
            completed.add(user)
        predictions.pop(user, None)

    pool = ProcessPoolExecutor(max_workers=prep_workers,
                               mp_context=multiprocessing.get_context("spawn"))
    pending = []
    try:
        queue_depth = fetch_limit
        for group_users, group in groups:
            group_rows = sum(int(key[3]) for key in group)
            oversized = group_rows > group_capacity
            if oversized:
                # Never retain multiple over-cap user sections in RAM at once.
                while pending:
                    item = pending.pop(0)
                    _process_group(item, model, config.DEVICE, record_sample, predictions, score_user,
                                   remaining, failed, skipped, timings, torch)
            while len(pending) >= queue_depth:
                item = pending.pop(0)
                _process_group(item, model, config.DEVICE, record_sample, predictions, score_user,
                               remaining, failed, skipped, timings, torch)
            future = pool.submit(_prepare_group_job,
                                 (config.DATASET_LMDB_PATH, config.DATASET_LMDB_SIZE, group))
            pending.append((group_users, group, future))
            if oversized:
                item = pending.pop(0)
                _process_group(item, model, config.DEVICE, record_sample, predictions, score_user,
                               remaining, failed, skipped, timings, torch)
                while pending:
                    item = pending.pop(0)
                    _process_group(item, model, config.DEVICE, record_sample, predictions, score_user,
                                   remaining, failed, skipped, timings, torch)
        while pending:
            item = pending.pop(0)
            _process_group(item, model, config.DEVICE, record_sample, predictions, score_user,
                           remaining, failed, skipped, timings, torch)
    except BaseException:
        for _, _, future in pending:
            future.cancel()
        for _, _, future in pending:
            if future.done() and not future.cancelled():
                future.exception()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        filter_env.close()

    for user in users:
        if user not in failed and user not in completed:
            skipped["incomplete"].append({"user": user, "reason": "not all samples completed"})
            failed.add(user)
    for kind in ("oom", "nan", "incomplete"):
        unique = {}
        for entry in skipped[kind]:
            unique[(entry.get("user"), tuple(entry.get("users", ())), entry.get("reason"), entry.get("source"))] = entry
        skipped[kind] = list(unique.values())
    skipped["oom_users"] = sorted({entry["user"] for entry in skipped["oom"] if "user" in entry})
    skipped["oom_users"] = sorted(set(skipped["oom_users"]).union(
        user for entry in skipped["oom"] for user in entry.get("users", [])))
    skipped["nan_users"] = sorted({entry["user"] for entry in skipped["nan"] if "user" in entry})
    skipped["incomplete_users"] = sorted({entry["user"] for entry in skipped["incomplete"] if "user" in entry})
    skipped_users = set(skipped["oom_users"]) | set(skipped["nan_users"]) | set(skipped["incomplete_users"])
    if not failed.issubset(skipped_users):
        raise RuntimeError(f"failed users lack a skip record: {sorted(failed - skipped_users)}")
    if set(users) != completed | failed:
        raise RuntimeError("internal accounting error: users were not all completed or skipped")
    timings["completed_users"] = len(completed)
    timings["skipped_users"] = len(failed)
    timings["oom_users"] = len(skipped["oom_users"])
    timings["nan_users"] = len(skipped["nan_users"])
    timings["incomplete_users"] = len(skipped["incomplete_users"])
    timings["largest_input_section_rows"] = max(
        (int(key[3]) for sections in user_keys.values() for key in sections), default=0)
    timings["total_eval_seconds"] = time.perf_counter() - eval_started
    timings["eval_ended_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    timings["actual_cuda_max_allocated_gib"] = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                                                   if torch.cuda.is_available() else 0.0)
    timings["actual_cuda_max_reserved_gib"] = (torch.cuda.max_memory_reserved() / (1024 ** 3)
                                                  if torch.cuda.is_available() else 0.0)
    timings["input_samples"] = sum(len(sections) for sections in user_keys.values())
    timings["input_review_rows"] = sum(int(key[3]) for sections in user_keys.values() for key in sections)
    timings["output_rows_ahead"] = len(outputs["ahead"])
    timings["output_rows_imm"] = len(outputs["imm"])
    return outputs, timings, skipped


def _process_group(item, model, device, record_sample, predictions, score_user,
                   remaining, failed, skipped, timings, torch):
    group_users, group, future = item
    started = time.perf_counter()
    samples, sample_users, sample_count, worker_time = future.result()
    prepare_wait = time.perf_counter() - started
    timings["prepare_seconds"] += prepare_wait
    timings["prepare_worker_seconds"] += worker_time
    timings["prepare_queue_wait_seconds"] += max(0.0, prepare_wait - worker_time)
    timings["samples"] += sample_count
    if set(sample_users) != set(group_users):
        raise RuntimeError(f"group user metadata mismatch: expected {group_users}, got {sorted(set(sample_users))}")
    if len(samples) != len(sample_users) or len(samples) != sample_count:
        raise RuntimeError("prepared sample metadata length mismatch")
    if any(sample.user_id != user for sample, user in zip(samples, sample_users)):
        raise RuntimeError("prepared sample user does not match its LMDB key")

    # Keep each original section as a separate model forward. This matches get_result's
    # per-section RNG seed, reduction order, and whole-user OOM/NaN skip behavior.
    for sample, user in zip(samples, sample_users):
        if user in failed:
            remaining[user] -= 1
            continue
        started = time.perf_counter()
        try:
            records = _forward_group(model, [sample], device)
        except torch.OutOfMemoryError as exc:
            timings["forward_seconds"] += time.perf_counter() - started
            failed.add(user)
            skipped["oom"].append({"user": user, "error": str(exc)})
            print(f"FAST_EVAL_OOM user={user} section_rows={sample.num_data} error={exc}", flush=True)
            predictions.pop(user, None)
            remaining[user] -= 1
            torch.cuda.empty_cache()
            continue
        timings["forward_seconds"] += time.perf_counter() - started
        if records is None:
            failed.add(user)
            skipped["nan"].append({"user": user, "source": "single-section NaN"})
            print(f"FAST_EVAL_NAN user={user}", flush=True)
            predictions.pop(user, None)
        else:
            record_sample(user, records[0])
        remaining[user] -= 1
    for user in group_users:
        if remaining[user] == 0 and user not in failed:
            score_user(user)
    timings["groups"] += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group-capacity", type=int, default=50000,
                        help="Approximate total review rows per inference group")
    parser.add_argument("--prep-workers", type=int, default=2)
    parser.add_argument("--users", help="Comma-separated user IDs; default uses config range")
    parser.add_argument("--users-file", help="JSON array of explicit user IDs")
    parser.add_argument("--allow-partial-reference", action="store_true",
                        help="Permit parity only on common users; never use with --result-prefix")
    parser.add_argument("--timing-only", action="store_true",
                        help="Run evaluation and save timing/skip report without reference comparison")
    parser.add_argument("--exclude", help="Comma-separated user IDs to skip")
    parser.add_argument("--compare", action="store_true",
                        help="Compare user sets, sizes, and stored metrics with references")
    parser.add_argument("--reference-prefix", help="Reference ahead tag; imm tag is derived")
    parser.add_argument("--result-prefix", help="Write non-overwriting ahead/imm JSONL files")
    parser.add_argument("--fail-on-skips", action="store_true",
                        help="Return nonzero if any user has an OOM, NaN, or incomplete skip")
    parser.add_argument("--output", help="Timing and skip report JSON path")
    parser.add_argument("--phase-label", default="adhoc",
                        help="Phase label for the default output path; use a new value per run")
    args = parser.parse_args()

    import tomli
    import torch
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with open(args.config, "rb") as handle:
        values = tomli.load(handle)
    config = argparse.Namespace(**values)
    _apply_model_environment(config)
    config.DEVICE = torch.device(values.get("DEVICE", "cpu"))
    config.DTYPE = torch.bfloat16 if values.get("DTYPE") == "bfloat16" else torch.float32
    if args.users_file and args.users:
        parser.error("use only one of --users and --users-file")
    if args.users_file and not Path(args.users_file).is_file():
        parser.error(f"user roster file does not exist: {args.users_file}")
    if args.users_file:
        user_values = json.loads(Path(args.users_file).read_text(encoding="utf-8"))
        if not isinstance(user_values, list) or not all(type(item) is int for item in user_values):
            parser.error("--users-file must contain a JSON array of integer IDs")
        users = user_values
    elif args.users:
        try:
            users = [int(item) for item in args.users.split(",")]
        except ValueError:
            parser.error("--users must be comma-separated integer IDs")
    else:
        users = list(range(config.USER_START, config.USER_END + 1))
    requested_users = list(users)
    args.requested_users = requested_users
    if len(set(requested_users)) != len(requested_users):
        parser.error("requested user roster contains duplicates")
    excluded = {int(item) for item in (args.exclude or "").split(",") if item.strip()}
    users = [user for user in requested_users if user not in excluded]
    result, timings, skipped = evaluate(config, users, args.group_capacity, args.prep_workers,
                                        phase_label=args.phase_label)
    if args.compare and not args.reference_prefix:
        parser.error("--compare requires --reference-prefix")
    if args.timing_only and args.compare:
        parser.error("--timing-only cannot be combined with --compare")
    if args.timing_only:
        print("FAST_EVAL_TIMING_ONLY: no metric parity check was requested", flush=True)

    if args.compare:
        if not args.reference_prefix.startswith("RWKV-"):
            parser.error("--reference-prefix must start with RWKV-")
        if args.reference_prefix.startswith("RWKV-P-"):
            parser.error("--reference-prefix must name the ahead RWKV result file, not the RWKV-P file")
        if Path(args.reference_prefix).name != args.reference_prefix:
            parser.error("--reference-prefix must be a simple filename stem")
        common_users = None
        for mode in ("ahead", "imm"):
            tag = (args.reference_prefix if mode == "ahead"
                   else "RWKV-P-" + args.reference_prefix[len("RWKV-"):])
            reference = ROOT / "result" / f"{tag}.jsonl"
            reference_rows = [json.loads(line) for line in reference.read_text(encoding="utf-8").splitlines() if line.strip()]
            expected = {row["user"]: row for row in reference_rows}
            if len(expected) != len(reference_rows):
                raise SystemExit(f"{mode}: reference contains duplicate user rows")
            observed = {row["user"]: row for row in result[mode]}
            if len(observed) != len(result[mode]):
                raise SystemExit(f"{mode}: fast outputs contain duplicate user rows")
            common = set(observed) & set(expected) & set(users)
            if common_users is None:
                common_users = common
            elif common != common_users:
                raise SystemExit("ahead and imm reference user intersections differ")
            all_ahead_users = {row["user"] for row in result["ahead"]}
            all_imm_users = {row["user"] for row in result["imm"]}
            if all_ahead_users != all_imm_users or all_ahead_users != set(observed):
                raise SystemExit("fast ahead and imm output user sets differ")
            for user in sorted(common):
                row, baseline = observed[user], expected[user]
                if row["size"] != baseline["size"]:
                    raise SystemExit(f"{mode}: size mismatch for user {user}")
                for metric in ("RMSE", "LogLoss", "RMSE(bins)", "AUC"):
                    left, right = row["metrics"][metric], baseline["metrics"][metric]
                    equal_nan = isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isnan(left) and math.isnan(right)
                    if left != right and not equal_nan:
                        raise SystemExit(f"{mode}: {metric} mismatch for user {user}: {left} != {right}")
            fast_only = set(observed) - set(expected)
            ref_only = set(expected).intersection(users) - set(observed)
            unexpected = set(observed) - set(users)
            if unexpected:
                raise SystemExit(f"{mode}: fast evaluator returned unrequested users: {sorted(unexpected)}")
            print(f"{mode} parity common={len(common)} fast_only={len(fast_only)} reference_only={len(ref_only)}", flush=True)
            if fast_only and not args.allow_partial_reference:
                raise SystemExit(f"{mode}: fast outputs users absent from reference: {sorted(fast_only)}")
            if common != set(users) and not args.allow_partial_reference:
                raise SystemExit(f"{mode}: incomplete reference set; use --allow-partial-reference only for diagnostic checks")
        if not common_users:
            raise SystemExit("no common users available for reference parity")
        print("FAST_EVAL_PARITY_OK", flush=True)

    if args.fail_on_skips and any(skipped[f"{kind}_users"] for kind in ("oom", "nan", "incomplete")):
        raise SystemExit(f"evaluation had skipped users: {skipped}")

    if args.result_prefix:
        if not args.output:
            parser.error("--result-prefix requires an explicit --output report path")
        if len(set(users)) != len(users):
            parser.error("result-writing requires a unique user roster")
        if args.allow_partial_reference:
            parser.error("--allow-partial-reference cannot be used with --result-prefix")
        if not args.compare:
            parser.error("--result-prefix requires --compare as a parity gate")
        if args.exclude or users != args.requested_users:
            parser.error("result-writing requires the full, unfiltered requested roster")
        if not args.phase_label or args.phase_label == "adhoc":
            parser.error("--result-prefix requires a unique --phase-label")
        if Path(args.phase_label).name != args.phase_label:
            parser.error("--phase-label must be a simple filename stem")
        if skipped["oom_users"] or skipped["nan_users"] or skipped["incomplete_users"]:
            raise SystemExit("refusing to write result JSONL while users were skipped")
        if Path(args.result_prefix).name != args.result_prefix or not args.result_prefix or args.result_prefix.startswith("RWKV-"):
            parser.error("--result-prefix must be a simple filename stem without the RWKV- prefix")
        ahead_path = ROOT / "result" / f"RWKV-{args.result_prefix}.jsonl"
        imm_path = ROOT / "result" / f"RWKV-P-{args.result_prefix}.jsonl"
        if not args.output:
            parser.error("--result-prefix requires an explicit --output report path")
        report_path = Path(args.output)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        if ahead_path.exists() or imm_path.exists() or report_path.exists():
            raise SystemExit(f"refusing to overwrite result/report files: {ahead_path}, {imm_path}, {report_path}")
        for path, rows in ((ahead_path, result["ahead"]), (imm_path, result["imm"])):
            with path.open("x", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row) + "\n" for row in rows)

    report = {"ahead": result["ahead"], "imm": result["imm"],
              "timing": timings, "skipped": skipped, "mode": "batched-bf16"}
    output = Path(args.output) if args.output else ROOT / "result" / f"fast-batch-{Path(args.config).stem}-{args.phase_label}.json"
    if not output.is_absolute():
        output = ROOT / output
    if args.result_prefix and not args.output:
        raise SystemExit("--result-prefix requires an explicit --output report path")
    if output.exists():
        raise SystemExit(f"refusing to overwrite evaluation report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, separators=(",", ":")))
    print(f"FAST_EVAL_OK users={timings['completed_users']}/{timings['users']} "
          f"prepare={timings['prepare_seconds']:.1f}s forward={timings['forward_seconds']:.1f}s "
          f"output={output}", flush=True)
    print("FAST_EVAL_TIMING " + json.dumps(timings, separators=(",", ":")), flush=True)
    print("FAST_EVAL_SKIPS " + json.dumps(skipped, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
