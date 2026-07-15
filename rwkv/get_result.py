"""
This script takes a trained model and a list of users and produces a result file.
"""

import json
import multiprocessing
import os
from pathlib import Path
import traceback
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
import torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.data_fetcher import DataFetcher
from rwkv.model.srs_model import SrsRWKV, extract_p
from rwkv.parse_toml import parse_toml
from rwkv.prepare_batch import prepare_data
from rwkv.utils import load_tensor, save_tensor  # type: ignore

# Prefetch depth in USERS (was 20; cut 2026-07-08 for RAM): eval batches are single-user and can
# reach 800k reviews, so 20-deep queues several GB of prepared batches while the GPU-bound shards
# wait ~4 ms per fetch anyway. Depth only throttles the workers -- numerics/order untouched.
FETCH_AHEAD = 5


# label_filter env opened ONCE per process and cached (2026-07-08): the old per-user
# lmdb.open(map_size=40GB) never closed its envs -- after ~2000 users the accumulated
# environments exhausted process resources and the next open died with a bogus
# "No such file or directory" (killed shard 0 of the first champ5k eval at user 2007,
# swallowed by main()'s catch-all -> partial results, caught by the n=5000 finish gate).
# readonly + lock=False: the db is a frozen deterministic cache, readers need no lock.
_equalize_envs = {}


def get_benchmark_info(db_path, db_size, user_id):
    equalize_env = _equalize_envs.get(db_path)
    if equalize_env is None:
        equalize_env = lmdb.open(db_path, db_size, readonly=True, lock=False,
                                 max_readers=1024)
        _equalize_envs[db_path] = equalize_env
    key_review_ths = f"{user_id}_review_ths"
    key_rmse_bins = f"{user_id}_rmse_bins"
    with equalize_env.begin(write=False) as txn:
        if txn.get(key_review_ths.encode()) is not None:
            return (
                load_tensor(txn, key_review_ths, device="cpu").tolist(),
                load_tensor(txn, key_rmse_bins, device="cpu").tolist(),
            )
    return [], []


def _eq_gather(d, eq_int64, what):
    """Vectorized eq-ordered gather of d[th] for numpy-valued dicts: same values and the
    same presence guarantee as the old per-element `assert th in d` loop."""
    keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
    vals = np.asarray(list(d.values()))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    pos = np.searchsorted(keys, eq_int64)
    if len(keys):
        pos_clip = np.minimum(pos, len(keys) - 1)
        found = keys[pos_clip] == eq_int64
    else:
        pos_clip = pos
        found = np.zeros(len(eq_int64), dtype=bool)
    assert bool(found.all()), f"{eq_int64[~found][0]} not found in {what}"
    return vals[order[pos_clip]]


def _get_stats_fast(user_id, equalize_review_ths, rmse_bins_dict, pred_dict, label_rating_dict):
    """Vectorized get_stats for numpy-scalar dicts (the get_result path). Produces the exact
    arrays the old loop fed sklearn/pandas (same dtypes, same within-bin row order), so all
    metrics are bit-identical; only the per-review Python loops are gone."""
    eq64 = np.asarray(equalize_review_ths, dtype=np.int64)
    preds = _eq_gather(pred_dict, eq64, "pred_dict")
    ratings = _eq_gather(label_rating_dict, eq64, "label_rating_dict")
    label_ys = np.clip(ratings, a_min=0, a_max=1)  # 0-3 -> 0-1
    bins_v = _eq_gather(rmse_bins_dict, eq64, "rmse_bins_dict")

    rmse_raw = root_mean_squared_error(y_true=label_ys, y_pred=preds)
    logloss = log_loss(y_true=label_ys, y_pred=preds, labels=[0, 1])
    try:
        auc = round(roc_auc_score(y_true=label_ys, y_score=preds), 6)
    except Exception:
        auc = None
    if auc is not None and np.isnan(auc):
        auc = None

    # The old path built pd.DataFrame(list-of-rows), whose dtype numpy promotes from the
    # mixed row scalars; probe one row the same way so the frame dtype (and therefore the
    # groupby-mean arithmetic) is unchanged. Row order within a bin also matches the old
    # bin-blocked frame, so per-group accumulation order -- and the sums -- are identical.
    probe_dtype = np.array([[bins_v[0], label_ys[0], preds[0], 1]]).dtype
    tmp = pd.DataFrame(
        {
            "bin": bins_v.astype(probe_dtype),
            "y": label_ys.astype(probe_dtype),
            "p": preds.astype(probe_dtype),
            "weights": np.ones(len(eq64), dtype=probe_dtype),
        }
    )
    tmp = (
        tmp.groupby("bin")
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    rmse_bins = root_mean_squared_error(
        tmp["y"], tmp["p"], sample_weight=tmp["weights"]
    )

    print(
        f"rmse raw: {rmse_raw:.4f}, logloss: {logloss:.4f}, rmse_bins: {rmse_bins:.4f}, auc: {np.nan if auc is None else auc:.4f}, len: {len(equalize_review_ths)}"
    )
    if len(equalize_review_ths) >= 5e5:
        print("Emptying cache.")
        torch.cuda.empty_cache()

    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "AUC": auc,
        },
        "user": int(user_id),
        "size": len(equalize_review_ths),
    }
    raw = {
        "user": int(user_id),
        "size": len(equalize_review_ths),
        "p": preds.tolist(),
        "y": label_ys.tolist(),
        "review_th": equalize_review_ths,
    }
    return stats, raw


def get_stats(
    user_id, equalize_review_ths, rmse_bins_dict, pred_dict, label_rating_dict
):
    # Fast path for the get_result eval (numpy-scalar dicts from extract_p). The RNN/trace
    # callers (run_as_rnn, export_rnn_trace) pass tensor-valued dicts and keep the old loop.
    if len(equalize_review_ths) and isinstance(
        next(iter(pred_dict.values()), None), np.generic
    ) and isinstance(next(iter(label_rating_dict.values()), None), np.generic):
        return _get_stats_fast(
            user_id, equalize_review_ths, rmse_bins_dict, pred_dict, label_rating_dict
        )
    gather_pred = []
    gather_y = []
    bin_pred = {}
    bin_y = {}
    y_dict = {}
    for label_review_th in equalize_review_ths:
        assert label_review_th in pred_dict, f"{label_review_th} not found in pred_dict"
        assert label_review_th in label_rating_dict, (
            f"{label_review_th} not found in label_rating_dict"
        )
        label_y = np.clip(
            label_rating_dict[label_review_th], a_min=0, a_max=1
        )  # 0-3 -> 0-1
        y_dict[label_review_th] = label_y
        pred = pred_dict[label_review_th]
        gather_pred.append(pred)
        gather_y.append(label_y)

        bin = rmse_bins_dict[label_review_th]
        if bin not in bin_pred:
            bin_pred[bin] = []
        bin_pred[bin].append(pred)
        if bin not in bin_y:
            bin_y[bin] = []
        bin_y[bin].append(label_y)

    assert len(equalize_review_ths) == len(gather_pred)
    rmse_raw = root_mean_squared_error(y_true=gather_y, y_pred=gather_pred)
    logloss = log_loss(y_true=gather_y, y_pred=gather_pred, labels=[0, 1])

    try:
        auc = round(roc_auc_score(y_true=gather_y, y_score=gather_pred), 6)
    except Exception:
        auc = None
    if auc is not None and np.isnan(auc):
        auc = None

    rows = []
    for bin in bin_pred.keys():
        for y, pred in zip(bin_y[bin], bin_pred[bin]):
            rows.append([bin, y, pred, 1])
    assert len(rows) == len(equalize_review_ths)

    tmp = pd.DataFrame(rows, columns=["bin", "y", "p", "weights"])
    tmp = (
        tmp.groupby("bin")
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    rmse_bins = root_mean_squared_error(
        tmp["y"], tmp["p"], sample_weight=tmp["weights"]
    )

    print(
        f"rmse raw: {rmse_raw:.4f}, logloss: {logloss:.4f}, rmse_bins: {rmse_bins:.4f}, auc: {np.nan if auc is None else auc:.4f}, len: {len(equalize_review_ths)}"
    )
    if len(equalize_review_ths) >= 5e5:
        print("Emptying cache.")
        torch.cuda.empty_cache()

    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "AUC": auc,
        },
        "user": int(user_id),
        "size": len(equalize_review_ths),
    }

    raw = {
        "user": int(user_id),
        "size": len(equalize_review_ths),
        "p": [pred_dict[review_th].tolist() for review_th in equalize_review_ths],
        "y": [y_dict[review_th].tolist() for review_th in equalize_review_ths],
        "review_th": equalize_review_ths,
    }
    return stats, raw


def get_test_keys_batch(config, users):
    dataset = lmdb.open(config.DATASET_LMDB_PATH, map_size=config.DATASET_LMDB_SIZE)
    keys = {}
    with dataset.begin(write=False) as txn:
        for user_id in users:
            user_batches_raw = txn.get(f"{user_id}_batches".encode())
            if user_batches_raw is None:
                print("No data found for user", {user_id})
                continue

            batches = json.loads(user_batches_raw)
            keys[user_id] = list(map(lambda x: (user_id, x[0], x[1], x[2]), batches))
    dataset.close()
    return keys


def run(
    config,
    task_queue,
    batch_queue,
    users,
    ahead_users_result,
    ahead_users_raw,
    imm_users_result,
    imm_users_raw,
    ahead_path_result,
    ahead_path_raw,
    imm_path_result,
    imm_path_raw,
):
    data_fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)

    master_model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = (
        SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        .selective_cast(config.DTYPE)
        .to(config.DEVICE)
    )
    print("Loading:", config.MODEL_PATH)
    master_model.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))
    model.copy_downcast_(master_model, dtype=config.DTYPE)
    model.eval()

    all_db_keys = get_test_keys_batch(config, users)

    # RAW db opened ONCE: lmdb refuses to re-open the same env in-process, so the old
    # per-user open crashed on the second user (dormant RAW-path bug, hit 2026-07-03).
    raw_db = (
        lmdb.open(config.RAW_DB_PATH, map_size=config.RAW_DB_SIZE) if config.RAW else None
    )

    for i in range(min(len(users), FETCH_AHEAD)):
        user_id = users[i]
        batches = all_db_keys[user_id]
        for batch_i, batch in enumerate(batches):
            data_fetcher.enqueue((f"validate-{user_id}-{batch_i}", [batch]))

    with torch.no_grad():
        for i, user_id in enumerate(users):
            batches = all_db_keys[user_id]
            print("User:", user_id, "key:", batches)
            equalize_review_ths, rmse_bins = get_benchmark_info(
                config.LABEL_FILTER_LMDB_PATH, config.LABEL_FILTER_LMDB_SIZE, user_id
            )
            rmse_bins_dict = {
                equalize_review_ths[i]: rmse_bins[i]
                for i in range(len(equalize_review_ths))
            }
            if i + FETCH_AHEAD < len(users):
                fetch_user_id = users[i + FETCH_AHEAD]
                next_batch = all_db_keys[fetch_user_id]
                for batch_i, batch in enumerate(next_batch):
                    data_fetcher.enqueue(
                        (f"validate-{fetch_user_id}-{batch_i}", [batch])
                    )

            ahead_ps = {}
            imm_ps = {}
            label_ratings = {}
            label_elapsed_seconds = {}
            imm_ps_all = {}
            w_list = []
            nan_batches = []
            for batch_i, batch in enumerate(batches):
                print("batch_i, batch:", batch_i, batch)
                batch = data_fetcher.get(f"validate-{user_id}-{batch_i}")
                if nan_batches:
                    # user already failed -- drain the prefetched batch (frees fetcher RAM)
                    # but skip the forward
                    continue
                batch = batch.to(config.DEVICE)
                if os.environ.get("RWKV_EVAL_CAST_FP32", "0") == "1":
                    # LMDB batches are stored bf16; an fp32 eval (DTYPE="float") needs the
                    # float tensors upcast or matmuls die on mixed dtypes. Diagnostic path
                    # (A0 NaN probe 2026-07-15); default off = byte-identical.
                    batch.start = batch.start.float()
                    if batch.labels.dtype == torch.bfloat16:
                        batch.labels = batch.labels.float()

                with torch.inference_mode():
                    stats = model.get_loss(batch)
                    if stats is None:
                        # get_loss's NaN guard fired (model emitted NaN logits -- first hit
                        # 2026-07-15: the 1-ep d=128 A0 NaNs on 1,048,576-token mega-chunks
                        # it never saw at MAX=32768 training). Record + skip the USER
                        # entirely (no partial rows -- partial stats would silently change
                        # that user's equalized "size").
                        print(f"NAN_SKIP user {user_id} batch {batch_i} {batches[batch_i]}")
                        nan_batches.append(batch_i)
                        torch.cuda.empty_cache()
                        continue
                    print(
                        f"{user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f}, imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}"
                    )
                    dict_stats = extract_p(stats)
                    # in-place update == the old {**a, **b} rebuild (same last-wins
                    # semantics), minus the per-batch full-dict copy
                    ahead_ps.update(dict_stats.ahead_ps)
                    imm_ps.update(dict_stats.imm_ps)
                    label_ratings.update(dict_stats.label_ratings)
                    label_elapsed_seconds.update(dict_stats.label_elapsed_seconds)
                    imm_ps_all.update(dict_stats.imm_ps_all)
                    w_list.append(dict_stats.w)
                    if len(dict_stats.label_ratings) > 300000:
                        print("Emptying cache.")
                        torch.cuda.empty_cache()

                    dict_stats = None  # future-proofing

            # stats = stats_batch[0]
            # for i in range(1, len(stats_batch)):
            #     print(type(stats), type(stats_batch[i]))
            #     stats = add_stats(stats, stats_batch[i])

            # if len(stats_batch) > 1:
            #     print(f"ALL {user_id} ahead_loss: {stats.ahead_equalize_avg.item():.3f}, imm_loss: {stats.imm_binary_equalize_avg.item():.3f}, imm_n: {stats.imm_binary_equalize_n}")

            if (i + 1) % 20 == 0:
                print("Emptying cache.")
                torch.cuda.empty_cache()

            if nan_batches:
                with open(f"result/{config.FILE_AHEAD}.nanskip.jsonl", "a") as f:
                    f.write(json.dumps({"user": user_id, "nan_batches": nan_batches,
                                        "n_batches": len(batches)}) + "\n")
                continue

            ahead_stats, ahead_raw = get_stats(
                user_id, equalize_review_ths, rmse_bins_dict, ahead_ps, label_ratings
            )
            imm_stats, imm_raw = get_stats(
                user_id, equalize_review_ths, rmse_bins_dict, imm_ps, label_ratings
            )
            # ahead_raw["label_elapsed_seconds"] = [dict_stats.label_elapsed_seconds[review_th] for review_th in equalize_review_ths]
            # ahead_raw["w"] = dict_stats.w
            # print(type(ahead_raw['w'][0][0]))
            # ahead_raw["s"] = [dict_stats.s[review_th] for review_th in equalize_review_ths]
            # ahead_raw["d"] = [dict_stats.d[review_th] for review_th in equalize_review_ths]
            # vectorized gathers; .tolist() -> plain floats, same values the old per-th
            # comprehensions produced (np scalars/arrays are not JSON serializable --
            # dormant RAW-path bug, hit 2026-07-03 by the entropy-floor analysis)
            eq64 = np.asarray(equalize_review_ths, dtype=np.int64)
            imm_raw["label_elapsed_seconds"] = _eq_gather(
                label_elapsed_seconds, eq64, "label_elapsed_seconds"
            ).tolist()
            imm_raw["p_all"] = _eq_gather(imm_ps_all, eq64, "imm_ps_all").tolist()
            # print(ahead_raw["s"])
            # print(imm_raw["p_all"])

            def write(data, filter_set, path):
                if user_id not in filter_set:
                    with open(path, "a") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

            write(ahead_stats, ahead_users_result, ahead_path_result)
            write(imm_stats, imm_users_result, imm_path_result)
            if config.RAW:
                w_tensor = torch.cat(w_list, dim=0)
                w_equalized = w_tensor[equalize_review_ths]
                with raw_db.begin(write=True) as txn:
                    save_tensor(txn, f"{user_id}_w", w_equalized)

                write(ahead_raw, ahead_users_raw, ahead_path_raw)
                write(imm_raw, imm_users_raw, imm_path_raw)


def sort_jsonl(file):
    data = list(map(lambda x: json.loads(x), open(file).readlines()))
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8") as jsonl_file:
        for json_data in data:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    return data


def main(config):
    # Optional explicit user list (JSON array of ids) -- used by optimization/eval_sharded.py to
    # run size-balanced shards in parallel processes. Absent -> the original USER_START..USER_END
    # range, byte-identical behavior. Selection only; per-user numerics are untouched.
    users_file = getattr(config, "USERS_FILE", "")
    if users_file:
        with open(users_file, encoding="utf-8") as fh:
            target_users = sorted(json.load(fh))
    else:
        target_users = list(range(config.USER_START, config.USER_END + 1))

    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)
    path_ahead_result = Path(f"result/{config.FILE_AHEAD}.jsonl")
    path_imm_result = Path(f"result/{config.FILE_IMM}.jsonl")
    path_ahead_raw = Path(f"raw/{config.FILE_AHEAD}.jsonl")
    path_imm_raw = Path(f"raw/{config.FILE_IMM}.jsonl")

    def fetch(path):
        if path.exists():
            data = sort_jsonl(path)
            result = set(map(lambda x: x["user"], data))
            assert len(data) == len(result)
        else:
            result = set()
        return result

    ahead_users_result = fetch(path_ahead_result)
    imm_users_result = fetch(path_imm_result)
    ahead_users_raw = fetch(path_ahead_raw)
    imm_users_raw = fetch(path_imm_raw)

    # Users recorded as NaN-skipped (get_loss NaN guard) count as processed on resume --
    # re-running them would just re-NaN and duplicate skip lines.
    nanskip_path = Path(f"result/{config.FILE_AHEAD}.nanskip.jsonl")
    nanskip_users = set()
    if nanskip_path.exists():
        with open(nanskip_path, encoding="utf-8") as fh:
            nanskip_users = {json.loads(line)["user"] for line in fh if line.strip()}
        print(f"nanskip resume: {len(nanskip_users)} users already recorded as NaN-skipped")

    unprocessed_users = []
    for user_id in target_users:
        if user_id in nanskip_users:
            continue
        if (
            config.RAW
            and user_id in ahead_users_result
            and user_id in imm_users_result
            and user_id in ahead_users_raw
            and user_id in imm_users_raw
        ):
            continue
        if (
            not config.RAW
            and user_id in ahead_users_result
            and user_id in imm_users_result
        ):
            continue
        unprocessed_users.append(user_id)

    unprocessed_users.sort()
    print("Unprocessed users length:", len(unprocessed_users))

    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()

        prepare_processes = []
        for _ in range(config.NUM_FETCH_PROCESSES):
            process = multiprocessing.Process(
                target=prepare_data,
                args=(
                    config.DATASET_LMDB_PATH,
                    config.DATASET_LMDB_SIZE,
                    task_queue,
                    batch_queue,
                    800000,
                    1234,
                ),
            )
            process.start()
            prepare_processes.append(process)

        try:
            run(
                config,
                task_queue,
                batch_queue,
                unprocessed_users,
                ahead_users_result,
                ahead_users_raw,
                imm_users_result,
                imm_users_raw,
                path_ahead_result,
                path_ahead_raw,
                path_imm_result,
                path_imm_raw,
            )
        except Exception:
            # Print AND re-raise: the old swallow made a mid-run crash exit 0 (hit 2026-07-15:
            # AttributeError at user 6701 -> shard "succeeded" with 1700/5000 users).
            traceback.print_exc()
            raise
        finally:
            for process in prepare_processes:
                process.terminate()

            print("Killed processes.")
            # exists-guard: a run whose only users were all NaN-skipped never creates the
            # result files (hit by the fp32 probe 2026-07-15)
            for _p in ([path_ahead_result, path_imm_result]
                       + ([path_ahead_raw, path_imm_raw] if config.RAW else [])):
                if _p.exists():
                    sort_jsonl(_p)
            print("Sorted files.")


if __name__ == "__main__":
    config = parse_toml()
    main(config)
