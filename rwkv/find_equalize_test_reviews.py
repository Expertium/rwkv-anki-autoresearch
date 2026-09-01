import psutil
from multiprocessing import Pool
import torch
from tqdm import tqdm
from config import Config, create_parser
from utils import get_bin
from features import create_features
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import lmdb

from rwkv.parse_toml import parse_toml
from rwkv.utils import save_tensor
import rwkv.id_features as _idf

rwkv_config = parse_toml()
lmdb_env = None

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)
config.model_name = rwkv_config.ALGO
config.include_short_term = bool(rwkv_config.SHORT)
config.use_secs_intervals = bool(rwkv_config.SECS)


def _open_lmdb_env():
    max_readers = getattr(rwkv_config, "LMDB_MAX_READERS", None)
    if max_readers is None:
        processes = getattr(rwkv_config, "PROCESSES", 1)
        max_readers = max(128, processes * 8)
    return lmdb.open(
        rwkv_config.LABEL_FILTER_LMDB_PATH,
        map_size=rwkv_config.LABEL_FILTER_LMDB_SIZE,
        max_readers=max_readers,
    )


def process(user_id):
    global lmdb_env
    if lmdb_env is None:
        lmdb_env = _open_lmdb_env()

    key_review_ths = f"{user_id}_review_ths"
    key_rmse_bins = f"{user_id}_rmse_bins"
    with lmdb_env.begin(write=False) as txn:
        if (
            txn.get(key_review_ths.encode()) is not None
            and txn.get(key_rmse_bins.encode()) is not None
        ):
            print(f"Found for {user_id}.")
            return

    df = pd.read_parquet(rwkv_config.DATA_PATH / "revlogs" / f"{user_id=}")

    # ★★ THE EQUALIZE FILTER MUST SEE THE INTERVAL THE MODEL IS ACTUALLY TRAINED AND SCORED ON
    # (Andrew 2026-09-01: "We should have delta_t > 0 though, to make our methodology closer to
    # that of srs-benchmark").
    #
    # We already HAVE `delta_t > 0` -- it comes from `create_features` itself
    # (features/base.py:284), which is why our `size` reproduces srs-benchmark's published jsonls.
    # What was missing is that this file reads the parquet DIRECTLY and never went through
    # `data_processing.get_rwkv_data`, so the filter was evaluated on END-TO-END intervals while
    # training and eval had moved to end-to-start. With `SECS = true` that filter is NOT
    # interval-independent: `delta_t := elapsed_seconds / 86400` (base.py:127,227), so WHICH rows
    # floor to zero depends on the interval definition.
    #
    # Consequence of leaving it: rows whose end-to-start gap is zero stayed in the scored set, so
    # the model was asked to predict recall across a zero-length gap on reviews srs-benchmark's
    # own rule deletes. Measured on 60 eval users / 3,151,582 rows: 0.1907% of rows, and they are
    # 1.46x EASIER than average (7.07% vs 10.31% failure, -9.8 sigma), so scoring them lowers
    # mean LogLoss for free.
    #
    # ⚠ THE SAME TWO FUNCTIONS `get_rwkv_data` CALLS, IN THE SAME ORDER, AND THAT IS THE POINT.
    # The two datasets need DIFFERENT formulas (published subtracts THIS review's duration, -id
    # the PREVIOUS one), and applying the wrong one is silently wrong -- no shape changes, no
    # error, just a different number. Re-deriving it here would be a second implementation to
    # keep in sync; calling the same functions makes divergence impossible. Gated on the DATASET
    # exactly as they are, so a published build is untouched unless RWKV_E2S_PUBLISHED=1.
    if "review_time" in df.columns:
        df = _idf.elapsed_end_to_start(df)
        df = _idf.clamp_negative_gaps(df)
    else:
        df = _idf.elapsed_end_to_start_published(df)

    try:
        df = create_features(df.copy(), config=config)
    except ValueError as err:
        # Some users can lose every row during outlier/non-continuity filtering; skip those users.
        if "No data after handling outliers" in str(err):
            print(f"Skipping {user_id}: {err}")
            return
        raise
    if len(df) == 0:  # that one user
        return

    # Get RMSE (bins) indices
    # Perf (2026-07-28): df.iloc[i] per row rebuilds a mixed-dtype Series (dtype-promotion
    # over ALL columns) each call -- profiled at ~17s/60k-row user, the single biggest cost
    # in this file. get_bin() only reads 4 columns; pull those once as plain Python lists
    # and index into them instead. Verified bit-identical (scratchpad/content_aware/
    # verify_find_equalize_patch.py), ~10-16x faster.
    r_hist_col = df["r_history"].tolist()
    t_hist_col = df["t_history"].tolist()
    delta_t_col = df["delta_t"].tolist()
    i_col = df["i"].tolist()
    bins = [
        get_bin(
            {
                "r_history": r_hist_col[k],
                "t_history": t_hist_col[k],
                "delta_t": delta_t_col[k],
                "i": i_col[k],
            }
        )
        for k in range(len(df))
    ]

    bins_set = set(bins)
    bins_ind = {}
    for i, x in enumerate(bins_set):
        bins_ind[x] = i

    # Get review_th that are included in the benchmark
    review_th_col = df["review_th"].tolist()
    tscv = TimeSeriesSplit(n_splits=5)
    test_label_review_th = []
    test_label_rmse_bins = []
    for _, (_, test_index) in enumerate(tscv.split(df)):
        for i in test_index:
            test_label_review_th.append(review_th_col[i])
            test_label_rmse_bins.append(bins_ind[bins[i]])

    assert sorted(test_label_review_th) == test_label_review_th
    review_ths_tensor = torch.tensor(test_label_review_th, dtype=torch.int32)
    rmse_bins_tensor = torch.tensor(test_label_rmse_bins, dtype=torch.int32)

    with lmdb_env.begin(write=True) as txn:
        save_tensor(txn, key_review_ths, review_ths_tensor)
        save_tensor(txn, key_rmse_bins, rmse_bins_tensor)

    print("Done:", user_id, "Size:", len(test_label_review_th))


def set_low_priority():
    try:
        p = psutil.Process()
        if hasattr(psutil, "IDLE_PRIORITY_CLASS"):
            p.nice(psutil.IDLE_PRIORITY_CLASS)
        else:
            # POSIX: nice level 19 is the lowest priority
            p.nice(19)
    except Exception as e:
        print(f"Failed to set priority: {e}")


def init_worker():
    global lmdb_env
    set_low_priority()
    lmdb_env = _open_lmdb_env()


def main():
    # Optional USER_IDS list in the config targets a specific (possibly scattered) set of
    # users — used for the seeded verification user-set — instead of a contiguous range.
    user_ids = getattr(rwkv_config, "USER_IDS", None)
    if user_ids is None:
        user_ids = list(range(rwkv_config.USER_START, rwkv_config.USER_END + 1))

    with Pool(processes=rwkv_config.PROCESSES, initializer=init_worker) as pool:
        _ = list(tqdm(pool.imap(process, user_ids), total=len(user_ids)))


if __name__ == "__main__":
    main()
