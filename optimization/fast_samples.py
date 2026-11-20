"""CPU sample construction for the fast evaluation path."""
from __future__ import annotations

import numpy as np
import torch

from rwkv.config import RWKV_SUBMODULES
from rwkv.data_processing import CARD_FEATURE_COLUMNS, ModuleData, RWKVSample, add_segment_features
from rwkv import extra_features as _extra
from rwkv import id_features as _idf

KEEP = [
    "card_id", "day_offset", "day_offset_first", "elapsed_days", "elapsed_days_cumulative",
    "elapsed_seconds", "elapsed_seconds_sin", "elapsed_seconds_cos", "elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos", "user_id", "review_th",
    "note_id", "deck_id", "preset_id", "note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan",
    "i", "is_first_review", "cum_new_cards", "diff_new_cards", "diff_reviews", "cum_new_cards_today",
    "cum_reviews_today", "scaled_elapsed_days", "scaled_elapsed_days_cumulative", "scaled_elapsed_seconds",
    "scaled_elapsed_seconds_cumulative", "day_offset_diff", "day_of_week", "is_query",
]
if _idf.enabled():
    KEEP.extend(_idf.active_new_columns())
if _extra.row_features():
    KEEP.append("review_time")
REJECT = [
    "rating", "duration", "scaled_duration", "state", "scaled_state", "y", "rating_1", "rating_2",
    "rating_3", "rating_4", "skip", "has_label", "label_elapsed_seconds", "label_elapsed_days",
    "label_y", "label_rating", "label_review_th", "label_is_equalize",
]
LABELS = [
    "label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating", "has_label",
    "label_is_equalize", "is_query",
]


def _matrix(columns, direct=False):
    if direct:
        result = np.empty((len(columns[0]), len(columns)), dtype=np.float64)
        for index, column in enumerate(columns):
            result[:, index] = column
        return result
    return np.column_stack([column.astype(np.float64) for column in columns])


def batched_id_table(ids, dimensions, dtype):
    """Match prepare()'s per-id torch draws while using one random tensor per stream."""
    from rwkv.config import ID_SPLIT

    identifiers = list(set(ids.tolist()))
    unique, inverse = np.unique(ids.numpy(), return_inverse=True)
    draws = torch.randint(0, ID_SPLIT, (len(identifiers), dimensions), device="cpu").to(dtype)
    draws -= (ID_SPLIT - 1) / 2
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    order = torch.tensor([positions[identifier] for identifier in unique], dtype=torch.long)
    return draws.index_select(0, order), torch.from_numpy(inverse)


def section_features(samples, seed):
    from rwkv.config import DAY_OFFSET_ENCODE_PERIODS, ID_ENCODE_DIMS

    torch.random.default_generator.manual_seed(seed)
    result = []
    with torch.no_grad():
        for sample in samples:
            pieces = [sample.card_features]
            for name in RWKV_SUBMODULES:
                if name == "user_id":
                    continue
                ids = sample.ids[name].cpu()
                table, indices = batched_id_table(ids, ID_ENCODE_DIMS[name], sample.card_features.dtype)
                pieces.append(table[indices])
            for period in DAY_OFFSET_ENCODE_PERIODS:
                baseline = torch.randint(0, period, (1,))
                factor = 2 * np.pi / period
                angle = factor * ((baseline + sample.day_offsets.cpu()) % period)
                first = factor * ((baseline + sample.day_offsets_first.cpu()) % period)
                pieces.extend((torch.stack((angle.sin(), angle.cos()), -1).to(sample.card_features.dtype),
                               torch.stack((first.sin(), first.cos()), -1).to(sample.card_features.dtype)))
            result.append(torch.cat(pieces, dim=-1))
    return result


def _module_data(values):
    ordering = np.argsort(values, kind="stable")
    unique, starts = np.unique(values[ordering], return_index=True)
    ends = np.r_[starts[1:], len(values)]
    locations = {key: ordering[start:end] for key, start, end in zip(unique, starts, ends)}
    grouped = {}
    for key in set(values):
        locs = locations[key]
        grouped.setdefault(len(locs), []).append(locs)
    flat, split_len, split_B = [], [], []
    for size in sorted(grouped):
        split_len.append(size)
        split_B.append(len(grouped[size]))
        flat.extend(grouped[size])
    locs = np.concatenate(flat)
    inverse = np.empty_like(locs)
    inverse[locs] = np.arange(len(locs))
    return ModuleData(
        split_len=np.asarray(split_len, dtype=np.int32),
        split_B=np.asarray(split_B, dtype=np.int32),
        from_perm=torch.from_numpy(locs.astype(np.int32, copy=False)),
        to_perm=torch.from_numpy(inverse.astype(np.int32, copy=False)),
    )


def create_sample_fast(user_id, frame):
    frame = add_segment_features(frame.copy()).reset_index(drop=True)
    data = {name: frame[name].to_numpy() for name in frame.columns}
    data["skip"] = np.zeros(len(frame), dtype=bool)
    expected = set(KEEP) | set(REJECT)
    if set(data) != expected or len(data) != len(KEEP) + len(REJECT):
        raise ValueError(f"Sample columns mismatch: missing={expected - set(data)}, extra={set(data) - expected}")
    query_rows = np.flatnonzero(data["is_first_review"] == False)  # noqa: E712
    if len(query_rows):
        query = {name: values[query_rows].copy() for name, values in data.items()}
        rating = query["rating"]
        if not np.isin(rating, (1, 2, 3, 4)).all():
            raise KeyError("Rating outside the add_queries label map")
        for name in REJECT:
            query[name] = np.zeros(len(query_rows), dtype=np.int64)
        query["skip"] = np.ones(len(query_rows), dtype=bool)
        query["label_rating"] = rating
        query["label_y"] = np.where(rating == 1, 0, 1).astype(np.int64)
        query["label_review_th"] = query["review_th"]
        query["has_label"] = np.ones(len(query_rows), dtype=bool)
        query["is_query"] = np.ones(len(query_rows), dtype=np.int64)
        query["label_is_equalize"] = np.zeros(len(query_rows), dtype=np.int64)
        data = {name: np.concatenate((data[name], query[name])) for name in data}
        order = np.lexsort((~data["skip"].astype(bool), data["review_th"]))
        data = {name: values[order] for name, values in data.items()}
    label_rating = data["label_rating"]
    if label_rating.min() < 1 or label_rating.max() > 4:
        raise AssertionError("label_rating outside 1-4")
    if _idf.enabled():
        from rwkv.data_processing import STORED_CARD_FEATURE_COLUMNS
        stored = list(STORED_CARD_FEATURE_COLUMNS)
        if set(stored) - set(data):
            raise ValueError(f"Missing stored card features: {sorted(set(stored) - set(data))}")
        card_columns = [name for name in stored if name not in _idf.dropped_columns()]
        if _idf.final_layout_enabled():
            card_columns = [name for name in card_columns if name not in _idf.FINAL_LAYOUT_DROPPED]
        card_columns += list(_idf.active_new_columns())
        card = _matrix([data[name] for name in card_columns], direct=True)
    else:
        card = _matrix([data[name] for name in CARD_FEATURE_COLUMNS], direct=True)
    if _extra.width():
        extra = _extra.values(user_id, data["review_th"],
                              sibgap=(data["scaled_sibling_gap"] if "sibgap_none" in _extra.active() else None))
        card = np.column_stack((card, extra))
    modules = {name: _module_data(data[name]) for name in RWKV_SUBMODULES}
    ids = {name: torch.as_tensor(data[name], dtype=torch.int64) for name in RWKV_SUBMODULES}
    return RWKVSample(
        user_id=user_id,
        start_th=int(data["review_th"].min()),
        end_th=int(data["review_th"].max()),
        length=len(data["review_th"]),
        modules=modules,
        ids=ids,
        card_features=torch.as_tensor(card, dtype=torch.float32),
        global_labels=torch.as_tensor(_matrix([data[name] for name in LABELS]), dtype=torch.float32),
        review_ths=torch.as_tensor(data["review_th"], dtype=torch.int32),
        label_review_ths=torch.as_tensor(data["label_review_th"], dtype=torch.int32),
        day_offsets=torch.as_tensor(data["day_offset"], dtype=torch.int32),
        day_offsets_first=torch.as_tensor(data["day_offset_first"], dtype=torch.int32),
        skips=torch.as_tensor(data["skip"], dtype=torch.bool),
    )


def make_samples_fast(section, user_id, chunk=128):
    return [create_sample_fast(user_id, section.iloc[offset:offset + chunk])
            for offset in range(0, len(section), chunk)]
