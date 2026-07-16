import math
import os

import lmdb
import numpy as np
import torch
from rwkv.config import (
    DAY_OFFSET_ENCODE_PERIODS,
    ID_ENCODE_DIMS,
    ID_SPLIT,
    RWKV_SUBMODULES,
)
from rwkv.data_processing import CARD_FEATURE_COLUMNS, ModuleData, RWKVSample
from rwkv.model.srs_model import PreparedBatch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.utils import load_tensor

# ---- iter 23 probe rows (MONOTONICITY_PLAN.md stage 2; scratchpad/iter23_pava/BUILD_NOTES.md)
# RWKV_PROBE_DENSITY > 0 inserts, per selected ahead-labeled real row, 4 counterfactual
# button-probe SKIP rows (grade one-hot swapped to Again..Easy, duration imputed) directly
# before it. Only the TRAIN branch of prepare_data_train_test passes the density through;
# validation and get_result eval always run probe-free. Default 0 = byte-identical.
_PROBE_DENSITY = float(os.environ.get("RWKV_PROBE_DENSITY", "0"))
# scale_duration(6433 ms) -- the train-set (users 1-5000) median review duration, frozen
# into the deploy contract (scratchpad/iter23_pava/duration_median.json). One shared value
# for all four probes: duration is spent BEFORE the press, so it cannot depend on the button.
_PROBE_DUR_SCALED = float(os.environ.get("RWKV_PROBE_DUR", "-0.12079481388911952"))
_COL_DUR = CARD_FEATURE_COLUMNS.index("scaled_duration")
_COL_R1 = CARD_FEATURE_COLUMNS.index("rating_1")
assert [CARD_FEATURE_COLUMNS[_COL_R1 + k] for k in range(4)] == [
    "rating_1", "rating_2", "rating_3", "rating_4"
], "grade one-hot columns not contiguous"
# global_labels column layout (create_sample): les, led, label_y, label_rating,
# has_label, label_is_equalize, is_query
_LBL_HAS_LABEL = 4
_LBL_IS_QUERY = 6


class ProbeMeta:
    """Local (per-sample) probe indices, remapped to flat b*global_T+t in prepare()."""

    def __init__(self, pos4, target, pressed, query):
        self.pos4 = pos4        # (m,4) int64 new-row positions, Again..Easy
        self.target = target    # (m,) probed real row's new position
        self.pressed = pressed  # (m,) actual rating-1 in 0..3
        self.query = query      # (m,) paired imm query row's new position


def insert_probes(data: RWKVSample, density: float, base_seed) -> tuple:
    """Insert 4 counterfactual button-probe skip rows before selected real rows.

    Selection is deterministic per (seed, user, chunk). Probes copy the target row's
    features/ids/timing; only the grade one-hot and the (imputed) duration differ, and
    their labels carry has_label=0 (out of every standard loss/metric) with the target's
    label_elapsed_seconds kept so the model's per-row curve evaluation lands at the
    pooled-comparison t. Streams are repacked exactly like data_processing.create_sample.
    """
    sk = data.skips.numpy()
    lab = data.global_labels.float().numpy()  # stored bf16; numpy can't view bf16
    cards = data.ids["card_id"].numpy()
    n = sk.shape[0]

    real = ~sk
    has_lab = lab[:, _LBL_HAS_LABEL] > 0.5
    real_idx = np.nonzero(real)[0]
    _, first_pos = np.unique(cards[real_idx], return_index=True)
    first_mask = np.zeros(n, dtype=bool)
    first_mask[real_idx[first_pos]] = True  # in-chunk first REAL occurrence of the card
    elig = real & has_lab & ~first_mask
    elig_rows = np.nonzero(elig)[0]
    if elig_rows.size == 0:
        return data, None

    seed = (int(base_seed) * 1000003 + int(data.user_id) * 7919
            + int(data.start_th) * 104729) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    pick = elig_rows[rng.random(elig_rows.size) < density]
    m = pick.size
    if m == 0:
        return data, None

    cf = data.card_features
    grade = cf.float().numpy()[pick, _COL_R1:_COL_R1 + 4]
    assert np.allclose(grade.sum(axis=1), 1.0), "target rows must carry a one-hot grade"
    pressed = grade.argmax(axis=1).astype(np.int64)  # the ACTUAL rating of the probed row

    # imm query row of each target: same review_th, is_query row (exists for every
    # non-first review; eligibility implies non-first)
    review_ths = data.review_ths.numpy()
    qmask = sk & (lab[:, _LBL_IS_QUERY] > 0.5)
    q_rows = np.nonzero(qmask)[0]
    q_map = {int(review_ths[q]): int(q) for q in q_rows}
    query_old = np.array([q_map[int(review_ths[r])] for r in pick], dtype=np.int64)

    # ---- build the new row order: 4 probes immediately BEFORE each target
    is_t = np.zeros(n, dtype=bool)
    is_t[pick] = True
    off = 4 * np.cumsum(is_t)          # inclusive: a target's own probes precede it
    new_pos_old = np.arange(n) + off   # old row r -> its new position
    new_n = n + 4 * m
    src = np.empty(new_n, dtype=np.int64)
    probe_rating = np.zeros(new_n, dtype=np.int64)  # 0 = not a probe, else 1..4
    src[new_pos_old] = np.arange(n)
    pos4 = np.empty((m, 4), dtype=np.int64)
    for k in range(4):
        pos_k = new_pos_old[pick] - 4 + k
        src[pos_k] = pick
        probe_rating[pos_k] = k + 1
        pos4[:, k] = pos_k

    src_t = torch.from_numpy(src)
    pm = torch.from_numpy(probe_rating > 0)

    cf_new = cf[src_t].clone()
    cf_new[pm, _COL_DUR] = torch.tensor(_PROBE_DUR_SCALED, dtype=cf_new.dtype)
    cf_new[pm, _COL_R1:_COL_R1 + 4] = 0
    for k in range(4):
        rows_k = torch.from_numpy(pos4[:, k])
        cf_new[rows_k, _COL_R1 + k] = 1

    gl_new = data.global_labels[src_t].clone()
    gl_new[pm, _LBL_HAS_LABEL] = 0  # probes enter NO standard loss/metric
    sk_new = data.skips[src_t].clone()
    sk_new[pm] = True
    lrt_new = data.label_review_ths[src_t].clone()
    lrt_new[pm] = -1

    ids_new = {}
    modules_new = {}
    for sub in RWKV_SUBMODULES:
        ids_sub = data.ids[sub][src_t].clone()
        ids_new[sub] = ids_sub
        arr = ids_sub.numpy()
        order = np.argsort(arr, kind="stable")  # groups by id, row order kept in-group
        sorted_ids = arr[order]
        starts = np.concatenate(([0], np.nonzero(np.diff(sorted_ids))[0] + 1))
        ends = np.concatenate((starts[1:], [new_n]))
        lens = ends - starts
        buckets = {}
        for s, e, l in zip(starts, ends, lens):
            buckets.setdefault(int(l), []).append(order[s:e])
        locs_parts = []
        split_len = []
        split_B = []
        for l in sorted(buckets):
            split_len.append(l)
            split_B.append(len(buckets[l]))
            locs_parts.extend(buckets[l])
        from_perm = np.concatenate(locs_parts)
        to_perm = np.empty(new_n, dtype=np.int64)
        to_perm[from_perm] = np.arange(new_n)
        modules_new[sub] = ModuleData(
            split_len=np.array(split_len, dtype=np.int32),
            split_B=np.array(split_B, dtype=np.int32),
            from_perm=torch.tensor(from_perm, dtype=torch.int32),
            to_perm=torch.tensor(to_perm, dtype=torch.int32),
        )

    data_new = RWKVSample(
        user_id=data.user_id,
        start_th=data.start_th,
        end_th=data.end_th,
        length=new_n,
        card_features=cf_new,
        modules=modules_new,
        ids=ids_new,
        global_labels=gl_new,
        review_ths=data.review_ths[src_t].clone(),
        label_review_ths=lrt_new,
        day_offsets=data.day_offsets[src_t].clone(),
        day_offsets_first=data.day_offsets_first[src_t].clone(),
        skips=sk_new,
    )
    meta = ProbeMeta(
        pos4=pos4,
        target=new_pos_old[pick],
        pressed=pressed,
        query=new_pos_old[query_old],
    )
    return data_new, meta


def prepare(data_list: list[RWKVSample], target_len=None, seed=None,
            probe_density: float = 0.0) -> PreparedBatch:
    if seed is not None:
        torch.manual_seed(seed)

    probe_metas = None
    if probe_density > 0:
        base_seed = seed if seed is not None else int(torch.randint(0, 2**31, (1,)).item())
        new_list = []
        probe_metas = []
        for data in data_list:
            data2, meta = insert_probes(data, probe_density, base_seed)
            new_list.append(data2)
            probe_metas.append(meta)
        data_list = new_list

    with torch.no_grad():
        global_T = max([data.card_features.size(0) for data in data_list])
        data_list_t_sum = sum([data.card_features.size(0) for data in data_list])

        def add_encodings(card_features, day_offsets, day_offsets_first, ids):
            def generate_id_encoding(submodule):
                ENCODE_DIM = ID_ENCODE_DIMS[submodule]
                return torch.randint(
                    low=0,
                    high=ID_SPLIT,
                    size=(ENCODE_DIM,),
                    device=card_features.device,
                    requires_grad=False,
                ).to(card_features.dtype) - ((ID_SPLIT - 1) / 2)

            gather = [card_features]
            for submodule in RWKV_SUBMODULES:
                if submodule == "user_id":
                    continue
                unique_ids = set(ids[submodule].tolist())
                encode = {id: generate_id_encoding(submodule) for id in unique_ids}

                encodings = []
                for id in ids[submodule].numpy():
                    encodings.append(encode[id])
                gather.append(torch.stack(encodings))
                # print("WARNING: zeroing out ids and rng")
                # gather.append(torch.zeros_like(torch.stack(encodings)))

            for period in DAY_OFFSET_ENCODE_PERIODS:
                # Randomly sampled baseline to improve generalization
                baseline = torch.randint(low=0, high=period, size=(1,))
                f = 2 * np.pi / period
                encodings_sin = torch.sin(f * ((baseline + day_offsets) % period)).to(
                    card_features.dtype
                )
                encodings_cos = torch.cos(f * ((baseline + day_offsets) % period)).to(
                    card_features.dtype
                )
                encodings = torch.stack((encodings_sin, encodings_cos), dim=-1)
                gather.append(encodings)
                # print("WARNING: zeroing out ids and rng")
                # gather.append(torch.zeros_like(encodings))
                encodings_first_sin = torch.sin(
                    f * ((baseline + day_offsets_first) % period)
                ).to(card_features.dtype)
                encodings_first_cos = torch.cos(
                    f * ((baseline + day_offsets_first) % period)
                ).to(card_features.dtype)
                encodings_first = torch.stack(
                    (encodings_first_sin, encodings_first_cos), dim=-1
                )
                gather.append(encodings_first)
                # print("WARNING: zeroing out ids and rng")
                # gather.append(torch.zeros_like(encodings_first))

            return torch.cat(gather, dim=-1)

        card_features_with_ids = [
            add_encodings(
                data.card_features, data.day_offsets, data.day_offsets_first, data.ids
            )
            for data in data_list
        ]
        start_tensor = torch.cat(
            [
                torch.nn.functional.pad(
                    card_features, (0, 0, 0, global_T - card_features.size(0))
                )
                for card_features in card_features_with_ids
            ],
            dim=0,
        )

        # Interpretation: the element representing a review_th of i is currently at a[i] where a[i] is a 1D tensor that holds all the data
        boundary_offset = 0
        current_locs_list = [
            i * global_T
            + torch.arange(0, data.card_features.size(0), 1, dtype=torch.long)
            for i, data in enumerate(data_list)
        ]

        # total used mem = x(1+f) where x is the sum of seq lens, f is the factor
        # at MAX and t, we use MAX*(1+t) memory
        # so f = MAX*(1+t)/x - 1
        factor = 0.9
        if target_len is None:
            splits = greedy_splits(data_list, factor=factor)
        else:
            splits = greedy_splits(
                data_list, factor=target_len * (1 + factor) / data_list_t_sum - 1
            )
        sub_gather = []
        sub_skip_gather = []
        sub_time_shift_gather = []
        sub_gather_lens = []
        for submodule_name, _ in DEFAULT_ANKI_RWKV_CONFIG.modules:
            assert submodule_name in splits
            split = splits[submodule_name]

            all_offset = 0
            next_locs_list = [
                np.zeros(data.card_features.size(0), dtype=np.int64)
                for data in data_list
            ]
            gather_lens = []
            gather = []
            skip_gather = []
            time_shift_gather = []
            for split_i in range(len(split)):
                l = 0 if split_i == 0 else split[split_i - 1]
                r = split[split_i]
                gather_lens.append(r)
                take_list = []
                skip_list = []
                time_shift_list = []

                for data_i, (data, current_locs) in enumerate(
                    zip(data_list, current_locs_list)
                ):
                    split_len = data.modules[submodule_name].split_len
                    split_B = data.modules[submodule_name].split_B
                    boundary_offset = 0
                    boundaries = []
                    for s_l, s_b in zip(split_len, split_B):
                        boundaries.append(boundary_offset)
                        boundary_offset += s_l * s_b

                    boundaries.append(boundary_offset)
                    assert boundary_offset == data.card_features.size(0)

                    module_data = data.modules[submodule_name]
                    for module_data_i, (data_split_B, data_split_len) in enumerate(
                        zip(module_data.split_B, module_data.split_len)
                    ):
                        if l < data_split_len and data_split_len <= r:
                            from_slice = module_data.from_perm[
                                boundaries[module_data_i] : boundaries[
                                    module_data_i + 1
                                ]
                            ]
                            take_from = torch.index_select(
                                current_locs, dim=0, index=from_slice
                            ).view(data_split_B, data_split_len)

                            # Maybe random instead of 0 padding to reduce collisions
                            take_from = torch.nn.functional.pad(
                                take_from,
                                (0, r - data_split_len),
                                mode="constant",
                                value=-1,
                            )
                            take_list.append(take_from)

                            skip = torch.index_select(
                                data.skips, dim=0, index=from_slice
                            ).view(data_split_B, data_split_len)
                            skip_arr = skip.numpy()
                            time_shift_select = np.zeros((data_split_B, data_split_len))
                            assert (skip_arr[0] == False).any(), (
                                "Cannot skip the start; otherwise we need to be careful for consecutive Trues at the start."
                            )
                            for b in range(data_split_B):
                                last = 0
                                for t in range(data_split_len):
                                    time_shift_select[b, t] = last
                                    if not skip_arr[b, t]:
                                        last = t

                            skip = torch.nn.functional.pad(
                                skip,
                                (0, r - data_split_len),
                                mode="constant",
                                value=True,
                            )
                            skip_list.append(skip)
                            time_shift_select = torch.nn.functional.pad(
                                torch.tensor(
                                    time_shift_select,
                                    dtype=torch.int32,
                                    device=skip.device,
                                ),
                                (0, r - data_split_len),
                                mode="constant",
                                value=0,
                            )
                            time_shift_list.append(time_shift_select)

                            for seq_unpadded in from_slice.view(
                                data_split_B, data_split_len
                            ):
                                for x in seq_unpadded:
                                    next_locs_list[data_i][x] = all_offset
                                    all_offset += 1

                                all_offset += r - data_split_len
                gather.append(torch.cat(take_list, dim=0).flatten())
                skip_gather.append(torch.cat(skip_list, dim=0).flatten())
                time_shift_gather.append(
                    torch.cat(time_shift_list, dim=0).flatten().long()
                )

            sub_gather.append(gather)
            next_locs_list = [torch.tensor(x) for x in next_locs_list]
            current_locs_list = next_locs_list
            sub_gather_lens.append(gather_lens)
            sub_skip_gather.append(skip_gather)
            sub_time_shift_gather.append(time_shift_gather)

        def pad_labels(labels):
            return torch.nn.functional.pad(
                labels, (0, 0, 0, global_T - labels.size(0)), mode="constant", value=0
            )

        padded_labels = torch.stack(
            list(map(lambda data: pad_labels(data.global_labels), data_list))
        )

        def pad_review_ths(labels):
            return torch.nn.functional.pad(
                labels, (0, global_T - labels.size(0)), mode="constant", value=-1
            )

        padded_label_review_th = torch.stack(
            list(map(lambda data: pad_review_ths(data.label_review_ths), data_list))
        )
        probe_rows_t = probe_target_t = probe_pressed_t = probe_query_t = None
        if probe_metas is not None and any(m is not None for m in probe_metas):
            rows, tgts, prs, qs = [], [], [], []
            for i, meta in enumerate(probe_metas):
                if meta is None:
                    continue
                base = i * global_T
                rows.append(torch.from_numpy(meta.pos4 + base))
                tgts.append(torch.from_numpy(meta.target + base))
                prs.append(torch.from_numpy(meta.pressed))
                qs.append(torch.from_numpy(meta.query + base))
            probe_rows_t = torch.cat(rows).long()
            probe_target_t = torch.cat(tgts).long()
            probe_pressed_t = torch.cat(prs).long()
            probe_query_t = torch.cat(qs).long()
        return PreparedBatch(
            num_data=len(data_list),
            start=start_tensor,
            sub_gather=sub_gather,
            sub_gather_lens=sub_gather_lens,
            skips=sub_skip_gather,
            time_shift_selects=sub_time_shift_gather,
            labels=padded_labels,
            label_review_th=padded_label_review_th,
            probe_rows=probe_rows_t,
            probe_target=probe_target_t,
            probe_pressed=probe_pressed_t,
            probe_query=probe_query_t,
        )


def greedy_splits(
    data_list: list[RWKVSample], factor, allowed_excess_in_one_step=20000
):
    """'factor' puts a limit on the memory complexity.
    'allowed_excess_in_one_step' captures the notion that at some point it is better to just separate the work into sequential calls
    example: if we are given [1, 1e6] then it would be worse to pad the 1 just to fit within the same batch.
    """
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        if submodule == RWKV_SUBMODULES[-1]:
            longest = 0
            for data in data_list:
                module_data = data.modules[submodule]
                longest = max(longest, module_data.split_len.max().item())
            splits_dict[submodule] = [longest]
            continue

        freqs = {}
        for data in data_list:
            module_data = data.modules[submodule]
            for l, b in zip(module_data.split_len, module_data.split_B):
                if l not in freqs:
                    freqs[l] = 0
                freqs[l] += b

        lens = list(reversed(sorted(freqs.keys())))
        splits = []
        l = 0
        while l < len(lens):
            r = l
            used = lens[l] * freqs[lens[l]]
            waste = 0
            while r + 1 < len(lens):
                next_used = used + lens[r + 1] * freqs[lens[r + 1]]
                extra_waste = (lens[l] - lens[r + 1]) * freqs[lens[r + 1]]
                next_waste = waste + extra_waste
                if (
                    factor * next_used >= next_waste
                    and extra_waste <= allowed_excess_in_one_step
                ):
                    used = next_used
                    waste = next_waste
                    r += 1
                else:
                    break

            splits.append(lens[l])
            l = r + 1

        splits.reverse()
        splits_dict[submodule] = splits

    return splits_dict


def naive_splits(data_list: list[RWKVSample]):
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        longest = 0
        for data in data_list:
            module_data = data.modules[submodule]
            longest = max(longest, module_data.split_len.max().item())

        if submodule == RWKV_SUBMODULES[-1]:
            splits_dict[submodule] = [longest]
            continue

        splits = []
        while longest > 0:
            splits.append(longest)
            longest = -1 + math.ceil(longest / 1.5)

        splits.reverse()
        splits_dict[submodule] = splits
    return splits_dict


def get_data(txn, key, device) -> RWKVSample:
    user_id, start_th, end_th, len = key
    prefix = f"{user_id}_{start_th}-{end_th}_{len}_"
    modules = {}
    ids = {}
    for submodule in RWKV_SUBMODULES:
        module_key = prefix + submodule + "_"
        split_len = load_tensor(txn, module_key + "split_len", device=device).numpy()
        split_B = load_tensor(txn, module_key + "split_B", device=device).numpy()
        from_perm = load_tensor(txn, module_key + "from_perm", device=device)
        to_perm = load_tensor(txn, module_key + "to_perm", device=device)
        modules[submodule] = ModuleData(
            split_len=split_len, split_B=split_B, from_perm=from_perm, to_perm=to_perm
        )
        ids[submodule] = load_tensor(txn, prefix + submodule + "_id_", device=device)

    card_features = load_tensor(txn, prefix + "card_features", device=device)
    global_labels = load_tensor(txn, prefix + "global_labels", device=device)
    review_ths = load_tensor(txn, prefix + "review_ths", device=device)

    label_review_ths = load_tensor(txn, prefix + "label_review_ths", device=device)
    day_offsets = load_tensor(txn, prefix + "day_offsets", device=device)
    day_offsets_first = load_tensor(txn, prefix + "day_offsets_first", device=device)
    skips = load_tensor(txn, prefix + "skips", device=device)

    return RWKVSample(
        user_id=user_id,
        start_th=start_th,
        end_th=end_th,
        length=len,
        card_features=card_features,
        modules=modules,
        ids=ids,
        global_labels=global_labels,
        review_ths=review_ths,
        label_review_ths=label_review_ths,
        day_offsets=day_offsets,
        day_offsets_first=day_offsets_first,
        skips=skips,
    )


def prepare_data(
    lmdb_path,
    lmdb_size,
    task_queue,
    batch_queue,
    target_len=66000,
    fixed_seed=None,
):
    env = lmdb.open(lmdb_path, map_size=lmdb_size)
    with env.begin(write=False) as txn:
        while True:
            task = task_queue.get()
            if task is None:
                return

            group_i, group = task
            result = prepare(
                [get_data(txn, key, device="cpu") for key in group],
                target_len=target_len,
                seed=fixed_seed,
            )
            batch_queue.put((group_i, result))


def prepare_data_train_test(
    train_lmdb_path,
    train_lmdb_size,
    all_lmdb_path,
    all_lmdb_size,
    task_queue,
    batch_queue,
    target_len=66000,
    fixed_seed=None,
):
    train_env = lmdb.open(train_lmdb_path, map_size=train_lmdb_size)
    all_env = lmdb.open(all_lmdb_path, map_size=all_lmdb_size)
    with train_env.begin(write=False) as train_txn:
        with all_env.begin(write=False) as all_txn:
            while True:
                task = task_queue.get()
                if task is None:
                    return

                group_i, group = task
                if "train" in group_i:
                    result = prepare(
                        [get_data(train_txn, key, device="cpu") for key in group],
                        target_len=target_len,
                        seed=fixed_seed,
                        # probes are TRAIN-ONLY: validation below and get_result's
                        # prepare_data stay probe-free (density 0 default)
                        probe_density=_PROBE_DENSITY,
                    )
                elif "validate" in group_i:
                    result = prepare(
                        [get_data(all_txn, key, device="cpu") for key in group],
                        target_len=800000,
                        seed=fixed_seed,
                    )
                else:
                    raise ValueError("No key.")
                batch_queue.put((group_i, result))
