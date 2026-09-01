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
from rwkv import deck_tree as _deck_tree

# The STREAM CHAIN prepare() builds gathers for. Equals RWKV_SUBMODULES unless
# RWKV_DECK_TREE inserts the deck-ancestor levels after deck_id.
_CHAIN = _deck_tree.chain_submodules(RWKV_SUBMODULES)

# ---- iter 23 probe rows (MONOTONICITY_PLAN.md stage 2; scratchpad/iter23_pava/BUILD_NOTES.md)
# RWKV_PROBE_DENSITY > 0 inserts, per selected ahead-labeled real row, 4 counterfactual
# button-probe SKIP rows (grade one-hot swapped to Again..Easy, duration imputed) directly
# before it. Only the TRAIN branch of prepare_data_train_test passes the density through;
# validation and get_result eval always run probe-free. Default 0 = byte-identical.
_PROBE_DENSITY = float(os.environ.get("RWKV_PROBE_DENSITY", "0"))
# scale_duration(6433 ms) -- the train-set (users 1-5000) median review duration, frozen
# into the deploy contract (scratchpad/iter23_pava/duration_median.json). One shared value
# for all four probes: duration is spent BEFORE the press, so it cannot depend on the button.
#
# ⚠ 2026-07-26 (Andrew): the DEPLOY convention is 0.0, NOT this median. At deploy Anki must
# draw the four button intervals BEFORE the press, so the current review's duration does not
# exist yet; recomputing live as it accrues would churn the displayed numbers and feed back
# into the user's dwell time. 0.0 is the pipeline's own "no press yet" encoding (query rows
# zero scaled_duration -- data_processing.reject_columns), so the model has seen it on every
# query row. Because scale_duration(x) = (log(10+x) - 8.9)/1.07, a literal 0.0 implies
# ~7.3 s, close to the 6,433 ms median it replaces. Runs from iter 31 on set
# RWKV_PROBE_DUR=0.0; the default below is kept only so iters 23-30 stay reproducible.
_PROBE_DUR_SCALED = float(os.environ.get("RWKV_PROBE_DUR", "-0.12079481388911952"))
_COL_DUR = CARD_FEATURE_COLUMNS.index("scaled_duration")
# Rectified EVAL (2026-07-26, Andrew: "Eval should score the rectified model, of course").
# RWKV_EVAL_PAVA=1 makes the eval workers insert the same 4 counterfactual button probes at
# density 1.0 -- restricted to rows the benchmark actually SCORES (label_is_equalize), which
# keeps the row inflation to the scored subset -- so srs_model can replace each scored ahead
# prediction with the PAVA-rectified value at the pressed button. Probe rows are skip rows
# and the WKV kernel restores the pre-step state on a skip (rwkv7_cuda.cu: `if (skip)
# state_xy = in_state_xy`), so inserting them changes NO other prediction. Default 0 =
# eval is byte-identical to every stored result.
# 2 = probes, no pooling; 3 = probes, no substitution at all (the bf16-noise control -- probe
# insertion re-buckets the batch and is NOT numerically free; see srs_model's mode table)
_EVAL_PAVA = os.environ.get("RWKV_EVAL_PAVA", "0") in ("1", "2", "3")
_COL_R1 = CARD_FEATURE_COLUMNS.index("rating_1")
assert [CARD_FEATURE_COLUMNS[_COL_R1 + k] for k in range(4)] == [
    "rating_1", "rating_2", "rating_3", "rating_4"
], "grade one-hot columns not contiguous"
# global_labels column layout (create_sample): les, led, label_y, label_rating,
# has_label, label_is_equalize, is_query
_LBL_HAS_LABEL = 4
_LBL_IS_EQUALIZE = 5
_LBL_IS_QUERY = 6


class ProbeMeta:
    """Local (per-sample) probe indices, remapped to flat b*global_T+t in prepare()."""

    def __init__(self, pos4, target, pressed, query):
        self.pos4 = pos4        # (m,4) int64 new-row positions, Again..Easy
        self.target = target    # (m,) probed real row's new position
        self.pressed = pressed  # (m,) actual rating-1 in 0..3
        self.query = query      # (m,) paired imm query row's new position


_unpairable_seen = 0


def _note_unpairable(data, n_dropped, n_picked):
    """Report probe targets dropped for having no imm query row. Bounded, but never silent.

    Rare by construction (2 rows across 768 chunks / 100 users on train_db_5k_h1_id3), so the
    first few are printed in full. It is reported at all because a silent drop is exactly the
    failure class this project keeps paying for: the run would complete, the number would look
    fine, and nothing would say the probe set had quietly shrunk. If this ever prints at volume,
    the frame ordering has drifted much further than the 60 s duration cap explains -- that is a
    dataset problem, not a probe problem, and it should be investigated rather than tolerated.
    """
    global _unpairable_seen
    _unpairable_seen += 1
    if _unpairable_seen <= 5 or _unpairable_seen % 1000 == 0:
        print(
            "[probe] user %s chunk %s-%s: dropped %d of %d probe targets with no imm query row "
            "(genuine first review not first in frame order); %d so far in this worker"
            % (data.user_id, data.start_th, data.end_th, n_dropped, n_picked, _unpairable_seen),
            flush=True,
        )


def insert_probes(data: RWKVSample, density: float, base_seed,
                  equalize_only: bool = False) -> tuple:
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
    if equalize_only:
        # rectified eval: only rows the benchmark scores need a rectified prediction
        elig = elig & (lab[:, _LBL_IS_EQUALIZE] > 0.5)
    elig_rows = np.nonzero(elig)[0]
    if elig_rows.size == 0:
        return data, None

    seed = (int(base_seed) * 1000003 + int(data.user_id) * 7919
            + int(data.start_th) * 104729) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    pick = elig_rows[rng.random(elig_rows.size) < density]

    # ---- the paired imm query row, and why this is a FILTER rather than an assumption ----
    # This used to read "exists for every non-first review; eligibility implies non-first" and
    # index q_map directly. That implication is FALSE, and it killed featB's fetch worker twice
    # (2026-08-21 and again 2026-09-01, both `KeyError` here).
    #
    # The two notions of "first" are computed from different things and nothing keeps them
    # aligned:
    #   * `first_mask` above is POSITIONAL -- the first real row of each card_id in this chunk.
    #   * a query row exists iff `is_first_review == False`, and `is_first_review` is
    #     `elapsed_days == -1`, which the -id builder sets from `state == 0`
    #     (build_parquet_id.py:138) and only THEN sorts the frame by review_time (:142).
    #
    # So on the -id set a card's genuine first review need not be its first row. Anki caps
    # `taken_millis` at 60 s, and `review_time = id - taken_millis` subtracts that CAP, so a
    # capped neighbouring review can acquire a show time up to a minute early and sort ahead of
    # the card's real first review. Ground truth, user 477 / card 1708127478116: review_th 73724
    # is the first review (elapsed_days -1, took 11.5 s) yet 73723 (duration exactly 60000, the
    # cap) sorts before it and carries elapsed_seconds -17. The genuine first review then passes
    # the positional mask, has no query row, and raises.
    #
    # A first review CANNOT be probed -- the imm task needs a prior review, so there is nothing
    # to pair against. Dropping such picks is therefore the correct semantics, not a workaround.
    #
    # ⚠ BIT-IDENTITY IS DELIBERATE. The rng draw happens BEFORE the filter and is untouched, and
    # the filter is a no-op wherever the old implication held -- which is every published/e2s
    # database, since neither `elapsed_end_to_start*` re-sorts the frame. So no existing number
    # moves. Verified by scratchpad/features_rebuild/probe_query_mismatch.py.
    review_ths = data.review_ths.numpy()
    qmask = sk & (lab[:, _LBL_IS_QUERY] > 0.5)
    q_rows = np.nonzero(qmask)[0]
    q_map = {int(review_ths[q]): int(q) for q in q_rows}
    if pick.size:
        pairable = np.array([int(review_ths[r]) in q_map for r in pick], dtype=bool)
        if not pairable.all():
            _note_unpairable(data, int((~pairable).sum()), int(pick.size))
            pick = pick[pairable]

    m = pick.size
    if m == 0:
        return data, None

    cf = data.card_features
    grade = cf.float().numpy()[pick, _COL_R1:_COL_R1 + 4]
    assert np.allclose(grade.sum(axis=1), 1.0), "target rows must carry a one-hot grade"
    pressed = grade.argmax(axis=1).astype(np.int64)  # the ACTUAL rating of the probed row

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
        split_len, split_B, from_perm, to_perm = _deck_tree.build_module_data(
            ids_sub.numpy(), new_n
        )
        modules_new[sub] = ModuleData(
            split_len=split_len,
            split_B=split_B,
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


def build_ahead_query(data, base: int) -> np.ndarray:
    """(iter 46) For each row, the flat index of the QUERY row that scores the SAME review this
    row's ahead label refers to; -1 where there is none. Used only as a self-distillation teacher
    index (RWKV_SELFKD_BETA); -1 everywhere = the historical hard-label behaviour.

    THE PAIRING, from data_processing.py:
      * a REAL row's label is the NEXT review of the same card
        (`label_review_th = groupby("card_id")["review_th"].shift(-1)`, :292-303), so its ahead
        target is review `label_review_th`;
      * a QUERY row carries `label_review_th = its own review_th` with the press hidden (:442-453).
    So real row r and the query row q with `review_th[q] == label_review_th[r]` score the SAME
    event -- which is exactly why extract_p keys both metric dicts on label_review_th and the
    per-user `size` counts match. q is the better-informed estimate of r's own target.

    ⚠ NOT the same join as the probe channel's `probe_query`, which uses `review_th[q] ==
    review_th[r]` -- review r's OWN decision point. That is correct there (PAVA weights the
    pooling by what the user would press AT r) and wrong here (r's label is review r+1).

    QUERY ROWS ARE EXCLUDED (-1): a query row's label_review_th is its own review_th, so the join
    would return the row itself and the teacher would be its own output.
    """
    sk = data.skips.numpy().astype(bool)
    lab = data.global_labels.float().numpy()
    n = sk.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    isq = lab[:, _LBL_IS_QUERY] > 0.5
    has_lab = lab[:, _LBL_HAS_LABEL] > 0.5
    q_rows = np.nonzero(sk & isq)[0]
    if q_rows.size == 0:
        return out
    rt = data.review_ths.numpy()
    # label_review_th is NOT fillna'd upstream (unlike label_y/label_rating), so it is NaN on
    # rows with no next review -- cast only where a label exists.
    lrt_f = data.label_review_ths.numpy().astype(np.float64)
    src = has_lab & (~isq) & np.isfinite(lrt_f)
    if not src.any():
        return out
    lrt_i = np.zeros(n, dtype=np.int64)
    lrt_i[src] = lrt_f[src].astype(np.int64)
    q_rt = rt[q_rows].astype(np.int64)
    order = np.argsort(q_rt, kind="stable")
    q_rt_s, q_rows_s = q_rt[order], q_rows[order]
    pos = np.searchsorted(q_rt_s, lrt_i)
    pos_c = np.clip(pos, 0, q_rt_s.size - 1)
    hit = src & (pos < q_rt_s.size) & (q_rt_s[pos_c] == lrt_i)
    out[hit] = q_rows_s[pos_c][hit] + base
    return out


def add_deck_levels(data: RWKVSample):
    """Derive the deck-ancestor streams for one chunk, in place. Returns {name: active mask}.

    Called AFTER probe insertion, so the ancestor ids are derived from the already-expanded
    deck_id column and every derived stream lines up row-for-row with the real ones.

    Rows with no k-th ancestor get a row-unique negative id (=> a singleton sequence, T=1) and
    active=False. They MUST still be grouped -- prepare() chains each stream's gather against the
    previous stream's layout, so a row absent from a stream has no slot for the next stream to
    reference. The bypass is therefore "compute and discard" (the model never scatters an
    inactive row back), not "omit".
    """
    n = data.card_features.size(0)
    deck = data.ids["deck_id"].numpy().astype(np.int64)
    active = {}
    for k in range(1, _deck_tree.num_levels()):
        name = _deck_tree.stream_name(k)
        ids_k, act = _deck_tree.ancestor_ids(data.user_id, deck, k)
        data.ids[name] = torch.from_numpy(ids_k)
        split_len, split_B, from_perm, to_perm = _deck_tree.build_module_data(ids_k, n)
        data.modules[name] = ModuleData(
            split_len=split_len,
            split_B=split_B,
            from_perm=torch.tensor(from_perm, dtype=torch.int32),
            to_perm=torch.tensor(to_perm, dtype=torch.int32),
        )
        active[name] = act
    return active


def prepare(data_list: list[RWKVSample], target_len=None, seed=None,
            probe_density: float = 0.0, probe_equalize_only: bool = False) -> PreparedBatch:
    if seed is not None:
        torch.manual_seed(seed)

    probe_metas = None
    if probe_density > 0:
        base_seed = seed if seed is not None else int(torch.randint(0, 2**31, (1,)).item())
        new_list = []
        probe_metas = []
        for data in data_list:
            data2, meta = insert_probes(data, probe_density, base_seed,
                                        equalize_only=probe_equalize_only)
            new_list.append(data2)
            probe_metas.append(meta)
        data_list = new_list

    # RWKV_DECK_TREE: derive the ancestor streams. AFTER probes (ids must be row-aligned with the
    # expanded chunk) and BEFORE greedy_splits (which now sizes the whole chain).
    tree_active = None
    if _deck_tree.enabled():
        tree_active = [add_deck_levels(d) for d in data_list]

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
        # iter 46 self-distillation teacher index: (B, global_T) flat b*global_T+t, -1 = none.
        # GATED ON THE FLAG so that with RWKV_SELFKD_BETA unset NOT ONE new line executes in the
        # fetch workers -- the change is then inert by construction, not merely unused. (Written
        # this way after noticing the hazard live: a phase that starts LATER imports whatever is
        # on disk THEN, so editing a module mid-chain silently changes the next phase of a
        # gate-critical run. Same family as the "never rewrite a running runner's tree" rule.)
        # Uses numpy only -- no torch RNG -- so augmentation draws, and hence the KD dump's
        # per-step labels checksum, are untouched either way.
        ahead_query_t = None
        if float(os.environ.get("RWKV_SELFKD_BETA", "0")) != 0.0:
            ahead_query_t = torch.from_numpy(
                np.stack([
                    np.pad(build_ahead_query(d, i * global_T),
                           (0, global_T - d.card_features.size(0)),
                           mode="constant", constant_values=-1)
                    for i, d in enumerate(data_list)
                ])
            ).long()
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
        # Per-stream row-activity, in the SAME order as sub_gather (config module order, which is
        # NOT RWKV_SUBMODULES order -- the default arch runs card->deck->note). An empty tensor
        # means "every row active", i.e. the ordinary streams, so the model's fast path is a
        # numel() check rather than an Optional (TorchScript-friendly).
        stream_active = []
        for _name, _ in DEFAULT_ANKI_RWKV_CONFIG.modules:
            if tree_active is None or _name not in tree_active[0]:
                stream_active.append(torch.empty(0, dtype=torch.bool))
                continue
            m = torch.zeros(len(data_list) * global_T, dtype=torch.bool)
            for _i, _act in enumerate(tree_active):
                a = _act[_name]
                m[_i * global_T : _i * global_T + a.shape[0]] = torch.from_numpy(a)
            stream_active.append(m)
        return PreparedBatch(
            num_data=len(data_list),
            start=start_tensor,
            stream_active=stream_active,
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
            ahead_query=ahead_query_t,
        )


def greedy_splits(
    data_list: list[RWKVSample], factor, allowed_excess_in_one_step=20000
):
    """'factor' puts a limit on the memory complexity.
    'allowed_excess_in_one_step' captures the notion that at some point it is better to just separate the work into sequential calls
    example: if we are given [1, 1e6] then it would be worse to pad the 1 just to fit within the same batch.
    """
    splits_dict = {}
    for submodule in _CHAIN:
        if submodule == _CHAIN[-1]:
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
    for submodule in _CHAIN:
        longest = 0
        for data in data_list:
            module_data = data.modules[submodule]
            longest = max(longest, module_data.split_len.max().item())

        if submodule == _CHAIN[-1]:
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
                # RWKV_EVAL_PAVA: probe EVERY scored row so srs_model can replace its ahead
                # prediction with the rectified pressed-button value. Density 1.0 (not the
                # training 0.08) because scoring needs a rectified value for every row the
                # benchmark counts, not a random sample.
                probe_density=1.0 if _EVAL_PAVA else 0.0,
                probe_equalize_only=True,
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
