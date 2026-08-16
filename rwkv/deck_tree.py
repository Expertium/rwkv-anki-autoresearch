"""RWKV_DECK_TREE -- the deck hierarchy as extra links in the stream chain.

Andrew's ask (2026-08-15): card->note->deck->preset->global becomes
card->note->(deck, depth_level)->preset->global. Each depth level is one more pass of the SAME
deck module over the reviews grouped by the deck's k-th ANCESTOR, so the model pools evidence at
every scope the user's deck tree actually defines instead of only the leaf.

WHY THIS NEEDS NO LMDB REBUILD, which is the whole reason it is cheap: data_processing drops
`parent_id` (:228) but NEVER factorizes or remaps `deck_id` -- the only rewrite is NaN ->
ID_PLACEHOLDER (:243-245). So the parquet's own deck_id -> parent_id mapping applies directly to
ids already sitting in the LMDBs. Confirmed at scale in optimization/FUTURE_FEATURES.md.

REACH (40 users, all chunks, 3.67 M reviews, review-weighted -- scratchpad/deck_tree/level_reach.py):
    ancestor at distance 1: 49.21%   2: 38.29%   3: 31.20%   4: 20.93%   5: 7.80%
The chain-depth histogram PEAKS AT 4, not 1 -- Anki users nest decks deeply -- which is why this
is a loop over levels and not a single "parent deck" link.

ROWS WITH NO k-TH ANCESTOR get a row-unique negative id, so they form SINGLETON sequences (T=1,
the cheapest thing the WKV kernel can be handed) and are additionally marked INACTIVE. The model
never scatters an inactive row's output back, so those rows pass through the level EXACTLY
unchanged. Singletons rather than one shared sequence on purpose: a shared sequence would be
T=N sequential work for output that is thrown away.

⚠ Every row must still belong to exactly one group in EVERY stream -- prepare() chains each
stream's gather against the previous stream's LAYOUT (current_locs_list), so a row that is absent
from a stream has no position for the next stream to reference. Hence singleton-and-mask, never
omit-from-the-gather.
"""
import os
import threading

import numpy as np

# RWKV_DECK_TREE=L: L is the TOTAL number of deck levels, so L=1 is the status quo (leaf only)
# and L=3 adds the parent and grandparent levels. Unset/0/1 == off == byte-identical.
_LEVELS = int(os.environ.get("RWKV_DECK_TREE", "0") or "0")
if _LEVELS == 1:
    _LEVELS = 0
_MAP_PATH = os.environ.get(
    "RWKV_DECK_TREE_MAP", "scratchpad/deck_tree/parent_maps.parquet"
)

_lock = threading.Lock()
_parents = None  # {user_id: {deck_id: parent_id}}, -1 = no resolvable parent
_warned = set()


def enabled() -> bool:
    return _LEVELS >= 2


def num_levels() -> int:
    """Total deck levels including the leaf; 0 when the lever is off."""
    return _LEVELS


def stream_name(k: int) -> str:
    """Chain-stream name for ancestor distance k (k >= 1). Level 0 is plain `deck_id`."""
    return f"deck_id@{k}"


def stream_names() -> list:
    return [stream_name(k) for k in range(1, _LEVELS)] if enabled() else []


def expand_modules(modules: list) -> list:
    """Insert the ancestor levels into a config's (name, RWKV7Config) list, after `deck_id`.

    Each level gets a DEEP COPY of the deck config -- same depth, same LoRA dims, same dropout --
    while srs_model shares the module OBJECT across them (that is where the zero-extra-parameters
    property comes from).

    ⚠ The copy is load-bearing, and sharing the config was a real bug: srs_model stamps
    `cfg.stream_name = name` per entry, so one shared object ends up carrying the LAST level's
    name, and `RWKV_STRIP_CMIX=deck_id:1,deck_id:2` silently stops matching -- +26,070 params
    appear with no error anywhere. Configs are mutated in place by several consumers
    (stream_name, the QAT scopes); one object per stream entry is the only safe shape.
    """
    if not enabled():
        return modules
    import copy

    out = []
    for name, cfg in modules:
        out.append((name, cfg))
        if name == "deck_id":
            for k in range(1, _LEVELS):
                out.append((stream_name(k), copy.deepcopy(cfg)))
    assert len(out) == len(modules) + _LEVELS - 1, "deck_id not found in the module list"
    return out


def _load():
    global _parents
    with _lock:
        if _parents is not None:
            return _parents
        import pandas as pd

        pm = pd.read_parquet(_MAP_PATH)
        d = {}
        for uid, g in pm.groupby("user_id"):
            d[int(uid)] = dict(
                zip(g["deck_id"].astype(np.int64), g["parent_id"].astype(np.int64))
            )
        _parents = d
        return _parents


def ancestor_ids(user_id: int, deck_ids: np.ndarray, k: int):
    """Walk `deck_ids` up k levels. Returns (ids, active).

    `ids` is int64 with a ROW-UNIQUE negative sentinel wherever the walk ran out of ancestors
    (root, deleted deck, ID_PLACEHOLDER, or a cycle), so those rows group as singletons.
    `active` is the bool mask of rows that DID reach distance k.
    """
    pm = _load()
    uid = int(user_id)
    par = pm.get(uid)
    if par is None:
        # ⚠ A user missing from the map would make EVERY row inactive -- the tree would silently
        # do nothing for them and the lever would measure as weaker than it is, with no error.
        # Loud once per user per process; the map must cover the train AND eval ranges.
        if uid not in _warned:
            _warned.add(uid)
            print(f"[deck-tree] WARNING: user {uid} has NO parent map entry -- every ancestor "
                  f"level is inactive for them (map={_MAP_PATH})")
        par = {}
    n = deck_ids.shape[0]
    # resolve per DISTINCT id, not per row: a user has ~56 decks and up to 16k rows
    uniq, inv = np.unique(deck_ids, return_inverse=True)
    res = np.empty(uniq.shape[0], dtype=np.int64)
    ok = np.zeros(uniq.shape[0], dtype=bool)
    for i, v in enumerate(uniq):
        cur = int(v)
        seen = {cur}
        good = True
        for _ in range(k):
            nxt = par.get(cur, -1)
            # a cycle would loop forever; the map builder guards too, belt and braces
            if nxt == -1 or nxt in seen:
                good = False
                break
            seen.add(nxt)
            cur = nxt
        res[i] = cur if good else 0
        ok[i] = good
    ids = res[inv]
    active = ok[inv]
    if not active.all():
        dead = np.nonzero(~active)[0]
        ids[dead] = -(dead.astype(np.int64) + 1)  # row-unique => singleton groups
    return ids, active


def chain_submodules(base: list) -> list:
    """RWKV_SUBMODULES with the ancestor levels inserted after `deck_id`.

    This is the STREAM CHAIN (what prepare() builds gathers for), which is deliberately not the
    same list as RWKV_SUBMODULES -- that one still names the five streams the LMDB actually
    stores keys for and the four that get id encodings in the input feature vector. Ancestor
    levels get NEITHER: no LMDB key (they are derived) and no id encoding (which would change
    the input dim and the augmentation RNG draw order, i.e. two confounds for free).
    """
    if not enabled():
        return list(base)
    out = []
    for name in base:
        out.append(name)
        if name == "deck_id":
            out.extend(stream_name(k) for k in range(1, _LEVELS))
    return out


def build_module_data(ids_np: np.ndarray, n: int):
    """Group rows by id -> (split_len, split_B, from_perm, to_perm), the ModuleData quadruple.

    Extracted verbatim from prepare_batch.insert_probes so the derived ancestor streams are
    grouped by EXACTLY the code path the probe rebuild uses -- a second copy would be free to
    drift, and a grouping mismatch is the silent kind of bug (wrong sequences, no assert).
    """
    order = np.argsort(ids_np, kind="stable")  # groups by id, row order kept in-group
    sorted_ids = ids_np[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(sorted_ids))[0] + 1))
    ends = np.concatenate((starts[1:], [n]))
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
    to_perm = np.empty(n, dtype=np.int64)
    to_perm[from_perm] = np.arange(n)
    return (
        np.array(split_len, dtype=np.int32),
        np.array(split_B, dtype=np.int32),
        from_perm,
        to_perm,
    )
