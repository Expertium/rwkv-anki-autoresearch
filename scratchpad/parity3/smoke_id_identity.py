"""THREE-WAY PARITY: do TRAINING and DEPLOY agree on which rows belong to the same ENTITY?

WHY THIS FILE EXISTS. On 2026-08-21 a single `dtype=torch.int32` in data_processing.create_sample
was found to have been silently destroying entity identity for the whole project:

  * TRAINING groups rows by the id STORED IN THE LMDB, which was cast to int32;
  * DEPLOY (`run_as_rnn`) keys its state dicts on the RAW FRAME VALUE at full precision
    (`self.note_states[row["note_id"]]`).

The NaN-metadata fill is written `ID_PLACEHOLDER + card_id` (3.14e17) precisely to give each such
card a UNIQUE note; int32 saturated them all to INT32_MIN. Measured on published user 101: TRAINING
saw **1** note entity, DEPLOY saw **3,277**. Each path was self-consistent in isolation, which is
exactly why no gate caught it -- the §9 failure mode, verbatim.

WHY IT CANNOT LIVE IN parity_train_vs_rnn.py. That harness feeds identical weights and inputs
through RWKV7 vs RWKV7RNN. It is SINGLE-STACK and takes the grouping as given, so it cannot see a
disagreement about WHAT THE GROUPS ARE. Same reason the RWKV_ID_FEATURES width question needed
`smoke_id_features_width.py` rather than a case in the parity harness.

WHAT IT ASSERTS, per user, per stream:
  1. the built sample's id tensor is not narrower than int64 (the direct regression guard);
  2. its DISTINCT-ENTITY COUNT equals the frame's -- the semantic check, and the one that fails on
     saturation, which is not an error and raises nothing;
  3. no negative ids (the unmistakable wrap signature for raw epoch-ms ids);
  4. the actual PARTITION matches: rows grouped together in training are exactly the rows deploy
     would key to the same state. Equal counts could in principle be reached by a different
     partition, so compare the partitions, not just their sizes.

USERS ARE CHOSEN FOR NaN-METADATA RATE, not at random: user 1 has 0.0% (healthy even when broken,
so it is the negative control), 101 has 66.8%, 417 has 99.6%. A smoke that sampled only user 1
would have passed on the broken build.

Usage: .venv/Scripts/python.exe scratchpad/parity3/smoke_id_identity.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.setdefault("RWKV_ID_FEATURES", "0")

from rwkv.data_processing import create_sample, get_rwkv_data  # noqa: E402

PUB = Path(r"C:\Users\Andrew\anki-revlogs-10k")
IDD = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
STREAMS = ("card_id", "note_id", "deck_id", "preset_id")

fails = []
# streams/users where a simulated int32 store WOULD change the partition -- the evidence that
# the PARTITION check is not vacuous. Not every stream can be corrupted (published card_id is a
# small factorized int; NaN decks already share ONE placeholder by design), so this is collected
# globally and asserted once, rather than demanded per case.
bites = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)


def partition(labels):
    """Canonical form of a partition: frozenset of frozensets of row positions."""
    out = {}
    for i, v in enumerate(labels):
        out.setdefault(v, []).append(i)
    return frozenset(frozenset(v) for v in out.values())


CASES = [("PUBLISHED", PUB, (1, 101, 417))]
if IDD.exists():
    CASES.append(("-id", IDD, (1, 101)))

for label, root, users in CASES:
    print("=== %s" % label)
    for uid in users:
        df = get_rwkv_data(root, uid)
        smp = create_sample(uid, df, [], torch.float32, "cpu")
        # DEPLOY's view: run_as_rnn iterates the frame and keys on the raw value. The sample adds
        # query/skip rows, so align on the REAL rows by review_th.
        for name in STREAMS:
            if name not in df.columns:
                continue
            t = smp.ids[name]
            a = t.numpy()
            frame_vals = df[name].to_numpy()

            check("%s u%-4d %-9s stored dtype is int64" % (label, uid, name),
                  t.dtype == torch.int64, str(t.dtype))
            check("%s u%-4d %-9s no negative ids" % (label, uid, name),
                  int((a < 0).sum()) == 0, "%d negative" % int((a < 0).sum()))
            want, got = int(df[name].nunique()), int(len(np.unique(a)))
            check("%s u%-4d %-9s entity count train==deploy" % (label, uid, name),
                  got == want, "train %d vs deploy %d" % (got, want))

        # 4. the PARTITION itself, on the real rows only.
        real = ~smp.skips.numpy().astype(bool)
        rt = smp.review_ths.numpy()[real]
        order = np.argsort(rt, kind="stable")
        for name in STREAMS:
            if name not in df.columns:
                continue
            train_lab = smp.ids[name].numpy()[real][order]
            deploy_lab = df.sort_values("review_th", kind="stable")[name].to_numpy()
            if len(train_lab) != len(deploy_lab):
                check("%s u%-4d %-9s row counts align" % (label, uid, name), False,
                      "%d vs %d" % (len(train_lab), len(deploy_lab)))
                continue
            ok = partition(train_lab) == partition(deploy_lab)
            check("%s u%-4d %-9s PARTITION train==deploy" % (label, uid, name), ok,
                  "" if ok else "grouping differs -- train and deploy disagree on what an entity IS")

            # ★ PROVE THE GUARD BITES. A check that has only ever been observed PASSING is
            # unproven: it might be structurally incapable of failing. Re-run the same comparison
            # against a simulated int32 store -- the exact corruption this file exists to catch --
            # and require it to FAIL. If it does not, the check is vacuous and says nothing.
            # ⚠ THE SIMULATION MUST MATCH THE REAL CONVERSION, and the first version of it did not.
            # numpy's .astype(int32) WRAPS (modulo 2^32), which is almost injective and barely
            # changes a partition. The old code did `torch.tensor(frame_values, dtype=torch.int32)`
            # where the frame column is float64 after the NaN-fill merge -- and float->int32 in
            # torch SATURATES, pinning every oversized value to INT32_MIN. Saturation is what
            # collapsed 3,277 notes into 1; wrapping never would. Reproduce the real path.
            if not (label == "PUBLISHED" and uid == 1):  # u1 published was healthy even when broken
                broken = torch.tensor(deploy_lab.astype(np.float64),
                                      dtype=torch.int32).numpy().astype(np.int64)
                if partition(broken) != partition(deploy_lab):
                    bites.append("%s u%d %s" % (label, uid, name))

print("")
print("NON-VACUITY: a simulated int32 store would change the partition in %d case(s):" % len(bites))
for b in bites:
    print("   " + b)
# The whole file is worthless if no case can distinguish the broken build from the fixed one.
# PUBLISHED note_id is Bug A (ID_PLACEHOLDER saturation); -id card/note/deck is Bug B.
check("the PARTITION check is NON-VACUOUS (>=1 case detects the real int32 store)",
      len(bites) >= 1, "%d cases" % len(bites))

print("")
print("ALL PASS" if not fails else "FAILURES (%d): %s" % (len(fails), fails[:4]))
sys.exit(1 if fails else 0)
