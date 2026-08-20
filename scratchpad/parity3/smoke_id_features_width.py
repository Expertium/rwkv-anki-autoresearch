#!/usr/bin/env python
"""RWKV_ID_FEATURES: do TRAIN, EVAL and DEPLOY agree on the input width? CPU, seconds.

The §9 three-way-parity rule says every new arch env flag gets a case in
`parity_train_vs_rnn.py`. This flag cannot have one: that harness compares a SINGLE RWKV7 stack
against RWKV7RNN, and the input width lives one level up, in `SrsRWKV` / `SrsRWKVRnn`. So the
equivalent check is here -- and it is the same question in the same shape:

  * TRAIN and EVAL are the same class (`SrsRWKV`), so they cannot disagree;
  * DEPLOY is `SrsRWKVRnn`, a separate file that carried its OWN `card_features_dim = 92` literal.
    Two hardcoded copies of one number is exactly the shape of bug this project keeps paying for
    (the Rust positional stream bug, STRIP_CMIX living only in rwkv_model.py), so the literal is
    gone and both sides now call `id_features.input_width()`. This asserts they still match.
  * `data_processing.CARD_FEATURE_COLUMNS` is the third leg: the LMDB is written with that many
    columns, so if it ever disagrees with the model the mismatch surfaces as a shape error at the
    first batch of a multi-hour run, not at import.

One subprocess per flag value, because the flag is read at import.

ASCII output only.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, sys, torch
sys.path.insert(0, os.environ["PYTHONPATH"])
from rwkv import id_features as idf
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.model.srs_model_rnn import SrsRWKVRnn

want = int(os.environ["WANT_WIDTH"])
train = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
rnn = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
w_train = train.features2card[0].in_features
w_rnn = rnn.features2card[0].in_features
w_cols = len(CARD_FEATURE_COLUMNS)

print(f"helper={idf.input_width()} train={w_train} rnn={w_rnn} card_cols={w_cols} want={want}")
assert idf.input_width() == want, "helper width"
assert w_train == want, "SrsRWKV input Linear"
assert w_rnn == want, "SrsRWKVRnn input Linear -- the deploy path disagrees with training"
assert w_cols + idf.ID_ENCODING_DIMS == want, (
    "CARD_FEATURE_COLUMNS disagrees with the model width -- the LMDB would be written with a "
    "different number of columns than the model consumes"
)
# the mask guard must refuse the obsolete RWKV_ZERO_FEATURES=22 under the new layout
if idf.enabled():
    os.environ["RWKV_ZERO_FEATURES"] = "22"
    try:
        SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    except AssertionError:
        print("ZERO_FEATURES guard fired as intended")
    else:
        raise SystemExit("ZERO_FEATURES=22 was ACCEPTED under RWKV_ID_FEATURES=1")
print("WIDTH_OK")
"""


def run(flag, want):
    env = dict(os.environ, PYTHONPATH=REPO, RWKV_ID_FEATURES=flag, WANT_WIDTH=str(want))
    env.pop("RWKV_ZERO_FEATURES", None)
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    print(f"--- RWKV_ID_FEATURES={flag} (expect {want})")
    for ln in out[-5:]:
        print("    " + ln)
    return p.returncode == 0 and any("WIDTH_OK" in ln for ln in out)


def main():
    # 114 = 68 ID-encoding dims + (24 base - 1 dropped card-state + 23 new). The 23 became
    # 21 + 2 on 2026-08-20 when Andrew's coverage audit found scaled_sibling_gap and
    # card_predates_first_review had been designed and never implemented. The literal is
    # deliberate: deriving it from id_features would make this smoke agree with itself.
    ok = [run("0", 92), run("1", 114)]
    print("\n" + ("WIDTH_ALL_PASS" if all(ok) else "WIDTH_FAILED"))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
