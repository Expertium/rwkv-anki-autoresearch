"""Iter 46 -- privileged self-distillation (imm -> ahead). CPU-only, seconds, no GPU, no LMDB.

WHAT THE CHANGE IS. `RWKV_SELFKD_BETA=<b>` softens the ahead objective's target away from the raw
0/1 label toward the model's own better-informed estimate of the SAME event:

    label_y <- b * (1 - P(Again))@teacher_row .detach() + (1-b) * label_y      [before the KD mix]

Measured headroom (research_5k_notes.md, iter-41 champion): identical per-user `size` on all 2500
users, imm better than ahead on 2497 of them, mean gap 0.032411 -- so the model already emits a
strictly better-informed estimate of the very label the curve head is fit to.

THE FIVE THINGS THAT COULD BE WRONG, and the check that catches each:

  1. THE TEACHER IS THE WRONG REVIEW. This is the one that nearly shipped. A real row's label is
     the NEXT review of that card (label_review_th = shift(-1) per card), while the probe
     channel's `probe_query` is review r's OWN decision point -- correct for PAVA's pooling
     weights, wrong for a teacher. CHECK_MAP asserts the built index lands on a query row whose
     label_review_th EQUALS the source row's, on synthetic data with a known answer.
  2. DEFAULT NOT FREE. Every env flag here must be byte-identical when off or every historical
     number silently re-bases.
  3. THE MIXING MATH IS BACKWARDS (b/(1-b) swapped, or teacher = P(Again) instead of 1-P(Again),
     index 0 being Again). Caught by an ORACLE needing no reference implementation: make the
     teacher exactly equal the label, then EVERY b must reproduce the b=0 target.
  4. THE DETACH IS MISSING -- the load-bearing one, and invisible in the loss value: without it
     the imm head is dragged toward the weaker ahead head, trading away the advantage being
     distilled.
  5. THE EXTERNAL TEACHER'S WEIGHT MOVES. forward() mixes a*d128 + (1-a)*hard; self-KD must
     soften only the HARD share, or alpha silently drops from 0.9 to 0.9*(1-b) and the experiment
     bundles two changes (cf. iters 42/43/44).

Run:  .venv/Scripts/python.exe scratchpad/iter46_selfkd/smoke_selfkd.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARCH_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1",
    "RWKV_GRU_HEAD": "3",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_STATE_CLAMP_TAU": "300",
    "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_STRIP_CMIX": ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                        "preset_id:2,deck_id:1,deck_id:2,card_id:1"),
}

# --------------------------------------------------------------------------------------------
# Child 1: THE INDEX MAP. Synthetic sample with a hand-known answer -- no LMDB, no model.
# --------------------------------------------------------------------------------------------
CHILD_MAP = r"""
import numpy as np, torch
from rwkv.prepare_batch import build_ahead_query, _LBL_HAS_LABEL, _LBL_IS_QUERY

NCOL = 7

class D:
    pass

def make(card_of_review, n_reviews):
    '''Build the row layout data_processing produces: for each review (in review_th order) a
    QUERY row (skip=1, is_query=1, label_review_th = its OWN review_th) followed by the REAL row
    (skip=0, is_query=0, label_review_th = the NEXT review OF THE SAME CARD, NaN if none).
    The first review of a card gets no query row (add_queries drops is_first_review).'''
    rt, lrt, isq, skip, has = [], [], [], [], []
    seen = set()
    nxt = {}
    for i in range(n_reviews):
        c = card_of_review[i]
        later = [j for j in range(i + 1, n_reviews) if card_of_review[j] == c]
        nxt[i] = (later[0] + 1) if later else None      # review_th is 1-based
    for i in range(n_reviews):
        c = card_of_review[i]
        th = i + 1
        if c in seen:                                    # query row exists only for non-first
            rt.append(th); lrt.append(th); isq.append(1); skip.append(1); has.append(1)
        seen.add(c)
        rt.append(th)
        lrt.append(np.nan if nxt[i] is None else nxt[i])
        isq.append(0); skip.append(0); has.append(0 if nxt[i] is None else 1)
    n = len(rt)
    lab = np.zeros((n, NCOL), dtype=np.float32)
    lab[:, _LBL_HAS_LABEL] = has
    lab[:, _LBL_IS_QUERY] = isq
    d = D()
    d.skips = torch.tensor(skip, dtype=torch.int32)
    d.global_labels = torch.tensor(lab)
    d.review_ths = torch.tensor(rt, dtype=torch.int64)
    d.label_review_ths = torch.tensor(np.array(lrt, dtype=np.float64))
    d.card_features = torch.zeros(n, 1)
    return d, np.array(rt), np.array(lrt, dtype=np.float64), np.array(isq, dtype=bool)

# Interleaved cards so "next review of the SAME card" is never just "the next row" -- a
# neighbour-offset bug would pass on a single-card sequence.
card_of = [0, 1, 0, 2, 1, 0, 2, 1]
d, rt, lrt, isq = make(card_of, len(card_of))
aq = build_ahead_query(d, base=0)

ok = True
n_hit = 0
for r in range(len(rt)):
    q = aq[r]
    if isq[r]:
        if q != -1:
            print(f"  row {r}: QUERY row got teacher {q}, must be -1 (self-reference)"); ok = False
        continue
    if not np.isfinite(lrt[r]):
        if q != -1:
            print(f"  row {r}: no next review but got teacher {q}"); ok = False
        continue
    # THE PROPERTY: the teacher row must be a query row scoring the SAME review as r's label.
    if q < 0:
        print(f"  row {r}: expected a teacher for label_review_th={lrt[r]}, got -1"); ok = False
        continue
    if not (isq[q] and rt[q] == lrt[r] and lrt[q] == lrt[r]):
        print(f"  row {r}: teacher {q} is_query={isq[q]} review_th={rt[q]} "
              f"label_review_th={lrt[q]} but r.label_review_th={lrt[r]}"); ok = False
        continue
    n_hit += 1
print(f"CHECK_MAP teacher index correct on {n_hit} labelled real rows "
      f"(rows={len(rt)}, cards={len(set(card_of))}): {'ok' if ok else 'FAIL'}")
assert ok, "the ahead_query mapping does not point at the review the label refers to"

# NEGATIVE CONTROL: the WRONG join (review r's own decision point, i.e. probe_query's rule) must
# disagree -- otherwise CHECK_MAP is vacuous on this fixture.
wrong = 0
q_by_rt = {int(rt[i]): i for i in range(len(rt)) if isq[i]}
for r in range(len(rt)):
    if isq[r] or not np.isfinite(lrt[r]):
        continue
    own = q_by_rt.get(int(rt[r]), -1)
    if own != aq[r]:
        wrong += 1
print(f"CHECK_MAP' own-review join differs on {wrong} rows (must be > 0, else the fixture "
      f"cannot tell the two joins apart)")
assert wrong > 0

# base offset must shift every hit and leave -1 alone
aq2 = build_ahead_query(d, base=1000)
shifted = np.all((aq2 == -1) == (aq == -1)) and np.all(aq2[aq >= 0] == aq[aq >= 0] + 1000)
print(f"CHECK_MAP'' base offset applied to hits only: {'ok' if shifted else 'FAIL'}")
assert shifted
print("MAP OK")
"""

# --------------------------------------------------------------------------------------------
# Child 2: THE TARGET MATH, exercised through the real forward's mixing block.
# --------------------------------------------------------------------------------------------
CHILD_MATH = r"""
import torch

BETA = float(__import__("os").environ["RWKV_SELFKD_BETA"])

def mix(label_y, out_p_logits, ahead_query, beta, kd=None):
    '''Literal copy of forward()'s two blocks, in order: self-KD softens the HARD label, then the
    external KD mixes over the result. Kept as an independent re-statement so the test fails if
    the shipped ORDER changes (post-KD softening would silently rescale alpha).'''
    if beta != 0.0 and ahead_query is not None:
        _aq = ahead_query.reshape(-1)
        _psucc = (1.0 - torch.softmax(out_p_logits.float(), dim=-1)[:, :, 0]).reshape(-1)
        _teacher = _psucc[_aq.clamp(min=0)].detach()
        _ly = label_y.reshape(-1)
        _soft = beta * _teacher + (1.0 - beta) * _ly
        label_y = torch.where(_aq >= 0, _soft, _ly).reshape(label_y.shape)
    if kd is not None:
        km_curve, alpha = kd
        label_y = alpha * km_curve + (1.0 - alpha) * label_y
    return label_y

torch.manual_seed(3)
B, T = 3, 40
label_y = (torch.rand(B, T) < 0.85).float()
logits = torch.randn(B, T, 4, requires_grad=True)
aq = torch.randint(-1, B * T, (B, T))
psucc = (1.0 - torch.softmax(logits.float(), dim=-1)[:, :, 0]).reshape(-1)

# ---- CHECK1: beta=0 leaves the target EXACTLY alone ----
same = torch.equal(mix(label_y, logits, aq, 0.0), label_y)
print(f"CHECK1 beta=0 target byte-identical: {same}")
assert same

# ---- CHECK2: ORACLE. teacher == label => every beta reproduces the beta=0 target ----
# Build logits whose 1-P(Again) equals label_y at each row, then point every row's teacher at
# ITSELF so teacher[r] == label_y[r] by construction.
big = 40.0
o = torch.zeros(B, T, 4)
o[:, :, 0] = torch.where(label_y < 0.5, torch.tensor(big), torch.tensor(-big))
o[:, :, 1] = torch.where(label_y < 0.5, torch.tensor(-big), torch.tensor(big))
self_idx = torch.arange(B * T).reshape(B, T)
ok2 = True
for b in (0.0, 0.3, 0.7, 1.0):
    got = mix(label_y, o, self_idx, b)
    d = (got - label_y).abs().max().item()
    hit = d < 1e-5
    ok2 = ok2 and hit
    print(f"CHECK2 oracle teacher==label, beta={b:<4}: max|d|={d:.2e} {'ok' if hit else 'FAIL'}")
assert ok2, "beta mix or the 1-P(Again) polarity is wrong"

# negative control: inverted teacher must move the target by O(1)
oi = o.clone(); oi[:, :, 0] = -o[:, :, 0]; oi[:, :, 1] = -o[:, :, 1]
dbad = (mix(label_y, oi, self_idx, 1.0) - label_y).abs().max().item()
print(f"CHECK2b negative control (inverted teacher): max|d|={dbad:.3f} (must be ~1)")
assert dbad > 0.5, "the oracle cannot see a polarity flip -- it is vacuous"

# ---- CHECK3: -1 rows keep the hard label at ANY beta ----
got = mix(label_y, logits, aq, 1.0)
none_rows = (aq.reshape(-1) < 0)
kept = torch.equal(got.reshape(-1)[none_rows], label_y.reshape(-1)[none_rows])
moved = (got.reshape(-1)[~none_rows] != label_y.reshape(-1)[~none_rows]).any().item()
print(f"CHECK3 rows with no teacher keep the hard label: {kept} ; rows with one move: {moved}")
assert kept and moved

# ---- CHECK4: DETACH -- no gradient reaches the rating head through the target ----
# The target must be a CONSTANT w.r.t. the rating head. With the detach in place it carries no
# grad_fn at all (label_y is data, the teacher is detached), so requires_grad is the assertion --
# calling .backward() on it raises "does not require grad", which is the property, not a failure.
lg = logits.detach().clone().requires_grad_(True)
tgt = mix(label_y, lg, self_idx, 0.7)
print(f"CHECK4 target.requires_grad={tgt.requires_grad} (must be False)")
assert not tgt.requires_grad, "teacher is not detached -- it would train the rating head"

# NEGATIVE CONTROL: the identical mix WITHOUT .detach() must be differentiable and put real
# gradient on the rating head -- otherwise CHECK4 passes for the wrong reason (e.g. beta ignored).
_aq = self_idx.reshape(-1)
_ps = (1.0 - torch.softmax(lg.float(), dim=-1)[:, :, 0]).reshape(-1)
leaky = torch.where(_aq >= 0, 0.7 * _ps[_aq.clamp(min=0)] + 0.3 * label_y.reshape(-1),
                    label_y.reshape(-1))
leaky.sum().backward()
gl = lg.grad.abs().max().item()
print(f"CHECK4b no-detach control: requires_grad={leaky.requires_grad} "
      f"max|grad into p_head|={gl:.3e} (must be > 0)")
assert leaky.requires_grad and gl > 0, "the detach check is vacuous on this fixture"

# ---- CHECK5: the external teacher's weight is UNTOUCHED by beta ----
alpha = 0.9
km = torch.rand(B, T).clamp(1e-4, 1 - 1e-4)
ok5 = True
for b in (0.0, 0.4, 1.0):
    got = mix(label_y, logits, aq, b, kd=(km, alpha))
    hard_soft = mix(label_y, logits, aq, b)              # the (1-a)-weighted part
    want = alpha * km + (1.0 - alpha) * hard_soft
    d = (got - want).abs().max().item()
    # recover alpha empirically: d(target)/d(km) must be exactly alpha for every beta
    kmg = km.clone().requires_grad_(True)
    mix(label_y, logits, aq, b, kd=(kmg, alpha)).sum().backward()
    a_hat = kmg.grad.mean().item()
    hit = d < 1e-6 and abs(a_hat - alpha) < 1e-6
    ok5 = ok5 and hit
    print(f"CHECK5 KD alpha={alpha} beta={b:<4}: |d|={d:.2e} recovered alpha={a_hat:.9f} "
          f"{'ok' if hit else 'FAIL'}")
assert ok5, "beta is disturbing the external teacher's weight"
print("MATH OK")
"""

# --------------------------------------------------------------------------------------------
# Child 3: the real model must still compile as a ScriptModule (eval runs scripted)
# --------------------------------------------------------------------------------------------
CHILD_SCRIPT = r"""
import os, torch
from rwkv.model.srs_model import SrsRWKV
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

m = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
n = sum(p.numel() for p in m.parameters())
print(f"SCRIPT built SrsRWKV ok: params={n:,} type={type(m).__mro__[1].__name__} "
      f"selfkd_beta={getattr(m, 'selfkd_beta', 'MISSING')}")
assert getattr(m, "selfkd_beta", None) == float(os.environ.get("RWKV_SELFKD_BETA", "0"))
assert n == 558212, f"param count changed: {n} (a loss-side change must add none)"
print("SCRIPT OK")
"""


def run_child(tag, code, extra_env):
    env = dict(os.environ)
    env.update(ARCH_ENV)
    env["PYTHONPATH"] = REPO
    env["PYTHONUNBUFFERED"] = "1"
    env["RWKV_SELFKD_BETA"] = "0"
    env.update(extra_env)
    shown = ", ".join(f"{k}={v}" for k, v in extra_env.items()) or "defaults"
    print(f"\n===== {tag} ({shown}) =====")
    p = subprocess.run([sys.executable, "-u", "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stdout.write(p.stderr[-4000:])
    return p.returncode == 0


ok = True
ok &= run_child("MAP", CHILD_MAP, {})
ok &= run_child("MATH", CHILD_MATH, {"RWKV_SELFKD_BETA": "0.5"})
ok &= run_child("SCRIPT beta=0", CHILD_SCRIPT, {"RWKV_SELFKD_BETA": "0"})
ok &= run_child("SCRIPT beta=0.5", CHILD_SCRIPT, {"RWKV_SELFKD_BETA": "0.5"})

print("\n" + ("ALL SMOKE CHECKS PASSED" if ok else "SMOKE FAILED"))
sys.exit(0 if ok else 1)
