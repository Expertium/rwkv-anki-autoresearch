"""Smoke for RWKV_CLOCK_AT_PREV_ANSWER (rwkv/clock_fix.py), CPU only, real gen-5 chunks.

Four parts, each in its own process because the flag is read at import:
  1. INERT   flag unset: prepare() is BIT-IDENTICAL to HEAD's prepare_batch on gen-5 test and train
             chunks (probes off, eval density 1.0, train density 0.08) and on a published e2s chunk.
             This is what protects the running chain: its later phases import prepare_batch fresh.
  2. ON      T=1800 at the sample level: real-row features untouched; a query row changes only if
             its gap is an in-session one; the query transform equals phase L's counterfactual on
             every column both shift; a label changes only on a labelled real row, by exactly the
             in-session gap of its target review (recomputed independently); probes take their
             target's gap; the probe selection is unchanged.
  3. END2END prepare() ON vs OFF: the batch differs only at query / probe rows (start) and on the
             label time of labelled real rows; every other tensor is identical.
  4. PARITY  the deploy path (run_as_rnn.imm_predict + clock_fix.ahead_t) against the LMDB path for
             the same user: the same rows shift, by the same delta, to the same values within bf16,
             and the fix does not degrade the agreement the two paths had with the flag off.

Usage: python scratchpad/leak/smoke_clock_fix.py        (exit 0 = PASS)
"""
import json
import os
import subprocess
import sys

REPO = "C:/Users/Andrew/rwkv-anki-autoresearch"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
TMP = os.path.join(os.environ.get("TEMP", "."), "smoke_clock_fix")
TEST_DB = "F:/rwkv_lmdb/test_db_5k_id5"
TRAIN_DB = "F:/rwkv_lmdb/train_db_5k_h1_id5"
E2S_DB = "F:/rwkv_lmdb/test_db_5k_e2s"
ID_DATA = "C:/Users/Andrew/anki-revlogs-10k-id"
T_FIX = "1800"

COMMON = r'''
import os, sys, json, dataclasses
sys.path.insert(0, "C:/Users/Andrew/rwkv-anki-autoresearch"); os.chdir("C:/Users/Andrew/rwkv-anki-autoresearch")
import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
import numpy as np, torch, lmdb
from rwkv import prepare_batch as PB
from rwkv import clock_fix as CF

def load(db, user, which):
    env = lmdb.open(db, map_size=400_000_000_000, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        bl = json.loads(txn.get(f"{user}_batches".encode()))
        b = bl[which]
        d = PB.get_data(txn, (user, b[0], b[1], b[2]), device="cpu")
    env.close()
    return d

def fields(pb):
    return {f.name: getattr(pb, f.name) for f in dataclasses.fields(pb)}

def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b
'''

CHILD_INERT = COMMON + r'''
import importlib.util, subprocess
# the reference is the last committed prepare_batch WITHOUT the hook: the parent of the commit that
# introduced it, or HEAD while it is not committed yet
intro = subprocess.run(["git", "log", "--format=%H", "-S", "clock_fix", "--", "rwkv/prepare_batch.py"],
                       capture_output=True, text=True, check=True).stdout.split()
rev = (intro[-1] + "^") if intro else "HEAD"
src = subprocess.run(["git", "show", rev + ":rwkv/prepare_batch.py"], capture_output=True, text=True, check=True).stdout
assert "clock_fix" not in src, rev
print("reference prepare_batch = " + rev)
p = os.path.join(sys.argv[1], "prepare_batch_head.py"); open(p, "w", encoding="utf-8").write(src)
spec = importlib.util.spec_from_file_location("prepare_batch_head", p)
HEAD = importlib.util.module_from_spec(spec); spec.loader.exec_module(HEAD)
assert not CF.enabled()
cases = json.loads(sys.argv[2])
res = []
for db, user, which in cases:
    for dens, eq in ((0.0, False), (1.0, True), (0.08, False)):
        a = fields(PB.prepare([load(db, user, which)], target_len=65536, seed=1234, probe_density=dens, probe_equalize_only=eq))
        b = fields(HEAD.prepare([load(db, user, which)], target_len=65536, seed=1234, probe_density=dens, probe_equalize_only=eq))
        bad = [k for k in a if not same(a[k], b[k])]
        res.append({"db": db, "user": user, "chunk": which, "density": dens, "rows": int(a["start"].shape[0]), "differ": bad})
print("RESULT " + json.dumps(res))
'''

CHILD_ON = COMMON + r'''
import importlib.util
assert CF.enabled() and CF.T == float(os.environ["RWKV_CLOCK_AT_PREV_ANSWER"])
os.environ["LEAK_CF_T"] = os.environ["RWKV_CLOCK_AT_PREV_ANSWER"]; os.environ["RWKV_EVAL_PAVA"] = "1"
spec = importlib.util.spec_from_file_location("cf", "scratchpad/leak/get_result_cf.py")
LCF = importlib.util.module_from_spec(spec); spec.loader.exec_module(LCF)
PB.insert_probes = LCF._orig_insert_probes          # the counterfactual patches it on import: undo
from rwkv.data_processing import CARD_FEATURE_COLUMNS as C
IANY, T = C.index("scaled_t_since_any_review"), CF.T
BOTH = [C.index(n) for n in ("scaled_t_since_any_review", "scaled_sibling_gap", "scaled_elapsed_seconds",
        "scaled_elapsed_seconds_cumulative", "elapsed_seconds_sin", "elapsed_seconds_cos",
        "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos", "tod_sin", "tod_cos",
        "tod_dev_sin", "tod_dev_cos")]
ONLY_FIX = [C.index("scaled_user_tenure"), C.index("scaled_deck_age_at_review")]
out = []
for db, user, which in json.loads(sys.argv[1]):
    d0 = load(db, user, which)
    d1 = CF.apply_to_sample(d0)
    A, B = d0.card_features.double().numpy(), d1.card_features.double().numpy()
    lab0, lab1 = d0.global_labels.double().numpy(), d1.global_labels.double().numpy()
    sk = d0.skips.numpy().astype(bool)
    isq = sk & (lab0[:, 6] > 0.5)
    real = ~sk
    g = np.maximum(CF._inv(A[:, IANY], *CF._ANY), 0.0)
    ins = (A[:, IANY] != 0.0) & (g < T) & (g > 0.0)
    changed = (A != B).any(axis=1)
    fwd0 = torch.tensor(CF._fwd(0.0, *CF._ANY)).to(d0.card_features.dtype).double().item()
    big = isq & ins & (g >= 2.0)
    r = {"db": db[-14:], "user": user, "rows": int(A.shape[0]), "query": int(isq.sum()),
         "query_in_session": int((isq & ins).sum()), "query_changed": int((changed & isq).sum()),
         "real_changed": int((changed & real).sum()), "changed_outside_query_insession": int((changed & ~(isq & ins)).sum()),
         "any_not_zeroed": int((B[big, IANY] != fwd0).sum()),
         "cols_changed": sorted({C[j] for j in np.nonzero((A != B).any(axis=0))[0]})}
    # the query transform against phase L's counterfactual, on the columns both shift
    cf2, _, _ = LCF.transform(d0.card_features, isq)
    L = cf2.double().numpy()
    r["vs_counterfactual_both_cols_mismatch"] = int((L[:, BOTH] != B[:, BOTH]).sum())
    r["vs_counterfactual_only_fix_cols_changed"] = int((L[:, ONLY_FIX] != B[:, ONLY_FIX]).sum())
    # labels: recompute independently, row by row
    rt = d0.review_ths.numpy().astype(np.int64)
    by_rt = {int(rt[i]): i for i in np.nonzero(real)[0]}
    lrt = d0.label_review_ths.numpy().astype(np.int64)
    lbl = real & (lab0[:, 4] > 0.5) & (lab0[:, 6] < 0.5)
    exp = lab0[:, 0].copy(); missing = 0; shifted = 0
    for i in np.nonzero(lbl)[0]:
        k = by_rt.get(int(lrt[i]))
        if k is None:
            missing += 1; continue
        s = A[k, IANY]; gk = max(float(CF._inv(s, *CF._ANY)), 0.0)
        if s != 0.0 and 0.0 < gk < T:
            exp[i] = max(lab0[i, 0] - gk, 0.0); shifted += 1
    exp_t = torch.from_numpy(exp).to(d0.global_labels.dtype).double().numpy()
    lab_changed = lab0[:, 0] != lab1[:, 0]
    rel = np.where(lbl & (lab0[:, 0] > 0), (lab0[:, 0] - lab1[:, 0]) / np.maximum(lab0[:, 0], 1e-9), 0.0)
    r.update({"labels": int(lbl.sum()), "labels_expected_shift": shifted, "labels_target_missing": missing,
              "labels_changed": int(lab_changed.sum()), "labels_changed_not_labelled_real": int((lab_changed & ~lbl).sum()),
              "labels_mismatch_vs_independent": int((exp_t != lab1[:, 0]).sum()),
              "labels_moved_ge_10pct_share": float((rel >= 0.10).sum() / max(lbl.sum(), 1)),
              "label_other_cols_changed": int((lab0[:, 1:] != lab1[:, 1:]).sum())})
    # probes: target's gap, selection unchanged
    e0, m0 = PB.insert_probes(d0, 1.0, 1234, equalize_only=True)
    e1, m1 = PB.insert_probes(d1, 1.0, 1234, equalize_only=True)
    if m0 is not None:
        f1 = CF.apply_to_probes(e1, m1)
        P0, P1 = e1.card_features.double().numpy(), f1.card_features.double().numpy()
        pm = np.zeros(P0.shape[0], bool); pm[m1.pos4.reshape(-1)] = True
        gp = np.maximum(CF._inv(P0[:, IANY], *CF._ANY), 0.0)
        pins = (P0[:, IANY] != 0.0) & (gp < T) & (gp > 0.0)
        pch = (P0 != P1).any(axis=1)
        tgt_lab = e1.global_labels.double().numpy()[m1.target, 0]
        pr_lab = e1.global_labels.double().numpy()[m1.pos4[:, 0], 0]
        r.update({"probe_rows": int(pm.sum()), "probe_in_session": int((pm & pins).sum()),
                  "probe_changed": int((pch & pm).sum()), "nonprobe_changed_by_probe_step": int((pch & ~pm).sum()),
                  "probe_changed_outside_in_session": int((pch & ~pins).sum()),
                  "probe_selection_same": bool(np.array_equal(m0.pos4, m1.pos4) and np.array_equal(m0.target, m1.target)),
                  "probe_label_equals_target_label": bool(np.array_equal(tgt_lab, pr_lab))})
    out.append(r)
print("RESULT " + json.dumps(out))
'''

CHILD_DUMP = COMMON + r'''
db, user, which, path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pb = PB.prepare([load(db, user, which)], target_len=65536, seed=1234, probe_density=1.0, probe_equalize_only=True)
torch.save(fields(pb), path)
print("RESULT " + json.dumps({"on": CF.enabled(), "path": path}))
'''

CHILD_PARITY = COMMON + r'''
from pathlib import Path
import rwkv.run_as_rnn as R
from rwkv.data_processing import CARD_FEATURE_COLUMNS as C, get_rwkv_data
db, user, data, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
class Stub(R.RNNProcess):
    def run(self, row, skip):
        self.last = np.array([float(row[c]) for c in C], dtype=np.float64)
        return (None, None, None, None), torch.tensor(0.5)
srs = Stub(path=None, device=torch.device("cpu"), dtype=torch.float32)
df = get_rwkv_data(Path(data), user).sort_values("review_th", kind="stable").reset_index(drop=True)
q_dep, t_dep = {}, {}
for _, row in df.iterrows():
    rth = int(row["review_th"])
    imm = row.copy(); imm.drop(columns=["rating", "duration"], inplace=True)
    srs.imm_predict(imm); q_dep[rth] = srs.last.copy()
    t_dep[rth] = float(CF.ahead_t(row))
    srs.process_row(row)
env = lmdb.open(db, map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    bl = json.loads(txn.get(f"{user}_batches".encode()))
    chunks = [PB.get_data(txn, (user, b[0], b[1], b[2]), device="cpu") for b in bl]
env.close()
q_lm, t_lm = {}, {}
for d in chunks:
    if CF.enabled():
        d = CF.apply_to_sample(d)
    F = d.card_features.double().numpy(); lab = d.global_labels.double().numpy()
    sk = d.skips.numpy().astype(bool); rt = d.review_ths.numpy(); lrt = d.label_review_ths.numpy()
    for i in np.nonzero(sk & (lab[:, 6] > 0.5))[0]:
        q_lm[int(rt[i])] = F[i]
    for i in np.nonzero(~sk & (lab[:, 4] > 0.5) & (lab[:, 6] < 0.5))[0]:
        t_lm[int(lrt[i])] = lab[i, 0]
keys = sorted(set(q_dep) & set(q_lm)); lk = sorted(set(t_dep) & set(t_lm))
np.savez(path, keys=np.array(keys), dep=np.stack([q_dep[k] for k in keys]), lm=np.stack([q_lm[k] for k in keys]),
         lkeys=np.array(lk), tdep=np.array([t_dep[k] for k in lk]), tlm=np.array([t_lm[k] for k in lk]),
         shifted=np.array(CF.SHIFTED_COLUMNS if CF.enabled() else []), iany=C.index("scaled_t_since_any_review"))
print("RESULT " + json.dumps({"on": CF.enabled(), "reviews": len(df), "query_pairs": len(keys), "label_pairs": len(lk)}))
'''


def child(code, args, flag, id_features=True):
    os.makedirs(TMP, exist_ok=True)
    src = os.path.join(TMP, "child.py")
    open(src, "w", encoding="utf-8").write(code)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_") and k != "LEAK_CF_T"}
    env.update({"RWKV_NO_JIT": "1", "PYTHONUNBUFFERED": "1"})
    if id_features:
        env.update({"RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1"})
    if flag:
        env["RWKV_CLOCK_AT_PREV_ANSWER"] = flag
    cp = subprocess.run([PY, src] + [str(a) for a in args], env=env, capture_output=True, text=True, cwd=REPO)
    line = [l for l in cp.stdout.splitlines() if l.startswith("RESULT ")]
    if cp.returncode != 0 or not line:
        print(cp.stdout[-3000:]); print(cp.stderr[-4000:])
        sys.exit("child failed")
    return json.loads(line[0][7:]), cp.stdout


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print("== 1. INERT: flag unset, prepare() vs HEAD")
    cases = [[TEST_DB, 5030, 0], [TEST_DB, 5033, 0], [TRAIN_DB, 101, 1]]
    r, _ = child(CHILD_INERT, [TMP, json.dumps(cases)], None)
    for x in r:
        check(not x["differ"], f"gen-5 {x['db'][-20:]} user {x['user']} chunk {x['chunk']} density {x['density']} "
                               f"({x['rows']} rows): differ {x['differ']}")
    r, _ = child(CHILD_INERT, [TMP, json.dumps([[E2S_DB, 5030, 0]])], None, id_features=False)
    for x in r:
        check(not x["differ"], f"published e2s user {x['user']} density {x['density']}: differ {x['differ']}")

    print(f"== 2. ON: T={T_FIX} s, sample level")
    r, out = child(CHILD_ON, [json.dumps(cases)], T_FIX)
    check("[clock-fix pid" in out and "ON: T=1800" in out, "the ON banner prints")
    for x in r:
        tag = f"{x['db']} user {x['user']}"
        print("  " + json.dumps(x))
        check(x["real_changed"] == 0, f"{tag}: no real row's features change")
        check(x["changed_outside_query_insession"] == 0, f"{tag}: only in-session query rows change")
        check(x["query_changed"] > 0, f"{tag}: the fix engages ({x['query_changed']} of {x['query']} query rows)")
        check(x["any_not_zeroed"] == 0, f"{tag}: every in-session gap >= 2 s is zeroed")
        check(x["vs_counterfactual_both_cols_mismatch"] == 0, f"{tag}: equals phase L's counterfactual on its 12 columns")
        check(x["labels_mismatch_vs_independent"] == 0, f"{tag}: label shift equals the independent recomputation")
        check(x["labels_changed_not_labelled_real"] == 0 and x["label_other_cols_changed"] == 0,
              f"{tag}: only the label time of labelled real rows moves")
        check(x["labels_changed"] > 0, f"{tag}: labels engage ({x['labels_changed']} of {x['labels']}; "
                                       f"{x['labels_target_missing']} targets not in chunk)")
        if "probe_rows" in x:
            check(x["probe_selection_same"], f"{tag}: probe selection unchanged")
            check(x["probe_label_equals_target_label"], f"{tag}: probes carry their target's shifted label")
            check(x["nonprobe_changed_by_probe_step"] == 0 and x["probe_changed_outside_in_session"] == 0,
                  f"{tag}: the probe step changes only in-session probe rows")
            check(x["probe_changed"] > 0, f"{tag}: probes engage ({x['probe_changed']} of {x['probe_rows']})")

    print("== 3. END2END: prepare() ON vs OFF, eval probes")
    import numpy as np
    import torch
    p_off, p_on = os.path.join(TMP, "pb_off.pt"), os.path.join(TMP, "pb_on.pt")
    child(CHILD_DUMP, [TEST_DB, 5030, 0, p_off], None)
    child(CHILD_DUMP, [TEST_DB, 5030, 0, p_on], T_FIX)
    a, b = torch.load(p_off, weights_only=False), torch.load(p_on, weights_only=False)
    other = [k for k in a if k not in ("start", "labels") and not _same(a[k], b[k])]
    check(not other, f"every tensor but start/labels identical (differ: {other})")
    S0, S1 = a["start"].double().numpy(), b["start"].double().numpy()
    L0, L1 = a["labels"].double().numpy().reshape(-1, 7), b["labels"].double().numpy().reshape(-1, 7)
    qpos = np.nonzero(L0[:, 6] > 0.5)[0]
    ppos = a["probe_rows"].numpy().reshape(-1) if a["probe_rows"] is not None else np.zeros(0, np.int64)
    allowed = np.zeros(S0.shape[0], bool); allowed[qpos] = True; allowed[ppos] = True
    srow = (S0 != S1).any(axis=1)
    check(srow.any() and not (srow & ~allowed).any(),
          f"start differs on {int(srow.sum())} rows, all of them query/probe rows")
    lrow = (L0 != L1).any(axis=1)
    pmask = np.zeros(S0.shape[0], bool); pmask[ppos] = True
    real_lbl = (L0[:, 4] > 0.5) & (L0[:, 6] < 0.5)
    check(lrow.any() and not (L0[:, 1:] != L1[:, 1:]).any() and not lrow[qpos].any()
          and not (lrow & ~(real_lbl | pmask)).any(),
          f"labels differ on {int(lrow.sum())} rows, only in the time column, only on labelled real rows "
          f"and the probes that copy them ({int((lrow & pmask).sum())})")

    print("== 4. PARITY: deploy path vs LMDB path, user 5030, OFF then ON")
    got = {}
    for flag in (None, T_FIX):
        p = os.path.join(TMP, f"parity_{'on' if flag else 'off'}.npz")
        r, _ = child(CHILD_PARITY, [TEST_DB, 5030, ID_DATA, p], flag)
        print("  " + json.dumps(r))
        got[flag] = np.load(p)
    on = got[T_FIX]
    cols = on["shifted"]; iany = int(on["iany"])
    std, mean = 1.3649, 2.2682

    def within(dep, lm):
        return np.abs(dep - lm) <= np.abs(dep) * 2.0 ** -7 + 2e-3

    # The label tolerance carries a delta term: the LMDB path recovers delta from the bf16-stored
    # t_since_any_review (~1% of delta, i.e. up to ~18 s at T=1800), the deploy path from float64.
    # Measured on the first run: all 10 pairs outside a delta-free tolerance had t' << delta and sat
    # inside 1.1% of delta. It is the storage precision of the LMDB, not a disagreement between paths.
    t_raw = got[None]["tdep"]
    for flag, z in got.items():
        w = within(z["dep"][:, cols], z["lm"][:, cols]).mean()
        d = np.maximum(t_raw - z["tdep"], 0.0)
        tl = (np.abs(z["tdep"] - z["tlm"]) <= np.abs(t_raw) * 2.0 ** -7 + 0.011 * d + 1.0).mean()
        print(f"  {'ON ' if flag else 'OFF'}: clock columns within bf16 {w:.5f} | label t within bf16 {tl:.5f}")
        got[flag] = (w, tl, z)
    (w_off, tl_off, off), (w_on, tl_on, _) = got[None], got[T_FIX]
    check(w_on >= w_off - 1e-3, f"clock-column agreement not degraded by the fix ({w_off:.5f} -> {w_on:.5f})")
    check(tl_on >= tl_off - 1e-3, f"label-time agreement not degraded by the fix ({tl_off:.5f} -> {tl_on:.5f})")
    g_dep = np.maximum(np.exp(off["dep"][:, iany] * std + mean) - 1 - 1e-5, 0)
    g_lm = np.maximum(np.exp(off["lm"][:, iany] * std + mean) - 1 - 1e-5, 0)
    clear = (off["dep"][:, iany] != 0) & (g_dep >= 1.0) & (g_dep < 0.98 * float(T_FIX))
    sh_dep = (on["dep"] != off["dep"]).any(axis=1)
    sh_lm = (on["lm"] != off["lm"]).any(axis=1)
    check(clear.sum() > 0 and (sh_dep[clear] == sh_lm[clear]).all() and sh_dep[clear].all(),
          f"both paths shift the same {int(clear.sum())} clearly in-session query rows")
    rel = np.abs(g_dep[clear] - g_lm[clear]) / np.maximum(g_dep[clear], 1.0)
    check(rel.max() <= 0.02, f"delta agrees within bf16 (max relative difference {rel.max():.4f})")
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    sys.exit(0 if ok else 1)


def _same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


if __name__ == "__main__":
    main()
