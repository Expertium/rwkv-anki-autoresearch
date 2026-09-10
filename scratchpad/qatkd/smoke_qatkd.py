"""Smoke for RWKV_QAT_KD's teacher and targets -- CPU, fp32, the REAL model, a REAL gen-5 chunk.

It calls the two functions the training loop calls, build_qat_kd_teacher() and qat_kd_targets()
(rwkv/train_rwkv.py), so it tests the code that runs, not a copy of it. Three claims:

 1. THE TEACHER IS FULL PRECISION. Built under arm 2's QAT env and stripped, it must give the SAME
    targets as the same checkpoint built under arm 1's PLAIN env. Non-vacuity: the same QAT-env model
    WITHOUT stripping must give DIFFERENT targets (so the QAT hooks do act on this CPU forward, and
    the stripping is what removes them). If the unstripped QAT forward cannot run on CPU, that is
    reported and claim 1 rests on the hook fields alone.
 2. SELF-CONSISTENCY. With teacher == student (same weights, plain, eval mode), the KD term's gradient
    must VANISH: soft-CE and soft-BCE are minimised exactly when the target equals the student's own
    output, so any residual gradient means the target is not the quantity the student computes.
 3. NON-VACUITY OF 2. The pre-fix formula (plain forgetting_curve under the GRU head) must give a
    LARGE KD gradient on the same batch.

Arch flags are read at import and ScriptModule bakes the first construction's flags, so each env runs
in its own child process. Envs are parsed from the real runner files, never the ambient env.

Usage: python scratchpad/qatkd/smoke_qatkd.py      (~5-10 min CPU; run at BelowNormal beside a GPU run)
"""
import json
import os
import re
import subprocess
import sys

REPO = r"C:\Users\Andrew\rwkv-anki-autoresearch"
os.chdir(REPO)
CKPT = "scratchpad/ws10/w10p_d_21870.pth"       # arm 1's decayed final = the teacher
RUN_PLAIN = "scratchpad/w10plain/run_w10plain.cmd"
RUN_QAT = "scratchpad/w10qat/run_w10qat.cmd"
TMP = os.environ.get("TEMP", ".")

CHILD = r'''
import os, sys, json
os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch"); sys.path.insert(0, os.getcwd())
import lmdb, torch
torch.set_num_threads(6)
torch.manual_seed(0)
from rwkv.train_rwkv import build_qat_kd_teacher, qat_kd_targets, SrsRWKV, DEFAULT_ANKI_RWKV_CONFIG
from rwkv.prepare_batch import get_data, prepare
import dataclasses
def _cast(v):   # sam_probe.to_f32, inlined: importing sam_probe parses OUR argv at import time
    if torch.is_tensor(v):
        return v.float() if v.is_floating_point() else v
    if isinstance(v, list):
        return [_cast(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_cast(x) for x in v)
    return v
def to_f32(pb):
    for f in dataclasses.fields(pb):
        setattr(pb, f.name, _cast(getattr(pb, f.name)))
    return pb
mode, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
env = lmdb.open("F:/rwkv_lmdb/train_db_5k_h1_id5", map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    b = min(json.loads(txn.get(b"101_batches")), key=lambda x: x[2])
    pb = to_f32(prepare([get_data(txn, (101, b[0], b[1], b[2]), device="cpu")], seed=1234,
                        probe_density=0.08).to("cpu"))
env.close()
res = {}
if mode in ("stripped", "plain"):
    t, n = build_qat_kd_teacher(ckpt, torch.float32, "cpu")
    tp, tc, _ = qat_kd_targets(t, pb, 1.0)
    torch.save({"p": tp, "c": tc}, out)
    res = {"stripped_hooks": n}
elif mode == "unstripped":
    m = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    m.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    m = m.float().eval()
    try:
        tp, tc, _ = qat_kd_targets(m, pb, 1.0)
        torch.save({"p": tp, "c": tc}, out)
        res = {"ran": True}
    except Exception as exc:
        res = {"ran": False, "error": repr(exc)[:300]}
elif mode == "selfcons":
    s = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    s.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    s = s.float().eval()
    t, _ = build_qat_kd_teacher(ckpt, torch.float32, "cpu")
    new = qat_kd_targets(t, pb, 1.0)
    with torch.no_grad():   # the PRE-FIX formula, reproduced verbatim, as the negative control
        ta, tw, _w, tpl, _s, _d = t.forward_batch(pb.start, pb.sub_gather, pb.sub_gather_lens,
                                                  pb.time_shift_selects, pb.skips, pb.num_data)
        les = pb.labels.float()[..., 0].unsqueeze(-1)
        tcr = t.forgetting_curve(tw, les).clamp(1e-5, 1 - 1e-5)
        tcl = torch.log(tcr / (1 - tcr)) + t.interp(ta, les)
        old = (tpl.float(), torch.sigmoid(tcl).clamp(1e-5, 1 - 1e-5).float(), 1.0)
    def grads(kd):
        s.zero_grad(set_to_none=True)
        st = s.get_loss(pb, kd=kd)
        st.average_loss.backward()
        return float(st.average_loss), [p.grad.detach().clone() if p.grad is not None
                                        else torch.zeros_like(p) for p in s.parameters()]
    l0, g0 = grads(None)
    l1, g1 = grads(new)
    l2, g2 = grads(old)
    norm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs)))
    n0 = norm(g0)
    res = {"gru_on": bool(t.gru_on), "loss_hard": l0, "loss_kd_new": l1, "loss_kd_old": l2,
           "g_hard": n0,
           "ratio_new": norm([a - b for a, b in zip(g1, g0)]) / n0,
           "ratio_old": norm([a - b for a, b in zip(g2, g0)]) / n0}
print("RESULT " + json.dumps(res))
'''


def runner_env(path):
    env = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if ln.upper().startswith("REM"):
            continue
        m = re.match(r"set (RWKV_[A-Z0-9_]+)=(.*)$", ln)
        if m and "%" not in m.group(2):
            env[m.group(1)] = m.group(2)
    return env


def child(mode, runner, out=""):
    src = os.path.join(TMP, "qatkd_child.py")
    open(src, "w", encoding="utf-8").write(CHILD)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update(runner_env(runner))
    env["RWKV_NO_JIT"] = "1"
    cp = subprocess.run([r".venv\Scripts\python.exe", src, mode, CKPT, out or os.path.join(TMP, "x.pt")],
                        env=env, capture_output=True, text=True)
    for ln in cp.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    sys.exit(f"child {mode} failed rc {cp.returncode}\n{cp.stdout[-1500:]}\n{cp.stderr[-3000:]}")


def main():
    import torch
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("  PASS " if c else "  FAIL ") + msg, flush=True)
        ok = ok and c

    fs, fp, fu = (os.path.join(TMP, f"qatkd_{k}.pt") for k in ("stripped", "plain", "unstripped"))
    print("1. the teacher is full precision")
    rs = child("stripped", RUN_QAT, fs)
    rp = child("plain", RUN_PLAIN, fp)
    a, b = torch.load(fs), torch.load(fp)
    dp = float((a["p"] - b["p"]).abs().max())
    dc = float((a["c"] - b["c"]).abs().max())
    print(f"  stripped {rs['stripped_hooks']} QAT hooks (QAT env) vs {rp['stripped_hooks']} (plain env)")
    check(rs["stripped_hooks"] > 0, "the QAT-env teacher had hooks to strip")
    check(dp <= 1e-5 and dc <= 1e-5, f"stripped-QAT targets == plain targets (max |d| p {dp:.2e}, curve {dc:.2e})")
    ru = child("unstripped", RUN_QAT, fu)
    if ru.get("ran"):
        u = torch.load(fu)
        du = max(float((u["p"] - b["p"]).abs().max()), float((u["c"] - b["c"]).abs().max()))
        check(du > 1e-4, f"non-vacuity: the UNSTRIPPED QAT model differs from plain (max |d| {du:.2e})")
    else:
        print(f"  NOTE the unstripped QAT forward cannot run on CPU ({ru.get('error')}); claim 1 rests on the hook fields")

    print("2./3. self-consistency, teacher == student (plain env)")
    r = child("selfcons", RUN_PLAIN)
    print(f"  gru_on={r['gru_on']}  loss hard {r['loss_hard']:.6f}  +kd(new) {r['loss_kd_new']:.6f}"
          f"  +kd(old) {r['loss_kd_old']:.6f}")
    print(f"  ||grad of the KD term|| / ||grad of the hard loss||:  new {r['ratio_new']:.2e}   "
          f"old {r['ratio_old']:.2e}")
    check(r["gru_on"], "the GRU head is on (the case the old formula got wrong)")
    check(r["ratio_new"] < 1e-4, "2. the fixed target makes the KD gradient vanish at teacher == student")
    check(r["ratio_old"] > 1e-2, "3. the pre-fix target does NOT (the test can fail)")
    print("\nSMOKE PASS" if ok else "\nSMOKE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
