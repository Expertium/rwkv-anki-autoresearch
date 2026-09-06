"""CPU screens on realcyc (parquet -> create_sample -> prepare -> SrsRWKV.get_loss; no LMDB, no GPU).
(A) PCGrad premise: cos(g_ahead, g_imm) on the SHARED TRUNK per chunk; fraction of trunk tensors with cos<0.
    Kill: cos >= 0 on every chunk (PCGrad acts only on conflict -> literally inert).
(B) Annealed-dropout premise: loss(train mode, dropout on) - loss(eval) at the current rates.
    Kill: |gap| < 1e-4 (dropout already inert -> annealing it to 0 is a no-op).
(C) Cold-grade probe: does the trunk output at REAL rows already linearly carry the k+1 GRADE beyond
    what the curve's own logit R gives?  LOUO multinomial probes.  Kill: CE gain of [R,x] over [R] > 0.05 nats
    (already carried -> an aux head is trivially satisfied, trunk unchanged).
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
    "RWKV_NO_JIT": "1", "RWKV_DROPOUT_SCALE": "0.5",
}
for k, v in _ENV.items():
    os.environ[k] = v
from pathlib import Path
import dataclasses
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.data_processing import get_rwkv_data, create_sample
from rwkv.prepare_batch import prepare

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
CKPT = "scratchpad/realcyc/rc_d_10935.pth"
USERS = [int(a) for a in sys.argv[1:]] or [110, 107, 136, 156, 203, 120]
HEAD_TAGS = ["head", "p_linear", "s_linear", "d_linear", "w_linear", "ahead_linear", "gru_", "pava_", "prehead"]


def to_f32(pb):
    def cast(v):
        if torch.is_tensor(v):
            return v.float() if v.is_floating_point() else v
        if isinstance(v, list):
            return [cast(x) for x in v]
        if isinstance(v, tuple):
            return tuple(cast(x) for x in v)
        return v
    for f in dataclasses.fields(pb):
        setattr(pb, f.name, cast(getattr(pb, f.name)))
    return pb


def build(uid):
    df = get_rwkv_data(DATA, uid)
    df = df.iloc[:1500]
    s = create_sample(user_id=uid, section_df=df, equalize_review_ths=[], dtype=torch.float32, device=torch.device("cpu"))
    pb = prepare([s], seed=1234, probe_density=0.08)
    return to_f32(pb.to("cpu")), len(df)


torch.manual_seed(0)
model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
model = model.float()
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
is_head = [any(t in n for t in HEAD_TAGS) for n, _ in named]
params = [p for _, p in named]
cap = {}
h = model.prehead_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("x", o.detach()))


def grads(loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def cosine(ga, gi, sel):
    num = sum(float((a * b).sum()) for a, b, s in zip(ga, gi, sel) if s)
    na = sum(float((a * a).sum()) for a, s in zip(ga, sel) if s)
    ni = sum(float((b * b).sum()) for b, s in zip(gi, sel) if s)
    return num / (np.sqrt(na * ni) + 1e-20), np.sqrt(na), np.sqrt(ni)


probe = {"x": [], "R": [], "rat": [], "u": [], "bce": []}
rows = []
_ce = F.cross_entropy
for uid in USERS:
    t0 = time.time()
    pb, nrows = build(uid)
    model.eval()
    torch.set_grad_enabled(True)
    st = model.get_loss(pb)
    l_tot = st.average_loss
    g_tot = grads(l_tot)
    x = cap["x"]                                   # (B,T,C) normed trunk output
    lab = pb.labels.float()                        # (B,T,7)
    has, isq, y, rating, t = lab[..., 4], lab[..., 6], lab[..., 2], lab[..., 3], lab[..., 0]
    real = (has > 0) & (isq == 0)
    pc = st.p_curve.detach().float()
    R = torch.log(pc / (1 - pc))
    bce = F.binary_cross_entropy(pc.clamp(1e-6, 1 - 1e-6), y, reduction="none")
    probe["x"].append(x[real].half()); probe["R"].append(R[real]); probe["rat"].append((rating[real] - 1).clamp(min=0).long())
    probe["u"].append(torch.full((int(real.sum()),), uid)); probe["bce"].append(bce[real])
    # ahead-only gradient: zero the 4-way CE (the only consumer of F.cross_entropy in _get_loss)
    F.cross_entropy = lambda *a, **k: torch.zeros_like(_ce(*a, **k))
    st_a = model.get_loss(pb)
    F.cross_entropy = _ce
    g_a = grads(st_a.average_loss)
    g_i = [gt - ga for gt, ga in zip(g_tot, g_a)]
    trunk = [not hh for hh in is_head]
    c_tr, na, ni = cosine(g_a, g_i, trunk)
    per = np.array([float((a * b).sum() / (a.norm() * b.norm() + 1e-20)) for a, b, s in zip(g_a, g_i, trunk) if s and a.norm() > 0 and b.norm() > 0])
    stream_c = {}
    for sname in ["features2card", "card_id", "note_id", "deck_id", "preset_id", "user_id"]:
        sel = [(sname in n) and not hh for (n, _), hh in zip(named, is_head)]
        if any(sel):
            stream_c[sname] = cosine(g_a, g_i, sel)[0]
    # (B) dropout gap
    with torch.no_grad():
        model.train()
        ls = []
        for sd in range(3):
            torch.manual_seed(sd)
            ls.append(float(model.get_loss(pb).average_loss))
        model.eval()
    dgap = float(np.mean(ls)) - float(l_tot)
    rows.append((uid, nrows, c_tr, per, stream_c, na, ni, dgap, float(st.ahead_avg), float(st.imm_avg)))
    print(f"user {uid:>5} rows {nrows:>5}  cos(g_ahead,g_imm)[trunk] {c_tr:+.4f}  per-tensor median {np.median(per):+.4f} frac<0 {np.mean(per<0):.2f}"
          f"  ||g_a|| {na:.4f} ||g_i|| {ni:.4f}  streams " + " ".join(f"{k}:{v:+.3f}" for k, v in stream_c.items())
          + f"  | dropout gap {dgap:+.5f} (L_eval {float(l_tot):.4f} ahead {float(st.ahead_avg):.4f} imm {float(st.imm_avg):.4f})  [{time.time()-t0:.0f}s]", flush=True)

cs = np.array([r[2] for r in rows])
print(f"\n(A) PCGrad: trunk cos(g_ahead, g_imm) per chunk: min {cs.min():+.4f} median {np.median(cs):+.4f} max {cs.max():+.4f}; "
      f"chunks with cos<0: {int((cs<0).sum())}/{len(cs)}; per-tensor frac<0 mean {np.mean([np.mean(r[3]<0) for r in rows]):.2f}"
      + ("  -> DEAD (no conflict anywhere)" if cs.min() >= 0 else "  -> conflict present"))
dg = np.array([r[7] for r in rows])
print(f"(B) dropout gap (train-mode loss - eval loss): median {np.median(dg):+.5f} max {dg.max():+.5f}" + ("  -> DEAD (<1e-4)" if np.abs(dg).max() < 1e-4 else ""))

# (C) LOUO probes
X = torch.cat(probe["x"]).float(); Rr = torch.cat(probe["R"]).unsqueeze(1); Y = torch.cat(probe["rat"]); U = torch.cat(probe["u"]); B = torch.cat(probe["bce"])
X = (X - X.mean(0)) / (X.std(0) + 1e-6); Rr = (Rr - Rr.mean()) / (Rr.std() + 1e-6)


def fit(Xtr, Ytr, Xte, Yte, l2=1e-3, iters=200):
    W = torch.zeros(Xtr.shape[1], 4, requires_grad=True); b = torch.zeros(4, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); l = F.cross_entropy(Xtr @ W + b, Ytr) + l2 * (W ** 2).sum(); l.backward(); return l
    opt.step(closure)
    with torch.no_grad():
        return F.cross_entropy(Xte @ W + b, Yte, reduction="none")


res = {"R": [], "Rx": [], "x": []}; per_row = []
for uid in USERS:
    te = U == uid; tr = ~te
    if te.sum() < 50:
        continue
    ceR = fit(Rr[tr], Y[tr], Rr[te], Y[te]); ceRx = fit(torch.cat([Rr, X], 1)[tr], Y[tr], torch.cat([Rr, X], 1)[te], Y[te]); cex = fit(X[tr], Y[tr], X[te], Y[te])
    res["R"].append(float(ceR.mean())); res["Rx"].append(float(ceRx.mean())); res["x"].append(float(cex.mean()))
    per_row.append((ceRx, B[te]))
    print(f"  LOUO user {uid}: CE(grade | R) {float(ceR.mean()):.4f}  CE(grade | R,x) {float(ceRx.mean()):.4f}  CE(grade | x) {float(cex.mean()):.4f}  n={int(te.sum())}")
gain = np.mean(res["R"]) - np.mean(res["Rx"])
allce = torch.cat([a for a, _ in per_row]); allb = torch.cat([b for _, b in per_row])
corr = float(np.corrcoef(allce.numpy(), allb.numpy())[0, 1])
print(f"(C) cold-grade probe: mean CE(grade|R) {np.mean(res['R']):.4f}  CE(grade|R,x) {np.mean(res['Rx']):.4f}  gain {gain:+.4f} nats"
      f"  corr(probe CE row, ahead BCE row) {corr:+.3f}" + ("  -> DEAD (trunk already carries the grade linearly)" if gain > 0.05 else "  -> trunk does NOT linearly carry the grade beyond R"))
