"""Is the delta rule's INPUT-DEPENDENCE used? A CPU screen before spending a GPU run.

THE PROPOSAL (Andrew, 2026-08-19): simplify the delta rule to cut parameters for free, in the style
of the earlier param-reduction ladder.

THE PRIOR THAT MAKES IT PLAUSIBLE. The 2026-08-17 lit review measured that the delta term moves the
state-transition eigenvalue by only ~0.15 against a decay of ~0.98 -- "our trunk uses its WKV state
almost as a pure exponential-decay accumulator with a small rank-1 correction; RWKV-7's headline
innovation is barely engaged". It also found BOTH factors are freely learnable toward more delta
authority (reachable ~0.95, operating at ~0.13), so nothing is blocking them. A mechanism that is
barely engaged AND unblocked is the natural place to look for free parameters.

THE PRIZE, counted on the champion checkpoint:
    a_lora   (delta removal rate, per channel)   9,360 params = 1.68%
    k_scale  (kappa magnitude, per head)         5,265 params = 0.94%
                                                --------------------
                                                14,625 params = 2.62%

THE QUESTION THIS SCREEN ANSWERS, and it is NOT "is the delta rule needed". It is narrower and
cheaper: **is the INPUT-DEPENDENCE of `a` and `k_scale` used, or would a learned CONSTANT do?**
`a = sigmoid(a_lora(x))` and `k_scale = sigmoid(k_scale_linear(x))`. If the value a channel takes is
essentially fixed across tokens, the LoRA is an expensive way to store a bias, and replacing it with
a per-channel bias keeps the same function while deleting the A/B matrices.

THE DECOMPOSITION. For each channel c, collect a[t, c] over tokens. Then
    total variance  =  BETWEEN-channel variance   +   WITHIN-channel variance
                       (a per-channel bias gives      (only the LoRA can give this --
                        this for free)                 it is what the params BUY)
Report within/total. Small => the input-dependence is decorative and the cut is close to free.
Large => the LoRA is doing real work and the cut is NOT free, so do not queue it.

⚠ THIS IS A NECESSITY SCREEN, NOT A VERDICT. Three iterations (48, 50, 57) have now shown this model
will USE any freedom it is given while gaining nothing, so "the values vary" does not prove the
variation earns its keep -- but "the values barely vary" does prove a bias would reproduce them.
The screen can therefore KILL the idea cheaply or promote it to a GPU run; it cannot accept it.

Usage: .venv/Scripts/python.exe scratchpad/bughunt/delta_rule_screen.py [user] [max_calls]
"""
import os
import sys

ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1", "RWKV_ZERO_FEATURES": "22",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_MUON": "1", "RWKV_MUON_INCLUDE_LORA": "1", "RWKV_NO_JIT": "1",
}
for k, v in ENV.items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402

sys.path.insert(0, os.getcwd())

CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"
user = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
max_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

# name -> list of (n_tokens, C) tensors, one row per call
rec = {}
calls = {"n": 0}


class Enough(Exception):
    pass


def make_hook(tag):
    def hook(mod, inp, out):
        o = out.detach().float().reshape(-1, out.shape[-1])
        rec.setdefault(tag, []).append(o)
        calls["n"] += 1
        if calls["n"] >= max_calls:
            raise Enough()
    return hook


from rwkv.model import rwkv_rnn_model  # noqa: E402
from pathlib import Path  # noqa: E402
from rwkv.run_as_rnn import run as rnn_run  # noqa: E402

# Attach to every time mixer's a_lora / k_scale, tagged by stream+layer so per-site stats survive.
_orig_init = rwkv_rnn_model.RWKV7RNNTimeMixer.__init__


# ⚠ TAG BY MODULE INSTANCE, NOT BY layer_id. There are five streams, each with its own layer 0,
# and each has a DIFFERENT B matrix. Keying on layer_id alone merged five distinct rank-4 maps
# into one bucket -- which inflated the apparent rank above 4 (impossible for B(A(x)), a purely
# linear map) and contaminated the variance decomposition, since different streams give the same
# channel index different means. Found by noticing the 4 components summed to 0.819, not 1.0.
_seen = {"n": 0}


def patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    lid = getattr(self, "layer_id", "?")
    idx = _seen["n"]
    _seen["n"] += 1
    if hasattr(self, "a_lora_simple"):
        self.a_lora_simple.register_forward_hook(make_hook("a@m%02d_L%s" % (idx, lid)))
    if hasattr(self, "k_scale_linear"):
        self.k_scale_linear.register_forward_hook(make_hook("kscale@m%02d_L%s" % (idx, lid)))


rwkv_rnn_model.RWKV7RNNTimeMixer.__init__ = patched_init

print("screening %s on user %d (<= %d hook calls)" % (CKPT, user, max_calls))
try:
    rnn_run(data_path=Path("../anki-revlogs-10k"), model_path=CKPT,
            label_db_path="label_filter_db", label_db_size=40_000_000_000,
            user_id=user, verbose=False)
except Enough:
    print("(stopped at the call cap)")
except Exception as e:  # noqa: BLE001
    print("walk ended: %s: %s" % (type(e).__name__, e))

print("")
print("--- VARIANCE DECOMPOSITION of the post-sigmoid value, per site")
print("%-22s %7s %8s %8s %9s" % ("site", "tokens", "mean", "within%", "verdict"))
print("-" * 62)
agg = []
for tag in sorted(rec):
    X = torch.cat(rec[tag], dim=0)
    if X.shape[0] < 32:
        continue
    X = torch.sigmoid(X)                      # both sites are sigmoid-activated
    per_ch_mean = X.mean(dim=0, keepdim=True)
    within = (X - per_ch_mean).pow(2).mean().item()      # across-token, what the LoRA buys
    total = (X - X.mean()).pow(2).mean().item()          # everything
    frac = within / max(total, 1e-12)
    agg.append((tag, X.shape[0], X.mean().item(), frac))
    print("%-22s %7d %8.4f %7.1f%% %9s"
          % (tag, X.shape[0], X.mean().item(), 100 * frac,
             "bias-ok" if frac < 0.10 else ("marginal" if frac < 0.30 else "LoRA earns it")))

# ---- RANK SPECTRUM of the a_lora output (the surviving lever) ----------------------------
# The constant route is dead if within% is high, but a_logit is produced by a rank-4 bottleneck,
# so the question becomes: do all 4 directions carry variance? Singular values of the CENTERED
# pre-sigmoid output answer it directly -- centering removes the bias, which a rank cut keeps.
print("")
print("--- a_lora RANK SPECTRUM (centered pre-sigmoid; bias excluded, a rank cut keeps it)")
print("%-22s %34s %10s" % ("site", "variance share by component", "rank>=95%"))
print("-" * 70)
import math
shares = []
for tag in sorted(t for t in rec if t.startswith("a@")):
    X = torch.cat(rec[tag], dim=0)
    if X.shape[0] < 32:
        continue
    Xc = X - X.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(Xc.double())
    e = (sv ** 2)
    e = (e / e.sum())[:4]
    cum = torch.cumsum(e, 0)
    need = int((cum < 0.95).sum().item()) + 1
    shares.append(need)
    print("%-22s %34s %10d" % (tag, "  ".join("%.3f" % v for v in e.tolist()), need))
if shares:
    print("")
    print("components needed for 95%% of the variance: min %d, max %d, mean %.1f (of 4)"
          % (min(shares), max(shares), sum(shares) / len(shares)))

if agg:
    aa=[f for t,_,_,f in agg if t.startswith("a@")]
    kk=[f for t,_,_,f in agg if t.startswith("kscale@")]
    if aa: print("a_lora  sites: %d, mean within-channel share %.1f%%" % (len(aa), 100*sum(aa)/len(aa)))
    if kk: print("k_scale sites: %d, mean within-channel share %.1f%%" % (len(kk), 100*sum(kk)/len(kk)))
    w = sum(f for _, _, _, f in agg) / len(agg)
    print("")
    print("mean within-channel share across sites: %.1f%%" % (100 * w))
    print("")
    if w < 0.10:
        print("=> INPUT-DEPENDENCE IS DECORATIVE. A per-channel bias reproduces these values,")
        print("   so replacing the LoRA with a bias is close to free. WORTH A GPU RUN.")
    elif w < 0.30:
        print("=> MARGINAL. Some real token-to-token structure. A GPU run is defensible but the")
        print("   cut is not free; expect to pay something.")
    else:
        print("=> THE LoRA IS DOING REAL WORK. Most of the variance is token-to-token, which a")
        print("   bias cannot reproduce. DO NOT queue this cut on the free-parameters argument.")
