# Literature-review seeds for the architecture step (task 4)

> Periodic NN-architecture review for inspiration (Andrew's charter). Each idea is mapped to OUR
> constraints: 5-stream RWKV-7 (card->deck->note->preset->global), champion d=32/K=32/H=1, layers
> [1,4,3,3,3], **192,800 params (cap 225,000)**, **card & note state FIXED**, deck/preset/global state
> may grow freely. Accept only if BOTH modes improve >=0.0003 vs champion. The model ALREADY has
> RWKV-7's core (extended delta rule, vector-valued decay, in-context learning rate, token-shift,
> channel-mixer), so seeds must ADD something cheap or re-allocate params/depth, not re-derive RWKV-7.

## Ranked candidate ideas (cheap & on-constraint first)

1. **Output gating on stream / head outputs** (NeurIPS 2025 "Gated Attention for LLMs": non-linearity,
   sparsity, attention-sink-free). A simple element-wise `y * sigmoid(x @ W_g)` after a stream's output
   or before the SRS heads. Param cost ~ d*d per gate (1,024 @ d=32) -> a few gates fit the 32k headroom.
   Reported consistent gains from the added non-linearity. **Highest-ROI first try.**
   https://arxiviq.substack.com/p/neurips-2025-gated-attention-for

2. **Layer weight-tying within the multi-layer streams** (Tied-LoRA / weight-sharing, arXiv 2311.09578).
   deck=4, note=3, preset=3, global=3 layers. Tie the LoRA factors (or whole blocks) across a stream's
   layers -> frees params to spend on MORE depth or wider cheap streams at the SAME count. Note state is
   FIXED, but tying note's LAYER weights doesn't change its per-entity STATE size (state = [H,K,K]+shifts
   per layer; tying weights, not state). Lets us deepen note WITHOUT new params (state still grows though
   -> only OK for deck/preset/global). Use for deck/preset/global depth.

3. **Residual learning for linear attention** (arXiv 2509.25223, "Enhancing Linear Attention with
   Residual Learning"). Adds a residual path to the linear-attention state readout. Read for whether it's
   a cheap accuracy add compatible with the RWKV-7 recurrence (must keep the per-entity state size fixed
   for card/note).

4. **DeltaProduct: multiple delta steps per token** (generalizes DeltaNet; referenced in RWKV-7 lineage).
   More expressive state update at the cost of >1 update/step -> raises compute & possibly state. Likely
   too heavy for the card/note fixed-state budget; consider ONLY for deck/preset/global. Lower priority.

5. **Better init / normalization (no param cost)** — cheap to try in the tuner-adjacent sweeps: gate/decay
   bias init, head init scale, LayerNorm vs RMSNorm on the head input, output-gate bias toward open.
   Zero param cost, only-upside if it helps -> fold into the training-pipeline experiments.

## Notes / cautions
- Vector-valued decay + in-context LR are ALREADY in RWKV-7 (don't "add" them).
- Anything touching the card/note WKV state SHAPE is gated out (state caps). Param-only or
  deck/preset/global-state changes are the safe design space.
- Measure every idea on the 100/100 workbench; one idea per iteration; log accepted/rejected.

## ★★ TOP ALGORITHMIC QUEUE (lit search 2026-06-30) -- model is DATA-limited at 100 users (capacity adds
## reject: exp1/exp2/decay8 all failed). So the wins are GENERALIZATION / optimization, not capacity. All below
## are deterministic (preserve variance=0), ZERO param/state cost -> pass param/state gates automatically.

1. **WEIGHT AVERAGING (EMA / SWA)** -- the standout "free lunch": averaging weights over training -> flatter
   minima -> better generalization at no param/state cost. Deterministic. Lit even says averaging "can eliminate
   the need for LR decay" -> EMA may augment OR replace our decay phase. IMPL: env-guarded EMA in train_rwkv
   (RWKV_EMA_DECAY e.g. 0.999); eval the EMA weights. Test (a) EMA over WS (eval EMA, no decay) vs champion
   WS+decay; (b) EMA over WS + decay. Refs: Switch-EMA 2402.09240, EMA-dynamics 2411.18704, SWA. **HIGHEST ROI.**
2. **Schedule-Free AdamW** (Defazio) -- constant LR + averaging, no schedule; matches/beats cosine decay at
   MEDIUM batch (our B~4 is exactly that). Refs: 2507.09846, ScheduleFree+ 2605.19095. (Optimizer swap, med effort.)
3. **Checkpoint merging / WSM** (decay-free): save WS ckpts each epoch, average last K, eval. Easiest SWA variant
   (no training-loop change -- average existing ckpts). Ref: 2507.17634.
4. **Label smoothing / loss-term reweighting** -- the loss has many terms (curve/raw/p/w-div/ahead-mag/ahead-diff
   scales in srs_model._get_loss); the tuner never touched these. A small smoothing or rescale could regularize.

## ★ Concrete experiment QUEUE for task 4 (ordered by ROI; run AFTER the tuner, gated vs the TUNED champion)

All are **state-neutral** (touch params/heads, NOT the per-entity WKV/token-shift state) so they pass the
card/note-state-fixed gate automatically; all stay under the 225k param cap. Model flow (verified in
srs_model.py): features2card(92->32) -> 5 CHAINED RWKV streams (each refines x) -> prehead_norm -> 3 heads
(head_w/curve [drives imm], head_ahead [drives ahead], head_p/rating [drives imm]). Champion 192,800 params.

1. **Restore SRS-head resolution: num_curves/num_points 64 -> 128** (architecture.py DEFAULT_ANKI_RWKV_CONFIG).
   The champion HALVED these (iter29) ONLY to save ~16k params under the OLD +0.0015 floor gate. The new 225k
   cap has room (192,800 -> ~209k). num_curves drives the forgetting-curve mixture (imm/ahead), num_points the
   ahead interp -> a DIRECT lever on both gated modes, ZERO state cost. **Highest-ROI, cheapest, first.**
   (Try 96 too if 128 overshoots the cap or overfits 100 users.)
2. **channel_mixer_factor 1.0 -> 1.5** (all streams or just the cheap ones). Adds per-block FFN capacity
   (params, no state). The original d=128 model used 1.5-2.0; our d=32 is "capacity-starved" (arch comment).
3. **LoRA dims 16 -> 24** (decay/a/gate; v0_mix 8->12). Per-block low-rank capacity, no state. Already raised
   16 once for d=32; push further within the cap.
4. **Prehead output gate** (LIT_REVIEW idea 1): in head_and_out, `x = x * sigmoid(x @ W_g)` before the heads.
   ~1,056 params; adds non-linearity at the head boundary (NeurIPS 2025 Gated Attention). Flag-guarded so
   it's arch-agnostic / easy to A/B.
5. **Grow cheap streams** (deck/preset/user +1 layer): state-cheap (few decks/presets, 1 global per user) so
   it does NOT touch the card/note state gate. Buys capacity to recover any imm lost elsewhere.

Method: take the TUNED champion config as the new baseline, apply ONE change, retrain (sc8k WS 6ep aug-off),
eval 101-200, accept iff BOTH modes improve >=0.0003 AND params<=225k AND card/note state unchanged. Log
accepted/rejected per iter. Do NOT edit architecture.py while the tuner is running (it would corrupt in-flight
trials) -- this queue runs only after the tuner converges. Re-run the HP tuner only after a VERY big arch
change or several accumulated small ones.

## Assessed, NOT adopted (don't re-review)
- **Attention Residuals / AttnRes** (Kimi team, arXiv 2603.15031): replaces the fixed-weight residual with
  softmax attention ACROSS DEPTH (each layer aggregates all preceding layer outputs at the current position;
  Block AttnRes groups layers to cut memory). Validated on Kimi Linear (48B). **Verdict (2026-07-02): POOR
  FIT.** Its whole purpose is taming residual dilution across MANY layers; our stacks are 1-4 layers (card=1
  = literal no-op; 3-4 = almost nothing to attend over), so the motivating problem doesn't exist here. It's
  compatible with our invariants (operates on transient per-token layer outputs, NOT the persisted WKV state
  -> card/note state unchanged; Rust/CPU-deployable as a small per-token depth-matmul), BUT costs Q/K params
  on a tiny d=32 model and adds expressivity in a DATA-limited regime where capacity adds already reject.
  Only salvageable piece = a cheap **learned residual-mixing weight over the <=4 layer outputs** (few params,
  no per-layer attention); rank LOW, and only worth a single test AT 5k scale (not while data-limited).

## Cross-head mixing candidate — Paired Head Attention (KellerJordan/modded-nanogpt PR #191, 2026-07-02)
PHA is a SOFTMAX-attention mod: interleave adjacent heads' K/V (`[k1_h1,k1_h2,k2_h1,...]`) so each query
attends to its own AND the neighboring head's representation of every position in one softmax (shared flash
attn + staggered RoPE). Param-free; merged in nanoGPT for a small val-loss + speed win (~0.0006 loss).
**Literal fit to RWKV = NONE** (no softmax, no K/V cache, no flash attn, no RoPE — all transformer-specific).
**Transferable spirit = cross-head state-readout mixing:** let each head's readout also read the neighbor's
WKV state, `o_h = r_h·(S_h + S_other)` (param-free at K=16) or `+α·S_other` (1 gate param). Compatible with
our invariants: persisted state UNCHANGED (only the readout mixes -> card/note state fixed), Rust/CPU
deployable (extra mat-vec in readout), ~free params. Caveats: at H=2 "neighbor" = the other head -> full
cross-read (re-couples the heads we split, though state stays split -> possible Pareto: 512-float state +
richer readout); it's an expressivity add in a DATA-limited regime (~0.0006 sits at the +-0.0003 gate).
**Rank LOW-MEDIUM** (above AttnRes, below output-gating/EMA). It's essentially talking-heads mixing (cf. the
MHLA reference below). Worth ONE cheap test at 5k scale, not now.

## Hyper-Connections / xHC — residual-stream expansion (arXiv 2607.14530, Andrew 2026-07-21)
xHC (SJTU/Xiaohongshu) scales Hyper-Connections to N=16 residual streams: HC replaces the single
residual stream with N parallel streams + three learned per-layer mappings (aggregate N→1, distribute
1→N, and an N×N inter-stream mixer); **mHC** (the predecessor, inherited here) stabilizes the mixer by
projecting it onto the **Birkhoff polytope via Sinkhorn–Knopp** — doubly-stochastic (rows+cols sum to 1)
= mass-preserving mixing, so ∏H_res across depth cannot amplify/attenuate signal (normalization-free
depth stability). xHC adds write-back enrichment (multi-scale causal depthwise convs + Gram–Schmidt)
and sparse k-of-N stream routing; +4.0 avg downstream on an 18B MoE, 1.19–1.50× compute efficiency.
**Fit to us = NONE (Andrew's prior confirmed, stronger than the AttnRes case):** (1) the whole mechanism
manages information flow across DEPTH — our stacks are 1–4 layers (and shrinking: A7 removed a user
layer and IMPROVED both modes); at depth ≤4 the ∏H_res instability the Sinkhorn constraint solves does
not arise; (2) their gains are capacity gains at 2.5B–28B — our capacity-adds are 0-for-everything at
both d=32 and d=128 (data/regularization-bound, the opposite regime); (3) N residual streams multiply
per-entity hidden state ~N× in the Rust engine = a deploy-contract violation on the axis the entire
quant endgame compressed (9 B/card). **Salvageable primitive (the part worth remembering): the
Sinkhorn-Knopp doubly-stochastic projection itself** — a cheap, normalization-free way to make ANY
small learned mixing matrix stable/mass-preserving. If an inter-STREAM mixing family ever opens
(learned connections between the 5 streams at matched layers — hierarchy order preserved), SK-constrain
the mixer. Rank: mechanism NOT APPLICABLE; SK-projection noted as a stability primitive for future
learned-mixing designs.

## modded-nanogpt speedrun FULL SWEEP (Andrew 2026-07-21: "maybe you should just sweep the whole repo")
All 84 world-record entries (05/24 llm.c baseline 45 min → 05/26 1.32 min) triaged against our
constraints. Buckets:

**ALREADY HAVE (the speedrun converged on things RWKV-7/our repo already do):** ReLU² (our channel
mixer is `square(relu(Wk·x))`), QK-norm analog (RWKV-7 L2-normalizes k and v per head), zero-init
projections (our W_o/heads/deltas), smeared tokens (token-shift IS adjacent-token smearing),
per-layer skip/residual scaling analogs. Validates the shared recipe space; nothing to copy.

**NOT APPLICABLE (transformer/LLM-token-specific):** value embeddings (all variants), untied/re-tied
embed↔head, U-net skips, FlexAttention/sliding windows/YaRN/window warmup, RoPE tweaks, FP8 head,
logit softcap (our outputs are already clamped probabilities), bigram hash embeddings, paired-head
attention (reviewed separately), MUDD/hyper-connections/partitioned HC (reviewed — xHC entry above),
cross-stream attention XSA, EOS-aligned batching, multi-token prediction (our labels = next review
only; next-next labels would need an LMDB rebuild = inputs-invariant violation), all the
distributed/Triton/comm-overlap engineering (their bottleneck, not ours — our GPU speed work banked
its own equivalents).

**★ TRANSFERABLE — the one big one: the Muon-family optimizer line** (records #3/4 = the largest
single wins in speedrun history; refined by NorMuon #41/42, Polar Express #38, accelerated variants
#48, phase-scheduled SV transforms PR#291, cautious weight decay #43/50, interleaved Adam/Muon #57).
Muon = orthogonalized-momentum updates for 2D weight matrices (Newton-Schulz/Polar-Express on the
momentum), Adam for everything else. For us: **training-only (deploy untouched), params unchanged,
Rust-irrelevant, and a genuinely FRESH track-1 family (optimizer family: 0 attempts)** — distinct
from the closed HP-tuning family (structural optimizer change, not an HP re-tune). Unknowns: Muon's
wins are documented at 124M–28B; our 171k model with d=32 matrices (32×32 to 512×128) is far below
any tested scale, and our regime is data-limited (though Muon is a conditioning/step-quality story,
not capacity). Cheap to test: hybrid Muon(2D matrices)+AdamW(rest), env-gated in train_rwkv.
**Rank: HIGH — the next fresh track-1 family (iter 29 candidate). Cautious weight decay (apply wd
only where update agrees with weight sign) = the cheap standalone sibling (AdamW mod, a few lines);
NorMuon/Polar-Express = refinements to fold in only if base Muon shows signal.**

**Minor/HP-adjacent (LOW):** batch-size schedule + min-lr floor (breaks our step-pairing infra —
costly), Adam-beta fine-scheduling, decay-bias init tuning (#45/64 analogs — init family currently
0/1), output/attention gating variants (#28/55 — our prehead gate was null; output-gating idea 1
above remains the untested variant).
- RWKV-7 "Goose" (arXiv 2503.14456) — current core; baseline for "what's already there".
- Gated Attention, NeurIPS 2025 — output gating (idea 1).
- Tied-LoRA (arXiv 2311.09578) — weight tying (idea 2).
- Enhancing Linear Attention with Residual Learning (arXiv 2509.25223) — idea 3.
- MoE RWKV-7 meta-learner (arXiv 2504.08247), MHLA (arXiv 2601.07832) — multi-head expressivity; reference.
- xHC: Expanded Hyper-Connections (arXiv 2607.14530) — residual-stream expansion; not applicable, SK-projection primitive noted.

**Bonsai-demo (github.com/PrismML-Eng/Bonsai-demo, Andrew 2026-07-22): 1-bit (Q1_0) + ternary
(Q2_0, ~1.7 b/weight) WEIGHT quantization for the Bonsai LLM family (1.7B-27B), group-based
(group 64/128), merged into llama.cpp. Triage: NOT APPLICABLE as a param-count lever --
low-bit quantization keeps the parameter COUNT unchanged (shrinks bits/weight, not weights);
our gates count params, and our deploy-memory problem is per-card/note STATE (already 9 B/card
via the q72u joint-codebook scheme, which is conceptually beyond group-wise scalar low-bit).
Weight-size itself is a non-problem (d=32 champion ~684 KB fp32) and weight PTQ was already
rejected (no speed win, lesson bank). One crossover thought kept: '1-bit-survivability' as a
redundancy PROBE (if a stream's weights survive binarization, its fp32 capacity is redundant ->
prune it) -- but saliency-guided pruning already measures this more directly and is 5/5.
Rank: NOT APPLICABLE.**

---

# REVIEW 2026-08-17 (Andrew: "time to do another literature review + GitHub repos that use RWKV")

Previous sweep was 2026-07-21 -- before the A18 trunk consolidated, before the Muon-as-regularizer
finding, and before the expressiveness-vs-capacity distinction. The filter is also sharper now.

**THE FILTER.** 558k params / d=80 / H=5 / K=16 / 13 layer-steps; CPU+Rust deploy; card+note state
FIXED; no new inputs; must clear **+0.0001 in BOTH modes**. Two consequences that kill most of the
field on sight: (a) anything whose reported gain is "small" at 1B+ scale is below our noise floor;
(b) **capacity adds are 0/3 here**, so a paper that buys accuracy with parameters is the wrong
shape -- what we want is a RICHER FUNCTIONAL FORM at fixed parameters.

## RWKV-8 -- NOT APPLICABLE (checked, do not re-review)
Still experimental. Its three published features are all LLM-scale problems we do not have:
**DeepEmbed** (edge-friendly sparse MoE to cut VRAM -- we have no MoE and no vocabulary embedding),
**DeepEmbedAttention** (streamlining the KV cache -- we have no KV cache), and **ROSA** (an online
suffix automaton over tokens -- we have no tokens). Nothing to port.

## flash-linear-attention (fla-org) -- mostly INFRASTRUCTURE, not architecture
2026 additions are MoBA/FlashMoBA, a TileLang backend, attention-sink support, and Context Parallel
for KDA/GDN -- distributed-training and long-context plumbing for large models. The architectural
items are **Kimi Delta Attention** (Oct 2025) and **DeltaFormer** (Sep 2025), both delta-rule
variants aimed at LLM scale. Worth a look ONLY if the delta-rule direction below pays.

## *** THE ONE REAL LEAD: the eigenvalue range of the state-transition matrix
**Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues** (arXiv 2411.12537, ICLR
2025; code `github.com/automl/unlocking_state_tracking`). Finite-precision linear RNNs whose
state-transition eigenvalues are all POSITIVE provably cannot solve parity; extending the range from
[0,1] to [-1,1] fixes that and also improves code/math perplexity, **at zero parameter and zero
inference cost**. For DeltaNet the change is one constant -- `beta = 2*sigmoid(.)` instead of
`sigmoid(.)` -- which works because DeltaNet has `||k|| = 1` and a diagonal of exactly 1, so
`1 - beta` lands in `(-1, 1)`. Follow-up **DeltaProduct** (2502.10297) stacks Householder products
for more expressivity at multiplied compute -- capacity-shaped, so LOW for us.

### MEASURED ON OUR CHAMPION, and the measurement changes the proposal
Our update (`rwkv_ops.single_timestep`) is `S <- S(diag(w) - kappa (a*kappa)^T) + v k^T`, so for
constant `w` the eigenvalues are `w` (K-1 of them, always positive) and **`w - a*||kappa||^2`**
along the delta direction. Read from the champion checkpoint's reachable envelopes:

| quantity | value |
|---|---|
| `w` (decay) | 0.970 - 0.985 |
| `a` at rest / at its reachable max | 0.49-0.52 / 0.59-0.64 |
| **`||kappa||^2`** (= `k_scale^2`, another sigmoid) | **0.23 - 0.27** |
| **eigenvalue along kappa, at rest** | **0.837 - 0.864** |
| same, at the model's maximum reachable `a` | 0.809 - 0.840 |
| same, if `a` were DOUBLED | 0.645 - 0.698 |

**Two conclusions, and the second is why the paper does not port cleanly.**
1. **We are firmly in the positive-eigenvalue regime** -- 0.84, not marginally positive. So the
   representational limitation the paper describes does apply to us.
2. **The naive `a = 2*sigmoid` port would be nearly INERT.** It moves the eigenvalue 0.85 -> 0.67,
   still comfortably positive; reaching negative needs `a*||kappa||^2 > w ~ 0.98`, i.e. roughly a
   **6x** increase in the product. The blocker is not mainly `a` -- it is **`||kappa||^2 ~ 0.24`**,
   because RWKV-7 rescales the L2-normalised key by `k_scale = sigmoid(.)`. DeltaNet's trick works
   precisely because it has `||k|| = 1`; we do not.

### *** THE FINDING THAT IS WORTH MORE THAN THE PORT
`a * ||kappa||^2` contributes only **~0.15** of eigenvalue movement against a decay of ~0.98. So
our trained model uses its state almost as a **pure exponential-decay accumulator with a small
rank-1 correction** -- RWKV-7's headline innovation, the expressive delta rule, is barely engaged.
That is a previously invisible fact about this model, and it reframes the lever: the question is not
"extend the range" but **"why is the delta-rule authority so weak, and does a version with more of
it do better?"**

**=> PROPOSAL: DELTA-RULE AUTHORITY -- PROPOSED, THEN KILLED BY THE NEXT MEASUREMENT (same day).**
The idea was `a = c * sigmoid(.)` with `c` = 2 then 4. Two checks retired it within the hour.

### !!! SELF-CORRECTION, and it is the SAME MISTAKE THIS PROJECT ALREADY PAID FOR TWICE
The stability numbers first written here ("`c=4` gives ~0.35, `c=8` gives ~-0.3, so even `c=8` stays
inside [-1,1]") were **WRONG, in the dangerous direction**. They used the **resting**
`||kappa||^2 = 0.24` as if it were a bound. It is not: `k_scale = sigmoid(Linear(x))` has an
**UNBOUNDED input**, so `||kappa||^2 -> 1` is reachable. Only `a` has a true envelope, because its
LoRA passes through a `tanh`. Redone with genuine maxima (`a_max = 0.9552` over all 1040 channels,
`w_min = 0.5452` the architectural floor, `||kappa||^2_max = 1`):

| `c` | worst-case eigenvalue `w_min - c*a_max` | verdict |
|---|---|---|
| **1.0 (today)** | **-0.410** | safe |
| 1.5 | -0.888 | safe |
| **2.0** | **-1.365** | **UNSAFE, |lambda| > 1** |
| 4.0 | -3.276 | far unsafe |

So the safe range is `c <= ~1.5`, not `c <= 8`. **This is the third instance of the same failure
shape in this project** -- iter 51 fitted a schedule on a median and missed a 1.76e7 blow-up; the
`a`-is-dead probe used "not pressed against the bound" where a representational argument was needed;
and now a resting value stood in for a maximum. **The rule is not "report the max" as a style
preference -- it is that a typical-case statistic cannot bound a worst case, ever.**

### AND THE SAME NUMBERS KILL THE PROPOSAL OUTRIGHT
`c = 1.0` is TODAY, and its worst-case eigenvalue is already **-0.410**. So negative eigenvalues are
**not structurally impossible** in RWKV-7 as we run it -- the paper's "provably cannot solve parity"
argument applies to architectures whose eigenvalues are CONFINED to [0,1], and ours are not confined,
merely far from negative in practice. The theoretical motivation therefore does not transfer.

Worse for the proposal: **both factors of `a * ||kappa||^2` are freely learnable in the direction of
MORE delta authority.** `a` can reach 0.955 and sits at ~0.50; `k_scale` is a sigmoid of an
unbounded linear and could approach 1 but sits at ~0.49. The model can already reach
`a * ||kappa||^2 ~ 0.95` and instead operates at **~0.13**. **The delta rule is weakly engaged
because the model prefers it that way, not because anything blocks it** -- so raising `c` only
extends a range that is already unused. **DEAD. Do not run it.**

### What SURVIVES, and it is a fact rather than a lever
The empirical finding stands and is worth carrying: at typical operating points the delta term moves
the state-transition eigenvalue by only **~0.15** against a decay of ~0.98, i.e. **our trained model
uses its WKV state almost as a pure exponential-decay accumulator with a small rank-1 correction.**
RWKV-7's headline innovation is barely used here. That is a real characterisation of what this model
learned, and it should inform how the next architecture proposal is judged -- but the same
measurement says the cause is preference, not constraint, so it is not itself actionable.
⚠ One genuinely open question it raises, for a future review rather than a run: is the delta rule
weakly engaged because the TASK does not need it (spaced repetition may really be exponential
forgetting plus small corrections), or because our 1.25-epoch budget never gets there? The 10x
endgame run would answer that for free -- re-run this probe on its checkpoint.

### CORRECTION to the 2026-08-17 expressiveness probe in PROPOSALS.md
That probe listed "`a = sigmoid(a_lora)` is nowhere near its bounds, therefore DEAD". **That
reasoning is wrong for a representational barrier**, and this paper is why: if the useful region is
unreachable at ANY parameter setting, there is no gradient pressure to climb toward the bound, so
"not pressed against it" proves nothing. The measurement above supersedes it -- `a` is not dead, it
is WEAK, and the reason is the `k_scale` factor rather than the sigmoid.

## Sources
* RWKV-7 "Goose": arXiv 2503.14456 - RWKV wiki architecture history: wiki.rwkv.com
* Negative eigenvalues: arXiv 2411.12537 (ICLR 2025) + github.com/automl/unlocking_state_tracking
* DeltaProduct: arXiv 2502.10297
* flash-linear-attention: github.com/fla-org/flash-linear-attention

