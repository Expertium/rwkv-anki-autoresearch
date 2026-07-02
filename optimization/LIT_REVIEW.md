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

## Sources
- RWKV-7 "Goose" (arXiv 2503.14456) — current core; baseline for "what's already there".
- Gated Attention, NeurIPS 2025 — output gating (idea 1).
- Tied-LoRA (arXiv 2311.09578) — weight tying (idea 2).
- Enhancing Linear Attention with Residual Learning (arXiv 2509.25223) — idea 3.
- MoE RWKV-7 meta-learner (arXiv 2504.08247), MHLA (arXiv 2601.07832) — multi-head expressivity; reference.
