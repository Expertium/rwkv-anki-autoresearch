# Stateful BPTT — design + build plan (2026-06-29)

Goal (Andrew's revised plan, step 1): make the **1k research phase** fast by raising the WKV kernel's
batch dimension B. Today each user's history is split into fixed `MAX_BATCH_SIZE//2`-review chunks
(`data_processing.job`), each trained **cold** (state starts at 0). The **chained** 5-stream design
(card→deck→note→preset→global, `x` flows through all 5 within one chunk) means every stream shares the
same chunking, and the **global/user stream is 1 sequence per user over the full history** → at
`MAX_TRAIN_GLOBAL_LEN=66000` most groups are B=1 (one user's chunk fills the budget) → ~15% GPU util.

Stateful BPTT (truncated): chunk smaller → many chunks pack per group → **B≫1**; carry each entity's
state across that user's consecutive chunks so the model still sees long context AND training matches
eval (eval processes the full history with implicit carry). "2–3 birds."

## What state must be carried (blueprint = `srs_model_rnn.py` RNN-mode `run()`)

The RNN inference keeps **per-entity state dicts**: `card_states[card_id]`, `note_states[note_id]`,
`deck_states[deck_id]`, `preset_states[preset_id]`, one `global_state`. Updated only on non-skip
reviews. The 5 streams carry **independently** (the card→…→global chaining is *within* a step via the
between-stream reorder, not across chunks).

Per **layer** per **entity**, the carried state is **3 tensors** (`RWKV7RNNLayer`):
- time-mixer **WKV state** `[H,K,K]` (fp32) — handled by the new CUDA stateful kernel,
- time-mixer **token-shift** `x_shift` `[C]` = previous step's `layer_norm(in)`,
- channel-mixer **token-shift** `x_shift` `[C]` = previous step's `layer_norm(in)`.

## ✅ DONE — CUDA kernel foundation (verified this session)

`RWKV7_WKV_Stateful` (rwkv_ops.py) + `rwkv7_wkv_{forward,backward}_stateful_{float,bf16,half}`
(rwkv7_cuda.cu / rwkv7.cpp). Forward takes `state0_BHKK`, returns `(out, final_state_BHKK)`; ALWAYS
uses the sequential kernel (the time-parallel scan can't take an initial state — fine, stateful chunks
are small). Backward forces the sequential kernel: the saved `checkpoint[0]=state0`, so it's correct
for a nonzero start (incl. the nonzero w/a/k_deformed grads from decay acting on state0); truncated
BPTT drops the dS into state0 (state0 treated as constant). Non-stateful path is byte-identical
(nullptr → original). **Op-level parity (scratchpad/test_stateful_wkv.py):**
- (A) `stateful(state0=0)` == non-stateful op: **exactly 0** (fp32 & bf16).
- (B) forward split-equivalence `fwd([A;B]) == [fwd(A); fwd(B, state0=final_A)]`: **exactly 0**.
- (C) truncated-BPTT grads vs pure-PyTorch detached-carry reference: **3.8e-6** fp32 / bf16-noise.

## KEY de-risking finding: NO data rebuild / NO schema change

Chunks are already stored in **per-user time order** (`{user_id}_batches`, appended in range order) and
each review carries its entity IDs (`data.ids[submodule]`). So stateful carry is purely a
**training-loop + model-forward** change — `train_db` does NOT need new keys. `train_db` for users
1–1000 can be built normally (it's needed for the research phase regardless); only the **chunk size**
(smaller = higher B) is baked into the db, and with carry a smaller chunk costs no context.

## Remaining build (truncated BPTT; reversible, incremental, each step parity-tested)

1. **Model-forward state carry** (reversible, no data dep — DO NEXT):
   - `RWKV7TimeMixer.forward` (training): accept optional initial WKV state + token-shift token; return
     final WKV state + final token-shift. Token-shift carry = override `x_shift[:,0]` per entity row
     with the carried previous-token (the `[B,sub_len]` rows ARE per-entity). Use `RWKV7_WKV_Stateful`.
   - `RWKV7ChannelMixer.forward`: same token-shift carry.
   - Thread an optional `state` dict (per layer: WKV + 2 shifts) through `RWKV7.forward` and
     `SrsRWKV.forward_batch`, returning the per-stream final states.
   - **Test**: model-level split-equivalence on a SYNTHETIC same-entity batch (split a `[B,T,C]` input
     at T/2, carry, compare to the full-T forward) → forward exact, truncated grads ≈ ref.
2. **Per-entity state store + gather/scatter** by entity_id at chunk boundaries (the crux): map
   entity_id → batch row (from `from_perm`/`ids`), assemble `state0` per stream's split order, persist
   finals into 5 dicts keyed by entity_id. Entities appear/disappear (fresh → zero state).
3. **Synchronized stateful batching** in `get_groups` + the training step: process users' chunks in
   time order, batching across users at the same chunk index (B = #users), carrying each user's 5
   dicts. Breaks the current free chunk-shuffle (the biggest training-loop change). Detached carry
   across the chunk boundary = truncated BPTT.
4. **Eval** already carries implicitly (full history, 1 batch/user) — unaffected. Add a tiny ~3-user
   parity check that chunked-stateful training-mode forward == full-sequence forward.

## ★ Scope fork for Andrew (the expensive part is steps 2–3, NOT the kernel)

Given the **chained** streams force one shared chunking, the only simple B-boost is **smaller chunks**.
Two routes:
- **(R)ecommended first — MEASURE the cheap version:** rebuild `train_db` 1–100 with a smaller chunk
  size, train the champion @66000 (cold, no carry), measure wall-clock speedup + eval logloss vs the
  32768-chunk re-baseline (imm 0.299–0.300). If smaller cold chunks cost little accuracy, the full
  carry (steps 2–3, multi-day, intricate) may be **unnecessary** — the re-baseline already showed
  32768 cold beats the champion. ~30–40 min compute, reversible, answers "is full carry worth it?".
- **(F)ull stateful carry:** build steps 2–3. Delivers speed + long-context + train/eval match, but is
  the intricate per-entity-mapping + synchronized-batching refactor.

**Recommendation:** do **R** first (cheap, evidence-generating, Andrew explicitly asked "see if there
are other ways to speed up"), then build **F** only if R's accuracy cost is unacceptable. The kernel
foundation is kept either way (needed for F and for any per-persist QAT work).

Chunk size for the train_db 1–1000 build (needed regardless) depends on this — hold the big build until
the route is chosen to avoid a costly rebuild.
