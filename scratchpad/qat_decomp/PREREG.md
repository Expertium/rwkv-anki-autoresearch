# Pre-registration: where does arm 2's QAT tax live? (written 2026-09-10 19:00, BEFORE any number)

## Why this runs
Andrew 2026-09-10: *"We've already worked on reducing the QAT tax, but maybe we should try again"*, and
on state size: *"I'd rather not increase card/note state size, but if that's the only way, alright."*
Arm 2 (`w10qat`) gives the WHOLE tax at the 10+2 budget. It does not say which part of the quantizer
costs it, and every QAT-retry route attacks a different part:

| route | attacks | bytes |
|---|---|---|
| WKV/shift bit split (m2b12 -> m2b8 shift, b10 -> b12 WKV) | the WKV codebook, paid for by the shift side | 0 |
| any other WKV catalog improvement | the WKV codebook | 0 |
| rank-2 WKV state (Andrew's conditional grant) | rank-1 truncation | +11 bits/head |
| distillation from the fp twin (`w10qatkd`, queued) | how the weights adapt | 0 |

So ~2.5 h of GPU decides where ~30 h A/Bs should go.

## Design -- the same QAT checkpoint, three ways, users 5001-5300, rectified VAL metric
* **deploy**: every quantizer on, LEARNED catalogs = arm 2's own full eval, restricted to these users.
* **noshift**: deploy MINUS shift quantization (`RWKV_QAT_SHIFT_SCOPE`, `RWKV_QAT_SHIFT_PQ` cleared).
* **nowcb**: deploy MINUS the WKV codebook and WKV norm quant (`RWKV_QAT_PQ` cleared; rank-1 kept with
  factors at int8, which is near exact; shift norms stay at 1 bit, so `RWKV_QAT_NORM_BITS` is unchanged).

Parts: **T = deploy - arm 1** (whole tax), **S = deploy - noshift**, **C = deploy - nowcb**,
**R = T - S - C** (rank-1 truncation + the weights' drift under QAT + interactions).

**No whole-QAT-off arm, deliberately.** Rank-1 truncation is STRUCTURAL: the 2026-08-13 cell-3 eval of a
QAT checkpoint at full precision scored +0.105 ahead / +0.28 imm (`research_5k_notes.md`, "CELL 3
INVALIDATES ITS OWN LABEL"). The two arms here each remove one VALUE quantizer and keep rank-1, so R
cannot be split further, and this design says so instead of pretending.

**Single-variable, verified by execution (2026-09-10):** `assert_arm.py` inspects the FINAL arch config
and the env -- each arm's env PASSES as itself and FAILS as each of the other two (9/9, six refusals);
`check_banners.py` requires each arm's eval log to show its quantizers and forbids the removed ones
(tested on synthetic logs and on a real deploy log, which fails both arms as it must). The WKV codebook
is CUDA-only, so the CPU cannot prove it; the GPU banner check does.

**Validity before cost:** each arm first runs 10 users; if it is more than 0.005 WORSE than deploy in
either mode the model depends on that quantizer (the cell-3 signature) and the 300-user eval is skipped.

**Where it runs:** phase D of `scratchpad/w10qat/run_w10qat_eval.cmd`, after `EVAL_OK` and before its
terminal marker, so the chain waits for it (cmixgraft and everything after shift ~2.5 h). Never fatal:
it logs to `scratchpad/w10qat/decomp.log` and always returns. Report: `decomp_report.txt` beside it.

## Predictions
* **P1 -- separable.** Both arms pass the probe gate, and neither S nor C is significantly NEGATIVE.
  Risk: the LEARNED catalog co-adapts with the weights (the 2-seed soup lesson), so removing it could
  HURT; the report flags that and prints no share.
* **P2 -- the WKV codebook dominates: C >= 50% of T in BOTH modes.** iter 47 cut the exact rank-1 error
  43% / 75% and the loss did not move, so rank-1 truncation carries little; the 2026-08-12 PTQ matrix
  priced a catalog swap at +0.0032 / +0.0042; the learned catalog still reconstructs at ~0.4-0.56
  relative L2.
* **P3 -- shift is small: S <= 20% of T in both modes.** The PTQ matrix put the whole shift side at
  ~1/14 of the WKV side.
* **P4 -- the bit-split decision, mechanical** (`decomp_report.py`): predicted gain =
  `a*C - 0.11*S` with a = 0.066 (WKV b12's reconstruction gain on a thin held-out corpus, a lower
  bound) or 0.146 (the full-corpus ladder); 0.11 is shift m2b8's measured reconstruction loss. **QUEUE**
  the ~30 h A/B iff the conservative gain is >= 0.0001 in both modes; **MARGINAL** (report, do not
  queue) iff only the optimistic one is; else **DEAD**. The map is crude -- reconstruction does not
  rank working catalogs -- so it can only say whether the A/B has a chance, never that it wins.
  **Stated so the likely outcome is not a surprise:** with S ~ 10% of T, QUEUE needs C >= ~2/3 of T on
  ahead (T ~ 0.003) and ~1/2 on imm. A synthetic C = 60% gives MARGINAL. My expectation: MARGINAL.
* **Falsifier for the whole catalog direction:** C < 30% of T in both modes. Then no catalog change can
  recover much; the tax is structure or drift, the zero-byte catalog routes are exhausted, and the
  remaining routes are distillation (already queued) and Andrew's byte grant (rank-2).

## After QAT-KD
The same decomposition on `w10qatkd` would show WHICH part distillation moves. It is NOT wired into the
KD runner yet: `mk_w10qatkd.py` asserts write_eval_toml's folder appears exactly twice in the eval part,
and phase D adds a third. Decide after this report; wiring it means updating that assert, regenerating
both KD runners, and a 300-user phase the KD dry run should not execute.

## Cost note found after writing the rule (2026-09-10 19:20, still before any number)
**The bit split cannot run on the current kernel.** `rwkv7_cuda.cu` holds the WKV catalog in a static
`__device__ float g_pq_cb[32768]` (and `g_pq_cb_grad[32768]` for learning), sized for joint-uv
ncent <= 1024 x 32 dims. A b12 catalog is 4096 x 32 = 131,072 floats, and the upload REFUSES it
(`TORCH_CHECK(n <= 32768, "PQ codebook too large")`) -- loud, not silent. So a QUEUE verdict also
buys: grow both arrays and the check, rebuild the `.pyd` in isolation (the live one is locked by every
running QAT process), prove bit-exactness on the existing paths with
`scratchpad/qat_speed/golden_gen.py check`, and swap it in while the GPU is idle. The joint search is
also 4x longer per step (warm-start pruning should cut that; unmeasured). The Rust engine has no fixed
limit. None of this changes the rule; it changes what a QUEUE costs.
