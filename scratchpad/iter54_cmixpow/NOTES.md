# iter 54 — read this before scoring the verdict

Annotated in a separate file, never in the runner (the Ops rule that cost iters 43 and 46).

## ★ THE LEVER IS 4 PARAMETERS, NOT 13 — read off the step-8000 checkpoint

`RWKV_CMIX_POW=1` adds one learnable exponent per block, and the param count says 13
(558,225 = 558,212 + 13). But at step 8000 **nine of them sit at exactly 2.0000**, i.e. they have
never received a gradient:

| block | exponent @ step 8000 | |
|---|---|---|
| card L0 | **1.3911** | live |
| note L0 | **1.6313** | live |
| deck L0 | **1.5087** | live |
| deck L3 | **1.9267** | live |
| card L1, deck L1, deck L2, preset L0/L1/L2, user L0/L1/L2 | 2.0000 | **dead** |

The nine dead ones are **exactly** the blocks whose channel mixers `RWKV_STRIP_CMIX` removes
(`user_id:0,1,2`, `preset_id:0,1,2`, `deck_id:1,2`, `card_id:1`). A `cmix_pow` Parameter is created
per block regardless, but with no channel mixer to act on it is inert — which the launch log had
already said in passing: `[grad-stats] 76 params never received a grad yet ... cmix_pow`.

**Consequences for the verdict:**
1. Describe this as a **4-parameter** change, not 13. The nine dead Parameters are also 9 keys of
   state-dict bloat that a deploy port would carry for nothing.
2. **The lever IS engaged, and directionally informative:** all four live exponents moved *down*
   from 2.0, three of them substantially (1.39 / 1.51 / 1.63). RWKV-7's squared-ReLU exponent of 2
   is not what this model wants at the layers where it can choose — it prefers ≈1.4–1.6.
3. So a null result would mean "curvature at those 4 sites doesn't matter", **not** "the flag did
   nothing" — the same distinction the iter 53 diagnostic drew.
4. ⚠ If this is rejected but the exponents keep moving down, the honest follow-up is a **single
   shared exponent across the live blocks** (1 param, no dead keys), or applying the lever to the
   blocks that actually have channel mixers. Do not re-run the 13-param form.

## ⚠ This run was RESUMED from step 8000 after a power outage
The 2026-08-18 outage killed WS at step 8110. Resumed via `make_resume.py` +
`RWKV_RESUME_SKIP_GROUPS=1` from the step-8000 pair; `[resume-skip]` confirmed in
`ws_resume8000.log`. Weights and optimizer state are exact, **but the resumed tail's dropout draws
differ** from an uninterrupted run — statistically equivalent, not bit-reproducible. The WS step
trace lives in TWO logs (`ws_1714828104.log` steps 1–8110, `ws_resume8000.log` steps 8001–10935);
**concatenate them and drop the 110-step overlap** when extracting the trace for a paired
comparison.

## Gate basis
Built on the **iter-45** recipe, so its controlled comparison is vs iter 45
(0.297697 / 0.265375). But iter 53 was accepted 2026-08-18, so **the gate is vs iter 53**
(0.297523 / 0.265191): to be accepted, iter 54's iter-45 deltas must exceed +0.000174 / +0.000184
before clearing the 0.0001 bar. Report both numbers.
Deploy debt: **yes** — a forward-pass change, so it needs the Rust port plus a fresh parity trace.
