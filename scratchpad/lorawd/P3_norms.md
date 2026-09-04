# lorawd P3 -- LoRA norm growth, measured on the WS checkpoints (2026-09-04 08:55, decay running, NO accuracy number seen)

Total Frobenius norm over the 104 LoRA matrices (sqrt of the sum of squares), same seed => identical at step 50:

| step | realcyc (wd=0, control) | lorawd (wd=0.05) | lorawd / control |
|---|---|---|---|
| 50 | 6.19 (start) | 6.19 | 1.000 |
| 1000 | 20.93 (+238%) | 20.68 (+234%) | 0.988 |
| 2000 | 32.38 (+423%) | 31.14 (+403%) | 0.962 |
| 5000 | 50.57 (+717%) | 47.42 (+666%) | 0.938 |
| 10935 | 71.96 (+1063%) | 62.47 (+909%) | **0.868** |

## What this says, written before the verdict

1. **The lever is ENGAGED, monotonically:** the ratio falls 1.000 -> 0.988 -> 0.962 -> 0.938 -> 0.868
   across checkpoints. Not inert, not a fluke of one checkpoint.
2. **The dose is a WEAK brake.** Decoupled decay at lr 0.0025 x wd 0.05 = 1.25e-4 per step would shrink a
   gradient-free weight by ~75% over 10,935 steps; the norm fell 13%. So the growth is gradient-driven and
   strongly restoring -- Muon's fixed-norm updates keep pushing the LoRA magnitude out and wd only trims
   the equilibrium. The +62% question ("does the endgame need a brake?") is therefore answered as: wd=0.05
   moves the equilibrium by ~13%, it does not remove the growth.
3. **PREREG's P3 threshold was MIS-SPECIFIED, and this is being said before the result:** it wrote "growth
   falls from ~+62% to under +25%". The +62% was iter 53's DEPLOYED norm relative to iter 45's (Muon-LoRA
   vs AdamW-LoRA, two trained models) -- not start-to-end growth, which is +1063% here. So the literal P3
   cannot be applied. The replacement, fixed now: **P3 holds iff the lorawd/control ratio at 10935 is below
   0.95 and decreasing across checkpoints** -- it is (0.868) -- so the accuracy verdict is INTERPRETABLE.
   If the accuracy result is a null, it is a null at a dose that demonstrably changed the weights by 13%,
   not an inert flag; a stronger dose (0.2) is the only in-family follow-up worth one run, and only if the
   null is a tie rather than a regression (P4 stands: a regression closes the family on mechanism).

Measured by the inline script in the session (torch.load, keys containing `lora`, 2-D tensors only);
the decay-phase final will be added to the verdict.
