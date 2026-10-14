# What changed the model's LogLoss

This table shows measured changes in mean LogLoss. A positive number made LogLoss worse.
A negative number made LogLoss better. The source data is `optimization/LEDGER.md`.

The measurements come from different model versions and training runs. They do not prove
that each change alone caused the full difference. Rows that combine changes are marked.

| Change | Ahead change | Imm change | What the row means |
|---|---:|---:|---|
| Changed the training recipe, but kept the old d=128 model | +0.0037 | +0.0043 | The shorter recipe scored worse than the original pretrained model. |
| Made the model 4.95 times smaller | +0.0010 | +0.0005 | The smaller model lost some accuracy at the short training budget. |
| Added the curve constraint, a new curve head, and Muon | -0.0004 | -0.0008 | Three changes together improved both scores. This row cannot show each change's effect. |
| Added teacher-based training | -0.0006 | -0.0004 | The model learned from a larger d=128 model. |
| Enforced the deployment curve rule during scoring | +0.0019 | +0.0001 | This changed the ahead score definition. It is not a direct model-to-model comparison. |
| Tuned training settings and increased the batch limit | -0.0013 | -0.0010 | Hyperparameter tuning improved both scores and raised training speed. |
| Kept six accepted improvements | -0.0014 | -0.0010 | The group improved both scores. It combines six changes. |
| Removed distillation, fixed a data bug, and rebuilt the data | +0.0007 | +0.0004 | The run combined three changes. Their separate effects are unknown. |
| Changed the review interval to end-to-start | +0.0004 | +0.0005 | This interval matches what a live scheduler can compute. |
| Added 23 time-based input features | -0.0007 | -0.0029 | The features improved both scores against the previous control. |
| Fixed another data bug and rebuilt the scored review set | +0.0002 | +0.0003 | The scored review set changed. This is not only a model change. |
| Replaced estimated calendar cycles with real calendar cycles | +0.0000 | +0.0000 | The measured result was a tie. |
| Increased training from about 1 to 10 epochs | -0.0005 | -0.0015 | Longer training improved both scores, but less than expected. |
| Fixed a time leak in query rows and changed the input layout | +0.0002 | +0.0021 | The leak-free model scored worse, mainly on imm. The input-layout change was small. |
| Reduced the final decay from 2 epochs to 1 | -0.0003 | +0.0000 | The ahead score improved. The imm change was within the noise range. |
| Reduced the model from d=80 to d=64 | +0.0000 | +0.0004 | The smaller model had a small imm cost at the 10+1 training budget. |
| Added sibling-rating inputs and changed the optimizer | -0.0000 | -0.0001 | The combined change slightly improved imm. The ahead score was nearly unchanged. |
| Quantized card and note state on the d=80 model | +0.0022 | +0.0035 | Quantization caused the main accuracy loss on that model. |
| Added two residual WKV quantization stages on d=80 | +0.0016 | +0.0026 | More bits reduced the quantization loss, but did not remove it. |
| Quantized the d=64 model using one WKV stage | +0.0028 | +0.0040 | The smaller model paid a larger quantization cost. |
| Used four residual WKV stages on d=64 | +0.0018 | +0.0028 | Four stages reduced the quantization cost. This is the current deploy recipe. |
| **Deploy estimate if additive** | **+0.0048** | **+0.0039** | Estimated deployed `s64hqr4b` delta versus the old d=128 reference: `s64hc10` score plus the measured 4-stage QAT tax. This matches the measured deployed delta; it is a consistency check, not an independent prediction. |


## Training budget versus parameter count

The planned 1+2 cells (`lfb1`, `d128b1`) never ran, so the original 2x2 could not separate size from training budget. I paired all five completed cells on the same 2,279 users with matching review counts. This common cohort gives a consistent view of the grid, though the d=80 and d=64 budget comparisons still use different decay lengths. The d=80 short run also used a different machine. Treat these as comparisons of the measured recipes, not pure WS-epoch effects.

Andrew's concern is reasonable. The measured 1+1 to 10+1 and 1+1 to 10+2 changes improve by about 0.0007–0.0015, but they do not measure a tenfold increase from today's 10+1 model. They show the gain from moving up to 10 WS epochs. We cannot assume another tenfold increase will buy the same gain. Do not schedule a 100+10 run now. Keep tuning the current 10+1 recipe first. After tuning, ask Sol whether a longer run is worth its cost.

| Parameters | Architecture | Training budget | Ahead LogLoss | Imm LogLoss | Result |
|---:|---|---:|---:|---:|---|
| 332,922 | d=64 | 1 WS + 1 decay epoch | 0.297939 | 0.264325 | `s64h` |
| 332,922 | d=64 | 10 WS + 1 decay epoch | 0.296755 | 0.262865 | `s64hc10` |
| 561,732 | d=80 | 1 WS + 1 decay epoch | 0.297649 | 0.263827 | `s80`, rented 4090 |
| 561,732 | d=80 | 10 WS + 2 decay epochs | 0.296993 | 0.262388 | `w10lf` |
| 2,575,766 | d=128 | 10 WS + 2 decay epochs | 0.296280 | 0.261156 | `d128e10`; 220 users outside this common cohort were OOM-skipped |

All means use these same 2,279 users. Every pair also has identical per-user equalized review counts. Positive gains below mean that the later or larger model scored lower.

| Matched comparison | Ahead gain | Imm gain | Interpretation |
|---|---:|---:|---|
| d=64: 1+1 to 10+1 | +0.001184 | +0.001461 | At 333k parameters, the longer recipe helped. 74.6% improved ahead; 91.6% improved imm. |
| d=80: 1+1 to 10+2 | +0.000656 | +0.001439 | Same-size recipe comparison. 69.6% improved ahead; 91.6% improved imm. The WS and decay both changed; the short run used another GPU. |
| d=128 versus d=80, both 10+2 | +0.000713 | +0.001231 | The 4.59x larger model did better, but the d=128 run used half the maximum sequence length and twice as many optimizer steps per epoch. This is not a pure parameter effect. |

### What this says about the current 333k model

At 333k parameters, the measured 1+1 to 10+1 comparison improved LogLoss by **0.00118 ahead** and **0.00146 imm**. This is a strong user-paired gain. It measures a combined WS-and-decay recipe change, not the effect of WS epochs alone.

The 562k comparison shows a similar pattern: 1+1 to 10+2 improved by **0.00066 ahead** and **0.00144 imm**. These results support longer training as a useful path for full-precision models in this size range. But they do **not** show that 10 more epochs will produce another 0.0012–0.0015 gain. We have measured one budget jump at each size. The WS and decay budgets changed together. So **more training is viable for improving the 333k model, but the remaining benefit is unknown.**

The 333k model has only been measured through 10+1. We have no evidence yet for 20+1, or for quantized training at the same 10+1 budget. The current quantized deploy score is a 10+1 plain run followed by a 1-epoch QAT decay; it is not the same model as a 10+2 QAT run. Longer plain training may improve the 333k model, but it is not yet proven to close the deploy-model gap. The active Muon-ratio trials test settings at 10+1. Use those results before funding a longer WS. On the common cohort, `s64hc10` is about +0.00291 ahead and +0.00113 imm above the old d=128 reference. Review-set alignment with the published reference still needs care.

The d128e10 result does not say that parameters have little value: on 2,279 identical users, it improves over d=80 by 0.00071 ahead and 0.00123 imm. But it costs 4.59x the parameters, and its batch/step schedule differs. The 300-user 1+2 cells that would separate the size and epoch effects were not completed. So the data supports **a real but modest size benefit at 10+2**, not a clean estimate of benefit per parameter.

## Current position

The current quantized model is `s64hqr4b`. Its VAL-half scores are **0.299369 ahead** and
**0.267496 imm**. The old d=128 reference scores **0.294612 ahead** and **0.263561 imm**.
The remaining gaps are about **0.0048 ahead** and **0.0039 imm**.

The current card state uses 403 bits. The current note state uses 214 bits.
This is about a 1,000-fold reduction from the old model's unquantized card state.

The largest confirmed cost is still QAT, which means training with the deployment quantizer.
More WKV stages helped, but five stages did not reach the reference scores. The 1-bit norm
range and the 2-bit norm both failed to improve QAT. The QAT catalog-LR and decay-LR tests also
found the current LR is better than either tested change.

The d=128, 10+2 scaling diagnostic (`d128e10`) completed with ahead LogLoss **0.296280** and imm LogLoss **0.261156** across 2,279 scored users. It used the `-id5` review filter. Its means are not directly comparable with the published `base5k` results: 1,955 of the 2,279 overlapping users have different scored-review counts. The baseline was not rebuilt on `label_filter_db_id_e2s`.

The `wtmuon0p5` trial is running. The `wtmuon2` and `wtwarmup2000` trials follow it. Preserve `RWKV_EMPTY_CACHE_EVERY=1` and the 11 GiB VRAM cap. See `GPU_RUN_BOARD.md` for the current queue.
