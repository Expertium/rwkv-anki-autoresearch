# What changed the model's LogLoss

This table shows measured changes in mean LogLoss. A positive number made LogLoss worse.
A negative number made LogLoss better. The source data is `optimization/LEDGER.md`.

The measurements come from different model versions and training runs. They do not prove
that each change alone caused the full difference. Rows that combine changes are marked.

| # | Change, in simple words | Ahead change | Imm change | What the row means |
|---:|---|---:|---:|---|
| 1 | Changed the training recipe, but kept the old d=128 model | +0.0037 | +0.0043 | The shorter recipe scored worse than the original pretrained model. |
| 2 | Made the model 4.95 times smaller | +0.0010 | +0.0005 | The smaller model lost some accuracy at the short training budget. |
| 3 | Added the curve constraint, a new curve head, and Muon | -0.0004 | -0.0008 | Three changes together improved both scores. This row cannot show each change's effect. |
| 4 | Added teacher-based training | -0.0006 | -0.0004 | The model learned from a larger d=128 model. |
| 5 | Enforced the deployment curve rule during scoring | +0.0019 | +0.0001 | This changed the ahead score definition. It is not a direct model-to-model comparison. |
| 6 | Tuned training settings and increased the batch limit | -0.0013 | -0.0010 | Hyperparameter tuning improved both scores and raised training speed. |
| 7 | Kept six accepted improvements | -0.0014 | -0.0010 | The group improved both scores. It combines six changes. |
| 8 | Removed distillation, fixed a data bug, and rebuilt the data | +0.0007 | +0.0004 | The run combined three changes. Their separate effects are unknown. |
| 9 | Changed the review interval to end-to-start | +0.0004 | +0.0005 | This interval matches what a live scheduler can compute. |
| 10 | Added 23 time-based input features | -0.0007 | -0.0029 | The features improved both scores against the control in row 8. |
| 11 | Fixed another data bug and rebuilt the scored review set | +0.0002 | +0.0003 | The scored review set changed. This is not only a model change. |
| 12 | Replaced estimated calendar cycles with real calendar cycles | +0.0000 | +0.0000 | The measured result was a tie. |
| 13 | Increased training from about 1 to 10 epochs | -0.0005 | -0.0015 | Longer training improved both scores, but less than expected. |
| 14 | Fixed a time leak in query rows and changed the input layout | +0.0002 | +0.0021 | The leak-free model scored worse, mainly on imm. The input-layout change was small. |
| 14b | Reduced the final decay from 2 epochs to 1 | -0.0003 | +0.0000 | The ahead score improved. The imm change was within the noise range. |
| 14c | Reduced the model from d=80 to d=64 | +0.0000 | +0.0004 | The smaller model had a small imm cost at the 10+1 training budget. |
| 14d | Added sibling-rating inputs and changed the optimizer | -0.0000 | -0.0001 | The combined change slightly improved imm. The ahead score was nearly unchanged. |
| 15 | Quantized card and note state on the d=80 model | +0.0022 | +0.0035 | Quantization caused the main accuracy loss on that model. |
| 15b | Added two more WKV quantization stages on d=80 | +0.0016 | +0.0026 | More bits reduced the quantization loss, but did not remove it. |
| 16 | Quantized the d=64 model using one WKV stage | +0.0028 | +0.0040 | The smaller model paid a larger quantization cost. |
| 16b | Used four residual WKV stages on d=64 | +0.0018 | +0.0028 | Four stages reduced the quantization cost. This is the current deploy recipe. |

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

The next useful work is to finish the queued d=128 training check, then test other QAT training
settings without increasing card or note state. See `GPU_RUN_BOARD.md` for live and queued runs.
