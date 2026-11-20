Row 11 amendment (2026-11-19 22:05, Moscow). The first full QAT decay epoch (`qat_l1i8e1`, launched
21:35 from `adaptive_173416.pth`) was **STOPPED** at 21:45, because Andrew's anchor rule requires the
QAT redo to start from the checkpoint BEFORE the run's one decay epoch, and `adaptive_173416` is the
checkpoint AFTER it. The partial folder `scratchpad/quant_plan/qat_l1i8e1/` is kept but must not be
resumed or scored. The validated 0.04-epoch smoke `qat_l1i8smoke` still stands as the code-path check
(step 1,734, `MODEL SAVED.`, 0 skipped batches, quant banners active). Anchor resolution: the adaptive
attempts turned out to be one-epoch segments of a single flat-LR 4-epoch schedule
(`attempt_04/train.log`: `LR schedule continues at step 130062 of 173416 (factor 1.000000)`,
`LR decay shape = constant`), so the only clean pre/post-decay pairs in this checkout are the old-regime
runs `iter36` (WS-final step 124 -> step 558) and `iter44`/`iter45` (8/16-epoch decay-QAT from the same
WS-final). `scratchpad/quant_plan/score_anchors.py` is scoring `iter45_496`, `iter46_248`, `iter44_248`
and `iter36_558` unquantized on the 750-user roster against the current 0.300609/0.272009 baseline; the
anchor and then every result row will be re-based on the winner. `qat_decay.py` now takes
`--start-folder/--start-name/--start-optim/--peak-lr/--dry-run` so the epoch can be redone from any
checkpoint. Details: `scratchpad/quant_plan/NOTES.md` ("Anchor investigation") and `RESULTS.md`.