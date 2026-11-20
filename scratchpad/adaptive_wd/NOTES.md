# Adaptive weight-decay experiment notes

Purpose: preserve this experiment's fixed lineage, acceptance rules, and resume hazards. This is durable context, not live run status. Read `optimization/GPU_RUN_BOARD.md` for current attempt, process state, resource readings, and queue.

## Shared setup for the three-run comparison

The plan is sequential:

1. Adaptive WD: up to 10 accepted training epochs and 7 rejected training attempts (17 maximum total), then one separate final decay/QAT epoch.
2. Low fixed WD: exactly 10 training epochs at WD=0.01, then one separate final decay/QAT epoch at WD=0.01.
3. High fixed WD: exactly 17 training epochs at WD=0.64, then one separate final decay/QAT epoch at WD=0.01.

For all three arms, use the same QAT46 step-248 model and optimizer start; default 92-input layout with `RWKV_ID_FEATURES=0`, `RWKV_REAL_CYCLES=0`, `RWKV_FINAL_LAYOUT=0`, and `RWKV_E2S_PUBLISHED=0`; rebuilt databases `F:/rwkv_lmdb/train_db_5k_h1_adaptive92` and `F:/rwkv_lmdb/test_db_5k_adaptive92`, from `../anki-revlogs-10k-id` with `F:/rwkv_lmdb/label_filter_db_id_e2s`; same optimizer, training data/order, LR/schedule, internal validation, batch size/cap, one fetch worker, RAM guard, 11-GiB VRAM limit, fixed 750-user screen, final QAT environment, and canonical full-VAL evaluation. Stored card-feature width is 24; model input width is 92. Never use the old adaptive DBs (stored width 69) or the published-data interval transform.

Adaptive screening uses a fixed deterministic roster of 750 eligible one-batch VAL users from 5001 through 7500. Exclude the known multi-batch users 5626, 6630, 6701, and 6810. Use production `optimization/eval_sharded.py` / `rwkv.get_result`, not the experimental fast evaluator. An adaptive attempt is accepted only when both ahead and immediate mean LogLoss strictly improve; on rejection restore both incumbent model and optimizer and advance WD through 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64. Final decay/QAT is a separate stage. The fixed controls do not screen to decide whether to accept or to switch WD.

## Requested fixed-WD controls

- Low control: hold WD=0.01 for 10 training epochs and for the separate final decay/QAT epoch.
- High control: hold WD=0.64 for 17 training epochs (the maximum adaptive attempt budget), then use WD=0.01 for the separate final decay/QAT epoch, matching the shared final-QAT treatment.
- Run these in order only after the adaptive arm, including final QAT and canonical full VAL, completes. Run the low control before high control.
- Each fixed control needs its own isolated 0.04-epoch real-path smoke and normal preflight. Preserve all checkpoints, optimizer states, logs, evaluation files, and source artifacts. Compare on the same evaluation users and report skipped users.
- The high static control has 17 training epochs plus one separate final decay/QAT epoch (18 epochs total); do not describe it as 17 epochs total. The adaptive arm may stop before 17 attempts, while the fixed high arm runs all 17 training epochs. Report actual work and elapsed time; the maximum budget is not a guarantee that realized compute or wall time is equal.
- Before runner creation, inspect the saved adaptive configs and confirm seeds/order and exact LR schedule. Aim to keep WD policy and the declared training-epoch count as the only planned differences. Do not replace this comparison with `wtwarmup2000` or another model/data lineage.

## Resume and artifact safety

- The initial controller used the old 30-attempt cap. After attempt 4 passed, temporary stop-marker files blocked the next output directory and caused `FileExistsError`. The saved state was intact. The markers were removed only after checking their contents; the controller resumed from `adaptive_state.json` with the 7/10 caps.
- Resume from saved state; do not restart from the original checkpoint. Preserve all model/optimizer pairs, evaluation artifacts, and logs. Never remove a path unless it is verified to be a temporary marker, not an artifact or run directory.
- A controller can spend time opening large LMDBs and counting groups before it creates an attempt directory or writes more log output. Check process state, step logs, checkpoints, GPU, and RAM. A live process alone does not prove progress.
- The RAM-guard process can outlive its script in this checkout. Check fresh timestamps in `scratchpad/ram_guard.log`; process presence alone does not prove fresh guard activity. Track free RAM directly and keep at least 10 GiB free.
- Project setup references helpers such as `scratchpad/preflight_runner.py`, `scratchpad/mk_dryrun.py`, and detach scripts. Verify each exists before relying on them.

## Evidence boundaries

The 0.04-epoch smoke checks plumbing only. It is not accuracy evidence. Do not add smoke metrics or partial adaptive screen results as completed experiment results. Record adaptive results only after its full protocol, final decay/QAT, and canonical VAL evaluation complete and their artifacts are checked.

- One training attempt is exactly ONE epoch = 43,354 optimizer steps: the schedule is absolute (`total_steps = EPOCHS * len(groups)`) and `EPOCHS` is the CUMULATIVE epoch target, so `EPOCHS = n` with `STEP_OFFSET = (n-1)*43,354 + 1` trains just epoch n-1. CORRECTED 2026-11-15: the accepted attempts did NOT replay one slice. Attempt 1 ran offset 1 (epoch 0), attempt 2 offset 43,355 (epoch 1), attempt 3 offset 86,709 (epoch 2), attempt 4 offset 130,063 (epoch 3) -- four DIFFERENT group orders, each starting from the previous attempt's saved model. Attempts 5-11 all sat at epoch target 5, so they repeated offset 173,417 seven times. Any replica of an arm must mirror this chain, and the record must not claim a shared slice.
- Accepted epochs accumulate exposure (incumbent after attempt 4 = 4 epochs past the start checkpoint); rejected attempts spend a full epoch of compute and are rolled back, so they add no exposure to the incumbent.
