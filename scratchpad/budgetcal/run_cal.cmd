@echo off
REM ===========================================================================================
REM BUDGET CALIBRATION (2026-08-10) -- decide whether research iterations can run at 1/3 the
REM training budget, per Andrew: "it's easier to cut the epoch budget down by a factor of 3-4
REM than to improve the % of good ideas by a factor of 3-4".
REM
REM THE DESIGN: re-run three recipes whose FULL-budget answers we already know, at 1/3 budget.
REM   c41  the champion (interleave, _cnd order)      full: 0.297889 / 0.265479
REM   c42  order-only, sequential                     full: 0.298379 / 0.266090   (large loss)
REM   c43  interleave, ORIGINAL order                 full: 0.297964 / 0.265464   (verified TIE)
REM
REM WHAT EACH PAIR BUYS:
REM   c41 vs c43 is a KNOWN NULL, so whatever |delta| it shows at 1/3 budget IS the short-budget
REM     NOISE FLOOR -- the number needed to set an accept threshold. We own this control by
REM     accident of iter 43 tying, and it is the single most informative measurement here.
REM   c41 vs c42 is a KNOWN LARGE EFFECT (+0.00049/+0.00061). If its sign and rough magnitude
REM     survive, ranking transfers for schedule-type changes.
REM   Both surviving => adopt 1/3 budget as the default and spend the winnings on more runs AND
REM     on 2-seed pairs (the seed doctrine has never once been paid because it doubled cost;
REM     at 1/3 budget a pair costs less than one run does today).
REM   The null coming back "significant" => the budget is too short. Learned for 3 short runs
REM     instead of by promoting a phantom champion.
REM
REM ⚠ KNOWN BIAS TO WATCH, not measured by this test: short budgets favour changes that help
REM EARLY learning and penalise ones that only pay off late (regularization, added capacity).
REM Same mechanism as the vprune scope lesson (train-loss pruning was sign-biased against
REM regularization levers and nearly killed wd=0.1, which then WON at eval). These three arms
REM are all SCHEDULE changes, so they cannot detect that bias -- do not read a pass here as
REM licence to screen dropout/weight-decay/capacity candidates at short budget.
REM
REM COST ~5.1 h/arm (WS 1.1 + decay 1.1 + eval 2.9) = ~15.3 h total. Eval is untouched and
REM becomes 57% of each arm -- which is itself the finding that a budget cut ALONE only buys
REM 1.8x, not 3x.
REM ⚠ Do NOT git rebase/pull/checkout this path while it runs (iter 43's chain died that way).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\budgetcal
set LOG=%DIR%\budgetcal.log
set CND=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set ORIG=scratchpad/track2_a18/architecture_d80_lora4.py

echo ===== BUDGET CALIBRATION START %DATE% %TIME% ===== > "%LOG%"

call "%DIR%\run_arm.cmd" c41 %CND% 1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_C41FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 41
)
call "%DIR%\run_arm.cmd" c42 %CND% 0
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_C42FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 42
)
call "%DIR%\run_arm.cmd" c43 %ORIG% 1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_C43FAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 43
)

echo === all three arms done, pairing %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --champ-ahead result/RWKV-c41.jsonl --champ-imm result/RWKV-P-c41.jsonl --cand-ahead result/RWKV-c43.jsonl --cand-imm result/RWKV-P-c43.jsonl --intersect >> "%LOG%" 2>&1
echo --- (above: c41 vs c43 = the KNOWN NULL -^> its ^|delta^| is the short-budget noise floor) --- >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --champ-ahead result/RWKV-c41.jsonl --champ-imm result/RWKV-P-c41.jsonl --cand-ahead result/RWKV-c42.jsonl --cand-imm result/RWKV-P-c42.jsonl --intersect >> "%LOG%" 2>&1
echo --- (above: c41 vs c42 = the KNOWN LARGE EFFECT, full-budget +0.00049/+0.00061) --- >> "%LOG%"

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
