@echo off
REM ===========================================================================================
REM PHASE 3: featB -- the new-features arm. Andrew 2026-09-01: "let's move on to phase 3 then."
REM
REM Waits for the fixc arm, which currently owns the GPU. Anchored findstr -- the unanchored form
REM matches this file's own prose and fires instantly.
REM
REM ---- WHAT featB's NUMBER WILL AND WILL NOT MEAN ----
REM featB trains on the `-id` gen-3 dbs, so against its control (featA2) it bundles THREE changes:
REM   1. the 23 real-timestamp feature columns (width 92 -> 114, params 558,212 -> 565,252)
REM   2. END-TO-START intervals -- these are automatic on any `-id` build, gated on the DATASET
REM      rather than on a flag, so they cannot be switched off there
REM   3. the dataset itself (`-id` vs published), which shifts `size` for ~30% of users
REM Bug C is held CONSTANT: id3 was built 2026-08-24 and _fix on 08-21, both before the
REM nan_id_fill fix of 08-26. So that is one thing NOT in the bundle.
REM
REM ★ THE BUNDLE IS NOW PARTLY SEPARABLE, WHICH IT WAS NOT IN AUGUST. The e2sc/fixc pair measures
REM component 2 directly on the published set, so featB - featA2 minus that gives features+dataset.
REM Component 3 is irreducible: the new features REQUIRE real timestamps, which only `-id` has.
REM
REM ⚠ featB died 2026-08-21 on the int32 id bug (a wrap collision made a card's genuine first
REM review probe-eligible with no query row -> KeyError in insert_probes -> the fetch worker died
REM and the run DEADLOCKED at 0% GPU for 69 minutes). That is fixed, and id3 is verified healthy:
REM 3,275 distinct card ids and 2,087 note ids over 25 sampled keys, versus the n_unique==1 per
REM user that the broken generations produced.
REM
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\wait_featB.log
set PREVLOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm\fixc_arm.log

echo ===== featB waiter armed %DATE% %TIME% ===== >> "%LOG%"

:waitprev
findstr /B /C:"DONE_EXIT_" "%PREVLOG%" >nul 2>&1
if errorlevel 1 (
  ping -n 121 127.0.0.1 >nul
  goto waitprev
)
echo fixc arm finished %DATE% %TIME% >> "%LOG%"

REM featB does not DEPEND on fixc succeeding -- they are independent experiments sharing a GPU --
REM so this waits for the marker but does not gate on DONE_EXIT_0.
call C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\run_featB.cmd
echo featB returned %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
