@echo off
REM Export MORE gen-5 parity traces, purely to widen the PQ-codebook corpus.
REM 4,096 centroids per role/pos chunk fitted from ~10k vectors is thin: the 2026-08-12 shift
REM refit reached 0.1902 held-out where the 3-user gen-5 refit reaches 0.3933.
REM Writes to its OWN dir. reference_realcyc is the CERTIFIED parity trace and is not touched.
REM Trace files are model-independent INPUTS for --dump-corpus, so this set is reusable when the
REM catalogs are re-fitted on the ws10 final checkpoint.
REM BelowNormal: ws10 is training and lost about 10 percent of its step rate to the last CPU job.
REM Export a parity trace for the GEN-5 lineage reference (realcyc), at BelowNormal so it does not
REM slow the GPU run's dispatch thread. CPU only, no GPU.
REM
REM WHY NOW. The last certified Rust-parity trace is reference_iter41 -- a 92-dim model. gen 5 is
REM 109-dim (RWKV_ID_FEATURES + RWKV_REAL_CYCLES drop day_of_week and the 28 pseudo-cycle encodings
REM and add 24 real ones). On 2026-09-02 the engine's FEATURE_DIM stopped being a compile-time 92 and
REM is now derived from features2card.0.weight -- but the record states plainly that "the non-92 path
REM is UNTESTED and cannot be tested until a checkpoint exists". One does now. This produces it, and
REM the same artifacts feed the WKV codebook staleness check the endgame's QAT arm needs.
REM
REM The env is realcyc's, minus the training-only vars (see scratchpad/realcyc/run_realcyc.cmd).
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=4
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_gen5
set LOG=%DIR%\export_more.log
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_ZERO_FEATURES=
set RWKV_CHAMP_CKPT=scratchpad/realcyc/rc_d_10935.pth
set RWKV_REF_DIR=scratchpad/qat_gen5/traces
set RWKV_REF_USERS=112,128,141,163,178
REM the gen-5 lineage reads the REAL-TIMESTAMP dataset and is scored against the e2s-selected
REM label filter (rwkv/find_equalize_id_e2s.toml); the published defaults would raise KeyError.
set RWKV_EXPORT_DATA=../anki-revlogs-10k-id
set RWKV_EXPORT_LABEL_DB=F:/rwkv_lmdb/label_filter_db_id_e2s
echo ===== gen-5 CORPUS trace export START %DATE% %TIME% ===== > "%LOG%"
start "" /belownormal /b /wait cmd /c ".venv\Scripts\python.exe -u export_rnn_trace.py >> "%LOG%" 2>&1"
echo export rc %ERRORLEVEL% %TIME% >> "%LOG%"
if not %ERRORLEVEL%==0 (
  echo EXPORT FAILED >> "%LOG%"
  echo DONE_EXIT_31 %DATE% %TIME% >> "%LOG%"
  exit /b 31
)
if not exist "scratchpad\qat_gen5\traces\ref_metrics.json" (
  echo EXPORT produced no ref_metrics.json >> "%LOG%"
  echo DONE_EXIT_32 %DATE% %TIME% >> "%LOG%"
  exit /b 32
)
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
