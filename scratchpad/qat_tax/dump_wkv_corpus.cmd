@echo off
REM ===========================================================================================
REM Dump a d=80 WKV state corpus from the Rust engine (CPU-only, no GPU contention).
REM
REM WHY. The WKV codebook in the QAT recipe -- reference/pq_cb_wkv_q72u.txt, header
REM `1 10 32 16 1024` -- was fitted on d=32 / H=2 states. It is DIMENSIONALLY still valid here
REM (K=16 either way, so joint(u,v) is 32-dim), which is exactly why nobody noticed it is
REM DISTRIBUTIONALLY stale: this trunk is d=80 / H=5, five heads instead of two, and the rank-1
REM factor directions it has to encode are drawn from a different model. The shift catalog got
REM refitted for C=80 only because the old one hard-FAILED on a shape check; the WKV one fails
REM silently, by being merely worse.
REM
REM WKV is also the bigger half of the state budget: per card layer it is H*K*K = 1280 floats
REM against 80 for the token shift. If PRECISION DEGRADATION turns out to matter, this is the
REM first place to look, and the whole check is CPU-only.
REM
REM Stride 4 samples every 4th entity: ~5 (u,v) pairs per state (one per head), and 1024
REM centroids want ~50-100 pairs each, so a few thousand states per user is already plenty.
REM Full stride would emit ~15 KB per state line and buy nothing.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax
set LOG=%DIR%\dump_wkv.log
set BIN=rust\rwkv-infer\target\release\rwkv-infer.exe

REM ---- the arch the traces were exported under (must match, or the engine builds a different model)
set RWKV_WEIGHTS=reference_iter45/iter45_kddecay.safetensors
set RWKV_TRACE_DIR=reference_iter45
set RWKV_INTERLEAVE=1
set RWKV_STREAM_ORDER=card,note,deck,preset,user
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
REM NO quantization env here on purpose -- the corpus must be fp32 ground truth.

echo ===== DUMP WKV CORPUS START %DATE% %TIME% ===== > "%LOG%"
for %%U in (101 102 103 104 107 136 156) do (
  for %%S in (card note) do (
    %BIN% --dump-corpus %%U %%S 4 > "%DIR%\corpus\wkv_%%U_%%S.txt" 2>> "%LOG%"
    echo DUMP_%%U_%%S_EXIT_%ERRORLEVEL% >> "%LOG%"
  )
)
echo DUMP_DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
