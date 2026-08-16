"""Generate scratchpad/iter53_muonlora/run_iter53.cmd from iter 45's champion runner.

Iter 53 puts the LoRA projections on Muon (RWKV_MUON_INCLUDE_LORA=1). Unlike iter 52 this touches
the OPTIMIZER, so it cannot warm-start from the champion's WS-final -- it is a full WS + decay run.

Generated from run_iter45.cmd by a small set of textual substitutions so the env block, the phase
structure and every existing guard are inherited verbatim; the asserts below check what a diff
would not show. The added banner guard is the important one: the lever is a param-group change, and
a param-group change that silently does nothing produces a run that looks perfect and measures the
champion. `[muon] ... incl N LoRA in a wd=0 group` only prints when the group actually exists.
"""
import io
import re

BS = chr(92)
SRC = "scratchpad/iter45_kddecay/run_iter45.cmd"
DST = "scratchpad/iter53_muonlora/run_iter53.cmd"

s = io.open(SRC, encoding="utf-8", newline="").read().replace("\r\n", "\n")

# --- retarget: run dir, prefixes, tags ---
s = s.replace("scratchpad/iter45_kddecay", "scratchpad/iter53_muonlora")
s = s.replace("scratchpad" + BS + "iter45_kddecay", "scratchpad" + BS + "iter53_muonlora")
s = s.replace("i45_ws", "i53_ws").replace("i45_d", "i53_d")
s = s.replace("i45_decay.toml", "i53_decay.toml").replace("i45_eval.toml", "i53_eval.toml")
s = s.replace("RWKV-iter45_kddecay", "RWKV-iter53_muonlora")
s = s.replace("RWKV-P-iter45_kddecay", "RWKV-P-iter53_muonlora")
s = s.replace("iter45.log", "iter53.log")

# --- the lever: one env var, set beside the other Muon settings so it reads as one block ---
assert s.count("set RWKV_MUON_MOMENTUM=0.95\n") == 1
s = s.replace(
    "set RWKV_MUON_MOMENTUM=0.95\n",
    "set RWKV_MUON_MOMENTUM=0.95\n"
    "REM ---- THE LEVER: the LoRA projections join the Muon groups (own group, wd stays 0.0) ----\n"
    "set RWKV_MUON_INCLUDE_LORA=1\n",
)

# --- the guard that makes a silent no-op impossible ---
BANNER = [
    "",
    "REM The lever is a PARAM-GROUP change, which fails SILENTLY: a run whose grouping did not",
    "REM change looks perfect and measures the champion. This banner substring only prints when",
    "REM the LoRA group actually exists in the optimizer.",
    'findstr /C:"LoRA in a wd=0 group" "%DIR%' + BS + 'ws_%STAMP%.log" >nul',
    "if not %ERRORLEVEL%==0 (",
    '  echo ITER53 NO_LORA_GROUP %DATE% %TIME% >> "%LOG%"',
    '  echo DONE_EXIT_37 >> "%LOG%"',
    "  exit /b 37",
    ")",
]
anchor = 'echo WS OK %TIME% >> "%LOG%"'
assert s.count(anchor) == 1
s = s.replace(anchor, "\n".join(BANNER) + "\n" + anchor)

# --- header ---
HEADER = [
    "@echo off",
    "REM ===========================================================================================",
    "REM ITER 53: PUT THE LoRA PROJECTIONS ON MUON (RWKV_MUON_INCLUDE_LORA=1).",
    "REM Family: optimizer / regularization. Second of the 10 algorithmic iterations Andrew asked",
    "REM for on 2026-08-17.",
    "REM",
    "REM THE MECHANISM, and it is a PREDICTION rather than a hunch. The 2026-08-16 matched-pair",
    "REM measurement (iter 29 vs iter 26, only the three RWKV_MUON_* vars differ) showed Muon's",
    "REM TRAIN-loss advantage decays to zero over a run -- on ahead it inverts -- while its",
    "REM HELD-OUT advantage holds at +0.0019 in both modes. So at our budget Muon is a",
    "REM REGULARIZER, not a faster optimizer. The grouping rule in get_optimizer excludes any",
    "REM param whose name contains lora, so 27,520 params in 94 tensors have never received that",
    "REM regularization -- and being deliberately low-rank (shapes like 4x80 and 80x2) they are the",
    "REM most anisotropic matrices in the model, i.e. exactly where the mechanism predicts the",
    "REM largest effect.",
    "REM",
    "REM ** THE COUNTER-HYPOTHESIS IS PRE-REGISTERED: flattening the update of a deliberately",
    "REM low-rank factorization may destroy what the factorization is for. A null here is",
    "REM informative either way, and a REGRESSION is the outcome that would most sharply bound the",
    "REM regularizer reading.",
    "REM",
    "REM SINGLE VARIABLE. The moved params get their OWN group at wd=0.0, which is what they",
    "REM already had -- dropping them into decay_params would have changed the optimizer AND the",
    "REM weight decay at once. The LR is not separable and that is not a defect: a Muon group runs",
    "REM at RWKV_MUON_LR on a norm-normalised update, which has no comparable scale to AdamW's",
    "REM PEAK_LR. Verified inert with the flag off: same 5 groups, same partition, LoRAs still on",
    "REM AdamW (scratchpad/iter53_muonlora/smoke_muon_lora.py).",
    "REM",
    "REM FULL WS + decay, unlike iter 52: an optimizer change acts from step 1, so warm-starting",
    "REM from the champion's WS-final would measure something else.",
    "REM",
    "REM GATE: PLAIN basis vs iter 45 == 0.297697 ahead / 0.265375 imm on the VAL half, both-modes",
    "REM rule. Params UNCHANGED at 558,212 -- regrouping moves no weights, and nothing ships to",
    "REM Rust (the optimizer does not exist at inference).",
    "REM",
    "REM Do NOT edit this file while it runs (iters 43 and 46 died that way; git checkout is not a",
    "REM safe undo, because line endings shift the byte offset cmd.exe resumes from).",
    "REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.",
    "REM ===========================================================================================",
]
body = s[s.index("setlocal"):]
out = "\n".join(HEADER) + "\n" + body

low = out.lower()
assert "iter45" not in low and "i45_" not in low, "iter 45 reference leaked"
assert out.count("set RWKV_MUON_INCLUDE_LORA=1") == 1
assert "LoRA in a wd=0 group" in out, "banner guard missing"
assert "set RWKV_AUGMENT_SEED=4321" in out and "architecture_d80_lora4_cnd.py" in out
assert out.count("set RWKV_KD_ALPHA=0.9") == 1 and out.count("set RWKV_KD_ALPHA=0.5") == 1, (
    "the champion's KD schedule (0.9 in WS, 0.5 in decay) must be preserved verbatim"
)
for ln in out.split("\n"):
    if ln.strip().upper().startswith("REM"):
        bad = [c for c in "<>&|^" if c in ln]
        assert not bad, "redirection char " + str(bad) + " in REM line: " + ln
io.open(DST, "w", encoding="ascii", newline="\r\n").write(out)
print("wrote", DST, len(out), "bytes,", out.count(chr(10)), "lines")
