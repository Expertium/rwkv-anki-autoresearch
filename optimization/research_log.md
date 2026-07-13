# Research-phase experiment log (100/100 era) — HISTORICAL

> **⚠ SUPERSEDED (2026-07-13).** This is the closed 100/100-era log (train 1-100 / eval 101-200,
> June 2026). The live phase is 5k: front table [research_5k.md](research_5k.md), verbose AI notes
> [research_5k_verbose.md](research_5k_verbose.md), machine log `research_log.jsonl`.

(original header: 100/100, train 1-100 / eval 101-200, sc8k WS-15 + 4ep decay, aug-off)

Champion = WS-15 + 4ep decay, fp32 ahead **0.314807** / imm **0.280200** (192,800 params, curves/points 64),
ckpt `scratchpad/tuner/decay15/decay15_640.pth`. Deployed (card rank2-int4 lowrank + note int2 + shifts) =
est. imm ~0.2822 / ahead ~0.3158 -- beats/ties d=128. Gate: ACCEPT only if BOTH modes improve >= 0.0003 vs the
CURRENT champion AND params <= 225,000 AND card/note state unchanged. Each experiment = full tuned recipe
(WS-15 2400 steps ~55 min + 4ep decay + eval) ~= 1 hr. Recipe levers via env (architecture.py): RWKV_NUM_CURVES/
RWKV_NUM_POINTS, RWKV_CHANNEL_MIXER_FACTOR, RWKV_LORA. Training env: RWKV_CLIP, RWKV_WEIGHT_DECAY, RWKV_AUGMENT_SEED.

| exp | change | params | ahead | imm | vs champ ahead | vs champ imm | status | note |
|---|---|---|---|---|---|---|---|---|
| champion | WS-15 + 4ep decay (curves/points 64) | 192,800 | 0.314807 | 0.280200 | - | - | CHAMPION | beats d=128 on both; deployed targets met |
| exp1 | num_curves/num_points 64->128 | 209,312 | 0.315209 | 0.280210 | -0.000402 | -0.000010 | REJECT | head resolution costs +16.5k params for ZERO accuracy gain (ahead slightly worse, imm tied). iter29's halve-to-64 confirmed correct. State-neutral. |
| exp2 | channel_mixer_factor 1.0->1.5 | 207,136 | 0.315322 | 0.280401 | -0.000515 | -0.000201 | REJECT | +14k FFN-capacity params slightly HURT both modes. 2nd capacity negative -> the d=32 model is DATA-limited at 100 users, NOT capacity-limited (matches d=128 already). Pivot from capacity to training levers. |
| decay8 | WS-15 + 8-epoch decay (vs 4ep) | 192,800 | 0.315154 | 0.280071 | -0.000347 | +0.000129 | REJECT | longer decay: ahead slightly worse, imm +0.00013 (< gate). 4-epoch decay is fine. Training-lever negative -> champion (WS-15 + 4ep decay) is well-established; 3 clean negatives (capacity x2, decay-length). |
| ep18 | WS-18 + 4ep decay (vs WS-15) | 192,800 | 0.314940 | 0.279790 | -0.000133 | +0.000410 | REJECT | more WS epochs (18 vs 15): ahead -0.00013 (worse), imm +0.0004 (better) -> fails both-modes gate. EPOCHS SATURATED at 15 on 100 users. More training on the SAME 100 users doesn't help -> motivates the 1500-user (more-DATA) experiment. |
| t1500ws | **1 epoch WS on 1500 users (1000-2499)**, eval 101-200, WS-ONLY (no decay) | 192,800 | 0.312814 | 0.278762 | +0.001993 | +0.001438 | **ACCEPT (variety wins)** | ★ DATA VARIETY beats repetition: 1 epoch on 1500 varied users BEATS the WS-15+4ep-decay champion on BOTH modes (+0.0020/+0.0014) -- and this is WS-ONLY (decay not yet applied). vs champion WS-15-only (0.316252/0.281974) = +0.0034/+0.0032. Trained on FURTHER users (1000-2499) than eval-adjacent 1-100, so the win is variety, not proximity. 3351 WS steps (>2400) but epochs saturate at 100u (ep18), so variety is the driver. Confirms DATA-LIMITED -> supports the 5k direction. Decay phase next. |
