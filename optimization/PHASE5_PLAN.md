# Phase 5 -- HP tuning of the QAT decay phase (plan, drafted 2026-09-13 00:45, before any lfq arm ran)

**Scope (Andrew 2026-09-12: "let's only tune the decay/QAT phase").** Trials are QAT decays branched off
w10lf's shared leak-free WS. WS HPs stay fixed: a WS trial is ~33-38 h, and the ws10 gap screen says the
model is fit-limited there, so the WS knobs (wd, dropout, warmup 400) are baked in. Every previous HP tune
was PLAIN; the QAT tax (+0.0025 ahead / +0.0045 imm at 10 epochs) changes the landscape, so this is a new
phase, not a re-confirmation.

## What the chain answers BEFORE this phase starts (do not tune these)

| knob | answered by | rule |
|---|---|---|
| decay length L* (1 / 2 / 3 epochs) | `lfd1` / `w10lf` / `lfd3` | `scratchpad/lfd3/PREREG.md`, `pick_lstar.py` |
| QAT-KD on/off, teacher = the plain twin | `lfq<L*>` vs `lfqkd<L*>` | `scratchpad/lfqat/PREREG.md`, `pick_kd.py` |
| WKV catalog bits b10 vs b13 | `lfq*<L*>b13` | `scratchpad/lfqat/PREREG.md` P6-P9 |
| card-stream capacity (cmf2 / lora16) | `lfcmf2` / `lflora16` | `scratchpad/cardgraft/PREREG.md` |

**The INCUMBENT for phase 5 is whatever arm those rules adopt** (the deploy candidate at the end of the
chain, ~2026-09-20). Its recipe is the trial template; each trial changes ONE knob against it.

## The knobs (the four Andrew named), their env names, and the grid

All four act only in the decay phase, so a trial is `mk_lfqat.py`'s arm with one override. The
generator must ASSERT the override reached the runner (a `set` line) AND the log (the trainer's own
echo of the value) -- the tuner's recurring failure mode is a lever that silently does nothing and
measures as a perfect null (`hp_tuner_5k.py` header).

| # | knob | env / arg | incumbent | grid (one at a time, coordinate descent) | why this order |
|---|---|---|---|---|---|
| 1 | decay peak LR, **AdamW AND Muon together** | `write_decay_setup.py` arg 9 (`PEAK_LR`, 1e-3) and `RWKV_MUON_LR` (0.0025 in arm 2's decay), scaled by the SAME factor | 1.0x | 0.5x, 2.0x (extend to 0.25x / 4x only if an edge wins) | the largest lever in every earlier tune; `PEAK_LR` alone moves ~10% of the weights (57k of 558k) |
| 2 | learnable-catalog LR | `RWKV_CB_LR_MULT` (multiplies `PEAK_LR` for the catalog optim groups; 1) | 1x | 0.3x, 3x | the tax lives in the CODEBOOK and NORM terms; the catalogs learn during QAT and their LR was never tuned under QAT |
| 3 | KD weight (only if KD is adopted) | `RWKV_QAT_KD=<lambda>` (1.0) | 1.0 | 0.5, 2.0 | a soft-target weight; the plain-era KD alpha tune moved both modes |
| 4 | QAT-phase weight decay | `RWKV_WEIGHT_DECAY` + `RWKV_WEIGHT_DECAY_HEAD` (0.01; the LoRA Muon group stays wd=0, iter 62) | 0.01 | 0.0, 0.03 | at the full budget the model is FIT-limited on the WS, but the QAT phase re-fits under fake-quant noise; a regulariser can only be judged there |

Trials: 2 + 2 + (2) + 2 = 6-8, plus extensions. Knob 3 is skipped if the chain does not adopt KD.

## Cost, and why the plan cannot be a plain grid

At L* = 2 one arm = ~22 h QAT decay (0.25-0.33 steps/s) + ~10.5 h probe and quant-aware VAL-half eval
= ~1.35 days. 8 trials = ~11 days of GPU. Two instruments cut that:

1. **Validation-trace early stop (zero cost).** A decay branch's 10-user validation endpoint (users
   5001-5010, quant-aware because validation runs the same fake-quant model), read beside the
   incumbent's endpoint on the SAME 10 users, predicted arm 1's gate deltas within 0.0004
   (`scratchpad/ws10/plot_losses.py`; research_5k_verbose.md "ENDGAME ARM 1"). Arm 2's preview
   showed the QAT tax closes by ~step 2,000 and is flat after. So: compare the trial's validation at
   steps 4,000 / 6,000 / 8,000 (`RWKV_DECAY_VALIDATE_EVERY=2000`) with the incumbent's at the same
   steps; **kill iff BOTH modes are worse by > 0.0010 at two consecutive checkpoints** (2.5x the
   instrument's error, and the sign-correct magnitude rule from the plain tuner's `vprune`). A killed
   trial costs ~5-7 h. A 0.0004 instrument can REJECT, it cannot PICK: never call a winner from
   the validation trace.
2. **The 1000-user tune-eval subset (5001-6000).** A quant-aware eval on the subset is ~4 h instead
   of ~10.5 h (51.0 M of 128.8 M reviews). The incumbent's subset numbers are free (restrict its
   VAL-half jsonls). Sub-0.001 effects on 200 users inverted at n=5000 (champ5k_t1); 1000 users
   resolve ~0.001. **Any subset winner is CONFIRMED on the full VAL half (2,499 users) before it
   becomes the recipe**, and the confirmation is the deploy number.

Budget with both: a surviving trial ~26 h, a killed one ~6 h. If half the trials die, 8 trials =
~5.5 days + one ~10.5 h confirmation eval. Matches the "~1-1.5 weeks" estimate in AGENTS.md.

## Accept rule (per knob, then the whole phase)

* Objective = ahead + imm, rectified, quant-aware, by-user mean; both modes reported.
* A grid point REPLACES the incumbent on the subset iff it improves BOTH modes by raw >= 0.0001
  (the research bar) -- the same rule the plain tuner used. Ties keep the incumbent.
* The phase's RESULT = one full-VAL-half confirmation of the final incumbent vs the chain's adopted
  arm, paired Wilcoxon on 2,499 users (`paired_pvalue.py --intersect`). Accept iff BOTH modes improve
  raw >= 0.0001 at p < 1e-4. Otherwise the chain's arm ships unchanged and the phase is a null.
* Seed-pair doctrine: a confirmation margin < ~0.0005 needs the recipe re-run at
  `RWKV_AUGMENT_SEED=4321` (augmentation is off, but the seed still drives dropout draws).

## What has to be BUILT (all CPU, all before ~09-20)

1. **BUILT 2026-09-13 00:40: `scratchpad/phase5/mk_trial.py <incumbent runner> <knob> <value>`** --
   takes the adopted arm's runner (an `mk_lfqat.py` output) and applies ONE override; asserts on the
   OUTPUT that the `set` line exists exactly once, that the incumbent's value is gone, that every other
   env key is byte-identical, and that the runner carries a RUN-TIME guard on the trainer's own echo of
   the value (`[muon] ... muon_lr=<v>` + `PEAK_LR = <v>` in decay.toml, exit 90/94; `lr x<mult>)` on
   the learnable-catalog group line, 91; `lambda = <v>` on the `[KD] teacher` line, 92; the `[wd] reset
   ... intended [<v>, <v>, <v>, ` list, 93). The eval range becomes the 1000-user subset. Writes the dry
   twin via `mk_dryrun.py` (which now matches the runner's peak LR instead of assuming `1e-3`; proven
   byte-inert on `w10qat`). Tag = `p5<knob><value>` (`p5lr0p5`, `p5cblr3`, `p5wd0p03`, `p5kd2`).
   ⚠ The 16 runners in the repo today were generated from **`lfq2` / `lfqkd2` as STAND-INS** for the
   adopted arm and all pass `preflight_runner.py`; **regenerate them from the adopted arm's runner at
   phase start** (the tuner's `next` command does that), never arm the stand-in versions.
2. **BUILT 2026-09-13 00:50 -- the trainer's own val-prune, opened to decay runs.** `train_rwkv.py`'s
   `RWKV_VPRUNE_*` hook was WS-only (`if TRAIN_MODE != "WS": ref = ""`); `RWKV_VPRUNE_DECAY=1` now lets
   it run in a decay (unset = byte-identical; py_compile PASS; the chain's later phases import it
   inert). `scratchpad/phase5/mk_vprune_ref.py <tag> <decay log> <total steps> <out>` builds the
   reference json from the incumbent's decay log (steps reconstructed from the cadence 50 / every 2000 /
   total and asserted against the line count; `final_*` = its subset means; tested on arm 2's log: 12
   points, refuses a wrong total). `mk_trial.py` wires it: `RWKV_VPRUNE_DECAY=1`, `RWKV_VPRUNE_REF=
   scratchpad/phase5/vref_<incumbent>.json`, **deltas 0.0015 / 0.0015, min_step 6000, persist 2**
   (looser than the plan's 0.0010 @ 4000: a trial at another LR sits at a different point of its own
   schedule mid-decay, so only DISASTERS should die; a killed trial still saves ~15 h); phase 0v refuses
   without the reference (96); a guard requires the trainer's `[vprune] vs champion` line (95); exit 42
   from the decay is caught on ONE line (preflight's 40-line artifact-gate window), moves the trainer's
   `ws_prune_marker.json` to `<dir>/pruned.json` and ends `DONE_EXIT_42` -- a marker the tuner reads.
3. **BUILT 2026-09-13 01:00: `scratchpad/phase5/tuner.py`** (`init <adopted runner>` / `next` / `record
   <tag>` / `last <tag>` / `status`; state.json + journal.jsonl). Coordinate descent over SPACE in the
   plan's order; `next` writes the next trial via `mk_trial.py` from the CURRENT incumbent's runner
   (so knob 2 is tuned on knob 1's winner: `p5lr2` -> `p5lr2cblr3`, chain-generated and preflighted),
   building the incumbent's vprune reference first; a coordinate promotes its best point iff it beats
   the incumbent by raw >= 0.0001 in BOTH modes on the subset (an edge win is noted for a manual
   extension); `kd` is skipped when the incumbent has no live KD. Tested in a relocated state dir
   (`P5_STATE_DIR`): init from real subset results, a pruned record from a `DONE_EXIT_42` + pruned.json,
   a refused out-of-order record, a clear win promoting, a +0.00005 win NOT promoting.
4. **BUILT 2026-09-13 01:00: `scratchpad/phase5/run_phase5.cmd`** (`mk_run_phase5.py`): the loop
   `next` -> (the FIRST trial's dry twin, once) -> `call` the trial -> `record` -> `last` (a FAILED
   trial stops the loop with 98: broken tooling, not a bad HP) -> repeat; `DONE_EXIT_0` when the
   descent ends, 97 on a tuner error, 99 on a failed dry run. Executed in the foreground on its
   NO_STATE, DONE and tuner-error paths (97 / 0 / 97). **NOT a waiter, by design: it is launched by
   hand (detached) after `tuner.py init <adopted arm>`**, because choosing the adopted arm IS the
   chain's verdict step (`pick_lstar`, `pick_kd`, the bits rule) and the record needs a human there
   anyway.
5. **BUILT 2026-09-13 01:05: `scratchpad/phase5/mk_confirm.py gen|merge <winner>`** -- the confirmation.
   `gen` slices the winner's runner to its EVAL phase only (its decay checkpoint and learned catalogs
   already exist) on the REST of the VAL half, users 6001-7500 (~6 h instead of ~10.5 h; per-user
   results do not depend on the eval's other users, phase L), results `RWKV-<winner>c`; `merge`
   concatenates the subset and confirmation jsonls into `RWKV-<winner>full` (refuses on overlap or a
   missing VAL user) for `paired_pvalue.py --intersect <adopted arm> <winner>full`. The stand-in
   `p5lr2c` runner passes preflight. **The phase's result = that paired test: accept iff both modes
   improve raw >= 0.0001 at p < 1e-4; otherwise the adopted arm ships unchanged.**

## Open questions (none block the build; ask Andrew when the chain's arm is known)

* Whether to spend the confirmation eval on the subset winner only, or also on the runner-up when
  the two are within 0.0003 on the subset (the subset cannot separate them).
* Knob 1's factor ALSO scales the catalog LR by construction (`RWKV_CB_LR_MULT` multiplies
  `PEAK_LR`). The plan keeps that coupling and tunes the multiplier separately at knob 2.
