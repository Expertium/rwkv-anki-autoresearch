# cmixpow (`RWKV_CMIX_POW=1`) — annotations for a RUNNING experiment

Written 2026-08-19 06:55 while phase 2b is training. **Annotations live in a separate file, never in
the runner:** cmd.exe re-reads a batch file from a saved byte offset after every command returns.

## ⚠ THE LEVER IS LIVE ON 4 OF 13 CHANNEL MIXERS, NOT ON ALL OF THEM

Read this before writing the verdict. From the WS phase's own `grad_stats.json` (434 tensors):

| | count | which |
|---|---|---|
| `cmix_pow` params created | 13 | one per channel mixer |
| **receive gradients** | **4** | `card:0`, `note:0`, `deck:0`, `deck:3` |
| never receive a gradient | 9 | `card:1`, `deck:1`, `deck:2`, `preset:0–2`, `user:0–2` |

**The 9 dead ones are exactly `RWKV_STRIP_CMIX`** — verified as a set equality, not by eyeballing:

```
RWKV_STRIP_CMIX = user_id:0,1,2  preset_id:0,1,2  deck_id:1,2  card_id:1     (9 entries)
dead cmix_pow   = user:0,1,2     preset:0,1,2     deck:1,2     card:1        (9 entries)
dead == stripped -> True
```

So this is **not a bug**. A stripped channel mixer still constructs its parameters; they are simply
never used, so no gradient reaches them. The lever works exactly where a channel mixer still exists.

### Why it changes the reading of the verdict

"Learnable channel-mixer exponent" sounds like a change to the whole trunk. It is a change to **four**
mixers, three of which are the layer-0 entry mixers of card / note / deck plus deck's top layer. So:

* **If cmixpow returns null, the correct statement is "null on the 4 surviving mixers"** — not
  "learnable exponents do not help". The other 9 sites were never tested, because they do not exist
  in this configuration.
* That distinction is load-bearing for the **expressiveness-vs-capacity family** Andrew opened on
  2026-08-17. This is the family's first run, and closing a family on a lever that reached 4 of 13
  sites would repeat the mistake the family was created to correct — where "capacity-at-5k is 0/3"
  stood in for an argument it could not support.
* Note also the overlap with **iter 49**, which restored the `user/preset` layer-0 channel mixers and
  was rejected (+0.000067 ahead at p=0.11). Those are among the sites that here carry no exponent at
  all. The two results are about the same missing mixers from opposite directions.

### Minor, not a correctness issue

9 unused scalars are created and saved in the checkpoint. Params read 558,225 vs iter 45's 558,212,
i.e. +13 for 13 exponents of which 4 are trainable. Trivial in size; worth knowing only so the param
count is not read as 13 live parameters.

## The run itself is correct

Phase 2a decayed 3.3 h at KD alpha **0.9** — iter 55's lever — because the decay-only generator
sliced away the WS region containing the reset line, and its own guard caught it
(`DONE_EXIT_WRONGALPHA_DECAY`). **Phase 2b confirms the repair in its own log:**

```
[kd-mix] KD ON: ... alpha FIXED at 0.5      (74 per-step confirmations of alpha=0.5000)
[assert-params] OK
```

WS is the pre-outage one (`i54_ws_10935.pth`, resumed from step 8000 with `[resume-skip]` confirmed).
⚠ The resumed tail's dropout draws differ from an uninterrupted run — weights and optimizer state are
exact, so the number is fair, but the run is **not bit-reproducible**.

## When recording the verdict

* Gate against the **current champion iter 53** = `0.297523 / 0.265191` (VAL 5001–7500).
* Built on the **iter-45** recipe, so report **both** deltas: vs iter 45 = the controlled effect of
  the lever, vs iter 53 = the gate.
* **BOTH-modes rule.** A channel-mixer change is a trunk change; the curve-side exception does not
  apply.
* The number is assigned at verdict time under the completion-order convention. `iter54_cmixpow` is a
  directory name, not a claim — `decayshape` already took **56**.
