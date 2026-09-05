# durdrop P3 -- engagement, measured on the DECAYED checkpoint before the accuracy verdict (2026-09-05 06:03)

Same instrument, same 10 train-range users, 218,841 predicted rows (`screen_pass.py` with
`SCREEN_CKPT=scratchpad/durdrop/dd_d_10935.pth`): the by-user mean cost of replacing the current
review's duration by 0 at prediction time.

| checkpoint | mean cost | min | max | users with cost > 0 |
|---|---|---|---|---|
| realcyc (control, wd 0) | **+0.001388** | -0.000303 | +0.003581 | 8 / 10 |
| durdrop (p=0.25) | **+0.000881** | -0.000816 | +0.002384 | 7 / 10 |

**Reading, fixed before the number:** the lever is ENGAGED -- the model's dependence on the current
duration fell by 37% -- but it did NOT reach the pre-registered +0.0007 line. PREREG P3 said: "If it
does NOT fall, the dose was too small and the verdict is uninterpretable rather than a null; then
p=0.5 is the one retry." This is the in-between case, so the rule is applied as written for the part
that matters: **an accuracy null at p=0.25 is a null AT A PARTIAL DOSE and licenses exactly one retry
at p=0.5**; an accuracy gain is a gain at a partial dose (and p=0.5 is then the follow-up); an
accuracy regression closes the family (P4 stands regardless of dose).

Per-user note: the three users where zeroing HELPS (107, 136, 178) are the ones where realcyc was
already near zero cost; durdrop moved them further negative, i.e. the model over-shot toward
"duration-free" there. The large-cost users (2207, 3207, 4207) fell by 0.3-0.8 mdp each but remain the
bulk of the residual.
