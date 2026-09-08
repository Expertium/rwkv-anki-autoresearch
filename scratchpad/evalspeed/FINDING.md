# Eval-phase sharding: REFUTED on the gen-5 lineage. The "UNSHARDED" LIVE RULE stands.

2026-09-08, 40 min of otherwise-idle GPU. Prompted by Andrew's constraint ("10+2 takes >a day"):
the eval is ~4.2 h of a ~10.6 h iteration -- 40% of every experiment -- and every runner uses
`--shards 1 --solo-threshold 0`. `eval_sharded.py`'s own docstring predicts **~0.56*W (~1.8x)** from
its phased mode (mega-users solo first, the rest LPT-split across 2 shards), and the LIVE RULE
forbidding it ("d=128/d=80 runs: Evals UNSHARDED") predates the solo phase that was built to make
sharding safe. Worth 40 min to re-test rather than inherit.

**A/B on users 5001-5400 (400 users), muongates' checkpoint, identical env:**

| arm | setting | wall clock |
|---|---|---|
| 1 sequential (today) | `--shards 1 --solo-threshold 0` | **23 min 05 s**, rc 0, 400 users scored |
| 2 phased | `--shards 2 --solo-threshold 1000000` | **WEDGED** -- killed after 38 min with no progress |

**The wedge is the documented one, and both witnesses agree** (the 2026-08-17 rule: a power reading
alone is not a freeze signal; absence of forward progress is): both shard logs stopped writing within
55 s of launch and had not advanced 38 minutes later, while the GPU sat at 100% utilisation,
**11,850 of 12,282 MiB**, 45 W -- the paging-through-shared-memory signature.

**Why the solo phase did not save it.** It peels off users with >= 1,000,000 work. The two users that
wedged were **5020 (work 114,769)** and **5006** -- an order of magnitude under the threshold. On this
lineage the eval does NOT chunk (test db: median 1 chunk/user, ~51k rows), so even a mid-sized user
holds a whole-history sequence, and TWO of them together exceed the card. The 2026-09-03 measurement
said the same thing from the other end: a giant user's eval needs ~40 GB of GPU-addressable memory.

**=> Eval parallelism is MEMORY-bound here, not schedule-bound, and no solo threshold fixes it: the
threshold that would keep pairs safe is low enough that almost every user runs solo, which is
sequential again. The 4.2 h is not recoverable this way. The LIVE RULE is correct and now has a
measurement behind it instead of an inherited scare.**

⚠ What this does NOT close: a single shard with a smaller per-user memory footprint (chunked eval)
would change the picture, but chunking the eval changes what the metric measures (2026-09-07:
the eval's whole-user sequences are why there is no cold-start cost in it) and is not a speed
decision to take unilaterally.

Sequential arm's by-user ahead on the 400 users: 0.299364 (a subset number, not comparable to the
2,499-user gate figures). Result files deleted; no orphan processes left.
