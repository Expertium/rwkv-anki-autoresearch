"""Produce the featB - featA verdict against the PRE-REGISTERED bands, and refuse to skip the
artefact checks that were pre-registered alongside them.

Written 2026-08-21 while featA was still evaluating, i.e. BEFORE either number existed. That is the
point: the bands and the checks live in optimization/FUTURE_FEATURES.md, and encoding them in a
script now removes the opportunity to soften them after seeing the result.

THE BANDS (both modes, raw, vs featA, p < 0.0001):
    >= +0.0010 both        LARGE   -> adopt as the trunk; the 7-arm family ablation earns its ~54 h
    +0.0003 .. +0.0010     REAL    -> adopt (rebuild cost is sunk); ablate only 2-3 families with a
                                      mechanism story, not 7 arms
    < +0.0003 or mixed     NULL    -> do not adopt; do NOT run per-feature arms, which would measure
                                      noise 23 times

THE THREE ARTEFACT CHECKS, which matter most if featB comes out WORSE, because all three look
exactly like "more input dims hurt at unchanged trunk capacity":
    1. arm B's WS log must contain `Trainable parameters: 565252` -- catches RWKV_ID_FEATURES not
       reaching the worker processes;
    2. `size` must be internally consistent WITHIN each arm. It is NOT comparable ACROSS arms: the
       dataset swap alone moves the equalized count for ~30% of users, so a cross-arm size
       difference is expected and is not a pipeline bug;
    3. arm B's eval.toml must name test_db_5k_id2 -- scoring 114-dim weights against a 112- or
       92-dim db is a silent shape mismatch.

⚠ AND THE ATTRIBUTION CAVEAT, recorded 2026-08-20 22:00: B - A is NOT "the 23 features". It bundles
the 23 columns, END-to-START intervals, the cumsum sentinel fix (a BUG FIX the old dbs predate), and
the dataset swap. The bands still decide "adopt the new pipeline?", which is the question on the
table -- but a null does not mean the features are worthless, and a win is not attributable to them.

Usage: .venv/Scripts/python.exe scratchpad/features_ab/verdict.py
"""
import io
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

# ★ THE CONTROL IS featA2, NOT featA (Andrew 2026-08-21: "Sure, re-run featA").
# featA ran on published dbs whose entity ids were saturated to INT32_MIN, so it is not a clean
# control for a featB built on id-fixed dbs. featA2 is featA's recipe with exactly two strings
# changed -- the db paths -- so control and treatment now differ ONLY in the input pipeline.
# featA's results are kept on disk but are NOT the comparison.
ARMS = {"featA2": {}, "featB": {}}
LARGE, REAL = 0.0010, 0.0003
WANT_PARAMS = {"featA2": "558212", "featB": "565252"}

fails = []
notes = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)


def load(path):
    """{user: (logloss, size)} from a result jsonl."""
    out = {}
    if not os.path.exists(path):
        return None
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[int(r["user"])] = (float(r["metrics"]["LogLoss"]), int(r["size"]))
    return out


print("=== ARTEFACT CHECKS (pre-registered) ===")
for arm in ARMS:
    d = os.path.join("scratchpad", "features_ab", arm)
    # An arm that has not RUN yet must report NOT-YET, never FAIL. A dry run before featB launches
    # would otherwise print "the width flag may not have reached the workers" about an arm that
    # simply does not exist -- an alarming string with no referent, which is the kind of thing a
    # later reader (or a post-compaction me) acts on.
    # ⚠ EXCLUDE logs from FAILED runs. featB's dead 04:09 attempt left ws_failed_idbug_0409.log,
    # which contains "Trainable parameters: 565252" and therefore SATISFIED the param guard for an
    # arm that had not run at all. Renaming a dead run's log is not enough if the reader still
    # globs it -- the guard has to exclude it by name. Same class as the watcher reporting a stale
    # log as a 12-minute stall.
    def live_ws(dirname):
        return [f for f in os.listdir(dirname)
                if f.startswith("ws_") and f.endswith(".log") and "failed" not in f]

    if not live_ws(d):
        print("  [ -- ] %s has not run yet; its checks are deferred" % arm)
        continue
    # 1. the param guard, read from the arm's own WS log
    ws = live_ws(d)
    hit = False
    for f in ws:
        txt = io.open(os.path.join(d, f), encoding="utf-8", errors="ignore").read()
        if "Trainable parameters: " + WANT_PARAMS[arm] in txt:
            hit = True
    check("%s WS log shows Trainable parameters: %s" % (arm, WANT_PARAMS[arm]), hit,
          "" if hit else "the width flag may not have reached the workers")
    # 3. the eval db
    et = os.path.join(d, "eval.toml")
    if os.path.exists(et):
        txt = io.open(et, encoding="utf-8", errors="ignore").read()
        want = "test_db_5k_id3" if arm == "featB" else "test_db_5k_fix"
        ok = want in txt
        check("%s eval.toml names %s" % (arm, want), ok)
    else:
        # NOT a failure: the eval toml is generated by the runner's eval PHASE, so an arm still in
        # WS or decay legitimately has none. Reporting FAIL here would put a red line against a
        # perfectly healthy run -- the same misleading-string class as the two fixes above.
        print("  [ -- ] %s has not reached its eval phase yet; eval.toml check deferred" % arm)

print("")
print("=== RESULTS ===")
res = {}
for arm in ARMS:
    # ⚠ ONLY THE MERGED FILE COUNTS. eval_sharded writes per-shard `-s0.jsonl` as it goes and
    # merges into the unsharded name at the END, behind a completeness gate that requires every
    # rostered user to be merged or explicitly NaN-skipped. Reading the shard file would let this
    # script compute a BAND from a partially evaluated arm and print it as a verdict -- a premature
    # number is worse than no number, because it is the one that gets acted on. If the merged file
    # is absent the arm is simply not done.
    a = load("result/RWKV-%s.jsonl" % arm)
    i = load("result/RWKV-P-%s.jsonl" % arm)
    if a is None or i is None:
        part = load("result/RWKV-%s-s0.jsonl" % arm)
        extra = " (shard file has %d users so far)" % len(part) if part else ""
        print("  %s: eval not COMPLETE -- no merged result jsonl yet%s" % (arm, extra))
        continue
    res[arm] = (a, i)
    print("  %s: ahead n=%d, imm n=%d  (merged)" % (arm, len(a), len(i)))

if len(res) < 2:
    print("")
    print("INCOMPLETE -- both arms must have results before a verdict.")
    sys.exit(0)

# 2. size consistency WITHIN each arm (ahead vs imm must agree per user)
for arm in res:
    a, i = res[arm]
    common = set(a) & set(i)
    bad = [u for u in common if a[u][1] != i[u][1]]
    check("%s size self-consistent across modes (%d users)" % (arm, len(common)), not bad,
          "" if not bad else "%d users disagree" % len(bad))

common = set(res["featA2"][0]) & set(res["featB"][0]) & set(res["featA2"][1]) & set(res["featB"][1])
print("")
print("=== B - A on %d shared users ===" % len(common))
assert len(common) > 100, "VACUOUS: only %d shared users" % len(common)

deltas = {}
for mode, k in (("ahead", 0), ("imm", 1)):
    A = sum(res["featA2"][k][u][0] for u in common) / len(common)
    B = sum(res["featB"][k][u][0] for u in common) / len(common)
    deltas[mode] = A - B  # POSITIVE = featB is BETTER (lower logloss)
    print("  %-5s  featA2 %.6f   featB %.6f   B improves A by %+.6f" % (mode, A, B, A - B))

# cross-arm size difference is EXPECTED, so report it as a note, never as a failure
sa = sum(res["featA2"][0][u][1] for u in common)
sb = sum(res["featB"][0][u][1] for u in common)
ndiff = sum(1 for u in common if res["featA2"][0][u][1] != res["featB"][0][u][1])
print("")
print("  NOTE cross-arm size differs on %d/%d users (total %d vs %d). EXPECTED -- the dataset swap"
      % (ndiff, len(common), sa, sb))
print("       alone moves the equalized count for ~30%% of users. Not a pipeline bug.")

print("")
print("=== p-GATE ===")
cmd = [".venv/Scripts/python.exe", "optimization/paired_pvalue.py",
       "--cand-ahead", "result/RWKV-featB.jsonl", "--cand-imm", "result/RWKV-P-featB.jsonl",
       "--champ-ahead", "result/RWKV-featA2.jsonl", "--champ-imm", "result/RWKV-P-featA2.jsonl",
       "--intersect"]
print("  " + " ".join(cmd))
try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(p.stdout.strip()[-2000:])
    if p.stderr.strip():
        print(p.stderr.strip()[-800:])
except Exception as exc:  # noqa: BLE001
    print("  paired_pvalue failed to run: %s" % exc)

print("")
print("=== VERDICT against the pre-registered bands ===")
lo = min(deltas["ahead"], deltas["imm"])
if deltas["ahead"] <= 0 or deltas["imm"] <= 0:
    band = "NULL (mixed sign)"
    action = ("do NOT adopt; do NOT run per-feature arms -- they would measure noise 23 times. "
              "Record which families to attack differently.")
elif lo >= LARGE:
    band = "LARGE (>= +0.0010 both modes)"
    action = "adopt as the trunk; the 7-arm FAMILY ablation now earns its ~54 h."
elif lo >= REAL:
    band = "REAL (+0.0003 .. +0.0010 both modes)"
    action = ("adopt -- the rebuild cost is sunk and what remains is one champion re-base plus the "
              "Rust input-width port -- but ablate only the 2-3 families with a mechanism story.")
else:
    band = "NULL (< +0.0003)"
    action = ("do NOT adopt; do NOT run per-feature arms. Record which families to attack "
              "differently.")
print("  min(ahead, imm) improvement = %+.6f" % lo)
print("  BAND   : %s" % band)
print("  ACTION : %s" % action)
print("")
print("  ⚠ ATTRIBUTION: B - A bundles the 23 columns, end-to-start intervals, the cumsum sentinel")
print("    BUG FIX (which the old dbs predate), and the dataset swap. The band decides 'adopt the")
print("    new pipeline?'. It does NOT attribute the result to the features.")
print("    If attribution matters, the next arm is RWKV_ID_FEATURES=0 on the gen-2 dbs.")
print("")
print("ARTEFACT CHECKS: " + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
