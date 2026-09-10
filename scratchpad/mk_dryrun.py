"""Turn any endgame branch runner into a FAST END-TO-END DRY RUN: ~0.05 epochs, 20 eval users.

ANDREW 2026-09-09: *"Maybe we should have a rule like 'do a test run on 20 users every 5
iterations' or something, lol"* -- said lightly, and it is right. His own 2026-08-18 bug-hunt
directive already specified this shape (EPOCHS 0.05, 20 eval users), and this repo's audit says
plainly that every failure of that week would have been caught by one cheap end-to-end run. It was
used once and never made routine.

WHY THE TRIGGER IS NOT A COUNT. Today's bug did not appear because five iterations had passed. It
appeared because the endgame branches are the FIRST to decay from a SHARED WS, so the decay wrote
its checkpoints into `scratchpad/ws10` while the runner asked `write_eval_toml` for its own
directory. The trigger is STRUCTURAL NOVELTY -- a new runner family, a new phase shape, or an edit
to a shared tool -- and a count fires at the wrong times in both directions. Keep the periodic
version as a BACKSTOP against slow drift; it is just not the primary rule.

WHAT THIS CATCHES THAT A STATIC CHECK CANNOT. `preflight_runner.py` now covers the config-side
class (it grew a decay-writes-here-vs-eval-looks-there check the same day) and needs no GPU at
all. A dry RUN catches the rest:
  * the HOLLOW run -- ordcut raised inside `_get_loss` on every step while the banner and the
    parameter count were both correct, and `train_rwkv`'s per-batch except swallowed it;
  * a guard whose findstr string disagrees with the value the runner SETS (decayshape, iter 54);
  * `endlocal` before the terminal marker, which stranded a whole chain for 45 min;
  * a phase that exits 0 without writing its artifact.

WHAT IT DOES NOT CATCH, and this is not a small caveat: anything that only appears at SCALE. The
6701 OOM, the WDDM co-tenant deadlock and the giant-user freeze all need real users and real VRAM.
Twenty small users prove the PLUMBING, never the capacity.

COST: ~10 min of GPU against a 9-16 h branch, i.e. 1-2%.

⚠ The dry run takes its own tag, directory, log, checkpoint prefix and result jsonls, so it cannot
touch a real branch's artifacts. It DOES write its checkpoint into the shared ws10 directory (a
decay-only run writes into the SOURCE run's folder) as `<prefix>dry_d_*.pth` -- delete those after.

Usage: python scratchpad/mk_dryrun.py scratchpad/w10qat/run_w10qat.cmd
"""
import io
import os
import re
import sys

B = chr(92)
NL = chr(10)
EPOCHS = 0.05
USERS = "5001 5020"

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
src_path = sys.argv[1]
src = io.open(src_path, encoding="utf-8", newline="").read().replace("\r\n", NL)

# Parse only NON-REM lines. These runners carry a REM that mentions write_decay_setup.py in prose;
# a naive search matched THAT, captured an English word as the checkpoint prefix and rewrote prose
# instead of the checkpoint name -- silently, because the bogus word then passed the "no stale
# prefix" assert vacuously. preflight_runner.py documents this trap for its own parsers; it
# applies to every parser over a runner.
code = NL.join(l for l in src.split(NL) if not l.strip().upper().startswith("REM"))

m = re.search(r"set TAG=(\S+)", code)
if not m:
    raise SystemExit("no `set TAG=` in " + src_path)
tag = m.group(1)
dry = tag + "dry"

# The checkpoint prefix does NOT always follow the tag -- arm 2 is tagged `w10qat` but its decay
# prefix is `w10q_d`. Read it from the runner's own call rather than assuming.
mp = re.search(r"write_decay_setup\.py\s+\S+\s+\S+\s+(\S+)", code)
if not mp:
    raise SystemExit("no write_decay_setup call in " + src_path)
ckpt = mp.group(1)
ckpt_dry = (ckpt[:-2] + "dry_d") if ckpt.endswith("_d") else (ckpt + "dry")
# MUST match train_rwkv.py:967 EXACTLY -- `total_steps = int(config.EPOCHS * len(groups))`,
# i.e. TRUNCATION, not rounding. int(round(0.05 * 10935)) = 547 while the trainer produces
# 546, so the dry runner's own artifact guard asked for a checkpoint that could not exist and
# every dry run died at DECAY_SHORT after doing its job perfectly. It cost 2 h of idle GPU on
# 2026-09-10: the arm-2 waiter saw the dry run fail and correctly refused to launch.
# 10935 is len(groups) for THIS lineage's train db at MAX=65536; a different db moves it.
steps = int(EPOCHS * 10935)

out = src
out = out.replace("scratchpad" + B + tag, "scratchpad" + B + dry)
out = out.replace("scratchpad/" + tag, "scratchpad/" + dry)
out = out.replace("set TAG=" + tag, "set TAG=" + dry)
out = out.replace(tag + ".log", dry + ".log")
out = out.replace(ckpt, ckpt_dry)
out = out.replace(tag + " START", dry + " START")
# The decay length and the eval user range are the whole point of a dry run.
out = re.sub(r"(1 5000 )[0-9.]+( 1e-3)", r"\g<1>%g\g<2>" % EPOCHS, out)
out = re.sub(r"set STEPS=\d+", "set STEPS=%d" % steps, out)
out = re.sub(r"(write_eval_toml\.py.+?)\d+ \d+(\s*>)", r"\g<1>%s\g<2>" % USERS, out)

body = NL.join(l for l in out.split(NL) if not l.strip().startswith("REM"))
assert "set STEPS=%d" % steps in body, "decay step count not rewritten"
assert USERS in body, "eval user range not rewritten"
assert "1 5000 %g 1e-3" % EPOCHS in body, "decay epochs not rewritten"
# Whole-LINE test for the tag: "set TAG=w10qatdry" contains "set TAG=w10qat" as a substring, so a
# naive `in` fails a correct file. Same shape as the eqw banner exempted from mk_muongates' scan.
assert ("set TAG=" + tag) not in [l.strip() for l in body.splitlines()], "tag line not rewritten"
# Non-vacuous by construction: ckpt provably occurs in the source, so this assert can fail.
assert ckpt in code, "internal: prefix not found in the source runner"
assert ckpt not in body, "stale checkpoint prefix %r survived outside a REM" % ckpt
assert body.count("DONE_EXIT_0") == 1

os.makedirs("scratchpad/" + dry, exist_ok=True)
dst = "scratchpad/%s/run_%s.cmd" % (dry, dry)
io.open(dst, "w", newline="\r\n").write(out)
print("wrote %s" % dst)
print("  decay %g ep = %d steps, prefix %s -> %s, eval users %s"
      % (EPOCHS, steps, ckpt, ckpt_dry, USERS))
print("  then check: DONE_EXIT_0 in the log, 0 tracebacks, 20 rows in result/RWKV-%s.jsonl" % dry)
