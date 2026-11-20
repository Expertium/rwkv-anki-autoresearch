"""TA3 (Training-Time Testing) 4-epoch chain, mirroring the plain 4-epoch chain exactly.

PRE-REGISTERED ENDPOINTS AND RULES (written 2026-11-15, before the run started):

  * Primary endpoint: the 4-epoch TA3 checkpoint
    `scratchpad/ta3_fullrun_20261115/epoch_04/ta3_173416.pth` against the plain 4-epoch
    checkpoint `.../adaptive_fullrun_92_20261105/attempt_04_epoch_04_wd_0.01/adaptive_173416.pth`
    (screen 0.304392 ahead / 0.272142 imm on the fixed 750-user roster).
  * Gate: on the same 750-user roster BOTH modes must improve by >= 0.0001 with p < 1e-4
    (same rule the other arms use). One chain is screen-level evidence only; a method-level
    claim needs independent paired seeds.
  * Per-epoch screens are DIAGNOSTIC. There is no accept/reject, no rollback and no WD
    switching: every epoch is trained and kept.
  * Everything except RWKV_TA3 is identical to the plain chain: same start checkpoint
    (QAT46 step-248 model+optimizer), same 92-input data, same LR shape and offsets, WD 0.01,
    one fetch worker, 11 GiB VRAM cap, 2-user internal validation, 750-user screen.
  * Rationale and gate history: `optimization/PROPOSALS.md` (2026-11-13 / 2026-11-15 entries)
    and row 10 of `optimization/GPU_RUN_BOARD.md`.

Epoch plan (mirrors the plain chain; offset = 1 + (epoch_target-1)*len(groups), len(groups)=43354):
    epoch 1: EPOCHS=1, STEP_OFFSET=1      , from QAT46 step-248      -> ta3_43354.pth
    epoch 2: EPOCHS=2, STEP_OFFSET=43355  , from epoch 1             -> ta3_86708.pth
    epoch 3: EPOCHS=3, STEP_OFFSET=86709  , from epoch 2             -> ta3_130062.pth
    epoch 4: EPOCHS=4, STEP_OFFSET=130063 , from epoch 3             -> ta3_173416.pth

The controller is resumable: a re-run skips epochs whose checkpoint pair already exists and
whose training log passes the same checks. It never deletes or overwrites a checkpoint.
"""
import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad" / "adaptive_wd"))
import adaptive_train as ad  # noqa: E402  (edit_toml, run_checked, eval_users, ...)

PY = ad.PY
START_FOLDER = ROOT / "pretrain" / "rwkv" / "opt_qat46"
START_NAME = "rwkv_iter46_248"
START_OPTIM = START_FOLDER / "rwkv_iter46_optim_248.pth"
INTERNAL_VAL_USERS = [5001, 5003]  # same as the plain chain's attempts
PLAIN_SCREEN = {  # plain chain, same start checkpoint, same 750-user roster
    1: (0.308803, 0.275131),
    2: (0.307044, 0.274275),
    3: (0.305847, 0.272732),
    4: (0.304392, 0.272142),
}


def build_env(eta, scope, frac):
    env = dict(os.environ)
    env.update({
        "RWKV_ID_FEATURES": "0", "RWKV_REAL_CYCLES": "0",
        "RWKV_FINAL_LAYOUT": "0", "RWKV_E2S_PUBLISHED": "0",
        "RWKV_EXTRA_FEATURES": "", "RWKV_ZERO_FEATURES": "",
        "RWKV_WEIGHT_DECAY": "0.01", "RWKV_WEIGHT_DECAY_CMIX": "0.01",
        "RWKV_WEIGHT_DECAY_HEAD": "0.01",
        "RWKV_RESUME_SKIP_GROUPS": "1", "RWKV_DECAY_RESUME_SCHED": "1",
        "RWKV_DECAY_SHAPE": "constant",
        "RWKV_VRAM_CAP_GB": "11",
        "RWKV_TA3": "1", "RWKV_TA3_ETA": str(eta),
        "RWKV_TA3_SCOPE": str(scope), "RWKV_TA3_FRAC": str(frac),
    })
    # TA3 owns the step; these must not be set at the same time.
    for key in ("RWKV_TA3_PROBE", "RWKV_TA3_CHECK", "RWKV_SAM_RHO", "RWKV_MUON"):
        if env.get(key):
            raise RuntimeError(f"{key} is set in the ambient environment; TA3 chain must not inherit it")
        env.pop(key, None)
    return env


def main():
    ap = argparse.ArgumentParser(description="TA3 4-epoch chain mirroring the plain chain.")
    ap.add_argument("--run-dir", default=str(ROOT / "scratchpad" / "ta3_fullrun_20261115"))
    ap.add_argument("--train-config", default=str(ROOT / "rwkv" / "train_rwkv_config.toml"))
    ap.add_argument("--eval-config", default=str(ROOT / "rwkv" / "get_result_config_base5k.toml"))
    ap.add_argument("--train-db", default="F:/rwkv_lmdb/train_db_5k_h1_adaptive92")
    ap.add_argument("--val-db", default="F:/rwkv_lmdb/test_db_5k_adaptive92")
    ap.add_argument("--label-db", default="F:/rwkv_lmdb/label_filter_db_id_e2s")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--scope", default="cmix")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--no-screen", action="store_true", help="train only; skip the diagnostic screens")
    ap.add_argument("--retries", type=int, default=2,
                    help="extra from-scratch attempts for an epoch that crashes (default 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="pre-flight: build every epoch's config from the plan and exit (no training, no state file)")
    ap.add_argument("--start-model", default=str(START_FOLDER / START_NAME) + ".pth")
    ap.add_argument("--start-optim", default=str(START_OPTIM))
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = Path(args.train_config).resolve()
    eval_base = Path(args.eval_config).resolve()
    train_db, val_db, label_db = args.train_db, args.val_db, args.label_db
    for p in (train_db, val_db, label_db, train_cfg, eval_base):
        if not Path(str(p).replace("/", "\\")).exists():
            ap.error(f"missing input path: {p}")
    if args.epochs < 1 or args.epochs > 10:
        ap.error("--epochs must be in 1..10")

    from rwkv.id_features import input_width
    if input_width() != 92:
        raise RuntimeError(f"TA3 chain requires 92 inputs; controller layout is {input_width()}")

    from rwkv.train_rwkv import get_groups
    groups = get_groups(train_db, 400_000_000_000, 16384, list(range(1, 5001)))
    g = len(groups)
    del groups
    if g < 1:
        raise RuntimeError("training DB has no groups")
    print(f"[ta3-chain] groups={g}", flush=True)

    # Fixed, deterministic 750-user screen, drawn here (never from results) so every arm
    # screens the same roster. Internal validation is the trainer's own 2-user roster.
    screen_users, excluded = ad.choose_screen_users(val_db)
    if len(screen_users) != ad.SCREEN_COUNT:
        raise RuntimeError(f"screen roster is {len(screen_users)}, expected {ad.SCREEN_COUNT}")
    print(f"[ta3-chain] screen users={len(screen_users)} excluded_multibatch_or_missing={len(excluded)}",
          flush=True)

    if args.dry_run:
        # Pre-flight (no training, no state file): build every epoch's config from the PLAN so the
        # offsets, load names and save names can be inspected before a multi-day run starts.
        import tomllib
        d = run_dir / "dryrun_plan"
        lf, ln, lo = Path(args.start_model).parent, Path(args.start_model).stem, Path(args.start_optim)
        for epoch in range(1, args.epochs + 1):
            td = d / f"epoch_{epoch:02d}"
            td.mkdir(parents=True, exist_ok=True)
            (td / "validation_users.json").write_text(json.dumps(INTERNAL_VAL_USERS), encoding="utf-8")
            cfg = td / "train.toml"
            ad.make_train_config(train_cfg, cfg, td, lf, ln, lo, epoch, 1 + (epoch - 1) * g,
                                 train_db, val_db, label_db, INTERNAL_VAL_USERS)
            ad.edit_toml(cfg, {"SAVE_MODEL_PREFIX": "ta3"})
            data = tomllib.loads(cfg.read_text(encoding="utf-8"))
            print(json.dumps({k: data[k] for k in (
                "EPOCHS", "STEP_OFFSET", "LOAD_MODEL_FOLDER", "LOAD_MODEL_NAME", "LOAD_OPTIM_PATH",
                "SAVE_MODEL_FOLDER", "SAVE_MODEL_PREFIX", "VALIDATE_EVERY", "NUM_FETCH_PROCESSES",
                "MAX_TRAIN_GLOBAL_LEN", "TRAIN_MODE", "PEAK_LR", "VALIDATE_USERS_START",
                "VALIDATE_USERS_END")}), flush=True)
            lf, ln, lo = td, f"ta3_{epoch * g}", td / f"ta3_optim_{epoch * g}.pth"
        print("TA3_DRYRUN_OK", flush=True)
        return

    env = build_env(args.eta, args.scope, args.frac)
    state_path = run_dir / "ta3_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        run_tag = state.get("result_tag") or (run_dir.name + "_" + uuid.uuid4().hex[:8])
        if list(state.get("screen_users", [])) != list(screen_users):
            raise RuntimeError("saved screen roster differs from the deterministic roster")
    else:
        state = {"plan": {"epochs": args.epochs, "eta": args.eta, "scope": args.scope,
                          "frac": args.frac, "groups": g,
                          "start_model": args.start_model, "start_optim": args.start_optim,
                          "train_db": train_db, "val_db": val_db, "label_db": label_db,
                          "screen_users": list(screen_users),
                          "screen_excluded": list(excluded),
                          "internal_val_users": list(INTERNAL_VAL_USERS),
                          "plain_screen": {str(k): list(v) for k, v in PLAIN_SCREEN.items()},
                          "gate": "both modes improve >= 0.0001 with p < 1e-4 on the 750-user roster",
                          "screening": "diagnostic only; no accept/reject and no rollback"},
                 "epochs": []}
        run_tag = run_dir.name + "_" + uuid.uuid4().hex[:8]
    state["result_tag"] = run_tag
    state["ta3_env"] = {k: v for k, v in env.items() if k.startswith("RWKV_")}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    load_folder, load_name, load_optim = Path(args.start_model).parent, Path(args.start_model).stem, Path(args.start_optim)
    if not (load_folder / f"{load_name}.pth").exists() or not load_optim.exists():
        raise RuntimeError(f"start pair missing: {load_folder}/{load_name}.pth , {load_optim}")

    for epoch in range(1, args.epochs + 1):
        done = next((r for r in state["epochs"] if r.get("epoch") == epoch), None)
        td = run_dir / f"epoch_{epoch:02d}"
        end_step = epoch * g
        cp, op = td / f"ta3_{end_step}.pth", td / f"ta3_optim_{end_step}.pth"
        if done and cp.exists() and op.exists():
            print(f"[ta3-chain] epoch {epoch} already trained -> {cp.name}", flush=True)
        else:
            td.mkdir(parents=True, exist_ok=True)
            (td / "validation_users.json").write_text(json.dumps(INTERNAL_VAL_USERS), encoding="utf-8")
            # A crash mid-epoch must not idle the GPU until a human notices: the 2026-11-15 crash sat
            # dead for ~12 h. Every attempt rebuilds the SAME config and restarts the EPOCH FROM
            # SCRATCH from the epoch's own start pair, so a retried epoch still covers every group
            # once and stays comparable; only the log file name changes.
            failures = []
            for attempt in range(1, args.retries + 2):
                cfg = td / "train.toml"
                ad.make_train_config(train_cfg, cfg, td, load_folder, load_name, load_optim,
                                     epoch, 1 + (epoch - 1) * g, train_db, val_db, label_db,
                                     INTERNAL_VAL_USERS)
                ad.edit_toml(cfg, {"SAVE_MODEL_PREFIX": "ta3"})
                log = td / ("train.log" if attempt == 1 else f"train_retry{attempt - 1}.log")
                print(f"[ta3-chain] epoch {epoch} attempt {attempt}: offset={1 + (epoch - 1) * g} "
                      f"load={load_name} -> {cp.name}", flush=True)
                try:
                    ad.run_checked([PY, "-u", "-m", "rwkv.train_rwkv", "--config", cfg], ROOT, log, env)
                    tx = log.read_text(encoding="utf-8", errors="replace")
                    if "MODEL SAVED." not in tx:
                        raise RuntimeError("trainer did not save a checkpoint")
                    if "TA3 step failed -- ABORT" in tx or "TA3 measure-only step failed" in tx:
                        raise RuntimeError("the log reports a TA3 abort")
                    trainer_val = ad.read_validation_log(log)
                    steps_seen = [int(m.group(1)) for m in
                                  (re.match(r"^\d+ \d+ (\d+)( \[ta3\])?, all:", ln)
                                   for ln in tx.splitlines()) if m]
                    if not steps_seen or max(steps_seen) != end_step:
                        raise RuntimeError(f"last logged step is "
                                           f"{max(steps_seen) if steps_seen else None}, "
                                           f"expected {end_step}")
                    if not cp.exists() or not op.exists():
                        raise RuntimeError(f"checkpoint pair missing ({cp.name})")
                    break
                except BaseException as exc:
                    failures.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}",
                                     "log": str(log)})
                    print(f"[ta3-chain] epoch {epoch} attempt {attempt} FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    state["retries"] = failures
                    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                    if attempt == args.retries + 1:
                        print(f"[ta3-chain] epoch {epoch}: all {attempt} attempts failed", flush=True)
                        raise
            rec = {"epoch": epoch, "step_offset": 1 + (epoch - 1) * g, "end_step": end_step,
                   "model": str(cp), "optim": str(op), "trainer_val": trainer_val,
                   "loaded_model": str(load_folder / f"{load_name}.pth"),
                   "train_log": str(log)}
            if failures:
                rec["retry_failures"] = failures
            if done:
                done.update(rec)
                rec = done
            else:
                state["epochs"].append(rec)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        if not args.no_screen and "screen" not in rec:
            # Unique tag per run invocation: eval_sharded refuses to overwrite a previous result.
            tag = f"{run_tag}_e{epoch:02d}_{uuid.uuid4().hex[:6]}"
            res = ad.eval_users(run_dir, eval_base, cp, tag, screen_users,
                                val_db, label_db, env, shards=1, require_full=True)
            rec["screen"] = {"ahead": res["ahead"], "imm": res["imm"], "users": res["n"], "tag": tag}
            pa = PLAIN_SCREEN.get(epoch)
            if pa:
                rec["plain_screen"] = {"ahead": pa[0], "imm": pa[1],
                                       "delta_ahead": pa[0] - res["ahead"],
                                       "delta_imm": pa[1] - res["imm"]}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps({k: rec.get(k) for k in ("epoch", "end_step", "screen", "plain_screen")},
                         default=str), flush=True)

        load_folder, load_name, load_optim = td, f"ta3_{end_step}", op

    print("TA3_CHAIN_DONE", json.dumps({"run_dir": str(run_dir), "epochs": args.epochs}), flush=True)
    for rec in state["epochs"]:
        sc = rec.get("screen")
        print(f"  epoch {rec['epoch']}: " + (f"{sc['ahead']:.6f}/{sc['imm']:.6f} (n={sc['users']})"
              if sc else "no screen"), flush=True)


if __name__ == "__main__":
    main()
