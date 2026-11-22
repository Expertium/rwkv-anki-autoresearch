"""Corruption (finiteness) check: run each passing quantized model on the 50 users with the longest
average review histories, with their histories NOT truncated, and FAIL the check if even ONE card/note
state value or even ONE p(recall) is non-finite.

Gate (Andrew, 2026-11-21): "if even ONE card/note state has non-finite values or if even ONE
p(recall) is non-finite, it counts as failing the check". Histories are NOT truncated: the eval DB
stores one section per user covering the full history (probe_sections.py: user 5001 = 32,615 rows /
17,459 reviews in ONE section), and nothing in this path caps review count -- the 64/128-review cap
belongs to the training chunker.

Roster: corruption/corrupt50.json = the 50 longest-mean-history users among 5001-7500 whose FULL
section fits the VRAM cap (Andrew: "skip users that cannot fit into VRAM" -> MAX_ROWS=200,000
section rows; 62 too-big users are recorded in excluded_too_big and backfilled).

Per config this runs `probe_get_result.py` (a normal get_result eval with finite_probe.install()
applied) on ONE process -- 50 users, no sharding -- and writes corruption_<tag>.json with the verdict,
the per-user offender detail, the state-call counts (proof the state was actually observed) and the
LogLoss per head. RWKV_QAT_FUSED=0 is required: it forces the Python per-step QAT loop, the
bit-parity twin of the fused CUDA kernels, which is the only place the state passes through Python.

Usage:
  .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\check_corruption.py --tag ctl_fp32
  ... --configs l2i8,l3i8,l2i8s4,l3i8s4 --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(ROOT / "scratchpad" / "adaptive_wd"))

import finite_probe  # noqa: E402
import quant_sweep as qs  # noqa: E402
import adaptive_train as ad  # noqa: E402

BASE_TOML = ROOT / "rwkv" / "get_result_config_base5k.toml"
ROSTER = HERE / "corrupt50.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# name -> (checkpoint, quant_sweep candidate env key, what the RESULTS.md row is)
CONFIGS = {
    "ctl_fp32": (RUN / "qat_it36ctl" / "ctl_124.pth", "nj",
                 "fp32 control, no quantization (row 0 control of the it36ctl ladder)"),
    "l2i8": (RUN / "qat_it36l2i8" / "qat_124.pth", "l2i8", "RESULTS row 15"),
    "l3i8": (RUN / "qat_it36l3i8" / "qat_124.pth", "l3i8", "RESULTS row 16"),
    "l2i8s4": (RUN / "qat_it36l2i8s4" / "qat_124.pth", "l2i8s4", "RESULTS row 17 (axis-1 best)"),
    "l3i8s4": (RUN / "qat_it36l3i8s4" / "qat_124.pth", "l3i8s4", "RESULTS row 18"),
    # Rows that pass the log-loss gate after 2026-11-21 23:00. The corruption check only ever runs on
    # a deploy candidate, so the chain guards each of these on its own sweep verdict (ub95 <= +0.0005
    # on both heads) before spending GPU time here.
    "l2i4s4": (RUN / "qat_it36l2i4s4" / "qat_124.pth", "l2i4s4", "RESULTS row 23"),
    "l3i4s4": (RUN / "qat_it36l3i4s4" / "qat_124.pth", "l3i4s4", "chain 11 point 1"),
    "l2i8s2": (RUN / "qat_it36l2i8s2" / "qat_124.pth", "l2i8s2", "chain 11 point 2"),
    "n1i8s3": (RUN / "qat_it36n1i8s3" / "qat_124.pth", "n1i8s3", "chain 11 point 3"),
    "n2i4s3": (RUN / "qat_it36n2i4s3" / "qat_124.pth", "n2i4s3", "chain 11 point 4"),
    "l3i4s4n8s3": (RUN / "qat_it36l3i4s4n8s3" / "qat_124.pth", "l3i4s4n8s3", "chain 13 point 1"),
    "l3i4s4n3": (RUN / "qat_it36l3i4s4n3" / "qat_124.pth", "l3i4s4n3", "chain 13 point 2"),
    "l2i4s4n8s3": (RUN / "qat_it36l2i4s4n8s3" / "qat_124.pth", "l2i4s4n8s3", "chain 13 point 3"),
    "l3i4s4n4": (RUN / "qat_it36l3i4s4n4" / "qat_124.pth", "l3i4s4n4", "shrink point 1"),
    "l2i8s4n3": (RUN / "qat_it36l2i8s4n3" / "qat_124.pth", "l2i8s4n3", "shrink point 2"),
    "l2i8s2n4": (RUN / "qat_it36l2i8s2n4" / "qat_124.pth", "l2i8s2n4", "shrink point 3"),
    "l3i8s4n4s2": (RUN / "qat_it36l3i8s4n4s2" / "qat_124.pth", "l3i8s4n4s2", "shrink point 4"),
    "l2i8n2i8": (RUN / "qat_it36l2i8n2i8" / "qat_124.pth", "l2i8n2i8", "chain 14 point 1 (note rank-2 int8)"),
    "l3i8n2i8": (RUN / "qat_it36l3i8n2i8" / "qat_124.pth", "l3i8n2i8", "chain 14 point 2"),
    "l2i8n2i8s8": (RUN / "qat_it36l2i8n2i8s8" / "qat_124.pth", "l2i8n2i8s8", "chain 14 point 3"),
    "l2i8n3i8": (RUN / "qat_it36l2i8n3i8" / "qat_124.pth", "l2i8n3i8", "chain 14 point 4"),
    # FIXED-BASIS (codec) rows, chain 19 (2026-11-22): the note holds row 17's rank-1 payload size at
    # m=8 and beats it at m=6. Registered here so the check is one command once a sweep passes.
    "l2i8fb8n": (RUN / "qat_it36l2i8fb8n" / "qat_124.pth", "l2i8fb8n",
                 "chain 19 point 1 (fixed-basis note m=8, row 17's exact size)"),
    "l2i8fb6n": (RUN / "qat_it36l2i8fb6n" / "qat_124.pth", "l2i8fb6n",
                 "chain 19 point 2 (fixed-basis note m=6, 672 bits below row 17)"),
}


def users_of(path=ROSTER):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    users = doc["users"] if isinstance(doc, dict) else doc
    return [int(u) for u in users]


def load_results(tag):
    out = {}
    for mode, name in (("ahead", f"adw_{tag}.jsonl"), ("imm", f"adwP_{tag}.jsonl")):
        p = ROOT / "result" / name
        d = {}
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    d[int(r["user"])] = float(r["metrics"]["LogLoss"])
        out[mode] = d
    return out


def skips(tag):
    out = {}
    for kind in ("nanskip", "oomskip"):
        p = ROOT / "result" / f"adw_{tag}.{kind}.jsonl"
        if p.exists():
            out[kind] = sorted({int(json.loads(l)["user"])
                                for l in p.read_text(encoding="utf-8").splitlines() if l.strip()})
    return out


def run_one(name, users, device="cuda", limit=0, dry=False, log_suffix=""):
    model, cand, note = CONFIGS[name]
    if not Path(model).exists():
        raise SystemExit(f"{name}: checkpoint missing: {model}")
    env_extra, comment, _ = qs.CANDIDATES[cand]
    tag = f"corrupt_{name}{log_suffix}"
    cfg = RUN / f"eval_{tag}.toml"
    probe_dir = HERE / "probe_out" / tag
    env = qs.base_env()
    env.update({k: str(v) for k, v in env_extra.items()})
    env.update({
        "RWKV_NO_JIT": "1",
        # The Python per-step QAT loop is the ONLY place the state is visible in Python; it is the
        # documented bit-parity twin of the fused CUDA kernels (rwkv_ops.quant_aware_rwkv7).
        "RWKV_QAT_FUSED": "0",
        "RWKV_FINITE_PROBE": "1",
        "RWKV_FINITE_PROBE_DIR": str(probe_dir),
        "RWKV_EXIT_HARD": "0",
    })
    if limit:
        users = users[:limit]
    ad.eval_config(BASE_TOML, cfg, model, tag, users, qs.VAL_DB, qs.LABEL_DB)
    if device != "cuda":
        edits = {"DEVICE": device, "DTYPE": "float"}
        ad.edit_toml(cfg, edits)
        env["RWKV_EVAL_CAST_FP32"] = "1"
    cmd = [str(PY), "-u", str(HERE / "probe_get_result.py"), "--config", str(cfg)]
    print(f"\n=== {name}: {note}\n    model={model}\n    cand={cand} env={env_extra}\n"
          f"    users={len(users)} device={device} tag={tag}\n    cmd={' '.join(cmd)}", flush=True)
    if dry:
        return {"name": name, "tag": tag, "dry_run": True, "users": len(users)}
    probe_dir.mkdir(parents=True, exist_ok=True)
    log = HERE / f"run_{tag}.log"
    t0 = time.monotonic()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} env={env_extra} users={users} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    secs = time.monotonic() - t0
    verdict = finite_probe.verdict_of(sorted(probe_dir.glob("probe_*.jsonl")))
    ll = load_results(tag)
    n_ll = len(ll["ahead"])
    means = {m: (sum(d.values()) / len(d) if d else None) for m, d in ll.items()}
    sk = skips(tag)
    # A missing observation is NOT a pass: without the Python QAT loop (RWKV_QAT_FUSED=0) the state
    # never passes through Python, and an empty probe file means the probe was never exercised.
    # The fp32 control has no quantized stream at all, so state observation is not expected there.
    state_expected = cand != "nj"
    problems = []
    if rc != 0:
        problems.append(f"rc={rc}")
    if verdict["n_users"] == 0:
        problems.append("no probe records (probe not exercised)")
    if verdict["p_calls"] == 0:
        problems.append("no p(recall) tensors observed")
    if state_expected and verdict["state_calls"] == 0:
        problems.append("no state observed (is the Python QAT loop active?)")
    if n_ll < len(users):
        problems.append(f"scored {n_ll}/{len(users)} users")
    if sk.get("nanskip"):
        problems.append(f"eval NaN-guard skips: {sk['nanskip']}")
    if verdict["state_bad"] or verdict["p_bad"]:
        problems.append(f"non-finite: state_bad={verdict['state_bad']} p_bad={verdict['p_bad']}")
    status = "FAIL" if (verdict["state_bad"] or verdict["p_bad"]
                        or sk.get("nanskip")) else ("PASS" if not problems else "INCOMPLETE")
    note_txt = ("state observed on the Python QAT path" if state_expected else
                "no quantized stream in this config: the state path is the plain CUDA kernel, so only "
                "p(recall) is observable here (the run is the attribution control)")
    rec = {
        "name": name, "tag": tag, "row": note, "model": str(model), "candidate": cand,
        "env": {k: str(v) for k, v in env_extra.items()}, "rc": rc, "secs": secs,
        "users_requested": len(users), "users_scored": n_ll, "mean_logloss": means,
        "skips": sk, "log": str(log), "probe_dir": str(probe_dir), "status": status,
        "problems": problems, "state_observed": bool(state_expected), "state_note": note_txt,
        "state_calls": verdict["state_calls"], "state_bad": verdict["state_bad"],
        "p_bad": verdict["p_bad"], "p_calls": verdict["p_calls"],
        "pass": verdict["pass"], "offenders": verdict["offenders"],
        "totals": verdict["total"], "per_user": verdict["per_user"],
    }
    out = HERE / f"corruption_{tag}.json"
    slim = dict(rec)
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"    rc={rc} {secs/60:.1f} min  scored={n_ll}/{len(users)}  "
          f"state calls={rec['state_calls']:,} bad={rec['state_bad']}  "
          f"p calls={rec['p_calls']:,} bad={rec['p_bad']}  -> {status}", flush=True)
    if problems:
        print(f"    problems: {problems}", flush=True)
    if means["ahead"]:
        print(f"    mean LogLoss ahead={means['ahead']:.6f} imm={means['imm']:.6f}", flush=True)
    if rec["offenders"]:
        print(f"    OFFENDERS ({len(rec['offenders'])} users): "
              f"{json.dumps(rec['offenders'])[:1200]}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--tag", default="", help="alias for --configs (one name)")
    ap.add_argument("--users-file", default=str(ROSTER))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = [n.strip() for n in (args.tag or args.configs).split(",") if n.strip()]
    for n in names:
        if n not in CONFIGS:
            raise SystemExit(f"unknown config {n}; known: {sorted(CONFIGS)}")
    users = users_of(args.users_file)
    print(f"roster={args.users_file} users={len(users)} device={args.device} "
          f"configs={names}", flush=True)
    out = []
    for n in names:
        out.append(run_one(n, users, args.device, args.limit, args.dry_run, args.log_suffix))
    bad = [r["name"] for r in out if r.get("status") == "FAIL"]
    inc = [r["name"] for r in out if r.get("status") == "INCOMPLETE"]
    print("\nSUMMARY: " + ", ".join(
        f"{r['name']}={r.get('status', 'dry')}" for r in out))
    if inc:
        print(f"INCOMPLETE (not a pass): {inc}")
    if bad:
        print(f"CORRUPTION CHECK FAILED for: {bad}")
    return 1 if (bad or inc) else 0


if __name__ == "__main__":
    sys.exit(main())
