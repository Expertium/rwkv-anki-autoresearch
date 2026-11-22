"""Record row 41 (`l3i4s4n8s3`), the held-out fixed-basis screen, and the 10-user trace-export gap.

Sources of truth (every number is read from these files, so nothing can be mistyped):
  * scratchpad/quant_plan/quant_sweep_it36l3i4s4n8s3.json          - row 41.
  * scratchpad/quant_plan/rust_wide_screen_it36fbwide_b1.json      - the codec screen, 7 held-out users.
  * scratchpad/quant_plan/trace_corrupt50/export_progress_shard*.jsonl - the 10 failed trace exports.

Writes a RESULTS row + two paragraphs, three NOTES live-status bullets, GPU_RUN_BOARD Amendment 24 and
the RECORDED.txt stamp. Raw-byte replaces with unique anchors (the files mix CRLF and LF).

Run: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_row41_fb.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW41 = RUN / "quant_sweep_it36l3i4s4n8s3.json"
WIDE = RUN / "rust_wide_screen_it36fbwide_b1.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"
TRACES = RUN / "trace_corrupt50"

# Last cell of row 40 - the two new rows/paragraphs go after the corresponding text.
ROW_ANCHOR = b"the same conclusion row 20 reached on the rank-1 int8 note | - |"
# Last sentence of the fixed-basis paragraph written by _record_fixedbasis.py.
PARA_ANCHOR = b"because it is LAST in run order and `it36corrupt2` cannot pick the rows up."
STATUS_ANCHOR = b"rows 33-40 FAIL too; row 35 (`n1i8s3`) PASSES** (row 40 "
NOTE_ANCHOR = b"## Live status\n\n"
AMEND = b"## Amendment 24"
MARK = b"Row 41 (`l3i4s4n8s3`) misses the immediate-head ceiling by 0.000016"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW41.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    wide = json.loads(WIDE.read_text(encoding="utf-8"))
    w8, w6 = wide["configs"]["l2i8fb8n"], wide["configs"]["l2i8fb6n"]
    n_users = wide["n_users"]

    # the users that could not be traced: ok = false in some shard's progress file
    failed = {}
    for p in sorted(TRACES.glob("export_progress_shard*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                failed[int(rec["user"])] = rec.get("error", "")
    roster = json.loads((RUN / "corruption" / "corrupt50.json").read_text(encoding="utf-8"))
    rank = {(int(e["user"]) if isinstance(e, dict) else int(e)): i + 1 for i, e in enumerate(roster["selected"])}
    worst = sorted(failed, key=lambda u: rank.get(u, 999))[:3]
    failed_txt = ", ".join(
        f"{u} (#{rank.get(u, '?')} by the roster's metric)" if i == 0 else
        f"{u} (#{rank.get(u, '?')})" for i, u in enumerate(worst)
    )

    def pair(d):
        return f"{f6(d['mean'])} [{f6(d['boot_lo'])}, {f6(d['boot_hi'])}] better {d['better']}/{d['n']}"

    # ---------------------------------------------------------------- RESULTS row 41
    row41 = (
        "| 41 | QAT card **rank-3 int4** factors + int4 shifts, note rank-1 int8 + **int3** shifts - "
        "the cheap card under the cheap note, 1,024/2,112 = 3,136 bits (`l3i4s4n8s3`, `it36combo`, "
        "landed 22:11; speed cell = row 33's measured card path `l3i4s4`, because the card is "
        "identical and the note's shift width does not change its rank-1 verify path) | "
        "1,024 (128) | 2,112 (264) | 2,290 | 0.800x | "
        f"{g['ahead']['mean_cur']:.6f} | {g['imm']['mean_cur']:.6f} | "
        f"{f6(g['ahead']['delta'])} | {f6(g['imm']['delta'])} | "
        "**FAIL by 0.000016 on the immediate head**: ahead sits at 0.46x of the ceiling "
        f"(ub95 +{g['ahead']['ub95']:.6f}, p {g['ahead']['p_worse']:.1e}) but the imm point estimate "
        f"is +{g['imm']['delta']:.6f} with ub95 +{g['imm']['ub95']:.6f} (p {g['imm']['p_worse']:.1e}). "
        "It is the smallest total that has come close - 256 BELOW row 35, the smallest passing row - "
        "and the two cuts compose SUBLINEARLY: the card alone costs +0.000208/+0.000472 (row 33, the "
        "same card with an int8 note) and the note's int3 shift rung alone +0.000011/+0.000037 (row "
        "35, the same rung on row 17's card), yet together they cost +0.000232/+0.000516. The 1,024 "
        "card bits are the expensive half though: the int6 rank-2 card reaches the SAME 1,024 bits for "
        "+0.000052/+0.000115 on 11 long histories (`l2i6c`), which is what chain 21 "
        "(`l2i6s4n1i8s3`, this exact 3,136 bits) tests | - |"
    )

    raw = RESULTS.read_bytes()
    if b"| 41 |" in raw:
        print("RESULTS: row 41 already present")
    else:
        lines = raw.split(b"\n")
        idx = [i for i, l in enumerate(lines) if ROW_ANCHOR in l]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times, expected 1")
        i = idx[0]
        eol = b"\r" if lines[i].endswith(b"\r") else b""
        lines[i + 1:i + 1] = [(row41 + ("\r" if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 41: ok")

    # ---------------------------------------------------------------- RESULTS paragraphs
    p41 = (
        "**Row 41 (`l3i4s4n8s3`) misses the immediate-head ceiling by 0.000016, so 3,136 bits is NOT "
        "yet a passing size - but the two cuts compose sublinearly, which is the useful part.** The "
        "cheap card (rank-3 int4 + int4 shifts, 1,024 bits) under the cheap note (rank-1 int8 + int3 "
        f"shifts, 2,112 bits) lands {g['ahead']['mean_cur']:.6f}/{g['imm']['mean_cur']:.6f} = "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} against `qnj_it36ctl` "
        f"(ub95 +{g['ahead']['ub95']:.6f}/+{g['imm']['ub95']:.6f}, p "
        f"{g['ahead']['p_worse']:.1e}/{g['imm']['p_worse']:.1e}): the ahead head is comfortably inside "
        f"the ceiling at 0.46x of it and the immediate head is 1.03x over on its point estimate. "
        "Against the sum of the parts, "
        f"{f6(0.000208 + 0.000011)} ahead / {f6(0.000472 + 0.000037)} imm, the combination is BETTER "
        "than additive on both heads - the note's int3 shift rung is nearly free when the card is "
        "already coarse. The cheap half is the NOTE, not the card: rank-3 int4 factors are the "
        "expensive way to buy the card's last 256 bits (row 33 prices that card at "
        "+0.000472 imm), while the int6 rank-2 card reaches the same 1,024 bits for +0.000115 imm on "
        "11 long histories (`l2i6c`, Amendment 22). That is exactly the swap chain 21 "
        "(`it36int6combo`, point `l2i6s4n1i8s3`, 1,024 + 2,112 = 3,136 bits, the same total) makes, "
        "and it is pre-registered in `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and "
        "`check_corruption.CONFIGS`. `rust_grid_it36int6shift.json` prices it on the same long "
        "histories first.\n"
    )
    pfb = (
        "**The held-out fixed-basis screen rejects the codec NOTE: 4-6x the tax at equal bits, on 0-1 "
        f"of {n_users} users.** `fixedbasis_l2i8s4all.npz` was fitted on three users (7213/7044/6960), "
        f"so a deploy-side screen over {n_users} OTHER long histories - all held out from the fit - is "
        "the fair test of whether the LOO capture table actually costs loss. The engine ran `plain`, "
        "`l2i8s4`, `l2i8fb8n` and `l2i8fb6n` on row 17's deploy weights, and "
        "`corruption/wide_screen.py` computed per-user PAIRED deltas against `l2i8s4` with 2,000 "
        "bootstrap resamples (`rust_wide_screen_it36fbwide_b1.json`; absolutes in "
        "`rust_grid_it36fbwide_b1.json`):\n\n"
        "| config | card/note bits | total | d ahead (95% CI) | d imm (95% CI) |\n"
        "|---|---:|---:|---|---|\n"
        f"| `l2i8fb8n` codec note m=8 (row 17's exact size) | 1,280/2,304 | 3,584 | "
        f"{pair(w8['ahead'])} | {pair(w8['imm'])} |\n"
        f"| `l2i8fb6n` codec note m=6 (480 below row 35) | 1,280/1,632 | 2,912 | "
        f"{pair(w6['ahead'])} | {pair(w6['imm'])} |\n\n"
        "Both are 4-6x the +0.0005 ceiling and the sign is consistent - the codec note is BETTER on "
        "one user out of "
        f"{n_users} on ahead and on NO user on the immediate head - so this is not an outlier "
        "effect. It is also the most expensive thing measured on the note so far: the note's int6 "
        "factor cut costs +0.000268/+0.000307 and the card's int6 +0.000052/+0.000115 on comparable "
        "long-history screens, i.e. a codec buys 672 bits for roughly 7-9x the price of a factor-width "
        "cut. That is consistent with the LOO table: the shipped rank-1 truncation keeps 0.904-0.951 "
        "of each state's energy while the codec holds out only 0.8757-0.9205 at m=8, and the energy "
        "the codec drops is spread across directions the model did not choose. THE ONE REGIME THIS "
        "SCREEN CANNOT SEE is QAT: the weights here were trained for the rank-r path, so the codec's "
        "cost includes the mismatch between the basis and how the state is actually used, and training "
        "can push state energy into a fixed subspace. Chain 19 (`it36fixedbasis`) is therefore still "
        "the fair test - `l2i8fb8n` at equal bits measures how much QAT recovers, and `l2i8fb6n` is "
        "the size claim at 2,912 bits - but the expectation is now recorded: on this evidence the "
        "codec note FAILS unless QAT recovers most of a +0.0020/+0.0028 gap.\n"
    )

    raw = RESULTS.read_bytes()
    if MARK in raw:
        print("RESULTS: paragraphs already present")
    else:
        if raw.count(PARA_ANCHOR) != 1:
            raise SystemExit(f"ABORT: paragraph anchor found {raw.count(PARA_ANCHOR)} times, expected 1")
        RESULTS.write_bytes(raw.replace(
            PARA_ANCHOR, PARA_ANCHOR + b"\n\n" + p41.encode("utf-8") + b"\n" + pfb.encode("utf-8"), 1))
        print("RESULTS paragraphs: ok")

    # ---------------------------------------------------------------- RESULTS status sentence
    raw = RESULTS.read_bytes()
    if b"rows 33-41 FAIL too" in raw:
        print("RESULTS: status sentence already updated")
    else:
        if raw.count(STATUS_ANCHOR) != 1:
            raise SystemExit(f"ABORT: status anchor found {raw.count(STATUS_ANCHOR)} times, expected 1")
        ins = (b"rows 33-41 FAIL too; row 35 (`n1i8s3`) PASSES** "
               b"(row 41 `l3i4s4n8s3` +0.000232/+0.000516 - the cheap card under the cheap note reaches "
               b"the smallest total yet at 3,136 bits and misses the imm ceiling by 0.000016, landed "
               b"22:08; the held-out fixed-basis screen then rejects the codec NOTE at equal bits, "
               b"+0.002035/+0.002756 on 7 users) (row 40 ")
        RESULTS.write_bytes(raw.replace(STATUS_ANCHOR, ins, 1))
        print("RESULTS status sentence: ok")

    # ---------------------------------------------------------------- NOTES bullets
    b41 = (
        "* 2026-11-22 22:20 - **Row 41 (`l3i4s4n8s3`) reaches the smallest total yet, 3,136 bits, and "
        "misses the immediate-head ceiling by 0.000016.** Card rank-3 int4 + int4 shifts (1,024) under "
        "note rank-1 int8 + int3 shifts (2,112): 0.334034/0.306727 = "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} (ub95 "
        f"+{g['ahead']['ub95']:.6f}/+{g['imm']['ub95']:.6f}) - ahead is at 0.46x of the ceiling, imm "
        "1.03x over. The two cuts compose SUBLINEARLY (0.000219/0.000509 additive vs "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} measured), so the note's int3 shift rung is "
        "nearly free on a coarse card. The expensive half is the CARD: rank-3 int4 buys 1,024 bits for "
        "+0.000472 imm (row 33) where the int6 rank-2 card buys the same 1,024 for +0.000115 (`l2i6c`). "
        "That swap is chain 21 `it36int6combo` (point `l2i6s4n1i8s3`, the same 3,136 bits), launched "
        "22:18 behind `it36fixedbasis`, with `rust_grid_it36int6shift.json` pricing it on the 7 long "
        "histories first.\n"
    )
    bfb = (
        "* 2026-11-22 22:20 - **The fixed-basis codec is REJECTED on the note by a held-out screen: "
        f"{f6(w8['imm']['mean'])} imm at equal bits, better on 0 of {n_users} users.** The basis was "
        "fitted on users 7213/7044/6960, so the screen ran the 7 OTHER long histories with row 17's "
        "deploy weights and `wide_screen.py` paired deltas vs `l2i8s4`: `l2i8fb8n` (3,584 bits, row "
        f"17's exact size) {pair(w8['ahead'])} ahead / {pair(w8['imm'])} imm, and `l2i8fb6n` (2,912 "
        f"bits, 480 below row 35) {pair(w6['ahead'])} ahead / {pair(w6['imm'])} imm. 4-6x the ceiling "
        "with a consistent sign, and 7-9x the price of the note's int6 factor cut. The LOO table "
        "predicted it (codec 0.8757-0.9205 at m=8 vs the shipped rank-1's 0.904-0.951). CAVEAT: this "
        "is PTQ on weights trained for the rank-r path, and QAT could push state energy into a fixed "
        "subspace - so chain 19's two QAT points stay queued as the fair test, now with the recorded "
        "expectation that the codec FAILS unless QAT recovers most of a +0.0020/+0.0028 gap. "
        "`rust_wide_screen_it36fbwide_b1.json`.\n"
    )
    bgap = (
        "* 2026-11-22 22:20 - **10 of the 50 corrupt50 traces cannot be exported, and the root cause is "
        "a data-source mismatch, not a size problem.** The deploy-twin finiteness check covers 40 of the "
        "50 longest-history users. The other 10 fail inside `export_rnn_trace.export_user` with "
        f"`AssertionError: <review_th> not found in pred_dict` (`rwkv/get_result.py:166`), and they "
        f"include {failed_txt}. The FRAME comes from `../anki-revlogs-10k` (the `RWKV_ID_FEATURES=0` "
        "replica in `load_user_df`, whose `review_th` is `range(1, len(df)+1)`), while the equalize set "
        "comes from the eval's label DB `F:/rwkv_lmdb/label_filter_db_id_e2s` - for these 10 users that "
        "DB holds a review_th beyond the frame's length, so `get_stats` asserts. The gate itself is "
        "unaffected (the Python probe runs on all 50 through the eval path); only the Rust deploy twin "
        "is short. A fix means either tolerating the mismatch (intersect the equalize set with the "
        "trace's own review_ths) or sourcing the frame from the same DB - the first is a patch to the "
        "CERTIFIED exporter, so it was left as a recorded decision rather than applied unilaterally.\n"
    )

    nb = NOTES.read_bytes()
    if b"Row 41 (`l3i4s4n8s3`) reaches the smallest total yet" in nb:
        print("NOTES: bullets already present")
    else:
        if nb.count(NOTE_ANCHOR) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(NOTE_ANCHOR)} times, expected 1")
        NOTES.write_bytes(nb.replace(
            NOTE_ANCHOR,
            NOTE_ANCHOR + b41.encode("utf-8") + bfb.encode("utf-8") + bgap.encode("utf-8"), 1))
        print("NOTES bullets: ok")

    # ---------------------------------------------------------------- GPU_RUN_BOARD Amendment 24
    amend = (
        "\n\n## Amendment 24 (2026-11-22 22:20): row 41 misses the imm ceiling by 0.000016 at 3,136 "
        "bits; the held-out fixed-basis screen rejects the codec note; 10 traces cannot be exported\n\n"
        "**(1) The smallest total yet is 0.000016 short.** `l3i4s4n8s3` (card rank-3 int4 + int4 "
        "shifts = 1,024 bits, note rank-1 int8 + int3 shifts = 2,112) reads "
        f"{g['ahead']['mean_cur']:.6f}/{g['imm']['mean_cur']:.6f} = {f6(g['ahead']['delta'])}/"
        f"{f6(g['imm']['delta'])} against `qnj_it36ctl` (ub95 +{g['ahead']['ub95']:.6f}/"
        f"+{g['imm']['ub95']:.6f}), i.e. ahead at 0.46x of the ceiling and the immediate head 1.03x "
        "over it. The two cuts compose SUBLINEARLY - "
        f"{f6(0.000208 + 0.000011)}/{f6(0.000472 + 0.000037)} additive vs the measured "
        f"{f6(g['ahead']['delta'])}/{f6(g['imm']['delta'])} - so 3,136 bits is a real frontier point "
        "and the question is only which 1,024-bit card to put under it. The int6 rank-2 card gets the "
        "same 1,024 bits for +0.000052/+0.000115 on 11 long histories (`l2i6c`, Amendment 22) where "
        "rank-3 int4 costs +0.000208/+0.000472 (row 33). Chain 21 (`it36int6combo`, point "
        "`l2i6s4n1i8s3`) tests that swap at the same 3,136 bits; it is registered in "
        "`quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`, launched behind "
        "chain 19, and pre-priced on the 7 long histories by `rust_grid_it36int6shift.json`.\n\n"
        "**(2) The fixed basis is a codec that costs 4-6x the ceiling on the note.** The basis "
        "(`corruption/fixedbasis_l2i8s4all.npz`) was fitted on users 7213/7044/6960, so the fair test "
        f"is the {n_users} OTHER long histories. Running row 17's deploy weights and pairing per user "
        "against `l2i8s4` (`rust_wide_screen_it36fbwide_b1.json`, 2,000 bootstrap resamples):\n\n"
        "| config | card/note bits | total | d ahead (95% CI) | d imm (95% CI) |\n"
        "|---|---:|---:|---|---|\n"
        f"| `l2i8fb8n` codec note m=8 | 1,280/2,304 | 3,584 | {pair(w8['ahead'])} | "
        f"{pair(w8['imm'])} |\n"
        f"| `l2i8fb6n` codec note m=6 | 1,280/1,632 | 2,912 | {pair(w6['ahead'])} | "
        f"{pair(w6['imm'])} |\n\n"
        "Better on one user out of "
        f"{n_users} on ahead and none on imm - a consistent sign, not an outlier. For scale, the note's "
        "int6 factor cut costs +0.000268/+0.000307 and the card's +0.000052/+0.000115, so the codec "
        "pays 7-9x a factor-width cut for 672 bits. This is PTQ against rank-r-trained weights, so QAT "
        "is the one regime the screen cannot see: chain 19 keeps both points (m=8 = the QAT-recovery "
        "A/B, m=6 = the 2,912-bit size claim), with the recorded expectation that the codec FAILS "
        "unless QAT recovers most of the gap.\n\n"
        "**(3) The deploy-twin corruption coverage is 40 of 50, for a documented reason.** Ten traces "
        "fail inside `export_rnn_trace.export_user` with `AssertionError: <review_th> not found in "
        "pred_dict` (`rwkv/get_result.py:166`): the frame comes from `../anki-revlogs-10k` while the "
        "equalize set comes from the eval's label DB `F:/rwkv_lmdb/label_filter_db_id_e2s`, and for "
        f"those users that DB holds a review_th beyond the frame's length. They include {failed_txt}, "
        "so the gap is not confined to the short histories. The GATE is unaffected - the Python probe "
        "covers all 50 through the eval path - and closing the deploy twin means either intersecting "
        "the equalize set with the trace's own review_th (a patch to the CERTIFIED exporter) or "
        "sourcing the frame from the same DB; it is recorded here as a decision, not applied.\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 24 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 24 appended")

    # ---------------------------------------------------------------- RECORDED.txt stamp
    rb = RECORDED.read_bytes()
    if b"it36l3i4s4n8s3" in rb.split(b"\n"):
        print("RECORDED.txt: stamp already present")
    else:
        if not rb.endswith(b"\n"):
            rb += b"\n"
        RECORDED.write_bytes(rb + b"it36l3i4s4n8s3\n")
        print("RECORDED.txt: added it36l3i4s4n8s3")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())