"""Everything the generation-5 chain assumes, checked BEFORE it starts unattended.

Adapted from preflight_gen4.py; the checks that differ are the point:

  * NO label-filter phase -- gen 5 REUSES label_filter_db_id_e2s, so the check is that it EXISTS
    and that both gen-5 configs point at it (a fresh path here would silently re-select the
    scored set and break the size gate for the realcyc-vs-gen4base pair);
  * the flag must reach the process: RWKV_REAL_CYCLES=1 with RWKV_ID_FEATURES=1 must give 69
    card-feature columns and 109 input dims. A build without it reproduces gen 4 under a gen-5
    name and passes every count check -- which is why this is asserted on the WIDTH, the one
    number the flag changes;
  * gen 4 must survive -- phase 3 compares against it and it is the realcyc run's control.

Usage: preflight_gen5.py       (read-only; exit 0 = clear to run)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

failures = []
warnings = []


def check(cond, label, hard=True):
    print("  %-64s %s" % (label, "PASS" if cond else ("*** FAIL ***" if hard else "warn")))
    if not cond:
        (failures if hard else warnings).append(label)


def toml_get(path, key):
    import tomli
    with open(path, "rb") as f:
        return tomli.load(f).get(key)


def main():
    print("=" * 80)
    print("GEN-5 CHAIN PREFLIGHT (real-time cycles)")
    print("=" * 80)

    print("\n[configs]")
    tr = "rwkv/data_processing_train_5k_h1_id5.toml"
    te = "rwkv/data_processing_test_5k_h2_id5.toml"
    for p in (tr, te):
        check(os.path.exists(p), "%s exists" % p)
    if failures:
        return 1
    tr_db, te_db = toml_get(tr, "LMDB_PATH"), toml_get(te, "LMDB_PATH")
    tr_lf, te_lf = toml_get(tr, "LABEL_FILTER_LMDB_PATH"), toml_get(te, "LABEL_FILTER_LMDB_PATH")
    check("id5" in tr_db and "id5" in te_db, "both dbs are id5 targets")
    check(tr_lf == te_lf and tr_lf.endswith("label_filter_db_id_e2s"),
          "both dbs REUSE label_filter_db_id_e2s (same scored set as gen 4)")
    check(toml_get(tr, "DATA_PATH") == toml_get(te, "DATA_PATH") == "../anki-revlogs-10k-id",
          "both dbs read the -id dataset")

    print("\n[the reused label filter must exist; the targets must NOT]")
    check(os.path.isdir(tr_lf), "%s present (reused, not rebuilt)" % tr_lf)
    for label, p in (("train db", tr_db), ("test db", te_db)):
        exists = os.path.isdir(p)
        check(not exists, "%s does not already exist (%s)" % (label, p))
        if exists:
            print("      ^ data_processing SKIPS users already present, so an existing store")
            print("        reports success in seconds having done nothing.")

    print("\n[gen 4 must survive -- phase 3 compares against it; it is realcyc's control]")
    for p in ("F:/rwkv_lmdb/train_db_5k_h1_id4", "F:/rwkv_lmdb/test_db_5k_id4"):
        check(os.path.isdir(p), "%s still present" % p)

    print("\n[source data]")
    root = os.path.abspath("../anki-revlogs-10k-id/revlogs")
    check(os.path.isdir(root), "-id revlogs readable (%s)" % root)
    if os.path.isdir(root):
        n = sum(1 for d in os.listdir(root) if d.startswith("user_id="))
        check(n >= 10000, "-id has >=10,000 user partitions (found %d)" % n)

    print("\n[disk, by FREE SPACE -- these stores are sparse, so file length is fiction]")
    free_gb = shutil.disk_usage("F:/").free / (1024 ** 3)
    need_gb = 232 + 20
    check(free_gb > need_gb, "F: free %.0f GiB > %d GiB needed" % (free_gb, need_gb))

    print("\n[the flag must reach the process -- asserted on the width, the number it changes]")
    try:
        os.environ["RWKV_ID_FEATURES"] = "1"
        os.environ["RWKV_REAL_CYCLES"] = "1"
        import rwkv.id_features as idf
        from rwkv.data_processing import CARD_FEATURE_COLUMNS as C
        w, iw = idf.card_feature_width(), idf.input_width()
        check(w == 69 and iw == 109, "RWKV_REAL_CYCLES=1: 69 card cols / 109 input (got %d / %d)" % (w, iw))
        check(list(C[-24:]) == list(idf.CYCLE_COLUMNS), "the 24 cycle columns are the tail of CARD_FEATURE_COLUMNS")
        check("day_of_week" not in C and "scaled_state" not in C,
              "day_of_week AND scaled_state are dropped from the input vector")
        check("dow_sin" in C and "doy_sin" in C and "cyc7_sin" not in C,
              "real dow/doy present; the review-time 7 d cycle is NOT duplicated")
    except Exception as exc:  # noqa: BLE001
        check(False, "id_features/data_processing import clean under the flags (%s: %s)" % (type(exc).__name__, exc))

    print("\n" + "=" * 80)
    if failures:
        print("PREFLIGHT FAILED (%d) -- do NOT let the chain start:" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    if warnings:
        print("PREFLIGHT PASSED with %d warning(s):" % len(warnings))
        for w_ in warnings:
            print("  - %s" % w_)
    else:
        print("PREFLIGHT PASSED -- the gen-5 chain's assumptions all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
