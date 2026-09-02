"""Everything the ~7 h generation-4 chain assumes, checked BEFORE it starts unattended.

The chain is label filter (~3 h) -> train db (~2 h) -> test db (~2 h) -> comparisons, and it
fires on featB's terminal marker at roughly 06:10 with nobody watching. A wrong path or a full
disk discovered at hour three costs the night; discovered now it costs nothing. Every check here
is one an earlier failure in this repo actually taught:

  * targets must not already exist -- data_processing SKIPS users already present, so a
    pre-existing db "succeeds" in seconds having done nothing (gen 3's own header records this);
  * gen 3 must still exist -- phase 3 compares against it;
  * the label filter must be a NEW path -- writing into label_filter_db_id would silently re-base
    featB, which is scored against it;
  * disk, measured with free space rather than file length, because these stores are SPARSE and
    `Get-ChildItem | Measure Length` reports the map_size reservation as if it were allocation.

Usage: preflight_gen4.py       (read-only; exit 0 = clear to run)
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
    print("GEN-4 CHAIN PREFLIGHT")
    print("=" * 80)

    # ---- configs ----
    print("\n[configs]")
    tr = "rwkv/data_processing_train_5k_h1_id4.toml"
    te = "rwkv/data_processing_test_5k_h2_id4.toml"
    lf = "rwkv/find_equalize_id_e2s.toml"
    for p in (tr, te, lf):
        check(os.path.exists(p), "%s exists" % p)
    if failures:
        return 1

    lf_path = toml_get(lf, "LABEL_FILTER_LMDB_PATH")
    tr_lf = toml_get(tr, "LABEL_FILTER_LMDB_PATH")
    te_lf = toml_get(te, "LABEL_FILTER_LMDB_PATH")
    tr_db = toml_get(tr, "LMDB_PATH")
    te_db = toml_get(te, "LMDB_PATH")

    check(lf_path.endswith("label_filter_db_id_e2s"), "label filter target is the NEW e2s path")
    check(tr_lf == lf_path and te_lf == lf_path,
          "both gen-4 dbs point at the SAME new label filter")
    check("id4" in tr_db and "id4" in te_db, "both dbs are id4 targets")
    check(toml_get(tr, "DATA_PATH") == toml_get(te, "DATA_PATH") == "../anki-revlogs-10k-id",
          "both dbs read the -id dataset")
    check(toml_get(lf, "USER_START") == 1 and toml_get(lf, "USER_END") == 10000,
          "label filter covers users 1-10000 (train half AND test half)")

    # ---- the writing-into-a-live-store trap ----
    print("\n[targets must be NEW]")
    for label, p in (("label filter", lf_path), ("train db", tr_db), ("test db", te_db)):
        exists = os.path.isdir(p)
        check(not exists, "%s does not already exist (%s)" % (label, p))
        if exists:
            print("      ^ data_processing/find_equalize SKIP users already present, so an")
            print("        existing store reports success in seconds having done nothing.")

    print("\n[gen 3 must survive -- phase 3 compares against it, featB is scored on it]")
    for p in ("F:/rwkv_lmdb/train_db_5k_h1_id3", "F:/rwkv_lmdb/test_db_5k_id3",
              "F:/rwkv_lmdb/label_filter_db_id"):
        check(os.path.isdir(p), "%s still present" % p)

    # ---- data ----
    print("\n[source data]")
    root = os.path.abspath("../anki-revlogs-10k-id/revlogs")
    check(os.path.isdir(root), "-id revlogs readable (%s)" % root)
    if os.path.isdir(root):
        n = sum(1 for d in os.listdir(root) if d.startswith("user_id="))
        check(n >= 10000, "-id has >=10,000 user partitions (found %d)" % n)

    # ---- disk, by FREE SPACE: these stores are sparse, so file length is fiction ----
    print("\n[disk]")
    free_gb = shutil.disk_usage("F:/").free / (1024 ** 3)
    need_gb = 232 + 40  # ~116 GiB per db (measured, not the map_size) + label filter headroom
    check(free_gb > need_gb, "F: free %.0f GiB > %d GiB needed" % (free_gb, need_gb))
    print("      (measured by free space -- these are SPARSE, so file length reports the")
    print("       map_size reservation, not the allocation)")

    # ---- the code the chain will import ----
    print("\n[imports]")
    try:
        os.environ["RWKV_ID_FEATURES"] = "1"
        import rwkv.id_features as idf
        w = idf.card_feature_width()
        check(w == 46, "RWKV_ID_FEATURES=1 gives 46 feature columns (got %d)" % w)
    except Exception as exc:  # noqa: BLE001
        check(False, "id_features imports clean (%s: %s)" % (type(exc).__name__, exc))

    # `find_equalize_test_reviews` parses --config AT IMPORT, so importing it bare raises
    # SystemExit from argparse -- which `except Exception` does not catch, so the first version of
    # this preflight died here without printing its verdict. A checker that can exit before
    # reporting is worse than no checker: it looks like a crash rather than a result.
    saved = sys.argv[:]
    try:
        sys.argv = ["find_equalize_test_reviews", "--config", lf]
        import rwkv.find_equalize_test_reviews  # noqa: F401
        check(True, "find_equalize_test_reviews imports with the chain's own config")
    except BaseException as exc:  # noqa: BLE001  -- SystemExit is not an Exception
        check(False, "find_equalize imports (%s: %s)" % (type(exc).__name__, exc))
    finally:
        sys.argv = saved

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
        print("PREFLIGHT PASSED -- the gen-4 chain's assumptions all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
