"""Fix a factual slip in GPU_RUN_BOARD Amendment 31 (the int5 rung's imm is BETTER, not worse)."""
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "optimization" / "GPU_RUN_BOARD.md"

BEFORE = ("Note also that the int5 rung was bought at exactly the size of the closed int2-shift rung "
          "(2,560 bits) and is WORSE on the immediate head than that rung's +0.001642 (+0.001182 "
          "against +0.001642), so neither way of buying 2,560 bits works.")
AFTER = ("Note also that the int5 rung was bought at exactly the size of the closed int2-shift rung "
         "(2,560 bits): it is BETTER on the immediate head than that rung (+0.001182 against "
         "+0.001642) but 9x worse on ahead (+0.000948 against +0.000104), and both miss the imm "
         "ceiling any way round - so neither way of buying 2,560 bits works.")


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    if raw.count(BEFORE) != 1:
        print(f"FAIL: found {raw.count(BEFORE)} times")
        return 1
    out = raw.replace(BEFORE, AFTER, 1)
    P.write_bytes(out.encode("utf-8"))
    print("fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())