"""Update RESULTS' status sentence: row 45's finiteness check PASSED (it was written as "launched")."""
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "scratchpad" / "quant_plan" / "RESULTS.md"

BEFORE = (b"its finiteness check was "
          b"launched in the first free GPU window, as chain 17 ended")
AFTER = (b"its finiteness check PASSED in the first free GPU window, as chain 17 ended "
         b"(50/50 users, 158,986 state observations, 0 non-finite, 4.7 min)")


def main() -> int:
    raw = P.read_bytes()
    if raw.count(AFTER) == 1:
        print("already updated")
        return 0
    if raw.count(BEFORE) != 1:
        print(f"FAIL: anchor found {raw.count(BEFORE)} times")
        return 1
    P.write_bytes(raw.replace(BEFORE, AFTER, 1))
    print("status sentence updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())