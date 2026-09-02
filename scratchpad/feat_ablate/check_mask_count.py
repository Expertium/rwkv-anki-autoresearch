"""Did the feature mask reach the model, with the RIGHT number of dims?

`RWKV_ABLATE_FEATURES` raises on an unknown name, so a typo cannot pass silently. What CAN pass
silently is an EMPTY or partly-empty variable: cmd.exe expands an undefined `%VAR%` to nothing,
the arm then runs unmasked, and it reports "these features do not matter" for a treatment that
never happened. That is the vacuous-green shape, and it is why this checks the COUNT rather than
merely the presence of a banner.

Lives in a file rather than inline in the runner on purpose: the inline version needed `%%d`
inside a cmd subroutine that already uses `%~3`, which is precisely the escaping minefield this
repo has been bitten by (the REM-with-angle-brackets trap, the backslash-in-generated-content
trap). A script file has no cmd escaping at all.

Usage: check_mask_count.py <log> <expected_count>
Exit 0 = right count. 46 = wrong count or no banner.
"""
import io
import re
import sys

BANNER = re.compile(r"\[feat-mask\] zeroing input feature dims \[([^\]]*)\]")


def main():
    log, expected = sys.argv[1], int(sys.argv[2])
    text = io.open(log, encoding="utf-8", errors="replace").read()
    m = BANNER.search(text)
    if not m:
        print("[ablate] NO [feat-mask] banner -- the mask never reached the model")
        return 46
    dims = [d for d in m.group(1).split(",") if d.strip()]
    print("[ablate] masked %d dims (expected %d)" % (len(dims), expected))
    if len(dims) != expected:
        print("[ablate] *** WRONG COUNT -- the arm did not ablate what it claims")
        return 46
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
