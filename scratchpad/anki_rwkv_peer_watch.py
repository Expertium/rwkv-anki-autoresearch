#!/usr/bin/env python3
"""Daily read-only commit watcher for JSchoreels/anki-rwkv.

Records new public commits and writes a Markdown report for manual overlap review.
It does not clone, modify, or execute anything from the peer repository.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "scratchpad" / "anki_rwkv_peer_watch_state.json"
REPORT_DIR = ROOT / "scratchpad" / "anki_rwkv_peer_reports"
API = "https://api.github.com/repos/JSchoreels/anki-rwkv/commits?per_page=100"


def fetch_commits() -> list[dict]:
    request = Request(
        API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "rwkv-anki-research-peer-watch",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected GitHub API response: {type(payload).__name__}")
    return payload


def main() -> int:
    now = datetime.now().astimezone()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    commits = fetch_commits()
    if not commits:
        raise RuntimeError("GitHub returned no commits")
    latest = commits[0]["sha"]
    previous = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else None
    if previous is None:
        new_commits = commits[:20]
        status = (
            "Initial snapshot: these recent commits establish the baseline; "
            "they are not new since a prior recorded check."
        )
    else:
        old_sha = previous.get("latest_sha")
        new_commits = []
        found_old = False
        for commit in commits:
            if commit.get("sha") == old_sha:
                found_old = True
                break
            new_commits.append(commit)
        if not found_old:
            status = (
                "The saved commit was not in the latest 100 entries; possible history "
                "rewrite or long gap. Showing the newest 20 for manual review."
            )
            new_commits = commits[:20]
        elif new_commits:
            status = f"Found {len(new_commits)} commit(s) since the previous check. Review them for research overlap."
        else:
            status = "No new commits since the previous check."
    stamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# JSchoreels/anki-rwkv daily commit check",
        "",
        f"Checked: {stamp}",
        f"Latest commit: `{latest[:12]}`",
        "",
        status,
        "",
        "This report lists public commit metadata only. It does not determine semantic overlap; "
        "compare relevant changes with this repository's research queue and source before drawing conclusions.",
        "",
    ]
    for item in new_commits:
        meta = item.get("commit", {})
        message = html.escape((meta.get("message") or "(no commit message)").splitlines()[0].replace("\r", " "))
        date_raw = (meta.get("author") or {}).get("date", "unknown date")
        try:
            date = parsedate_to_datetime(date_raw).astimezone().strftime("%Y-%m-%d %H:%M %Z")
        except (TypeError, ValueError):
            date = date_raw
        lines.extend(
            [
                f"## {date} — {message}",
                f"- Commit: `{item.get('sha', '')[:12]}`",
                f"- URL: {item.get('html_url', '')}",
                "",
            ]
        )
    report = REPORT_DIR / f"peer_check_{now.strftime('%Y%m%d_%H%M%S')}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "repository": "JSchoreels/anki-rwkv",
                "latest_sha": latest,
                "checked_at": now.isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"REPORT={report}")
    print(f"STATUS={status}")
    print(f"COMMITS_LISTED={len(new_commits)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"peer-watch failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
