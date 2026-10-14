"""Idle-only OpenHands heartbeat. Reads the API key from its protected user file."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
import requests
ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERSATION_ID = "7541a74f-3313-46e7-8126-33ae36253b97"
API_BASE = "http://localhost:18000"
KEY_FILE = pathlib.Path.home() / ".openhands" / "agent-canvas" / "api-key.txt"
STATE_FILE = ROOT / "scratchpad" / "context_heartbeat_state.json"
LOG_FILE = ROOT / "scratchpad" / "context_heartbeat.log"
QUIET_AFTER = dt.timedelta(hours=1)
POLL_SECONDS = 3600
NUDGE = (
    "Heartbeat: continue useful project work autonomously. Check active runs and logs. "
    "If GPU work is active, advance safe CPU tasks. If the GPU is idle and work remains, "
    "run the next safe experiment. Update the run board and loss breakdown after each run."
)
def get_headers() -> dict[str, str]:
    if not KEY_FILE.exists():
        raise RuntimeError(f"Missing protected Agent Canvas API key file: {KEY_FILE}")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Agent Canvas API key file is empty")
    return {"X-Session-API-Key": key}
def poll(*, dry_run: bool = False) -> str:
    headers = get_headers()
    response = requests.get(
        f"{API_BASE}/api/conversations/{CONVERSATION_ID}", headers=headers, timeout=10
    )
    response.raise_for_status()
    conversation = response.json()
    status = conversation.get("execution_status")
    now = dt.datetime.now(dt.timezone.utc)
    usage = conversation.get("stats", {}).get("usage_to_metrics", {}).get("default", {})
    samples = usage.get("token_usages") or []
    prompt_tokens = samples[-1].get("prompt_tokens") if samples else None
    context_window = samples[-1].get("context_window") if samples else None
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    last_user = state.get("last_user_message_id")
    current_user = conversation.get("last_user_message_id")
    if current_user == last_user and state.get("quiet_since"):
        quiet_since = dt.datetime.fromisoformat(state["quiet_since"])
    else:
        event_response = requests.get(
            f"{API_BASE}/api/conversations/{CONVERSATION_ID}/events/{current_user}",
            headers=headers,
            timeout=10,
        )
        event_response.raise_for_status()
        event = event_response.json()
        event_time = event.get("timestamp") or conversation["updated_at"]
        quiet_since = dt.datetime.fromisoformat(event_time.replace("Z", "+00:00"))
        if quiet_since.tzinfo is None:
            quiet_since = quiet_since.replace(tzinfo=dt.timezone.utc)
        state["quiet_since"] = quiet_since.isoformat()
        state["last_user_message_id"] = current_user
    quiet_for = max(dt.timedelta(), now - quiet_since)
    if status not in {"idle", "finished"}:
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        return f"No message: conversation status is {status!r}; quiet {quiet_for.total_seconds()/3600:.2f} h."
    if quiet_for < QUIET_AFTER:
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        return f"No message: status {status!r}; quiet {quiet_for.total_seconds()/3600:.2f} h; threshold is 1 h."
    if dry_run:
        return f"DRY RUN: would wake {status!r} conversation after {quiet_for.total_seconds()/3600:.2f} h; prompt {prompt_tokens}/{context_window}."
    sent = requests.post(
        f"{API_BASE}/api/conversations/{CONVERSATION_ID}/events",
        headers=headers,
        json={"role": "user", "content": [{"type": "text", "text": NUDGE}], "run": True},
        timeout=10,
    )
    sent.raise_for_status()
    state["last_heartbeat"] = now.isoformat()
    state["quiet_since"] = now.isoformat()
    state["last_user_message_id"] = "pending-heartbeat"
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return f"Heartbeat sent to {status!r} conversation after {quiet_for.total_seconds()/3600:.2f} h; prompt {prompt_tokens}/{context_window}."
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            line = f"{dt.datetime.now().astimezone().isoformat()} {poll(dry_run=args.dry_run)}"
            print(line, flush=True)
            if args.log:
                with LOG_FILE.open("a", encoding="utf-8") as log:
                    log.write(line + "\n")
        except Exception as exc:
            print(f"{dt.datetime.now().astimezone().isoformat()} ERROR {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)
if __name__ == "__main__":
    sys.exit(main())