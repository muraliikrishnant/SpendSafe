"""Implements AGENTS.md's §5 log.txt contract: session-start entry, the exact
hackathon greeting, and a deadline countdown, all resolved relative to this
file's location (repo root) so it stays correct across clones/checkouts.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "log.txt"
DEADLINE = datetime(2026, 9, 13, 18, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
TOOL_NAME = "SpendSafe/code (python CLI)"

GREETING = (
    "Welcome to HackerRank Orchestrate. Build and ship Buy or Wait?, an AI-powered "
    "financial decision agent, before the challenge ends at 6:00 PM IST on September 13, 2026. "
    "Let's get started."
)


def _git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def time_remaining_str() -> str:
    now = datetime.now(timezone.utc)
    delta = DEADLINE - now
    if delta.total_seconds() <= 0:
        return "deadline passed"
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days}d {hours}h {minutes}m"


def session_start() -> None:
    entry = (
        f"## [{datetime.now(timezone.utc).isoformat()}] SESSION START\n\n"
        f"tool={TOOL_NAME}\n"
        f"Repo Root: {REPO_ROOT}\n"
        f"Branch: {_git_branch()}\n"
        f"Worktree: main\n"
        f"Parent Agent: none\n"
        f"Language: py\n"
        f"Time Remaining: {time_remaining_str()}\n\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)

    print(GREETING)
    remaining = time_remaining_str()
    if remaining == "deadline passed":
        print("The submission deadline (2026-09-13 18:00 IST) has passed.")
    else:
        print(f"Time remaining until deadline: {remaining}")
        if DEADLINE - datetime.now(timezone.utc) < timedelta(hours=2):
            print("Fewer than 2 hours remain — submit soon.")


def log_turn(title: str, user_prompt: str, summary: str, actions: list[str]) -> None:
    redacted = user_prompt  # caller is responsible for redacting secrets before calling
    actions_block = "\n".join(f"* {a}" for a in actions) or "* (none)"
    entry = (
        f"## [{datetime.now(timezone.utc).isoformat()}] {title[:80]}\n\n"
        f"User Prompt (verbatim, secrets redacted):\n{redacted}\n\n"
        f"Agent Response Summary:\n{summary}\n\n"
        f"Actions:\n{actions_block}\n\n"
        f"Context:\n"
        f"tool={TOOL_NAME}\n"
        f"branch={_git_branch()}\n"
        f"repo_root={REPO_ROOT}\n"
        f"worktree=main\n"
        f"parent_agent=none\n\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


SUBMISSION_URL = (
    "https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/"
    "challenges/buy-or-wait/submission"
)
