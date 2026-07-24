# Reconstruct MAXIMUM-precision narrative git history by replaying Cursor
# transcript Write/StrReplace ops chronologically, then syncing to final tree.
#
# Warning: deletes .git and rebuilds. Final file bytes come from the live workspace.
#
# Usage: python tools/_reconstruct_git_history_v2.py

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = Path(
    r"C:\Users\REDACTED\.cursor\projects\c-Users-25959-Downloads-SleepyDetect\agent-transcripts"
)
CHAT_IDS = [
    "1b48f9d0-110b-49af-9141-ddae9aa7348b",
    "afcf99e5-3de8-4a8d-960e-e2e2ea524e4a",
    "0e81ac8c-eae7-4b3b-b402-2b80d9c88d2d",
    "ad41efed-cd71-4b89-a2f7-ea6562687f6b",
]
SNAP = ROOT / "_history_final_snapshot"
TS_RE = re.compile(
    r"<timestamp>\s*(.*?)\s*</timestamp>",
    re.I | re.S,
)
QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.I | re.S)

SKIP_NAME_PARTS = (
    "_patch_",
    "_dump_",
    "_fix_",
    "_test_fatigue_",
    "_process_image_snippet",
    ".cursor",
)

GITIGNORE = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""


@dataclass
class Op:
    seq: int
    when: datetime
    chat: str
    kind: str  # Write | StrReplace | Delete | User
    path: str | None
    contents: str | None = None
    old: str | None = None
    new: str | None = None
    query: str | None = None


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(args))
    return subprocess.run(args, cwd=ROOT, check=check)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *args], check=check)


def commit(title: str, body: str) -> bool:
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not staged:
        return False
    git("commit", "-m", title, "-m", body)
    return True


def parse_ts(text: str, fallback: datetime) -> datetime:
    m = TS_RE.search(text or "")
    if not m:
        return fallback
    raw = re.sub(r"\s*\(UTC[+-]\d+\)\s*", "", m.group(1)).strip()
    # Thursday, Jul 23, 2026, 2:47 PM
    for fmt in (
        "%A, %b %d, %Y, %I:%M %p",
        "%A, %B %d, %Y, %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return fallback


def to_repo_rel(path: str) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        # normalize case
        s = str(p)
        marker = "SleepyDetect"
        idx = s.lower().rfind(marker.lower())
        if idx < 0:
            return None
        # after SleepyDetect\
        rest = s[idx + len(marker) :].lstrip("\\/")
        if not rest or rest.startswith("."):
            return None
        if any(x in rest.replace("\\", "/") for x in ("agent-transcripts", ".cursor")):
            return None
        return rest.replace("\\", "/")
    except Exception:
        return None


def should_skip_path(rel: str | None) -> bool:
    if not rel:
        return True
    name = Path(rel).name
    low = rel.lower()
    if any(s.lower() in low for s in SKIP_NAME_PARTS):
        return True
    if name.endswith(".exe"):
        return True
    return False


def extract_ops() -> list[Op]:
    ops: list[Op] = []
    seq = 0
    for chat in CHAT_IDS:
        f = TRANSCRIPTS / chat / f"{chat}.jsonl"
        if not f.exists():
            continue
        # rough chat start
        when = datetime(2026, 7, 21, 12, 0, 0)
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            role = obj.get("role")
            msg = obj.get("message") or {}
            content = msg.get("content")

            texts: list[str] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text") or "")
            elif isinstance(content, str):
                texts.append(content)
            blob = "\n".join(texts)

            if role == "user" and blob:
                when = parse_ts(blob, when)
                q = QUERY_RE.search(blob)
                seq += 1
                ops.append(
                    Op(
                        seq=seq,
                        when=when,
                        chat=chat[:8],
                        kind="User",
                        path=None,
                        query=(q.group(1).strip() if q else blob[:200]),
                    )
                )
                continue

            # walk tool calls in assistant message
            def walk(x, depth=0):
                nonlocal seq, when
                if depth > 16:
                    return
                if isinstance(x, dict):
                    name = x.get("name") or x.get("toolName")
                    args = x.get("input") or x.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if name in ("Write", "StrReplace", "Delete") and isinstance(args, dict):
                        rel = to_repo_rel(args.get("path"))
                        if should_skip_path(rel):
                            pass
                        else:
                            seq += 1
                            if name == "Write":
                                contents = args.get("contents")
                                if contents is None:
                                    contents = args.get("content")
                                if isinstance(contents, str):
                                    ops.append(
                                        Op(
                                            seq=seq,
                                            when=when,
                                            chat=chat[:8],
                                            kind="Write",
                                            path=rel,
                                            contents=contents,
                                        )
                                    )
                            elif name == "StrReplace":
                                old = args.get("old_string")
                                new = args.get("new_string")
                                if isinstance(old, str) and isinstance(new, str) and rel:
                                    ops.append(
                                        Op(
                                            seq=seq,
                                            when=when,
                                            chat=chat[:8],
                                            kind="StrReplace",
                                            path=rel,
                                            old=old,
                                            new=new,
                                        )
                                    )
                            elif name == "Delete" and rel:
                                ops.append(
                                    Op(
                                        seq=seq,
                                        when=when,
                                        chat=chat[:8],
                                        kind="Delete",
                                        path=rel,
                                    )
                                )
                    for v in x.values():
                        walk(v, depth + 1)
                elif isinstance(x, list):
                    for i in x:
                        walk(i, depth + 1)

            if role == "assistant":
                walk(obj)

    ops.sort(key=lambda o: (o.when, o.seq))
    return ops


def ensure_gitkeep() -> None:
    for rel in (
        "fatigue_detection_system/media/results/.gitkeep",
        "fatigue_detection_system/media/uploads/.gitkeep",
    ):
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("", encoding="utf-8")


def copy_snapshot() -> None:
    if SNAP.exists():
        shutil.rmtree(SNAP)
    SNAP.mkdir(parents=True)
    # copy selected roots
    for name in (
        ".gitignore",
        "README.md",
        "requirements.txt",
        "data.yaml",
        "start.bat",
        "stop.bat",
        "stop.ps1",
        "dlib-19.19.0-cp38-cp38-win_amd64.whl",
        "fatigue_detection_system",
        "deploy",
        "tools",
        "datasets",
    ):
        src = ROOT / name
        if not src.exists():
            # chinese readme
            continue
        dst = SNAP / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".DS_Store",
                    "db.sqlite3",
                    "venv",
                ),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for md in ROOT.glob("*.md"):
        if md.name == "README.md":
            continue
        shutil.copy2(md, SNAP / md.name)


def restore_from_snapshot() -> None:
    # overlay snapshot onto ROOT for tracked areas
    for dirpath, _, filenames in os.walk(SNAP):
        rel_dir = Path(dirpath).relative_to(SNAP)
        for fn in filenames:
            s = Path(dirpath) / fn
            d = ROOT / rel_dir / fn
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def apply_str_replace(path: Path, old: str, new: str) -> bool:
    if not path.exists():
        return False
    try:
        text = read_text(path)
    except UnicodeDecodeError:
        return False
    if old not in text:
        return False
    # replace once like typical editor apply
    text2 = text.replace(old, new, 1)
    write_text(path, text2)
    return True


def reverse_str_replace(path: Path, old: str, new: str) -> bool:
    # undo: new -> old
    return apply_str_replace(path, new, old)


def build_initial_touched_state(ops: list[Op]) -> None:
    """Start from final snapshot files; reverse StrReplace/Write to approximate pre-agent state for touched files."""
    restore_from_snapshot()
    file_ops: dict[str, list[Op]] = {}
    for op in ops:
        if op.kind in ("Write", "StrReplace", "Delete") and op.path:
            file_ops.setdefault(op.path, []).append(op)

    for rel, fops in file_ops.items():
        path = ROOT / rel
        # reverse chronological within this file
        for op in reversed(fops):
            if op.kind == "StrReplace" and op.old is not None and op.new is not None:
                if path.exists():
                    reverse_str_replace(path, op.old, op.new)
            elif op.kind == "Write":
                # before this write: look for earlier write content
                earlier = [x for x in fops if x.seq < op.seq and x.kind == "Write" and x.contents is not None]
                if earlier:
                    write_text(path, earlier[-1].contents or "")
                else:
                    # file likely created here
                    if path.exists():
                        path.unlink()
            elif op.kind == "Delete":
                # undoing delete => we don't have content; leave as-is from reverse chain
                pass


def main() -> int:
    os.chdir(ROOT)
    ensure_gitkeep()

    # ensure gitignore present
    gi = ROOT / ".gitignore"
    if not gi.exists() or "venv/" not in gi.read_text(encoding="utf-8", errors="replace"):
        gi.write_text(
            """__pycache__/
*.py[cod]
venv/
.venv/
.DS_Store
__MACOSX/
*.sqlite3
fatigue_detection_system/db.sqlite3
fatigue_detection_system/media/results/*
!fatigue_detection_system/media/results/.gitkeep
fatigue_detection_system/media/uploads/*
!fatigue_detection_system/media/uploads/.gitkeep
runs/
fatigue_detection_system/weights/*.bz2
fatigue_detection_system/weights/last.pt
*.exe
_patch_*.py
_dump_*.py
**/_patch_*.py
**/_dump_*.py
**/_fix_*.py
**/_test_fatigue_*.py
**/_process_image_snippet.py
_history_final_snapshot/
""",
            encoding="utf-8",
        )

    print("Extracting ops...")
    ops = extract_ops()
    file_ops = [o for o in ops if o.kind in ("Write", "StrReplace", "Delete")]
    users = [o for o in ops if o.kind == "User"]
    print(f"ops: {len(file_ops)} file changes, {len(users)} user turns")

    print("Snapshot final tree...")
    copy_snapshot()

    # wipe git (Windows may lock files; prefer rename then delete)
    gitdir = ROOT / ".git"
    if gitdir.exists():
        print("Removing existing .git for rebuild...")
        bak = ROOT / ".git_old_bak"
        if bak.exists():
            shutil.rmtree(bak, ignore_errors=True)
        try:
            gitdir.rename(bak)
            shutil.rmtree(bak, ignore_errors=True)
        except OSError:
            shutil.rmtree(gitdir, ignore_errors=True)
        if (ROOT / ".git").exists():
            raise SystemExit("Could not remove .git; close Git clients and retry.")

    print("Building approximate pre-agent state for touched files...")
    build_initial_touched_state(ops)

    git("init")
    git("branch", "-M", "main")

    # Commit 1: gitignore
    git("add", "-A", "--", ".gitignore")
    commit(
        "chore: add gitignore for env, media, installers, and scratch files",
        "Bootstrap reconstructed history. Source: Cursor chats afcf/1b48/0e81/ad41.",
    )

    # Commit 2: baseline = everything currently on disk that is not going to be
    # fully recreated purely from writes... Actually after reverse, touched files
    # are at earliest state; commit ALL current tracked files as baseline.
    git("add", "-A")
    # unstage snapshot dir if tracked
    subprocess.run(["git", "reset", "-q", "HEAD", "--", "_history_final_snapshot"], cwd=ROOT)
    commit(
        "chore: import baseline project state before replayed Cursor edits",
        "Approximate pre-agent tree: untouched files at final bytes; "
        "agent-touched files reversed via transcript StrReplace/Write where possible.\n\n"
        "NOTE: True original bytes for StrReplace-only files are best-effort.",
    )

    # Forward replay
    pending_paths: set[str] = set()
    last_user: str | None = None
    commits = 0

    def flush(reason: str, chat: str) -> None:
        nonlocal pending_paths, commits
        if not pending_paths:
            return
        git("add", "-A", "--", *sorted(pending_paths))
        title = reason[:72]
        body = (
            f"Replayed from Cursor chat {chat}.\n"
            f"Files: {', '.join(sorted(pending_paths))}\n"
        )
        if last_user:
            body += f"\nUser intent (truncated):\n{last_user[:500]}\n"
        if commit(title, body):
            commits += 1
        pending_paths = set()

    for op in ops:
        if op.kind == "User":
            # flush previous edit group at user boundaries
            flush(f"refactor: apply edits before next user request ({op.chat})", op.chat)
            last_user = op.query
            continue

        if op.kind == "Write" and op.path and op.contents is not None:
            write_text(ROOT / op.path, op.contents)
            pending_paths.add(op.path)
            # each full Write is a meaningful milestone — flush alone
            flush(f"feat: write {Path(op.path).name} ({op.chat})", op.chat)
        elif op.kind == "StrReplace" and op.path and op.old is not None and op.new is not None:
            ok = apply_str_replace(ROOT / op.path, op.old, op.new)
            if ok:
                pending_paths.add(op.path)
            else:
                print(f"  WARN StrReplace miss: {op.path} @ {op.chat} seq={op.seq}")
        elif op.kind == "Delete" and op.path:
            p = ROOT / op.path
            if p.exists():
                p.unlink()
                pending_paths.add(op.path)

    flush("refactor: apply remaining replayed edits", "final")

    # Sync to true final snapshot
    print("Syncing to final workspace snapshot...")
    restore_from_snapshot()
    # restore reconstruction scripts too from snap
    git("add", "-A")
    subprocess.run(["git", "reset", "-q", "HEAD", "--", "_history_final_snapshot"], cwd=ROOT)
    commit(
        "chore: sync working tree to final on-disk project state",
        "After transcript replay, force-align all tracked files to the live workspace "
        "(covers missed StrReplace, path=null edits, and manual tweaks).\n\n"
        f"Replay produced ~{commits} intermediate commits before this sync.",
    )

    # keep reconstruct tools
    git("add", "-A", "--", "tools")
    commit(
        "chore: keep git history reconstruction helpers",
        "tools/_reconstruct_git_history.py and v2 replay script for auditability.",
    )

    print("\n===== LOG =====")
    git("log", "--oneline", "--reverse")
    print("\n===== STATUS =====")
    git("status", "-sb")

    # cleanup snapshot optional — keep for audit then remove to avoid clutter
    if SNAP.exists():
        shutil.rmtree(SNAP)
        print("Removed _history_final_snapshot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
