import json
from pathlib import Path
from collections import Counter

base = Path(r"C:\Users\REDACTED\.cursor\projects\c-Users-25959-Downloads-SleepyDetect\agent-transcripts")
ids = [
    "1b48f9d0-110b-49af-9141-ddae9aa7348b",
    "afcf99e5-3de8-4a8d-960e-e2e2ea524e4a",
    "ad41efed-cd71-4b89-a2f7-ea6562687f6b",
]
ops = []
for tid in ids:
    f = base / tid / f"{tid}.jsonl"
    for li, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            o = json.loads(line)
        except Exception:
            continue

        def walk(x, depth=0):
            if depth > 15:
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
                    p = args.get("path")
                    ps = str(p or "")
                    if p and "SleepyDetect" in ps.replace("\\", "/") and ".cursor" not in ps:
                        contents = args.get("contents") or args.get("content")
                        old = args.get("old_string")
                        new = args.get("new_string")
                        ops.append(
                            {
                                "tid": tid[:8],
                                "line": li,
                                "name": name,
                                "path": p,
                                "clen": len(contents) if isinstance(contents, str) else 0,
                                "has_patch": isinstance(old, str) and isinstance(new, str),
                            }
                        )
                for v in x.values():
                    walk(v, depth + 1)
            elif isinstance(x, list):
                for i in x:
                    walk(i, depth + 1)

        walk(o)

print("ops", len(ops))
print(Counter(o["name"] for o in ops))
writes = [o for o in ops if o["name"] == "Write"]
print("writes with content", sum(1 for o in writes if o["clen"] > 0), "/", len(writes))
print("total write bytes", sum(o["clen"] for o in writes))
repl = [o for o in ops if o["name"] == "StrReplace"]
print("replaces with patch", sum(1 for o in repl if o["has_patch"]), "/", len(repl))
for o in writes:
    print(f"{o['tid']} L{o['line']} {o['clen']:6d}B {Path(o['path']).name}")
