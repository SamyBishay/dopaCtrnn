#!/usr/bin/env python3
"""PostToolUse hook: validate required frontmatter fields on paper-note files."""
import sys, json, os

data = json.load(sys.stdin)
file_path = data.get("tool_input", {}).get("file_path", "")

VAULT = "/home/samy/Documents/Obsidian Vault"
PAPERS_DIR = os.path.join(VAULT, "Papers")

if not file_path.endswith(".md"):
    sys.exit(0)
if os.path.dirname(os.path.abspath(file_path)) != PAPERS_DIR:
    sys.exit(0)
if os.path.basename(file_path) == "PAPERS_INDEX.md":
    sys.exit(0)

try:
    content = open(file_path).read()
except Exception:
    sys.exit(0)

if "type: paper-note" not in content:
    sys.exit(0)

required = ["citekey:", "status:", "verify:", "topics:", "pdf:"]
missing = [f for f in required if f not in content]
if missing:
    print(f"FRONTMATTER WARNING in {os.path.basename(file_path)}: missing {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
