# Sync CLAUDE.md file structure

Scan the vault directory tree and update the "Where things live" block in CLAUDE.md (Section 0) to reflect the current structure.

Steps:
1. Run `find . -maxdepth 3 -not -path "*/.*" -not -path "*/My Library/files/*" -not -path "*/__pycache__/*" -not -path "*/venv/*" -not -path "*/results/*" | sort` from the vault root to get the current tree.
2. Compare the output against the code block in CLAUDE.md Section 0 ("## 0. Where things live").
3. Edit the code block in CLAUDE.md to match reality — add any missing directories/files, remove any that no longer exist, update comments if needed.
4. Do NOT change any other part of CLAUDE.md.
5. Report a one-line diff summary of what changed.
