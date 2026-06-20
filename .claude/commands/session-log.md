# Session Log

Append today's entry to `logs/SESSION_LOG.md`. The file is append-only — never read past entries.

1. Check the most recent git commit hash: `git -C "/home/samy/Documents/Obsidian Vault" log -1 --format="%H %s"` (use `–` if no commit was made this session).
2. Summarise this session in 2–5 bullet points: what was done, what changed, any decisions or blockers.
3. Run this exact command (substituting the real values, YYYY-MM-DD = today):

```bash
printf '\n## YYYY-MM-DD — <title>\nCommit: <hash or –>\n\n- <bullet>\n- <bullet>\n' >> "/home/samy/Documents/Obsidian Vault/logs/SESSION_LOG.md"
```

Do not use Write or Edit on SESSION_LOG.md — only `printf >>`.
