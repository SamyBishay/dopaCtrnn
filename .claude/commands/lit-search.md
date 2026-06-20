# Literature Search

Search the vault for papers relevant to a topic using the retrieval ladder. Stop at the highest rung that answers the question.

The user provides a topic or question.

## Ladder (stop as soon as you have the answer)

**Rung 1 — Index scan** (always start here):
```bash
rg -i "<topic>" "/home/samy/Documents/Obsidian Vault/Papers/PAPERS_INDEX.md"
rg -l "<topic>" "/home/samy/Documents/Obsidian Vault/Papers/"*.md --include="*.md"
```

**Rung 2 — Paper notes** (open only the relevant `<citekey>.md` files found above):
Read `Papers/<citekey>.md` for the distilled take, key claims, and anchors.

**Rung 3 — Full text** (only when the note is insufficient):
Read `Papers/<Author Title>.md` for the full argument, methods, and exact quotes.

**Rung 4 — PDF** (last resort, for equations or figures):
Open `Papers/My Library/files/<N>/<paper>.pdf`.

## Output format

Return a summary with:
- Which papers are relevant and why (with `[[citekey]]` links)
- Key claims or data points found, each with a section anchor `(Title.md §Section)`
- Any gaps — questions the literature doesn't answer

Do not bulk-load notes. Load only papers that rung 1 identifies as relevant.
