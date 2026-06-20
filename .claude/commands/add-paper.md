# Add Paper

Ingest a new paper into the vault. The user provides a citekey (as it appears in `Papers/My Library/My Library.bib`).

## Steps

1. **Look up the citekey** in `Papers/My Library/My Library.bib` — extract title, authors, year, journal/venue, abstract.

2. **Create the paper note** at `Papers/<citekey>.md` using this exact structure:

```markdown
---
citekey: <citekey>
type: paper-note
status: to-read
verify: flag
topics: []
pdf: "My Library/files/<N>/<filename>.pdf"
full-text: "[[<Author Title>]]"
---

## What the paper does

## Key claims relevant to this project

## Mechanism / model details

## Implications for our model

## Open questions / caveats

---
[[<related-citekey>]]
```

   Set `status: read` and `verify: pass` only after actually reading and checking the content. Set `topics` based on the paper content (use existing tags from other notes where possible).

3. **Add a row to `Papers/PAPERS_INDEX.md`** with: `| [[<citekey>]] | <one-line description> | <year> |`

4. **Full-text conversion**: if a PDF exists at the pdf path, offer to convert it. Save the result as `Papers/<Author Title>.md` and update the `full-text` field.

5. **Commit**: `git add Papers/<citekey>.md Papers/PAPERS_INDEX.md && git commit -m "notes: add <citekey> paper note"`
