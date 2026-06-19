# CLAUDE.md — Operating manual for this research project (Obsidian vault)

You are the research engineer for this project. This file is your contract. Read it fully at the start of every session. When a rule here conflicts with a habit, follow the rule.

**This repo IS the Obsidian vault.** The user opens this same folder in Obsidian. That has consequences you must respect:

- Cross-references should be **wikilinks** `[[citekey]]`, not relative markdown links — wikilinks populate Obsidian's backlinks panel and graph view.
- Structured fields (status, citekey, topics) go in **YAML frontmatter** ("Properties" in Obsidian's UI) — this makes the vault queryable by the **Dataview** plugin.
- Dataview queries render _only inside the Obsidian app_. Navigate with `rg`/`glob` over real files and frontmatter text. Dataview is a courtesy layer for the human.
- Never break wikilinks: if you rename or move a note, update every `[[old-name]]` reference (`rg -l '\[\[old-name'` to find them) — Obsidian's own rename-tracking only fires from inside the app, not from filesystem edits.

---

## 0. Where things live

```
Papers/
  PAPERS_INDEX.md              # routing table; each entry links to [[citekey]]
  <citekey>.md                 # project-specific notes (concise, structured)
  <Author Title>.md            # full paper text (converted from PDF)
  My Library/
    My Library.bib             # Zotero BibTeX export — source of citekeys
    files/<N>/<paper>.pdf      # PDFs (Zotero IDs as folder names)
Mémoire/
  memoire_introduction_section_draft.md
  memoire_method_section_draft.md
Code/
  main/                        # Stage-1 implementation (see Code/main/README.md)
  visualisations/              # trajectory viewer
PROJECT_STATUS_AND_PLAN.md     # timeline, experiment scope, one-page mémoire outline
Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff.md
                               # settled design decisions — source of truth for the model
```

PDFs and their full-text conversions live **inside the vault** under `Papers/`. The `.bib` citekeys in `My Library.bib` are the authoritative identifiers — use them as note filenames (`Papers/<citekey>.md`).

---

## 1. The retrieval ladder

When you need information from the literature, climb the ladder and **stop at the highest rung that answers the question.** Never skip straight to the bottom.

| Rung | Location | Cost | Use it to… |
|------|----------|------|------------|
| 1 | `Papers/PAPERS_INDEX.md` (or `rg` over `Papers/<citekey>.md` frontmatter) | tiny | find which notes are relevant to a topic |
| 2 | `Papers/<citekey>.md` | small | get the project's distilled take + anchors |
| 3 | `Papers/<Author Title>.md` | medium | read the full argument/method when the note isn't enough |
| 4 | `Papers/My Library/files/<N>/<paper>.pdf` | visual | last resort — verify an equation or figure pixel-for-pixel |

Rules:
- Always start at rung 1. Do not load multiple notes in bulk "to be safe."
- **Any claim going into the mémoire must be verified at rung 3 or deeper at least once.** Record the anchor in the note (`(<Author Title>.md §Results)`) once verified, so the next check is one glance.
- Open rung-3+ files only for specific papers you actually need.

---

## 2. Provenance & accuracy rules

- Every factual claim you write traces to a `[[citekey]]` (literature) or an experiment result. No orphan claims.
- In notes, attach a section anchor to each load-bearing claim, e.g. `(<Author Title>.md §Results)`, plus a short verbatim quote (<15 words) for numbers and equations so they can be re-verified instantly.
- Equations and table values are the highest-risk items. Verify against the full-text `.md` or the PDF before using in the mémoire. If a note's `verify` frontmatter is `flag`, treat its math as suspect until checked.
- Never invent a citation, DOI, page number, or result. Leave a `TODO(verify)` marker if you cannot verify.

---

## 3. Frontmatter conventions (read by you AND by Obsidian/Dataview)

Use these properties consistently — they are the queryable backbone of the vault.

**`Papers/<citekey>.md` (project notes):**

```yaml
---
citekey: smith2024
type: paper-note
status: read          # to-read | read | verified
verify: pass          # pass | flag
topics: [dopamine, habit, CTRNN, striatum, D1/D2]
pdf: "My Library/files/<N>/<paper>.pdf"
full-text: "[[<Author Title>]]"
---
```

Inside Obsidian, the user can render `dataview` query blocks anywhere:

````
```dataview
table topics, verify, status from "Papers"
where type = "paper-note"
sort citekey asc
```
````

Maintain `Papers/PAPERS_INDEX.md` as a real, `rg`-able file regardless — Dataview's rendered output doesn't exist on disk for you to search.

---

## 4. Living documents

- `Papers/PAPERS_INDEX.md` — `rg`-able routing table for the literature.
- `Papers/<citekey>.md` — per-paper, per-project distillation.
- `PROJECT_STATUS_AND_PLAN.md` — timeline, scope, mémoire outline, decision log.
- `Project overview … handoff.md` — settled state of the project's vision, mechanism, and roadmap. **Source of truth for all design decisions.** Read this when in doubt about the model.
- `Mémoire/` — the manuscript drafts (introduction, method; more sections to be added).

---

## 5. Literature workflow

For any review touching more than ~3 new papers, spawn a subagent for the heavy reading so only the distilled result returns to the main context.

New paper:
1. Add PDF to Zotero, export updated library to `Papers/My Library/` to get the BibTeX citekey.
2. Convert PDF to full-text markdown → save as `Papers/<Author Title>.md`.
3. Write `Papers/<citekey>.md` using the structure from §3 frontmatter + these sections: **What the paper does** · **Key claims relevant to this project** · **Mechanism / model details** (if any) · **Implications for our model** · **Open questions / caveats** · wikilinks at the bottom.
4. Add a row to `Papers/PAPERS_INDEX.md` and a `[[citekey]]` wikilink on the heading line.

---

## 6. Mémoire & soutenance

- Drafts live in `Mémoire/`. Every sentence asserting a prior finding carries its `[[citekey]]`. Verify each cited claim at rung 3+ once before it ships (§1).
- Export with pandoc: pre-process `[[citekey]]` → `[@citekey]` before calling pandoc, using `Papers/My Library/My Library.bib` as the bibliography source.
- Soutenance slides: derive from the mémoire and `PROJECT_STATUS_AND_PLAN.md`. Do not restate papers in full.

---

## 7. Session log

At the end of every session, append a dated entry to `SESSION_LOG.md`:

```
## YYYY-MM-DD — <short commit message or "no commit">
Commit: <hash> (or "–" if nothing committed)

<exactly the bullet-point summary you output to the user at session end>
```

Do this even for short sessions. The log is append-only — never edit past entries. If you committed during the session, include the hash so the log entry is traceable to an exact repo state.

---

## 8. Git & Codeberg

Remote: **https://codeberg.org/samyb/dopaCtrnn.git** (branch: `main`)

Every change to the vault should be committed and pushed. Workflow:
1. Stage relevant files (`git add <files>` — never `git add -A` blindly).
2. Commit with a short message referencing what changed (e.g. `notes: add villet2025 paper note`).
3. Push to `origin main` — confirm with the user before the first push of a session if unsure of state.

Gitignored (see `.gitignore`): `.obsidian/workspace.json`, plugin caches, `Papers/My Library/files/` (PDFs — large, copyrighted), `Code/main/results/`.

Committed (safe, public): everything else — vault notes, code, `.bib`, mémoire drafts, `SESSION_LOG.md`.

---

## 9. Code

- Source in `Code/main/` — see `Code/main/README.md` for run commands, hypothesis table (H1–H7), and output structure.
- `Code/visualisations/` — standalone trajectory viewer (outputs `maze_viz.html`).
- Small, message-rich commits. Reference the hypothesis id (H1–H7) in the commit body when relevant.
