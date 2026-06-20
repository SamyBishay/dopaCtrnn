# Export Mémoire

Convert a mémoire draft to a pandoc-ready PDF/DOCX. The user specifies which draft file (default: all files in `Mémoire/`).

## Steps

1. **Pre-process wikilinks → citations**: replace every `[[citekey]]` with `[@citekey]` in a temporary copy.

```bash
DRAFT="Mémoire/<filename>.md"
TMP="/tmp/memoire_export_$(date +%s).md"
sed 's/\[\[\([^]]*\)\]\]/[@\1]/g' "$DRAFT" > "$TMP"
```

2. **Run pandoc**:

```bash
pandoc "$TMP" \
  --bibliography "/home/samy/Documents/Obsidian Vault/Papers/My Library/My Library.bib" \
  --citeproc \
  -o "/tmp/memoire_output.pdf"
```

   For DOCX: replace `-o /tmp/memoire_output.pdf` with `-o /tmp/memoire_output.docx`.

3. **Open the output** for review: `xdg-open /tmp/memoire_output.pdf`

4. Clean up the temp file when done.

## Notes
- Check that every `[[citekey]]` in the draft has a matching entry in `My Library.bib` before running — missing keys will silently drop citations.
- If pandoc is not installed: `sudo dnf install pandoc` or `sudo apt install pandoc`.
