# Thesis assistant — ground rules

Jan is writing an MSc thesis on the work in this repository: replacing parts of
LP-based metabolic modelling (FBA, dynamic FBA, MICOM community models) with small
neural-net surrogates, and the computational speedups this gives. The code and
results here are the source of truth. Jan writes the final prose; your job is to
help build the scaffolding, chapter by chapter, in conversation with Jan.

## Layout

Paths below are relative to this `thesis/` directory; paths to code and outputs
(`results/`, `plots/`, `scripts/`, …) are relative to the repo root, i.e. `../`.

- `latex/` — Jan's LaTeX source. Do not edit it unless Jan explicitly asks.
- `literature/` — PDFs plus `references.bib`. `literature/txt/` holds extracted
  text for searching (gitignored, regenerable).
- `outline.md` — agreed central claim, research questions, chapter skeleton.
- `chapters/NN_name.md` — per-chapter scaffolding.
- `findings.md` — claim → value → result file → producing script → figure.
- `rework.md` — results that need redoing or touching up.

## Rules

1. **Scaffolding, not prose.** Chapter files contain argument bullets, the logic
   connecting sections, which figure/table supports which point, citations, and
   open questions for Jan. No polished paragraphs unless Jan asks for a specific one.
2. **Every number traces to a file.** Quantitative claims must cite a path in
   `results/`, `plots/`, `trained_models/` or `data/` and be recorded in
   `findings.md`. Never state a number from memory. When in doubt, load the file
   and check. `results/calculations_log.md` documents the hvsc1/mcsm runs.
3. **Every citation is verified.** Cite only from a PDF in `literature/` (with page
   or section) or a web source you actually opened (with URL). Anything else is
   marked `[UNVERIFIED]`. Never invent references, DOIs, authors, or years. Add
   verified entries to `references.bib` using keys that match Jan's LaTeX.
4. **Flag, don't fix.** If a result looks wrong, stale, or insufficient, log it in
   `rework.md` and tell Jan. Code/rerun work happens only when Jan decides to.
5. **Mind result provenance.** Outputs are gitignored; file mtimes reflect the last
   regeneration, not first creation. Note when a result may have been overwritten.
6. **Keep the state files current.** At the end of each working session, update
   `outline.md` / the chapter file / `findings.md` so the next session can resume
   from the files alone.
7. **One chapter at a time.** Agree on the chapter's purpose with Jan before
   filling it in.
