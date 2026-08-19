# Notebook generators

Every `ChapterNN_*.ipynb` in the repository root is written by the script of the same number
here. `chNN.py` writes `ChapterNN_*.ipynb` — the numbering matches, one to one.

The notebooks are the artifact; these scripts are the source. Prose and code live here as plain
Python strings, which is what makes a change like renumbering the chapters or rewording a
cross-reference a normal text edit rather than surgery on notebook JSON.

## Running one

```bash
python generators/ch09.py     # writes Chapter09_Radiomics_Features.ipynb
```

The output path is derived from the script's own location, so the working directory does not
matter — but the script always writes into *this* checkout.

## The one thing to know

The committed notebooks are executed — 155 of their 158 code cells carry outputs, and readers
browsing on GitHub see those figures without running anything.

`build()` preserves them. Before writing, it reads the notebook already at the target path and
carries each executed code cell's outputs, execution count and metadata across to the cell with
**byte-identical source**. So:

- **Editing prose is safe.** Regenerating after a markdown-only change is a no-op for every code
  cell, and `git diff` shows exactly the prose you changed.
- **Editing code costs its outputs.** A code cell whose source changed no longer matches, so it
  comes back empty rather than carrying a stale result that no longer corresponds to the code.
  **Re-execute the notebook** before committing.

The line it prints says which happened: `wrote ...: 39 cells (16 code, 16 outputs kept)`. If the
kept count drops, you changed code, and that notebook needs re-executing.

Source text is the only key the two notebooks share, so **two code cells with identical source
cannot be told apart** and neither keeps its outputs. Given three identical cells and a generator
that now emits two, nothing in the text says which one was removed, and pairing them in file order
would attach the wrong figure. Both come back empty instead, the kept count drops, and re-executing
restores them. No chapter currently has a duplicated code cell.

To confirm the two have not drifted, just regenerate in place and check `git status`:

```bash
for f in generators/ch*.py; do python "$f"; done
git status --porcelain      # silent means the notebooks match their generators
```

## Shared code

`nbbuild.py` holds `build()`, the Colab badge helper, and the `SETUP` / `UNPACK` cell templates that
every chapter opens with. The Colab-vs-local bootstrap logic lives there, in one place.

## Not included

The one-off scripts that prepared derived data under `work/` (TotalSegmentator runs, the delta
radiomics timepoint pairing) are not here. They are not part of building the notebooks, they carry
absolute paths, and the chapters now fetch what they need through `medimage_data.py` instead.
