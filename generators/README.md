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

`build()` emits cells with **empty outputs and no execution counts**. The committed notebooks are
executed — 150 of their 153 code cells carry outputs, and readers browsing on GitHub see those
figures without running anything.

So regenerating is only half the job:

1. Edit `chNN.py`.
2. Run it, which produces the notebook with its outputs stripped.
3. **Re-execute the notebook** before committing, or the figures disappear from the diff.

For a small prose fix, editing the markdown cell in the `.ipynb` and making the identical edit in
`chNN.py` is the cheaper path — it keeps the outputs and keeps the two in step. That is how commit
`23386ad` was made.

To confirm the two have not drifted, generate into a scratch directory laid out like the repository
(`<scratch>/generators/`) and compare cell sources against the committed notebooks; they should
match exactly.

## Shared code

`nbbuild.py` holds `build()`, the Colab badge helper, and the `SETUP` / `UNPACK` cell templates that
every chapter opens with. The Colab-vs-local bootstrap logic lives there, in one place.

## Not included

The one-off scripts that prepared derived data under `work/` (TotalSegmentator runs, the delta
radiomics timepoint pairing) are not here. They are not part of building the notebooks, they carry
absolute paths, and the chapters now fetch what they need through `medimage_data.py` instead.
