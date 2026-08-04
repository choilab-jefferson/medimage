# medimage

## Quantitative Medical Image Analysis with Python

A hands-on, notebook-based course: **from opening your first DICOM file to measuring body
composition on CT and MR, and reproducing published radiomics results.**

Every chapter builds a piece of what comes later, and the explanations assume no prior background in
medical imaging.

| Chapter | What you learn | Why the course needs it |
|---|---|---|
| **Part I — Foundations** | | |
| [1. Exploration](Chapter01_Exploration.ipynb) | DICOM, headers, Hounsfield units, image orientation, viewing volumes | You cannot measure anything until the numbers mean something |
| [2. Masks and filters](Chapter02_Masks_and_Filters.ipynb) | Histograms, selecting pixels, denoising, morphology, edges | Fat and muscle are picked out by their HU range — after denoising |
| [3. Measurement](Chapter03_Measurement.ipynb) | Labeling, object selection, area and volume, mean HU, validating with Dice | Areas and mean HU *are* the body composition numbers |
| [4. Image comparison](Chapter04_Image_Comparison.ipynb) | Resampling, transformations, similarity metrics, normalization | Two patients are different sizes, so raw numbers cannot be compared |
| [5. Patient privacy](Chapter05_Patient_Privacy.ipynb) | Finding, removing and **verifying** the removal of PHI in DICOM | Before any of this can touch real clinical data |
| **Part II — Applications** | | |
| [6. Body composition from CT](Chapter06_Body_Composition_CT.ipynb) | Finding L3, verifying a pretrained model, muscle / SAT / VAT, the muscle index | The destination the first four chapters were building toward |
| [7. Fat quantification with MR](Chapter07_MR_Fat_Quantification.ipynb) | Dixon in/opposed-phase, fat-fraction maps, liver steatosis | The other modality that can measure fat, and why CT is still the default |
| [8. PET/CT](Chapter08_PET_CT.ipynb) | SUV, PET/CT fusion, cardiac FDG uptake, change between timepoints | Function as well as anatomy — and a published clinical application |
| **Part III — Quantitative methods** | | |
| [9. Radiomics features](Chapter09_Radiomics_Features.ipynb) | The three feature families, patterns, and what preprocessing does to each | Knowing which of your 1130 numbers survive someone else running the pipeline |
| [10. Registration](Chapter10_Registration.ipynb) | Two engines, scoring in millimeters, diagnosing a registration that fails silently | Aligning scans is where pipelines break without saying so |
| [11. Delta radiomics](Chapter11_Delta_Radiomics.ipynb) | Measuring change between timepoints, and measuring your own noise floor | Change is more informative than any single value — once you know what change means nothing |
| [12. Classification](Chapter12_Classification.ipynb) | Benchmarking eight models, a real data leak that produced AUC 1.000, and what a hold-out set can and cannot tell you at this size | An implausibly good result is a bug report, not a finding |
| [13. Reproducing published results](Chapter13_Reproducibility.ipynb) | The full `qr` analysis on Lung1, compared against the paper | Whether a pipeline reproduces is the question that matters |

### What the course is actually about

Two people can weigh the same and be in very different health. What matters is what the weight is
made of — muscle or fat, and where the fat sits. People with little muscle recover worse from
surgery and tolerate chemotherapy worse than their weight alone predicts.

A CT scan already contains that information. The course teaches you to get it out, correctly, and
then to check whether the number you produced can be believed.

## Running the notebooks

**Google Colab is the primary target.** Every notebook opens with a setup cell that detects Colab,
clones this repository and installs what the runtime is missing. Local execution is supported and
used for testing.

| Chapter | |
|---|---|
| 1. Exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter01_Exploration.ipynb) |
| 2. Masks and filters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter02_Masks_and_Filters.ipynb) |
| 3. Measurement | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter03_Measurement.ipynb) |
| 4. Image comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter04_Image_Comparison.ipynb) |
| 5. Patient privacy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter05_Patient_Privacy.ipynb) |
| 6. Body composition from CT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter06_Body_Composition_CT.ipynb) |
| 7. Fat quantification with MR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter07_MR_Fat_Quantification.ipynb) |
| 8. PET/CT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter08_PET_CT.ipynb) |
| 9. Radiomics features | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter09_Radiomics_Features.ipynb) |
| 10. Registration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter10_Registration.ipynb) |
| 11. Delta radiomics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter11_Delta_Radiomics.ipynb) |
| 12. Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter12_Classification.ipynb) |
| 13. Reproducing published results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter13_Reproducibility.ipynb) |

```bash
git clone https://github.com/choilab-jefferson/medimage.git
cd medimage
pip install -r requirements.txt
jupyter lab
```

## How the notebooks are built

The `.ipynb` files are generated, not hand-edited. Each chapter's prose and code live in
`generators/chNN.py` as plain Python strings, and `chNN.py` writes `ChapterNN_*.ipynb` — the
numbering matches one to one.

```bash
python generators/ch09.py     # rewrites Chapter09_Radiomics_Features.ipynb
```

Keeping the source as text is what makes a change like renumbering the chapters or rewording a
cross-reference an ordinary edit rather than surgery on notebook JSON.

One thing to know before running it: `build()` emits cells with **empty outputs**, while the
committed notebooks are executed and carry their figures. Regenerating therefore strips those
figures unless you re-execute afterwards. For a small prose fix it is cheaper to edit the markdown
cell in the `.ipynb` and make the identical edit in `chNN.py`, which keeps the two in step without
losing the outputs. See [generators/README.md](generators/README.md).

## Data policy

**This repository contains no imaging data.** Every notebook downloads what it needs from the
original public source on first run and caches it under `data/` (git-ignored). Each loader prints
its citation when it runs.

That keeps the repository small and the licensing clean, but the real reason is pedagogical: when
the provenance of an image is one function call away, it stays visible, and you always know what the
numbers you compute are actually measuring.

| Dataset | Used by | Source | License | Transfer |
|---|---|---|---|---|
| **Pancreas-CT** — contrast-enhanced abdominal CT, 80 subjects | Ch 1–6 | [TCIA](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU), public NBIA REST API | CC BY 3.0 | ~40 MB per subject |
| **CHAOS** — T1-DUAL in/out-of-phase abdominal MRI with reference organ masks | Ch 3, 7 | [Zenodo](https://doi.org/10.5281/zenodo.3431873) | CC BY-NC-SA 4.0 | ~9 MB (partial download) |
| **ACRIN-NSCLC-FDG-PET** (ACRIN 6668) — FDG PET/CT, 242 patients, serial timepoints | Ch 8, 11 | [TCIA](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/) | TCIA data usage policy | subset only |
| **NSCLC-Radiomics** (Lung1) — CT + GTV contours, 422 patients | Ch 9, 12, 13 | [TCIA](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) | CC BY-NC 3.0 | configurable subset |

Nothing is redistributed here. The CHAOS archive is 890 MB, but the notebook pulls only the few
megabytes it uses via HTTP range requests. All collections are de-identified at source.

`medimage_data.py` holds the loaders used by all chapters:

```python
import medimage_data as md

ct_dir = md.fetch_pancreas_ct(0)                  # downloads, caches, prints the citation
vol, spacing, datasets = md.load_series(ct_dir)   # HU volume, (dz, dy, dx) in mm

in_phase, out_phase, labels, spacing = md.load_chaos(1)   # MRI with reference masks
pet_dir = md.fetch_acrin(timepoint=0)                     # PET/CT, one timepoint
paths = md.fetch_lung1_cohort(60)                         # CT + contours + outcomes
```

Each loader prepares whatever it needs and caches the result, so **no notebook depends on another
having been run first.** `fetch_lung1_cohort` runs the whole `qr` chain — download, contour
conversion, crop and resample, feature extraction, clinical merge — and returns the paths, which is
why Chapters 9, 12 and 13 can each be opened on their own.

`load_series` does two things worth knowing about, because getting either wrong is a common and
quiet source of error:

- It honours `ImageOrientationPatient`, so images are displayed anterior-up rather than however the
  scanner happened to store the rows.
- It orders slices head first and derives slice spacing from the actual slice positions rather than
  trusting `SliceThickness`.

Chapter 6 is the one chapter that cannot use it: TotalSegmentator reads NIfTI, and its masks have to
stay on the same indices as the image, so the chapter goes through SimpleITK instead. It reaches the
same convention with `sitk.DICOMOrient(..., "LPI")`, applied to the scan **and every mask** through a
single helper. Orienting the image alone would leave the masks on the old axes — and the failure
would be silent, since the areas come out identical either way.

## Tools used in the later chapters

### qradiomics

[qradiomics](https://github.com/choilab-jefferson/qradiomics) (`qr`) is the research toolkit the
applied chapters build on. Its canonical flow is:

```
qr tcia download  →  qr convert  →  qr preprocess  →  qr extract  →  qr results merge  →  qr analyze
```

| Command | What it does |
|---|---|
| `qr tcia` | List, download and build manifests from TCIA collections |
| `qr convert` | DICOM ↔ NRRD, RTSTRUCT → mask; PET routes through QIBA-standard SUV conversion |
| `qr preprocess` | Crop to an ROI and resample image/mask pairs |
| `qr extract` | Radiomics features from a manifest, with curated patterns and multiple engines |
| `qr delta` | Delta and trend features across timepoints (see Chapter 11) |
| `qr shape` | Shape descriptors — AHSN, spiculation (see Chapter 13) |
| `qr results merge` | Join features with a clinical table into an analysis-ready CSV |
| `qr analyze` | Univariate Cox survival, classification, feature importance |
| `qr ml` | Multi-model training, benchmarking, evaluation and prediction (see Chapter 12) |
| `qr anonymize` / `qr phi` | Strip and audit PHI (see Chapter 5) |

### TotalSegmentator

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) is a pretrained nnU-Net that labels
117 anatomical structures. Chapter 6 uses it to find the L3 vertebral level automatically and to
separate the abdominal cavity from the abdominal wall — which is what makes visceral and
subcutaneous fat separable at all, since no HU threshold can tell them apart.

Install it in its own environment so its `torch` pin cannot clash with the analysis kernel:

```bash
python -m venv .venv-ts
.venv-ts/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-ts/bin/pip install TotalSegmentator
```

The notebook finds `TotalSegmentator` on `$PATH` or in `.venv-ts/bin/`. No GPU is required — the
3 mm (`--fast`) model runs on CPU in well under a minute.

## What the later chapters do

**Chapter 6 — body composition from CT.** Finds the L3 vertebra with TotalSegmentator and then
checks the answer, because three different run settings gave three different L3 levels on the same
scan (Dice 0.66 between two of them). Builds the abdominal cavity boundary from the organ labels,
which is what separates visceral from subcutaneous fat, and measures muscle 161 cm2, SAT 193 cm2 and
VAT 84 cm2 at mid-L3. Closes on the muscle index, and on the fact that no scan contains the patient
height the index needs.

**Chapter 7 — fat quantification with MR.** MRI separates fat from water physically rather than by
brightness, using the slightly different precession rates of their hydrogen. The chapter derives
water and fat images from in-phase and opposed-phase scans, then shows that the resulting fat
fraction cannot exceed 50% and silently swaps fat for water wherever fat dominates. It then applies
the method where it genuinely works — liver steatosis — measuring 2.9%, 3.7%, 6.8% and 24.8% across
four subjects against the 5% clinical threshold, with the spleen as a noise-floor control.

**Chapter 8 — PET/CT.** PET measures function rather than anatomy, and SUV is its calibrated unit —
the PET analog of the Hounsfield unit. The chapter converts an ACRIN 6668 PET series to SUV with
the QIBA-standard converter in qradiomics, resamples it onto the CT grid, and measures uptake in the
heart against the liver as reference. The liver reads SUV 2.27, right on the expected value, which
is what validates the conversion before anything is interpreted. Framed by Choi et al.,
*Novel Functional Radiomics for Prediction of Cardiac PET Avidity in Lung Cancer Radiotherapy*,
JCO Clinical Cancer Informatics 2024 ([PMID 38452302](https://pubmed.ncbi.nlm.nih.gov/38452302/)).

**Chapter 12 — classification, and a result that was too good.** Benchmarks eight models on two
labels. Histology comes back at chance. Two-year mortality comes back at
**AUC 1.000** with zero variance across folds — and the chapter treats that as a bug report rather
than a finding. The leak is that the label was derived from `OS_months` and `OS_event`, both still
sitting in the feature table; dropping two columns takes the best model from 1.000 to about 0.65.
It closes by asking why not simply hold out a test set instead. Splitting the same 59 patients six
ways, changing nothing but the seed, gives hold-out AUCs from 0.48 to 0.74 — and they do not track
the cross-validated score computed on the other side of the same split. The argument is about **n**,
not against hold-out testing: eighteen test patients, thirteen of them events, cannot measure
anything.

**Chapter 13 — reproducing published results.** Runs the full `qr` pipeline on a subset of the Lung1
cohort — download, contour conversion, feature extraction, clinical merge, cross-validated Cox — and
compares the result against the full-cohort reproduction (0.580) and the published figure from Aerts
et al. (0.650). At 59 patients the measured c-index is **0.482**, indistinguishable from chance, and
the chapter says so instead of tuning until it agrees. It separates two questions that are easy to
confuse: whether an *effect* reproduces, and whether a *number* does.

It then turns to a second, cleaner kind of reproduction. Where the Aerts signature reproduces 0.07
below its published value, three of four reproductions of the Choi lab's own methods land **at or
above** theirs — the spiculation feature set (Choi 2021 CMPB) matches almost exactly at AUC 0.816.
The chapter runs `qr shape` to compute those six interpretable descriptors on the cohort and uses
the contrast to make a general point: *methods* reproduce more reliably than *fitted models*,
because a geometric measurement is defined by its algorithm while a signature depends on the cohort,
the software version and the modeling choices.

**Chapter 5 — patient privacy.** Every scan in this course arrived already de-identified. Real data
does not. The chapter plants invented identifiers in a DICOM header, removes them under the HIPAA
Safe Harbor profile, and then *verifies* the removal with an audit that exits non-zero on any
finding — 24 findings before, clean after. It also shows why dates are shifted rather than deleted:
birth date and study date both move by the same 19 days, so age at scan is preserved exactly while
the link to the real calendar is broken. Closes on what header cleaning cannot catch — burned-in
pixel text, private tags, and faces reconstructable from head CT.

## Limitations

- The course measures body composition on **pancreas-protocol CT**, which covers L3 but was acquired
  for a different purpose. Published cut-offs come from cohorts scanned and analyzed differently.
- Every number these notebooks produce is a **demonstration of a method, not a clinical
  measurement**. Each chapter states its own limitations where they apply.
- Chapter 3 deliberately ends on a failed segmentation (Dice 0.45, volume 3× too large) and
  Chapter 4 on an over-corrected normalization, because knowing why a simple method fails is what
  tells you when to reach for a complicated one.

## License

Code and notebook text: MIT (see [LICENSE](LICENSE)). Each dataset remains under the terms of its
original source, listed above.
