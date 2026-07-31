# medimage

## Quantitative Medical Image Analysis with Python

A hands-on, notebook-based course: **from opening your first DICOM file to measuring body
composition on CT and MR, and reproducing published radiomics results.**

Every chapter builds a piece of what comes later, and the explanations assume no prior background in
medical imaging.

| Chapter | What you learn | Why the course needs it | Status |
|---|---|---|---|
| [1. Exploration](Chapter1_Exploration.ipynb) | DICOM, headers, Hounsfield units, image orientation, viewing volumes | You cannot measure anything until the numbers mean something | ✅ ready |
| [2. Masks and filters](Chapter2_Masks_and_Filters.ipynb) | Histograms, selecting pixels, denoising, morphology, edges | Fat and muscle are picked out by their HU range — after denoising | ✅ ready |
| [3. Measurement](Chapter3_Measurement.ipynb) | Labeling, object selection, area and volume, mean HU, validating with Dice | Areas and mean HU *are* the body composition numbers | ✅ ready |
| [4. Image comparison](Chapter4_Image_Comparison.ipynb) | Resampling, transformations, similarity metrics, normalisation | Two patients are different sizes, so raw numbers cannot be compared | ✅ ready |
| 5. Body composition from CT | Muscle, subcutaneous and visceral fat at the L3 vertebra | The destination | in progress |
| 6. Body composition from MR | Dixon water–fat separation, fat-fraction maps | The other modality that can do this, and why CT is still the default | in progress |
| 7. PET/CT | SUV, PET/CT fusion, cardiac FDG uptake, delta radiomics | Function as well as anatomy — and a published clinical application | in progress |
| 8. Reproducing published results | `qr` end to end on a public cohort, compared against the paper | Whether a pipeline reproduces is the question that matters | in progress |

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
| 1. Exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter1_Exploration.ipynb) |
| 2. Masks and filters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter2_Masks_and_Filters.ipynb) |
| 3. Measurement | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter3_Measurement.ipynb) |
| 4. Image comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter4_Image_Comparison.ipynb) |

```bash
git clone https://github.com/choilab-jefferson/medimage.git
cd medimage
pip install -r requirements.txt
jupyter lab
```

## Data policy

**This repository contains no imaging data.** Every notebook downloads what it needs from the
original public source on first run and caches it under `data/` (git-ignored). Each loader prints
its citation when it runs.

That keeps the repository small and the licensing clean, but the real reason is pedagogical: when
the provenance of an image is one function call away, it stays visible, and you always know what the
numbers you compute are actually measuring.

| Dataset | Used by | Source | License | Transfer |
|---|---|---|---|---|
| **Pancreas-CT** — contrast-enhanced abdominal CT, 80 subjects | Ch 1–5 | [TCIA](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU), public NBIA REST API | CC BY 3.0 | ~40 MB per subject |
| **CHAOS** — T1-DUAL in/out-of-phase abdominal MRI with reference organ masks | Ch 3, 6 | [Zenodo](https://doi.org/10.5281/zenodo.3431873) | CC BY-NC-SA 4.0 | ~9 MB (partial download) |
| **ACRIN-NSCLC-FDG-PET** (ACRIN 6668) — FDG PET/CT, 242 patients, serial timepoints | Ch 7 | [TCIA](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/) | TCIA data usage policy | subset only |
| **NSCLC-Radiomics** (Lung1) — CT + GTV contours, 422 patients | Ch 8 | [TCIA](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) | CC BY-NC 3.0 | configurable subset |

Nothing is redistributed here. The CHAOS archive is 890 MB, but the notebook pulls only the few
megabytes it uses via HTTP range requests. All collections are de-identified at source.

`medimage_data.py` holds the loaders used by Chapters 1–6:

```python
import medimage_data as md

ct_dir = md.fetch_pancreas_ct(0)                  # downloads, caches, prints the citation
vol, spacing, datasets = md.load_series(ct_dir)   # HU volume, (dz, dy, dx) in mm

in_phase, out_phase, labels, spacing = md.load_chaos(1)   # MRI with reference masks
```

`load_series` does two things worth knowing about, because getting either wrong is a common and
quiet source of error:

- It honours `ImageOrientationPatient`, so images are displayed anterior-up rather than however the
  scanner happened to store the rows.
- It orders slices head first and derives slice spacing from the actual slice positions rather than
  trusting `SliceThickness`.

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
| `qr delta` | Delta and trend features across timepoints |
| `qr shape` | Shape descriptors — AHSN, spiculation |
| `qr results merge` | Join features with a clinical table into an analysis-ready CSV |
| `qr analyze` | Univariate Cox survival, classification, feature importance |
| `qr ml` | Multi-model training, benchmarking, evaluation and prediction |
| `qr anonymize` / `qr phi` | Strip and audit PHI |

### TotalSegmentator

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) is a pretrained nnU-Net that labels
117 anatomical structures. Chapter 5 uses it to find the L3 vertebral level automatically and to
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

## What Chapters 5–8 do

**Chapter 5 — body composition from CT.** Locate L3, segment skeletal muscle, subcutaneous fat and
visceral fat, convert to areas and indices, extend to radiomics features, and scale the workflow to
a cohort.

**Chapter 6 — body composition from MR.** MRI is arguably the reference standard for adipose tissue,
because Dixon acquisitions separate fat and water physically rather than by brightness. The chapter
derives water and fat images from in-phase and out-of-phase scans, builds a fat-fraction map, and
then states plainly where 2-point Dixon falls short of true PDFF.

**Chapter 7 — PET/CT.** PET measures function rather than anatomy, and SUV is its calibrated unit —
the PET analogue of the Hounsfield unit. The chapter covers SUV conversion, PET/CT fusion, uptake in
the heart, and delta features between treatment timepoints. It is framed by Choi et al.,
*Novel Functional Radiomics for Prediction of Cardiac PET Avidity in Lung Cancer Radiotherapy*,
JCO Clinical Cancer Informatics 2024 ([PMID 38452302](https://pubmed.ncbi.nlm.nih.gov/38452302/)),
which classified cardiac FDG uptake from pretreatment PET/CT.

**Chapter 8 — reproducing published results.** Runs the full `qr` pipeline on a subset of the Lung1
cohort and compares the resulting Cox c-index against both the full-cohort reproduction and the
number published by Aerts et al. (2014). A small subset scores lower, and the chapter explains why
rather than tuning until it agrees.

## Honest limitations

- The course measures body composition on **pancreas-protocol CT**, which covers L3 but was acquired
  for a different purpose. Published cut-offs come from cohorts scanned and analysed differently.
- Every number these notebooks produce is a **demonstration of a method, not a clinical
  measurement**. Each chapter states its own limitations where they apply.
- Chapter 3 deliberately ends on a failed segmentation (Dice 0.45, volume 3× too large) and
  Chapter 4 on an over-corrected normalisation, because knowing why a simple method fails is what
  tells you when to reach for a complicated one.

## License

Code and notebook text: MIT (see [LICENSE](LICENSE)). Each dataset remains under the terms of its
original source, listed above.
