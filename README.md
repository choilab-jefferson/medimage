# medimage

## Quantitative Medical Image Analysis with Python

A hands-on, notebook-based course: **from reading your first DICOM file to quantifying body
composition on a real CT scan.**

Each chapter builds a piece of the final analysis. Nothing is introduced that the destination does
not need, and the destination is reached with the tools the earlier chapters teach.

| Chapter | You learn | Why the final analysis needs it | Status |
|---|---|---|---|
| 1. Exploration | Open a scan, read its header, Hounsfield units, orientation, viewing volumes | You cannot measure anything until the numbers mean something | ✅ ready |
| 2. Masks and filters | Histograms, selecting pixels, denoising, morphology, edges | Fat and muscle are picked out by their HU range — after denoising | ✅ ready |
| 3. Measurement | Labeling, object selection, area/volume, mean HU, validating with Dice | Cross-sectional areas and mean HU *are* the body composition metrics | ✅ ready |
| 4. Image comparison | Transformations, resampling, similarity metrics, normalization | Comparing patients requires common geometry and size-normalized indices | in progress |
| 5. Body composition from CT | Muscle / subcutaneous fat / visceral fat quantification at L3 | The destination | in progress |
| 6. Body composition from MR | Dixon water–fat separation, fat-fraction maps | The other modality that can do this, and why CT is still the default | in progress |

## Running the notebooks

**Google Colab is the primary target.** Every notebook opens with a setup cell that detects Colab,
clones this repository, and installs what the runtime is missing.

| Chapter | |
|---|---|
| 1. Exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter1_Exploration.ipynb) |
| 2. Masks and filters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter2_Masks_and_Filters.ipynb) |
| 3. Measurement | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choilab-jefferson/medimage/blob/main/Chapter3_Measurement.ipynb) |

Locally, the same setup cell walks up to the repository root instead:

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

This keeps the repository small and the licensing clean, but the real reason is pedagogical: when
the provenance of an image is one function call away, it stays visible, and you always know what the
numbers you compute are actually measuring.

| Dataset | Used by | Source | License | Transfer |
|---|---|---|---|---|
| **Pancreas-CT** — contrast-enhanced abdominal CT, 80 subjects | Ch 1–5 | [TCIA](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU), via the public NBIA REST API | CC BY 3.0 | ~40 MB per subject |
| **CHAOS** — T1-DUAL in/out-of-phase abdominal MRI | Ch 3, 6 | [Zenodo](https://doi.org/10.5281/zenodo.3431873) | CC BY-NC-SA 4.0 | ~9 MB (partial download) |

Neither dataset is redistributed here. The CHAOS archive is 890 MB, but the notebook pulls only the
few megabytes it uses via HTTP range requests. Both collections are de-identified at source.

`medimage_data.py` holds the loaders:

```python
import medimage_data as md

ct_dir = md.fetch_pancreas_ct(0)            # downloads + caches, prints citation
vol, spacing, datasets = md.load_series(ct_dir)   # HU volume, (dz, dy, dx) in mm

mr_dir = md.fetch_chaos_t1dual(1)           # partial download from Zenodo
```

## Chapters 5 and 6

**Chapter 5 (CT)** follows a `data → image → features → modeling` pipeline on an abdominal CT:
locate the L3 vertebral level, segment skeletal muscle, subcutaneous fat and visceral fat, derive
the standard cross-sectional area and index metrics, extend them to radiomics features, and scale
the workflow to a cohort. It uses two tools on top of the course material:

* **[qradiomics](https://github.com/choilab-jefferson/qradiomics)** — DICOM → NRRD conversion,
  manifest handling, PyRadiomics feature extraction, cohort workflow and statistics.
* **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** — a pretrained nnU-Net
  labeling 117 anatomical structures, used to pick the vertebral level automatically and to
  separate the abdominal cavity from the abdominal wall (something an HU threshold cannot do).

```bash
# TotalSegmentator in its own venv so its torch pin cannot clash with the analysis kernel
python -m venv .venv-ts
.venv-ts/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-ts/bin/pip install TotalSegmentator
```

**Chapter 6 (MR)** derives water and fat images from in-phase / out-of-phase acquisitions, builds a
fat-fraction map, and segments subcutaneous and visceral adipose tissue — then states plainly where
2-point Dixon falls short of true PDFF.

## License

Code and notebook text: MIT (see [LICENSE](LICENSE)). The datasets remain under the terms of their
original sources, listed above.
