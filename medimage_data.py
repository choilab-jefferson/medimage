"""Data access for the *medimage* course.

**No imaging data is stored in this repository.** Every dataset is downloaded
from its original public source the first time it is needed and cached under
``data/`` (git-ignored). Each loader prints where the data came from and how it
should be cited, so the provenance of anything you measure is always visible.

Sources used by the course
--------------------------

``Pancreas-CT``
    Contrast-enhanced abdominal CT, 80 subjects, from The Cancer Imaging
    Archive. Downloaded through the public NBIA REST API — no account needed.
    https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU  (CC BY 3.0)

``CHAOS``
    T1-DUAL in-phase / out-of-phase abdominal MRI, from the CHAOS challenge,
    hosted on Zenodo. Only the few megabytes actually used are pulled, via HTTP
    range requests, so the 890 MB archive is never downloaded in full.
    https://doi.org/10.5281/zenodo.3431873  (CC BY-NC-SA 4.0)
"""

from __future__ import annotations

import json
import pathlib
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pydicom

CACHE = pathlib.Path("data")

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
CHAOS_ZIP = "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1"

CITATIONS = {
    "pancreas-ct": (
        "Roth H, Farag A, Turkbey EB, Lu L, Liu J, Summers RM (2016). Data From "
        "Pancreas-CT. The Cancer Imaging Archive. "
        "https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU  (CC BY 3.0)"
    ),
    "chaos": (
        "Kavur AE, Gezer NS, Baris M, et al. (2021). CHAOS Challenge - combined "
        "(CT-MR) healthy abdominal organ segmentation. Medical Image Analysis, "
        "69:101950. Data: https://doi.org/10.5281/zenodo.3431873  (CC BY-NC-SA 4.0)"
    ),
}


def cite(key: str) -> None:
    """Print the citation for a dataset."""
    print("Source:", CITATIONS[key])


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response, open(dest, "wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    return dest


# --------------------------------------------------------------------------- #
# Pancreas-CT (TCIA)
# --------------------------------------------------------------------------- #

def pancreas_series(limit: int | None = None) -> list[dict]:
    """List the Pancreas-CT series, sorted so that indices are reproducible."""
    series = _get_json(f"{NBIA}/getSeries?Collection=Pancreas-CT")
    series.sort(key=lambda s: s["SeriesInstanceUID"])
    return series[:limit] if limit else series


def fetch_pancreas_ct(index: int = 0, quiet: bool = False) -> pathlib.Path:
    """Download one Pancreas-CT series and return the folder holding its DICOMs.

    ``index`` selects a subject from the sorted series list, so the same index
    always gives the same patient. About 40 MB per subject.
    """
    meta = pancreas_series()[index]
    uid = meta["SeriesInstanceUID"]
    out = CACHE / "pancreas-ct" / meta["PatientID"]

    if not out.is_dir() or not any(out.glob("*.dcm")):
        archive = CACHE / "pancreas-ct" / f"{meta['PatientID']}.zip"
        if not quiet:
            print(f"Downloading {meta['PatientID']} ({meta['ImageCount']} slices) from TCIA ...")
        _download(f"{NBIA}/getImage?SeriesInstanceUID={urllib.parse.quote(uid)}", archive)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
        archive.unlink()

    if not quiet:
        print(f"{meta['PatientID']}: {len(list(out.glob('*.dcm')))} slices in {out}")
        cite("pancreas-ct")
    return out


# --------------------------------------------------------------------------- #
# CHAOS T1-DUAL (Zenodo, partial download)
# --------------------------------------------------------------------------- #

def fetch_chaos_t1dual(subject: int = 1, quiet: bool = False) -> pathlib.Path:
    """Download one CHAOS subject's T1-DUAL series (~9 MB) and return its folder.

    Uses HTTP range requests so only the requested subject is transferred, not
    the full 890 MB archive.
    """
    out = CACHE / "chaos" / str(subject)
    marker = out / "T1DUAL" / "DICOM_anon" / "InPhase"

    if not marker.is_dir() or not any(marker.iterdir()):
        try:
            from remotezip import RemoteZip
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise SystemExit("pip install remotezip") from exc

        if not quiet:
            print(f"Fetching CHAOS subject {subject} from Zenodo (partial download) ...")
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"Train_Sets/MR/{subject}/T1DUAL/"
        with RemoteZip(CHAOS_ZIP) as zf:
            names = [n for n in zf.namelist() if prefix in n and not n.endswith("/")]
            if not names:
                raise FileNotFoundError(f"CHAOS subject {subject} not found")
            for name in names:
                target = out / name.split(f"/MR/{subject}/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    if not quiet:
        n = len(list(marker.glob("*.dcm")))
        print(f"CHAOS subject {subject}: {n} in-phase slices in {out}")
        cite("chaos")
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_series(folder, hounsfield: bool = True):
    """Read a folder of DICOM slices into a volume in radiological orientation.

    Slices are ordered head first, and each frame is flipped as required by
    ``ImageOrientationPatient`` so that ``imshow`` draws anterior at the top and
    the patient's left on the right. Ignoring those direction cosines is a
    common source of silently upside-down images: a scanner is free to store
    rows running back-to-front, and many abdominal series do.

    Returns ``(volume, spacing, datasets)`` where ``spacing`` is
    ``(dz, dy, dx)`` in millimetres. When ``hounsfield`` is true the rescale
    slope and intercept from the header are applied, so the values are in HU.
    """
    folder = pathlib.Path(folder)
    datasets = [pydicom.dcmread(p) for p in sorted(folder.glob("*.dcm"))]
    if not datasets:
        raise FileNotFoundError(f"no DICOM files in {folder}")

    # Head first: the most superior slice becomes index 0, so coronal and
    # sagittal reformats come out the right way up.
    datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]), reverse=True)

    # Direction cosines of the row and column axes, in LPS coordinates.
    orientation = [float(v) for v in getattr(datasets[0], "ImageOrientationPatient",
                                             [1, 0, 0, 0, 1, 0])]
    col_dir, row_dir = orientation[:3], orientation[3:]
    flip_rows = row_dir[1] < 0      # rows run posterior -> anterior
    flip_cols = col_dir[0] < 0      # columns run right -> left

    frames = []
    for d in datasets:
        frame = d.pixel_array.astype(np.float32)
        if hounsfield:
            frame = frame * float(getattr(d, "RescaleSlope", 1)) + float(
                getattr(d, "RescaleIntercept", 0)
            )
        if flip_rows:
            frame = frame[::-1, :]
        if flip_cols:
            frame = frame[:, ::-1]
        frames.append(frame)
    volume = np.stack(frames).astype(np.int16 if hounsfield else np.float32)

    dy, dx = (float(v) for v in datasets[0].PixelSpacing)
    if len(datasets) > 1:
        z = [float(d.ImagePositionPatient[2]) for d in datasets]
        dz = float(np.median(np.diff(z)))
    else:
        dz = float(getattr(datasets[0], "SliceThickness", 1) or 1)

    return volume, (abs(dz), dy, dx), datasets
