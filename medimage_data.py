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
import time
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
    "acrin": (
        "Machtay M, Duan F, Siegel BA, et al. (2013). ACRIN 6668/RTOG 0235 "
        "trial; data from The Cancer Imaging Archive: ACRIN-NSCLC-FDG-PET. "
        "https://doi.org/10.7937/tcia.2019.30ilqfcl  (TCIA data usage policy)"
    ),
    "lung1": (
        "Aerts HJWL, Wee L, Rios Velazquez E, et al. (2019). Data From "
        "NSCLC-Radiomics. The Cancer Imaging Archive. "
        "https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI  (CC BY-NC 3.0)"
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

def fetch_chaos_t1dual(subject: int = 1, quiet: bool = False,
                       attempts: int = 6) -> pathlib.Path:
    """Download one CHAOS subject's T1-DUAL series (~9 MB) and return its folder.

    Uses HTTP range requests so only the requested subject is transferred, not
    the full 890 MB archive. That means one request per file, which adds up:
    fetching several subjects in a row reliably trips Zenodo's rate limit and
    comes back as HTTP 429. Each file is therefore retried with an increasing
    delay, and already-written files are skipped so a retry resumes rather than
    starting over.
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

        for attempt in range(attempts):
            try:
                with RemoteZip(CHAOS_ZIP) as zf:
                    names = [n for n in zf.namelist()
                             if prefix in n and not n.endswith("/")]
                    if not names:
                        raise FileNotFoundError(f"CHAOS subject {subject} not found")
                    for name in names:
                        target = out / name.split(f"/MR/{subject}/", 1)[1]
                        if target.exists() and target.stat().st_size:
                            continue          # resume where a retry left off
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(name) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                break
            except FileNotFoundError:
                raise
            except Exception as exc:          # rate limiting, dropped connections
                if attempt == attempts - 1:
                    raise
                delay = 5 * 2 ** attempt
                if not quiet:
                    print(f"  {type(exc).__name__}: retrying in {delay}s "
                          f"({attempt + 1}/{attempts - 1})")
                time.sleep(delay)

    if not quiet:
        n = len(list(marker.glob("*.dcm")))
        print(f"CHAOS subject {subject}: {n} in-phase slices in {out}")
        cite("chaos")
    return out


# --------------------------------------------------------------------------- #
# ACRIN-NSCLC-FDG-PET  (ACRIN 6668)
# --------------------------------------------------------------------------- #

def acrin_series(patient: str = "ACRIN-NSCLC-FDG-PET-001") -> list[dict]:
    """List one ACRIN 6668 patient's series, oldest study first."""
    series = _get_json(f"{NBIA}/getSeries?Collection=ACRIN-NSCLC-FDG-PET")
    mine = [s for s in series if s["PatientID"] == patient]
    mine.sort(key=lambda s: (str(s.get("SeriesDate")), s["SeriesInstanceUID"]))
    return mine


def fetch_acrin(patient: str = "ACRIN-NSCLC-FDG-PET-001",
                modality: str = "PT",
                description: str = "PET WB",
                timepoint: int = 0,
                quiet: bool = False) -> pathlib.Path:
    """Download one ACRIN 6668 series and return its folder.

    ``timepoint`` selects among the matching series in date order, so 0 is the
    pre-treatment scan and 1 the follow-up.
    """
    matches = [s for s in acrin_series(patient)
               if s["Modality"] == modality
               and str(s.get("SeriesDescription", "")).strip() == description]
    if not matches:
        raise LookupError(f"no {modality} series described '{description}' for {patient}")
    meta = matches[timepoint]

    short = patient.rsplit("-", 1)[-1]
    out = CACHE / "acrin" / f"{short}_{modality}_{timepoint}"

    if not out.is_dir() or not any(out.glob("*.dcm")):
        archive = out.with_suffix(".zip")
        if not quiet:
            print(f"Downloading {patient} {modality} '{description}' "
                  f"timepoint {timepoint} ({meta['ImageCount']} slices) ...")
        _download(
            f"{NBIA}/getImage?SeriesInstanceUID={urllib.parse.quote(meta['SeriesInstanceUID'])}",
            archive)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
        archive.unlink()

    if not quiet:
        print(f"{out}: {len(list(out.glob('*.dcm')))} files "
              f"(study date {str(meta.get('SeriesDate'))[:10]})")
        cite("acrin")
    return out


# --------------------------------------------------------------------------- #
# NSCLC-Radiomics (Lung1) cohort — CT + tumor contour + clinical outcomes
# --------------------------------------------------------------------------- #

LUNG1_CLINICAL = ("https://www.cancerimagingarchive.net/wp-content/uploads/"
                  "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")


def fetch_lung1_cohort(n_patients: int = 60, work=None, quiet: bool = False):
    """Prepare a Lung1 subset and return the paths every later chapter needs.

    Runs the whole `qr` chain once — download, RTSTRUCT conversion, crop and
    resample, feature extraction, clinical merge — and caches every artifact,
    so any chapter can call this without depending on another having been run
    first. Returns a dict of paths.

    Requires the ``qr`` CLI plus ``rt-utils`` and ``opencv-python-headless``;
    the ordinary OpenCV build needs a graphics library that servers lack, and
    without the headless one every contour conversion fails on ``libGL.so.1``.
    """
    import subprocess

    import pandas as pd

    work = pathlib.Path(work or CACHE.parent / "work" / "lung1")
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "work": work,
        "series": work / "series.csv",
        "nrrd": work / "nrrd",
        "cropped": work / "cropped",
        "features": work / "features.csv",
        "clinical": work / "clinical.csv",
        "analysis_ready": work / "analysis_ready.csv",
    }
    say = (lambda *a: None) if quiet else print

    if not paths["series"].exists():
        say("Listing NSCLC-Radiomics series ...")
        subprocess.run(["qr", "tcia", "series", "--collection", "NSCLC-Radiomics",
                        "-o", str(paths["series"])], check=True, capture_output=True)
    series = pd.read_csv(paths["series"])

    with_ct = set(series.loc[series.Modality == "CT", "PatientID"])
    with_rt = set(series.loc[series.Modality == "RTSTRUCT", "PatientID"])
    patients = sorted(with_ct & with_rt)[:n_patients]
    targets = series[series.PatientID.isin(patients)
                     & series.Modality.isin(["CT", "RTSTRUCT"])]
    target_csv = work / "targets.csv"
    targets.to_csv(target_csv, index=False)

    dicom_dir = work / "dicom"
    if not dicom_dir.is_dir() or len(list(dicom_dir.glob("*"))) < len(patients):
        say(f"Downloading {len(patients)} patients from TCIA ...")
        subprocess.run(["qr", "tcia", "download", "--manifest", str(target_csv),
                        "-o", str(dicom_dir), "-j", "8"], check=True, capture_output=True)

    paths["nrrd"].mkdir(exist_ok=True)
    ct_rows = targets[targets.Modality == "CT"]
    rt_rows = targets[targets.Modality == "RTSTRUCT"]

    def series_dir(row):
        return dicom_dir / row.PatientID / row.StudyInstanceUID / row.SeriesInstanceUID

    # Per-patient conversions warn and skip rather than raising: the manifest below is
    # built from whatever landed on disk, so one unreadable series costs one patient
    # instead of the whole cohort. The cohort-wide steps after it do use check=True.
    for _, row in ct_rows.iterrows():
        out = paths["nrrd"] / f"{row.PatientID}_image.nrrd"
        if not out.exists():
            done = subprocess.run(["qr", "convert", "dicom-series", "-i", str(series_dir(row)),
                                   "-o", str(out)], capture_output=True, text=True)
            if done.returncode:
                say(f"  {row.PatientID}: CT conversion failed, skipping")
    for _, row in rt_rows.iterrows():
        out = paths["nrrd"] / f"{row.PatientID}_mask.nrrd"
        match = ct_rows[ct_rows.PatientID == row.PatientID]
        if out.exists() or match.empty:
            continue
        # --roi GTV-1 is not optional: without it the converter takes the first
        # contour in the structure set, which in Lung1 is often a lung or the cord.
        done = subprocess.run(["qr", "convert", "rtstruct", "-d", str(series_dir(match.iloc[0])),
                               "-r", str(series_dir(row)), "--roi", "GTV-1", "-o", str(out)],
                              capture_output=True, text=True)
        if done.returncode:
            say(f"  {row.PatientID}: no GTV-1 contour in the structure set, skipping")

    manifest = paths["nrrd"] / "manifest.csv"
    subprocess.run(["qr", "convert", "manifest-from-dir", "-d", str(paths["nrrd"]),
                    "--image-glob", "*_image.nrrd", "--mask-glob", "*_mask.nrrd",
                    "-o", str(manifest)], check=True, capture_output=True)
    subprocess.run(["qr", "preprocess", "-m", str(manifest), "-o", str(paths["cropped"]),
                    "--pad-mm", "5", "--resample", "1.0", "--jobs", "4",
                    "--out-manifest", str(paths["cropped"] / "manifest.csv")],
                   check=True, capture_output=True)

    if not paths["features"].exists():
        say("Extracting features (nsclc-survival pattern) ...")
        subprocess.run(["qr", "extract", "-m", str(paths["cropped"] / "manifest.csv"),
                        "-p", "nsclc-survival", "-o", str(paths["features"]), "-j", "4"],
                       check=True, capture_output=True)

    if not paths["clinical"].exists():
        _download(LUNG1_CLINICAL, paths["clinical"])

    if not paths["analysis_ready"].exists():
        subprocess.run(["qr", "results", "merge", "-f", str(paths["features"]),
                        "-c", str(paths["clinical"]), "--clinical-id-col", "PatientID",
                        "--time-col", "Survival.time", "--event-col", "deadstatus.event",
                        "-o", str(paths["analysis_ready"])], check=True, capture_output=True)

    if not quiet:
        ready = pd.read_csv(paths["analysis_ready"])
        print(f"Lung1 cohort ready: {len(ready)} patients, "
              f"{ready.shape[1] - 3} features in {work}")
        cite("lung1")
    return paths


CHAOS_LABELS = {63: "liver", 126: "right kidney", 189: "left kidney", 252: "spleen"}


def load_chaos(subject: int = 1):
    """Load one CHAOS subject: in-phase, out-of-phase and the reference masks.

    The reference masks ship as PNG files named in acquisition order, while
    :func:`load_series` returns slices head first. Pairing the two without
    accounting for that silently mismatches every slice, so the reordering is
    done here once.

    Returns ``(in_phase, out_phase, labels, spacing)``. ``labels`` holds the
    values in :data:`CHAOS_LABELS`; 0 is background.
    """
    import imageio.v2 as iio

    folder = fetch_chaos_t1dual(subject, quiet=True)
    base = folder / "T1DUAL"

    in_phase, spacing, datasets = load_series(base / "DICOM_anon" / "InPhase",
                                              hounsfield=False)
    out_phase, _, _ = load_series(base / "DICOM_anon" / "OutPhase", hounsfield=False)

    # Each reference PNG shares its file name with the DICOM slice it belongs to,
    # so pair them by name rather than by position. Guessing the order from the
    # slice geometry is what silently mismatches every slice.
    ground = base / "Ground"
    labels = np.stack([
        iio.imread(ground / (pathlib.Path(d.filename).stem + ".png"))
        for d in datasets
    ])

    return in_phase, out_phase, labels, spacing


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
    ``(dz, dy, dx)`` in millimeters. When ``hounsfield`` is true the rescale
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
