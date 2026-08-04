import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbbuild import build, badge, SETUP

FN = "Chapter05_Patient_Privacy.ipynb"

cells = [
("md", f"""\
# Chapter 5: Patient Privacy

{badge(FN)}

Every scan in this course came from a public archive, already de-identified by someone else. Your
own data will not.

A DICOM file that comes off a hospital scanner carries the patient's name, their medical record
number, their date of birth, the referring physician, the institution — several hundred fields, most
of which have nothing to do with the image. Move that file to a laptop, a shared drive, or a cloud
notebook, and you have moved identifiable medical records with it.

This chapter is about the step that has to happen before any of the previous eight can be applied to
real data.

By the end you will be able to:

1. Find the identifying information hidden in a DICOM header.
2. Remove it, at a defined standard, keeping everything the analysis needs.
3. **Verify** that it is gone, rather than assuming.
4. Explain why dates get shifted instead of deleted, and why a pseudonym is not the same as anonymity.

> **This chapter teaches a technique, not a compliance sign-off.** What is legally required depends
> on your jurisdiction, your institution and your study. Talk to your IRB or data protection office.
> No notebook can do that for you.
"""),

("md", "## Setup"),
("code", SETUP),
("code", """\
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "qradiomics"], check=True)
"""),
("code", """\
import pathlib
import shutil

import pydicom

import medimage_data as md

# Start from an empty directory. This chapter writes a file with invented identifiers,
# anonymizes it, and reads the mapping back. Leftovers from an earlier run would get
# anonymized a second time, and the pseudonyms below would chain off each other instead
# of off the original identifier.
WORK = pathlib.Path("work/privacy")
if WORK.exists():
    shutil.rmtree(WORK)
(WORK / "original").mkdir(parents=True, exist_ok=True)
"""),

("md", """\
## 1. Building a file that has something to hide

The public data has already been cleaned, so there would be nothing to find. To have something to
work with, we take four real slices and write **invented** identifiers into them — the sort of
values a scanner would actually produce.

Every name, number and date below is fictional.
"""),
("code", """\
FAKE_PHI = {
    "PatientName": "DOE^JANE^Q",
    "PatientID": "MRN-4471902",
    "PatientBirthDate": "19551103",
    "PatientSex": "F",
    "InstitutionName": "Example General Hospital",
    "InstitutionAddress": "1 Example Way, Springfield",
    "ReferringPhysicianName": "REFERRER^A",
    "StudyDate": "20240117",
    "StudyTime": "142233",
    "AccessionNumber": "ACC-99120043",
    "StudyDescription": "CT ABDOMEN W CONTRAST",
}

source = sorted(md.fetch_pancreas_ct(0).glob("*.dcm"))[:4]

for i, path in enumerate(source):
    ds = pydicom.dcmread(path)
    for tag, value in FAKE_PHI.items():
        setattr(ds, tag, value)
    ds.save_as(WORK / "original" / f"slice{i:03d}.dcm", enforce_file_format=True)

print(f"wrote {len(source)} files carrying invented identifiers")
"""),

("md", """\
## 2. What is actually in there

Most people picture a DICOM header as a handful of technical fields. Print the whole thing and the
scale of the problem becomes obvious.
"""),
("code", """\
ds = pydicom.dcmread(WORK / "original" / "slice000.dcm")

print(f"total header elements: {len(ds)}\\n")

for tag in FAKE_PHI:
    print(f"  {tag:24s} {getattr(ds, tag, '')!r}")
"""),

("md", """\
It is worth pausing on how little of that is needed to measure anything. Chapters 1 to 4 used exactly
five header fields: `Modality`, `PixelSpacing`, `SliceThickness`, `ImagePositionPatient` and
`RescaleSlope`/`RescaleIntercept`. Everything else in the list above is administrative.

That asymmetry is what makes de-identification practical: the fields that identify a person and the
fields that make an image measurable are almost entirely different sets.
"""),

("md", """\
## 3. Removing it

`qr anonymize` applies a named policy rather than a hand-written list of tags. Three are available:

| Profile | What it does |
|---|---|
| `basic` | Clears the obvious identifiers |
| `safe-harbor` | Follows the HIPAA Safe Harbor list — the strict option |
| `tcia` | Matches what public archives apply before publishing |

Two flags matter beyond the profile. `--replace-pid` swaps the patient identifier for a pseudonym
rather than blanking it, so the slices of one patient still group together. `--pid-salt` seeds that
pseudonym, so the same patient gets the same code across separate runs of the same study — and a
*different* code in a different study, which stops two datasets from being cross-linked.
"""),
("code", """\
result = subprocess.run([
    "qr", "anonymize",
    "-i", str(WORK / "original"),
    "-o", str(WORK / "clean"),
    "--profile", "safe-harbor",
    "--replace-pid",
    "--pid-salt", "CourseDemo2026",
    "--mapping", str(WORK / "pid_map.csv"),
], capture_output=True, text=True)

print(result.stdout.strip()[-400:])
"""),
("code", """\
before = pydicom.dcmread(WORK / "original" / "slice000.dcm")
after = pydicom.dcmread(WORK / "clean" / "slice000.dcm")

print(f"{'tag':<24}{'before':<32}{'after'}")
print("-" * 84)
for tag in list(FAKE_PHI) + ["Modality", "PixelSpacing", "RescaleIntercept"]:
    b = str(getattr(before, tag, "(absent)"))[:30]
    a = str(getattr(after, tag, "(absent)"))[:30]
    print(f"{tag:<24}{b:<32}{a}")
"""),

("md", """\
### Two things that did not simply disappear

**The patient identifier became a pseudonym**, not a blank. Blanking it would merge every patient in
the cohort into one anonymous blob, and no per-patient analysis would be possible. The pseudonym is
a salted hash, so it cannot be reversed without the salt, and the original-to-pseudonym mapping is
written to a separate file that stays behind wherever the identifiable data stays.

**The dates moved rather than vanished.** Compare them.
"""),
("code", """\
from datetime import datetime


def parse(value):
    return datetime.strptime(value, "%Y%m%d")


for tag in ["PatientBirthDate", "StudyDate"]:
    b, a = getattr(before, tag), getattr(after, tag)
    shift = (parse(a) - parse(b)).days
    print(f"{tag:20s} {b} -> {a}   shifted by {shift:+d} days")

print()
print("Both dates moved by the same amount, so every interval is preserved:")
b_gap = (parse(before.StudyDate) - parse(before.PatientBirthDate)).days
a_gap = (parse(after.StudyDate) - parse(after.PatientBirthDate)).days
print(f"  age at scan, before shifting: {b_gap / 365.25:.2f} years")
print(f"  age at scan, after  shifting: {a_gap / 365.25:.2f} years")
"""),

("md", """\
This is the detail people get wrong most often. Deleting dates outright feels safer and quietly
destroys the data: Chapter 8 compares two PET scans months apart, and the survival analysis in
Chapter 13 depends entirely on elapsed time. Both need intervals.

Shifting every date in a patient's record by the same random offset keeps every interval intact
while breaking the link to the real calendar. Note that the offset must be **per patient** — one
shared offset across the cohort would leave the relative timing of patients intact and could be
undone by anyone who knows one real date.
"""),
("code", """\
import pandas as pd

mapping = pd.read_csv(WORK / "pid_map.csv")
print(mapping.to_string(index=False))
print()
print("This file is the re-identification key. It belongs wherever the identifiable")
print("data belongs — never alongside the de-identified copy you share.")
"""),

("md", """\
One row, one direction — but only because the salt is fixed and the mapping was kept. `qr phi tokens`
reads that store from either end.
"""),
("code", """\
token = mapping.anon_pid.iloc[0]

forward = subprocess.run(["qr", "phi", "tokens", "lookup", str(mapping.original_pid.iloc[0]),
                          "--mapping", str(WORK / "pid_map.csv")],
                         capture_output=True, text=True, check=True)
reverse = subprocess.run(["qr", "phi", "tokens", "lookup", str(token),
                          "--mapping", str(WORK / "pid_map.csv"), "--by", "token"],
                         capture_output=True, text=True, check=True)

print(f"the pseudonym that gets shared:      {token}")
print(f"looked up from the original ID:      {forward.stdout.strip()}")
print(f"looked up backwards from the token:  {reverse.stdout.strip()}")
"""),

("md", """\
The last line is the point of the chapter's fourth objective. The pseudonym in the shared files is
not anonymous — it is *reversible by whoever holds this file*. That is a feature: it is how you
answer a query about a specific patient two years later, and how you withdraw someone who revokes
consent. It is also why the mapping never travels with the data it unlocks.

Delete the mapping and you have anonymized rather than pseudonymized: nobody can re-identify the
patients, including you, including when you need to.

## 4. Verifying, not assuming

An anonymizer that silently missed a tag looks exactly like one that worked. Chapter 3 introduced
Dice for the same reason: the output of a process is not evidence that the process was correct.

`qr phi audit` re-reads the files and searches for anything that still looks identifying — unblanked
name fields, values shaped like medical record numbers, identifiers embedded in file names. It exits
non-zero on any finding, which makes it usable as a gate in a script rather than something a human
has to remember to eyeball.
"""),
("code", """\
for label, folder in [("BEFORE anonymization", WORK / "original"),
                      ("AFTER anonymization", WORK / "clean")]:
    audit = subprocess.run(["qr", "phi", "audit", str(folder), "--limit", "6"],
                           capture_output=True, text=True)
    # The audit reports absolute paths; show them relative to the repository so the
    # output does not depend on where the checkout happens to live.
    root = str(pathlib.Path.cwd()) + "/"
    findings = [line.replace(root, "") for line in audit.stdout.splitlines() if line.strip()]
    print(f"=== {label} — exit code {audit.returncode} ===")
    for line in findings[:5]:
        print("   ", line.split("::")[-1].strip()[:100] if "::" in line else line[:100])
    print()
"""),

("md", """\
Non-zero before, zero after. That is the check worth building into any export step.

### What an audit like this cannot catch

Being clear about the limits matters more than the reassurance:

- **Pixel data.** Ultrasound images, screen captures and scanned documents often have the patient's
  name *burned into the image itself*. No header cleaner touches that; it needs the pixels examined.
- **Private tags.** Vendors store extra information in tags outside the standard, and identifiers
  turn up there.
- **Re-identification from the image.** A head CT contains the face. Rendering the surface of a
  full-head scan reconstructs recognizable features, which is why defacing tools exist for brain
  imaging.
- **Uniqueness.** A rare diagnosis in a small hospital in a known month can identify someone with no
  name attached at all. This is a property of the dataset, not of any one file.

**De-identification reduces risk. It does not reach zero**, and any statement that a dataset is
"anonymous" should be read as "we removed the identifiers we knew to look for".

## 5. The order of operations

Putting it together, the sequence for real data is:

1. Copy from the source — **never anonymize in place**, because a bug in the middle leaves you with
   neither the original nor a clean copy.
2. `qr anonymize --profile safe-harbor --replace-pid --pid-salt <study specific>`
3. `qr phi audit` on the output, and stop if it is non-zero.
4. Store the mapping file with the identifiable data, not with the shared copy.
5. Only then start Chapter 1.

## Exercises

1. Run the anonymizer with `--profile basic` instead of `safe-harbor` and audit the result. Which
   fields survive, and can you construct a case where one of them identifies someone?
2. Change `--pid-salt` and re-run. Does the same patient get the same pseudonym? Why does that
   matter for combining two datasets — and why is it also a risk?
3. `StudyInstanceUID` is not a name, but it is unique to one study. Check whether it changed. Should
   it have?
4. Write a check that confirms `PixelSpacing`, `RescaleSlope` and `ImagePositionPatient` are
   unchanged. Why is that check as important as the PHI audit itself?

## References

- [HIPAA Safe Harbor de-identification standard](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
- DICOM PS3.15 Annex E — *Attribute Confidentiality Profiles.*
  [dicom.nema.org](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html)
- Freymann JB, Kirby JS, Perry JH, Clunie DA, Jaffe CC. *Image data sharing for biomedical research —
  meeting HIPAA requirements for de-identification.* J Digit Imaging. 2012;25(1):14–24.
- Schwarz CG, Kremers WK, Therneau TM, et al. *Identification of anonymous MRI research participants
  with face-recognition software.* N Engl J Med. 2019;381(17):1684–1686.
"""),
]

build(pathlib.Path(__file__).resolve().parent.parent / FN, cells)
