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
print("installed")
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

`qr anonymize` applies a policy rather than a hand-written list of tags: it clears the fields that
identify a person and leaves the ones that make the image measurable.

Two flags matter. `--replace-pid` swaps the patient identifier for a pseudonym rather than blanking
it, so the slices of one patient still group together. `--pid-salt` seeds that pseudonym, so the same
patient gets the same code across separate runs of the same study — and a *different* code in a
different study, which stops two datasets from being cross-linked. `--mapping` writes the
original-to-pseudonym table to a file of your choosing.
"""),
("code", """\
result = subprocess.run([
    "qr", "anonymize",
    "-i", str(WORK / "original"),
    "-o", str(WORK / "clean"),
    "--replace-pid",
    "--pid-salt", "CourseDemo2026",
    "--mapping", str(WORK / "pid_map.csv"),
], capture_output=True, text=True)

# The tool reports absolute paths; show them relative to the repository so the
# output does not depend on where the checkout lives.
print(result.stdout.strip()[-400:].replace(str(pathlib.Path.cwd()) + "/", ""))
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

**The dates were not treated the same way.** Look at the two of them in the table above: the birth
date is blank, and the study date came through untouched.

That is worth stopping on, because it is the chapter's thesis applied to the tool itself. A
de-identifier's default policy is a policy, not a guarantee, and this one does not do what a
longitudinal study needs.

Deleting a date outright feels safer and quietly destroys the data. Chapter 8 compares two PET scans
months apart; the survival analysis in Chapter 13 depends entirely on elapsed time. Both need
*intervals*, and a blank birth date means no age at scan. Meanwhile a surviving study date is a real
identifier: combined with a rare diagnosis and a hospital, it narrows the field fast.

What you want instead is to move every date in a patient's record by the same offset. Intervals
survive; the link to the real calendar does not. The offset must be **per patient** — one shared
offset across the cohort would leave the relative timing of patients intact, and anyone who knows a
single real date could undo it.

The tool does not do that here, so do it explicitly.
"""),
("code", """\
import random
from datetime import datetime, timedelta

DATE_TAGS = ["PatientBirthDate", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]


def parse(value):
    return datetime.strptime(value, "%Y%m%d")


# Seeded on the original identifier so one patient gets one offset, stable across
# runs. A real study would draw this once, store it with the mapping file, and
# never regenerate it from the identifier.
offset = timedelta(days=random.Random(before.PatientID).randint(-365, 365))

for path in sorted((WORK / "clean").glob("*.dcm")):
    source = pydicom.dcmread(WORK / "original" / path.name)
    shifted = pydicom.dcmread(path)
    for tag in DATE_TAGS:
        value = getattr(source, tag, "")
        if value:
            setattr(shifted, tag, (parse(value) + offset).strftime("%Y%m%d"))
    shifted.save_as(path, enforce_file_format=True)

after = pydicom.dcmread(WORK / "clean" / "slice000.dcm")

for tag in ["PatientBirthDate", "StudyDate"]:
    b, a = getattr(before, tag), getattr(after, tag)
    print(f"{tag:20s} {b} -> {a}   shifted by {(parse(a) - parse(b)).days:+d} days")

print()
print("Both dates moved by the same amount, so every interval is preserved:")
b_gap = (parse(before.StudyDate) - parse(before.PatientBirthDate)).days
a_gap = (parse(after.StudyDate) - parse(after.PatientBirthDate)).days
print(f"  age at scan, before shifting: {b_gap / 365.25:.2f} years")
print(f"  age at scan, after  shifting: {a_gap / 365.25:.2f} years")
"""),

("md", """\
Same interval, different calendar. That is the property the analysis needs and the one the
identifier loses.

It is also a reminder that "we ran the anonymizer" is not a description of what happened to your
data. Somebody has to read the output and decide whether the policy it applied is the policy the
study requires.
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
One row, one direction — but the table reads just as well from the other end.
"""),
("code", """\
token = mapping.anon_pid.iloc[0]
original = mapping.original_pid.iloc[0]

forward = mapping.loc[mapping.original_pid == original, "anon_pid"].iloc[0]
reverse = mapping.loc[mapping.anon_pid == token, "original_pid"].iloc[0]

print(f"the pseudonym that gets shared:      {token}")
print(f"looked up from the original ID:      {forward}")
print(f"looked up backwards from the token:  {reverse}")
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

So write the check. It is short enough to read, and reading it is the point — an audit you cannot
inspect is another thing you are trusting.

It asks two questions of every element in every file. **Is a field that must be empty non-empty?**
That is a fixed list of tags: names, addresses, telephone numbers, institution, physicians, accession
and admission identifiers. **Does any value still look like an identifier?** — a `SURNAME^FORENAME`
token, a run of eight or more digits, an email address, a phone number.

The second question is only asked of free-text fields. Dates and UIDs are structural, and a shifted
`PatientBirthDate` of `19551122` is eight digits that mean nothing to anybody — scanning it would
report a finding that is not one.
"""),
("code", """\
import re

MUST_BE_BLANK = ("PatientName", "PatientAddress", "PatientTelephone",
                 "InstitutionName", "InstitutionAddress",
                 "ReferringPhysician", "PerformingPhysician", "RequestingPhysician",
                 "OperatorsName", "StationName", "DeviceSerialNumber",
                 "AccessionNumber", "AdmissionID", "IssuerOfPatientID")

LOOKS_IDENTIFYING = {
    "a name": re.compile(r"\\b[A-Z]{2,}\\^[A-Z]{2,}\\b"),
    "a record number": re.compile(r"\\b\\d{8,}\\b"),
    "an email address": re.compile(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b"),
    "a phone number": re.compile(r"\\b(?:\\(\\d{3}\\)\\s*|\\d{3}[-.\\s])\\d{3}[-.\\s]\\d{4}\\b"),
}

# Narrative fields only. DA (date), TM (time), UI (uid) and the numeric VRs hold
# structural values, and the patterns above would fire on them spuriously.
FREE_TEXT = {"LO", "SH", "LT", "ST", "UT", "PN", "AE"}


def audit(folder):
    \"\"\"Re-read every file and report anything that still looks identifying.\"\"\"
    findings = []
    for path in sorted(pathlib.Path(folder).glob("*.dcm")):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        for element in ds:
            keyword, value = element.keyword, str(element.value or "")
            if not keyword or not value:
                continue
            if keyword.startswith(MUST_BE_BLANK):
                findings.append((path.name, f"{keyword} not blanked: {value!r}"))
            elif keyword != "PatientID" and element.VR in FREE_TEXT:
                # PatientID is expected to hold a pseudonym rather than nothing.
                for description, pattern in LOOKS_IDENTIFYING.items():
                    found = pattern.search(value)
                    if found:
                        findings.append(
                            (path.name, f"{keyword} looks like {description}: {found.group(0)!r}"))
                        break
    return findings


for label, folder in [("BEFORE anonymization", WORK / "original"),
                      ("AFTER anonymization", WORK / "clean")]:
    findings = audit(folder)
    print(f"=== {label} — {len(findings)} finding(s) ===")
    for name, detail in findings[:5]:
        print(f"    {name}  {detail[:78]}")
    if len(findings) > 5:
        print(f"    ... and {len(findings) - 5} more")
    print()
"""),

("md", """\
Findings before, none after. Wrapped in an `assert` or a non-zero exit, that becomes a gate in an
export script rather than something a human has to remember to eyeball.

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
2. `qr anonymize --replace-pid --pid-salt <study specific> --mapping <kept separately>`
3. Audit the output, and stop if anything is found.
4. Store the mapping file with the identifiable data, not with the shared copy.
5. Only then start Chapter 1.

## Exercises

1. Add a tag to `FAKE_PHI` that the anonymizer does not clear — `PatientComments` is a good one —
   and audit the result. Does the audit catch it? If not, what would you add to the checks?
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
