import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbbuild import build, badge, SETUP

FN = "Chapter01_Exploration.ipynb"

cells = [
("md", f"""\
# Quantitative Medical Image Analysis with Python

{badge(FN)}

**A hands-on course: from opening your first medical image file to measuring how much muscle and
fat a person has, from their CT scan.**

Every chapter builds one piece of that final measurement.

| Chapter | What you learn | Why the course needs it |
|---|---|---|
| **Part I — Foundations** | | |
| **1. Exploration** | Open a CT scan, read the information attached to it, understand what the numbers mean | You cannot measure anything until the numbers mean something |
| **2. Masks and filters** | Select the pixels you care about, and clean up the noise first | Fat and muscle are picked out by their brightness values |
| **3. Measurement** | Count, label and measure the regions you selected | Areas and averages *are* the body composition numbers |
| **4. Image comparison** | Compare scans from different people fairly | Two patients are different sizes, so raw numbers cannot be compared directly |
| **5. Patient privacy** | Find, remove and *verify the removal of* identifying information | Before any of this can touch real clinical data |
| **Part II — Applications** | | |
| **6. Body composition from CT** | Measure muscle and fat at the standard anatomical level | The destination the first four chapters were building toward |
| **7. Fat quantification with MR** | Do the same with MRI, which separates fat differently | The other way to do it, and why CT is still the usual choice |
| **8. PET/CT** | Measure how hard tissue is *working*, not just where it is | Function as well as anatomy |
| **Part III — Quantitative methods** | | |
| **9. Radiomics features** | What the hundreds of numbers extracted from a tumor actually are | Knowing which of them survive someone else running your pipeline |
| **10. Registration** | Align two scans, and catch the failures that report success | Aligning scans is where pipelines break without saying so |
| **11. Delta radiomics** | Measure change between timepoints, and measure your own noise floor | Change means nothing until you know what change means nothing |
| **12. Classification** | Benchmark models, and recognize a result that is too good to be true | An implausibly good result is a bug report, not a finding |
| **13. Reproducing published results** | Run a full study and compare it against the paper | Whether a pipeline reproduces is the question that matters |

Chapters 1 to 5 are the foundations, and they run in order. After those, Part II and Part III can
be read in any order — each says at the top which earlier chapters it leans on.

### What is body composition, and why measure it?

Two people can weigh exactly the same and be in very different health. What matters is *what* the
weight is made of — muscle or fat, and where the fat sits. People with little muscle recover worse
from surgery, tolerate chemotherapy worse, and generally do worse than their weight alone predicts.

A CT scan already contains this information. It is a stack of cross-sections through the body, and
in each cross-section muscle and fat look different. If you can teach a computer to recognize and
count them, you get a measurement that a scale can never give you.

That is what this course builds, step by step.

### About the data

This repository contains **no medical images**. Each notebook downloads what it needs from the
public archive that hosts it, and prints where it came from. Nothing here is a mystery file of
unknown origin — you can always trace a number back to the scan it came from.

This chapter uses an abdominal CT scan from the **Pancreas-CT** collection at
[The Cancer Imaging Archive](https://www.cancerimagingarchive.net/). No account is needed.

---

# Chapter 1: Exploration

A medical image is two things at once: a grid of numbers, and a description of what those numbers
mean. Beginners see only the first, and that is where the mistakes come from.

By the end of this chapter you will be able to:

1. Download a real CT scan and load it as a 3D grid of numbers.
2. Read the information stored alongside the image, and explain why it is what makes measurement possible.
3. Convert the stored numbers into **Hounsfield units** — the scale that lets you tell fat from muscle.
4. Work out which way is up (this is less obvious than it sounds, and getting it wrong is common).
5. Turn a scan into a PyTorch tensor, ready for machine learning.
6. Look at a 3D scan from any direction, without distorting it.

### Before you start

| | |
|---|---|
| **Builds on** | Nothing — this is the first chapter |
| **You should know** | Basic Python, and what a NumPy array is |
| **Downloads** | One Pancreas-CT subject, about 40 MB |
| **Longest wait** | The download in section 1, roughly 20 seconds |
| **Beyond the setup cell** | Nothing extra to install |
| **Hardware** | Any laptop. No GPU needed |

Run the cells in order from the top — each one uses names defined by the one before it. If a cell
fails partway through the chapter, re-run from the setup cell rather than retrying it in place.
"""),

("md", """\
## Setup

Run this cell first. On Google Colab it downloads the course files and installs what is missing.
On your own computer it just finds the project folder.
"""),
("code", SETUP),

("md", """\
## 1. Getting a scan

The helper below downloads one patient's CT scan and remembers it, so you only wait once. It is
about 40 MB and takes roughly 20 seconds.
"""),
("code", """\
import medimage_data as md

ct_dir = md.fetch_pancreas_ct(0)
"""),

("md", """\
## 2. What is inside one file?

Medical images are stored in a format called **DICOM**. One DICOM file is one cross-section
("slice") through the body, and a whole scan is a folder full of them.

A DICOM file is not just a picture. Along with the pixels it stores a few hundred labeled facts:
how big each pixel is in millimeters, how far apart the slices are, what kind of scanner took it,
and so on. Those facts are called the **header**.

We read DICOM with a library called **pydicom**.
"""),
("code", """\
import pydicom

paths = sorted(ct_dir.glob("*.dcm"))
ds = pydicom.dcmread(paths[len(paths) // 2])     # a slice from the middle of the scan

print("Slices in this scan:", len(paths))
print("Pixels in one slice:", ds.pixel_array.shape)
print("Numbers stored as:  ", ds.pixel_array.dtype)
print("Smallest / largest: ", ds.pixel_array.min(), "/", ds.pixel_array.max())
"""),

("md", """\
### The header is what makes it a measurement

A photograph from your phone has pixels but no physical meaning — you cannot tell from it how many
centimeters wide something is. A CT slice can tell you, because the header says how much space each
pixel covers.

Here are the entries that matter most.
"""),
("code", """\
for tag in ["Modality", "BodyPartExamined", "PixelSpacing", "SliceThickness",
            "RescaleSlope", "RescaleIntercept", "PatientID", "PatientSex",
            "PatientAge", "InstitutionName"]:
    print(f"{tag:20s} {getattr(ds, tag, '(not present)')}")
"""),

("md", """\
Two things are worth noticing.

**Some entries are empty.** `PatientAge` and `PatientSex` are blank because the archive removes
anything that could identify a patient before publishing. So real code has to cope with missing
entries — that is why the code below asks for values with a fallback instead of assuming they exist.

**"Slice thickness" is not the same as "slice spacing".** Thickness is how thick a single
cross-section is. Spacing is the distance from one cross-section to the next. They are usually the
same, but not always — slices can overlap. The safe way to get the spacing is to measure the actual
distance between slice positions, which the loader used later does for you.
"""),

("md", """\
### Hounsfield units: the scale that makes CT measurable

The numbers stored in a CT file do not mean anything on their own. The header carries two values
that convert them onto a proper scale:

$$\\text{HU} = \\text{stored number} \\times \\text{RescaleSlope} + \\text{RescaleIntercept}$$

The result is in **Hounsfield units (HU)**, named after the engineer who invented the CT scanner.
The scale is fixed by definition:

- water is **0 HU**
- air is **−1000 HU**

Because those two anchors are always the same, a given tissue lands at roughly the same HU value on
any scanner, in any hospital, in any country. That is the entire reason this course works: a rule
like *"fat is between −190 and −30 HU"* can be written down once and used everywhere.

**Forgetting this conversion is the single most common beginner mistake.** The picture still looks
fine, and every number you calculate from it is wrong.
"""),
("code", """\
import numpy as np

raw = ds.pixel_array
slope = float(getattr(ds, "RescaleSlope", 1))
intercept = float(getattr(ds, "RescaleIntercept", 0))
hu = raw * slope + intercept

print("stored numbers:", raw.min(), "to", raw.max())
print("Hounsfield:    ", hu.min(), "to", hu.max())
print()
print("A corner pixel is outside the body, so it should be close to -1000 (air):",
      round(float(hu[5, 5]), 1))
"""),

("md", """\
Here is the whole scale in one table. The two highlighted rows are the ones this course is built on.

| Tissue | Hounsfield units |
|---|---|
| Air | about −1000 |
| Lung | −1000 to −500 |
| **Fat** | **−190 to −30** |
| Water | 0 |
| **Muscle** | **−29 to +150** |
| Bone | +300 and up |

Notice that fat and muscle do not overlap. That gap is what makes them separable, and it is why
Chapter 6 can measure them just by counting pixels in the right range.
"""),
("code", """\
TISSUE = {
    "air":    (-1024, -600),
    "lung":   (-600, -400),
    "fat":    (-190, -30),
    "muscle": (-29, 150),
    "bone":   (150, 3000),
}

for name, (lo, hi) in TISSUE.items():
    fraction = np.count_nonzero((hu >= lo) & (hu < hi)) / hu.size
    print(f"{name:8s} {lo:5d} to {hi:5d} HU   {fraction:6.1%} of the image")
"""),

("md", """\
The "air" share is huge because most of the picture is the empty space *around* the patient, not
the patient. Before measuring anything you have to find the body first — that is one of the first
jobs in Chapter 2. Measuring without doing that is another easy way to get a confident wrong answer.
"""),

("md", """\
## 3. Which way is up?

Now let us actually look at the slice. Plot the numbers exactly as they are stored, and notice
where the spine ends up.
"""),
("code", """\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.imshow(hu, cmap="gray", vmin=-160, vmax=240)
ax.set_title("as stored — the spine is at the top")
ax.axis("off")
plt.show()
"""),

("md", """\
The spine is at the top and the belly at the bottom. That is upside down.

The convention in medicine is that you view a cross-section **as if you were standing at the
patient's feet, looking up toward their head**. So the front of the body belongs at the top of the
picture, and the spine at the bottom.

Nothing is broken — the file simply says which way its rows and columns run, and we ignored it. The
header entry is called `ImageOrientationPatient`, and it holds six numbers.
"""),
("code", """\
orientation = [float(v) for v in ds.ImageOrientationPatient]

print("ImageOrientationPatient:", orientation)
print("  first three  (which way the columns run):", orientation[:3])
print("  second three (which way the rows run):   ", orientation[3:])
"""),

("md", """\
To read those six numbers you need the coordinate system they use, which is:

- **x** points to the patient's **left**
- **y** points to the patient's **back**
- **z** points up toward the patient's **head**

Each group of three says which of those directions you travel in as you move along the image.

A typical CT has rows running `[0, 1, 0]` — moving down the rows takes you toward the patient's
back. So the first row is the front of the body, and it gets drawn at the top. Correct.

This scan has rows running `[0, -1, 0]` — the opposite. Moving down the rows takes you toward the
*front*, so the first row is the back, and the spine gets drawn on top.

This matters more than it might seem. The upside-down picture still looks like a perfectly
reasonable abdomen, so nothing warns you. The same kind of mistake in the left–right direction
would swap a patient's left and right, and that has reached print in published papers. **Read the
six numbers. Never assume them.**
"""),
("code", """\
columns_run, rows_run = orientation[:3], orientation[3:]

if rows_run[1] < 0:        # rows run toward the front instead of the back
    hu = hu[::-1, :]       # flip top to bottom
if columns_run[0] < 0:     # columns run toward the right instead of the left
    hu = hu[:, ::-1]       # flip left to right

fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.imshow(hu, cmap="gray", vmin=-160, vmax=240)
ax.set_title("corrected — belly up, spine down")
ax.axis("off")
plt.show()
"""),

("md", """\
`md.load_series`, used from here on, applies this correction for you. It also puts the slices in
head-to-foot order, so that side views come out the right way up as well.
"""),

("md", """\
## 4. Windowing: choosing what you can see

A CT scan covers about 4000 Hounsfield units, from air to dense bone. A screen shows only 256
shades of gray, and your eye tells apart fewer than that.

So if you spread the whole range across the screen, every soft tissue — fat, muscle, liver, kidney —
is squeezed into a couple of nearly identical grays, and you can see none of it.

The fix is called **windowing**: pick a narrow slice of the range and stretch just that across the
full black-to-white scale. Everything below becomes solid black, everything above solid white, and
the part you care about gets all the contrast.

A window is described by its **center** (where it sits) and its **width** (how wide it is). In code
that is just `vmin` and `vmax`. The data never changes — only what you can see.
"""),
("code", """\
def window(center, width):
    \"\"\"Convert a radiology-style window into (vmin, vmax) for imshow.\"\"\"
    return center - width / 2, center + width / 2


views = {
    "everything at once\\n(useless)": (hu.min(), hu.max()),
    "soft tissue\\ncenter 40, width 400": window(40, 400),
    "fat\\ncenter -95, width 160": window(-95, 160),
    "bone\\ncenter 500, width 2000": window(500, 2000),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, (title, (vmin, vmax)) in zip(axes, views.items()):
    ax.imshow(hu, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.suptitle("The same slice, four windows — same data, different settings", y=1.04)
plt.tight_layout()
plt.show()
"""),

("md", """\
The fat window is the interesting one for this course. It is set to the fat range from the table
above, so fat lights up and almost everything else goes white — which is, in visual form, exactly
the measurement Chapter 6 makes by counting.
"""),

("md", """\
## 5. From image to PyTorch tensor

Later chapters hand images to neural networks — including the one Chapter 6 uses to find organs.
Networks do not read DICOM files; they take **tensors**, which are essentially NumPy arrays that
can run on a GPU.

Three conventions catch people out:

1. **Shape.** PyTorch expects four dimensions: `(batch, channel, height, width)`. One slice is only
   `(height, width)`, so two dimensions have to be added.
2. **Number type.** Networks want decimals (`float32`); DICOM gives whole numbers.
3. **Range.** Values from −1000 to +3000 make training unstable. Squash them into roughly 0 to 1.

One detail is easy to get wrong. We squash using **fixed** HU limits, not each image's own minimum
and maximum. If you scaled every scan by its own range, the same tissue would end up as a different
number in different scans — throwing away the very thing that makes CT measurable.
"""),
("code", """\
import torch


def ct_to_tensor(hu_image, center=40.0, width=400.0):
    \"\"\"Window a CT slice and return it as a (1, 1, H, W) float32 tensor.\"\"\"
    lo, hi = window(center, width)
    scaled = (np.clip(hu_image, lo, hi) - lo) / (hi - lo)      # now between 0 and 1
    return torch.from_numpy(scaled.astype(np.float32))[None, None]


x = ct_to_tensor(hu)

print("PyTorch version:", torch.__version__)
print("tensor shape:   ", tuple(x.shape), " (batch, channel, height, width)")
print("number type:    ", x.dtype)
print("range:          ", round(float(x.min()), 2), "to", round(float(x.max()), 2))
"""),

("md", """\
## 6. The third dimension

So far we have looked at one slice. A scan is a whole stack of them.

The idea is simple and worth saying out loud: a row of numbers is 1D, a grid of rows is 2D (one
slice), a stack of slices is 3D (the scan), and a series of scans over time would be 4D. Each time
you add a dimension you are stacking the previous thing.

By convention the new dimension goes first, so a scan is indexed `volume[slice, row, column]`.
"""),
("code", """\
fig, ax = plt.subplots(figsize=(7, 3.2))
for i, offset in enumerate([0, 0.9, 1.8, 2.7]):
    ax.add_patch(plt.Rectangle((offset, -offset * 0.28), 1.5, 1.0,
                               facecolor=f"C{i}", alpha=0.55, edgecolor="black"))
    ax.text(offset + 0.12, -offset * 0.28 + 0.12, f"slice {i}", fontsize=9)
ax.set_xlim(-0.3, 4.6)
ax.set_ylim(-1.2, 1.3)
ax.set_title("A scan is a stack of slices:  volume[slice, row, column]")
ax.axis("off")
plt.show()
"""),

("md", """\
### Loading the whole scan

You could read the files one by one and stack them, but there is a trap: file names do not
necessarily run in anatomical order. Stack them in the wrong order and the body gets scrambled,
quietly.

`load_series` reads the folder, sorts the slices by their real physical position, converts to
Hounsfield units, fixes the orientation, and works out the true spacing.
"""),
("code", """\
vol, spacing, datasets = md.load_series(ct_dir)
dz, dy, dx = spacing

print("scan shape:", vol.shape, " (slices, rows, columns)")
print(f"spacing:    {dz:.2f} mm between slices, {dy:.2f} x {dx:.2f} mm per pixel")
print("HU range:  ", vol.min(), "to", vol.max())
"""),

("md", """\
### Size in pixels versus size in centimeters

Three ideas that are easy to mix up, and mixing them up makes every measurement wrong by a fixed
factor:

- **Shape** — how many pixels there are. Just a property of the array.
- **Spacing** — how many millimeters one pixel covers. Comes from the header.
- **Field of view** — the real size of the whole scan: shape × spacing.

Chapters 3 and 6 measure areas by counting pixels and multiplying by the area of one pixel. So
spacing is the bridge between "number of pixels" and "square centimeters".
"""),
("code", """\
n_slices, n_rows, n_cols = vol.shape

print(f"pixels     : {n_slices} x {n_rows} x {n_cols}")
print(f"real size  : {n_slices * dz:.0f} x {n_rows * dy:.0f} x {n_cols * dx:.0f} mm")
print(f"             ({n_slices * dz / 10:.1f} cm of the body from top to bottom)")
print()
print(f"one pixel covers {dy * dx:.4f} mm^2  =  {dy * dx / 100:.6f} cm^2")
"""),

("md", """\
That last number does the real work in Chapter 6. A muscle area in square centimeters is nothing
more than *the number of muscle pixels* × *the area of one pixel*.
"""),

("md", """\
## 7. Looking at a 3D scan

You cannot show a 3D object in one 2D picture, so you slice it and show several slices at once.

One practical note: when you make a grid of plots, draw on each panel directly
(`axes[i].imshow(...)`) rather than calling `plt.imshow(...)`, which has no way of knowing which
panel you meant.
"""),
("code", """\
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, idx in zip(axes, np.linspace(0, n_slices - 1, 4).astype(int)):
    ax.imshow(vol[idx], cmap="gray", vmin=-160, vmax=240)
    ax.set_title(f"slice {idx}", fontsize=10)
    ax.axis("off")
fig.suptitle("Slices from the top of the scan down to the bottom", y=1.02)
plt.tight_layout()
plt.show()
"""),

("md", """\
### The other two directions

Slicing the stack the way we just did gives cross-sections — the **axial** view. But a 3D array can
be cut along any axis. Cut it the other two ways and you get the view from the front (**coronal**)
and the view from the side (**sagittal**). Same data, different direction.

There is a catch. In this scan a pixel is about 0.9 mm across, but the slices are 1 mm apart, and in
many scans that gap is much bigger — 5 mm or more. In the front and side views one axis is
therefore spaced differently from the other, and the picture comes out stretched or squashed unless
you tell matplotlib the real proportions with `aspect`.
"""),
("code", """\
coronal = vol[:, n_rows // 2, :]
sagittal = vol[:, :, n_cols // 2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(coronal, cmap="gray", vmin=-160, vmax=240, aspect=dz / dx)
axes[0].set_title("coronal — seen from the front", fontsize=10)
axes[1].imshow(sagittal, cmap="gray", vmin=-160, vmax=240, aspect=dz / dy)
axes[1].set_title("sagittal — seen from the side", fontsize=10)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),

("md", """\
The side view is how Chapter 6 finds where to measure. Body composition is measured at one standard
place — the level of the third lumbar vertebra, partway down the spine — and this is the view in
which you can count vertebrae to find it.

## Recap

The six things promised at the top, and where each of them landed:

| | |
|---|---|
| **A scan is a folder of slices** | `md.load_series` reads it into `vol[slice, row, column]` |
| **The header is what makes it a measurement** | `PixelSpacing` turns pixel counts into centimeters; ask for fields with a fallback, because some are empty |
| **Hounsfield units** | `stored × RescaleSlope + RescaleIntercept`; water 0, air −1000, fat −190 to −30, muscle −29 to +150 |
| **Orientation** | `ImageOrientationPatient` says which way rows and columns run — read it, never assume it |
| **Tensors** | Window with *fixed* HU limits, not each image's own range, or the same tissue gets a different number in every scan |
| **Three directions** | Axial, coronal and sagittal are the same array cut three ways; pass `aspect` or the picture is stretched |

Two of these are silent-failure traps, which is why they got the most space: an upside-down scan and
a scan that skipped the Hounsfield conversion both look completely reasonable, and every number you
compute from either is wrong.

**Next:** Chapter 2 takes this scan, whose numbers now mean something, and turns those numbers into
selections — deciding for each pixel whether it is fat, muscle, or not even part of the patient.

## Exercises

1. Run `md.fetch_pancreas_ct(1)` to fetch a second patient. Compare the pixel spacing and the field
   of view with the first. If you measured something in pixels, would the two patients be
   comparable? What if you measured in square centimeters?

   *Hint:* call `md.load_series` on both and print `spacing` and `shape` for each. Section 6's
   field-of-view calculation is the one to repeat.

2. Work out the tissue percentages for the very first and very last slice of `vol`. Why is there so
   much more lung in one of them?

   *Hint:* reuse the `TISSUE` loop from section 2 with `vol[0]` and `vol[-1]`. Then look at the two
   slices — the answer is anatomical, not numerical.

3. Take the correction in section 3 back out and plot the coronal view without it. What changes,
   and would you have noticed if nobody had told you?

   *Hint:* `md.load_series` already applied the fix, so undo it with `vol[:, ::-1, :]`. The point of
   the exercise is the second question.

4. Skip the Hounsfield conversion on purpose: calculate the "fat percentage" straight from
   `ds.pixel_array`, and compare it with the answer from `hu`.

   *Hint:* on **this** scan the two agree exactly, because its header says `RescaleSlope 1` and
   `RescaleIntercept 0` — the stored numbers are already Hounsfield units. That is the exercise. You
   cannot tell from the picture, or from the answer looking sensible, whether the conversion
   mattered; only the header says. Now simulate a scanner that stores it differently
   (`raw = hu + 1024`, intercept −1024, which is a common convention) and redo the count without
   converting. That is the size of the mistake you avoid by reading two header fields.

## References

- [pydicom documentation](https://pydicom.github.io/pydicom/stable/)
- [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- Hounsfield GN. *Computed medical imaging.* Science. 1980;210(4465):22–28.
- Roth H, Farag A, Turkbey EB, Lu L, Liu J, Summers RM (2016). *Data From Pancreas-CT.*
  The Cancer Imaging Archive.
  [doi:10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
"""),
]

build(pathlib.Path(__file__).resolve().parent.parent / FN, cells)
