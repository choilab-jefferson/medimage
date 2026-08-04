import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbbuild import build, badge, SETUP

FN = "Chapter12_Classification.ipynb"

cells = [
("md", f"""\
# Chapter 12: Classification, and the Result That Was Too Good

{badge(FN)}

Chapter 13 predicts survival time. This chapter predicts a yes-or-no answer — a **classification** —
and compares eight models on the same data to see which wins.

It also contains a mistake. Not a hypothetical one: a real data leak that appeared while writing
this notebook, produced a perfect score, and is reproduced here exactly as it happened. Learning to
recognize that pattern is worth more than any of the models.

By the end you will be able to:

1. Turn clinical records into a binary label without accidentally encoding the answer.
2. Rank features by how well each separates two groups.
3. Benchmark several models under nested cross-validation.
4. **Recognize a result that is too good, find the leak that caused it, and prove it.**
5. Explain why picking the best of eight models is itself a way to fool yourself.
"""),

("md", "## Setup"),
("code", SETUP),
("code", """\
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "qradiomics", "statsmodels", "lightgbm", "xgboost"], check=True)
"""),
("code", """\
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import medimage_data as md

paths = md.fetch_lung1_cohort(60)
WORK = paths["work"]

features = pd.read_csv(paths["analysis_ready"])
clinical = pd.read_csv(paths["clinical"])
print()
print(f"{len(features)} patients, {features.shape[1] - 3} radiomic features")
"""),

("md", """\
## 1. Two labels

**Histology: squamous or not.** A genuine radiomics question — can texture see the tumor subtype
that a pathologist reports?

**Survival to two years.** Derived from the follow-up record: a patient is labeled 1 if they died
within 24 months.

The second label needs care. "Not dead within 24 months" is not the same as "alive at 24 months" —
a patient last seen at 8 months is neither. In this cohort nobody is censored before 24 months, so
the label is clean. That is worth checking rather than assuming, because the alternative silently
mislabels the very patients the model most needs to get right.
"""),
("code", """\
labels = clinical[["PatientID", "Histology"]].rename(columns={"PatientID": "patient_id"})
data = features.merge(labels, on="patient_id")

data["squamous"] = data.Histology.str.contains("squamous", case=False, na=False).astype(int)
data["dead_2y"] = ((data.OS_months < 24) & (data.OS_event == 1)).astype(int)
data = data.drop(columns=["Histology"])

censored_early = ((data.OS_months < 24) & (data.OS_event == 0)).sum()
print(f"squamous          {data.squamous.sum():3d} / {len(data)}")
print(f"died within 2 yrs {data.dead_2y.sum():3d} / {len(data)}")
print(f"censored before 24 months (would be unusable): {censored_early}")
"""),

("md", """\
## 2. Which single features separate the groups?

`qr analyze classify` fits a logistic regression per feature and reports the area under the ROC
curve — the probability that a randomly chosen positive case scores higher than a randomly chosen
negative one. 0.5 is a coin toss.
"""),
("code", """\
data.to_csv(WORK / "classify_squamous.csv", index=False)

subprocess.run(["qr", "analyze", "classify", "-i", str(WORK / "classify_squamous.csv"),
                "--outcome", "squamous", "-o", str(WORK / "univariate.csv")],
               check=True, capture_output=True)

univariate = pd.read_csv(WORK / "univariate.csv")
print(f"{len(univariate)} features tested, {(univariate.p < 0.05).sum()} with p < 0.05")
print(f"expected by chance alone: about {0.05 * len(univariate):.0f}")
print()
print(univariate.nsmallest(6, "p").to_string(index=False))
"""),

("md", """\
The same arithmetic applies here as in Chapter 13. Testing about eleven hundred features at p < 0.05
yields roughly fifty-five "significant" ones from pure noise. A single small p-value drawn from a
list that long is not evidence of anything.

Which is why the real question is not "does any feature look associated" but "does a model built
from them predict patients it has never seen".
"""),

("md", """\
## 3. Benchmarking eight models

`qr ml benchmark` runs nested cross-validation: an outer loop holds out patients for testing, and
an inner loop tunes hyperparameters using only the training portion. That separation is what keeps
the tuning from quietly leaking test information into the model.
"""),
("code", """\
def benchmark(input_csv, outcome, outdir):
    subprocess.run(["qr", "ml", "benchmark", "-i", str(input_csv), "-o", outcome,
                    "--output-dir", str(outdir), "--cv", "5", "--inner-cv", "3"],
                   check=True, capture_output=True)
    return pd.read_csv(pathlib.Path(outdir) / "benchmark_cv_results.csv")


squamous_results = benchmark(WORK / "classify_squamous.csv", "squamous", WORK / "bench_squamous")
print(squamous_results[["model", "oof_auc", "cv_auc_mean", "cv_auc_std"]].to_string(index=False))
"""),

("md", """\
Every model lands between about 0.39 and 0.52 — chance, with noise. Radiomics does not predict
squamous histology in fifty-nine patients.

Note what the tool nonetheless reports: a **best** model. With eight candidates scored on the same
data, one of them wins by luck alone, and its score is biased upward precisely because it was chosen
for being highest. "We evaluated eight models and selected the best" is a multiple-comparisons
problem wearing a methods-section disguise. The honest summary of this table is that the *spread*
contains 0.5.
"""),

("md", """\
## 4. Now the other label

Same features, same procedure, different outcome.
"""),
("code", """\
mortality_results = benchmark(WORK / "classify_squamous.csv", "dead_2y", WORK / "bench_leaky")
print(mortality_results[["model", "oof_auc", "oof_ap", "cv_auc_mean", "cv_auc_std"]]
      .to_string(index=False))
"""),

("md", """\
### Stop.

Four models at **AUC 1.000**, with a standard deviation of zero across folds. Perfect prediction of
who dies within two years, from CT texture.

That is not a discovery. Cancer outcomes are not perfectly predictable from anything, least of all
from fifty-nine patients. A result like this has one realistic explanation: **the answer is
somewhere in the input.**

This is the single most useful reflex in this chapter. An implausibly good score is a bug report.
Chase it before you celebrate it.
"""),

("md", """\
## 5. Finding the leak

Where could the answer be hiding? The label was built like this:

```python
data["dead_2y"] = ((data.OS_months < 24) & (data.OS_event == 1)).astype(int)
```

and `OS_months` and `OS_event` are **still columns in the table** handed to the model. The label is
a deterministic function of two of its own inputs. Any model that finds them can reconstruct the
answer exactly — and a decision tree finds them immediately.

The proof takes one line.
"""),
("code", """\
reconstructed = ((data.OS_months < 24) & (data.OS_event == 1)).astype(int)
print("label reconstructable from OS_months and OS_event alone:",
      bool((reconstructed == data.dead_2y).all()))
print()
print("columns the model was given that derive from the outcome:")
for column in ["OS_months", "OS_event"]:
    print(f"  {column}")
"""),
("code", """\
clean = data.drop(columns=["OS_months", "OS_event", "squamous"])
clean.to_csv(WORK / "classify_clean.csv", index=False)

clean_results = benchmark(WORK / "classify_clean.csv", "dead_2y", WORK / "bench_clean")
print(clean_results[["model", "oof_auc", "oof_ap", "cv_auc_mean", "cv_auc_std"]]
      .to_string(index=False))
"""),
("code", """\
comparison = (mortality_results[["model", "oof_auc"]]
              .rename(columns={"oof_auc": "with_leak"})
              .merge(clean_results[["model", "oof_auc"]]
                     .rename(columns={"oof_auc": "leak_removed"}), on="model")
              .sort_values("with_leak", ascending=False))

print(comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
y = np.arange(len(comparison))
ax.barh(y - 0.2, comparison.with_leak, height=0.4, color="tab:red", alpha=0.85,
        label="with the leak")
ax.barh(y + 0.2, comparison.leak_removed, height=0.4, color="tab:blue", alpha=0.85,
        label="leak removed")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
ax.text(0.505, len(comparison) - 0.4, "chance", fontsize=8)
ax.set_yticks(y, comparison.model)
ax.set_xlabel("out-of-fold AUC")
ax.set_title("The same models, before and after removing two columns")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),

("md", """\
Removing two columns takes the best model from a perfect 1.000 to about 0.65. Nothing else changed —
same features, same models, same cross-validation.

### Why this kind of leak is so easy to miss

The mistake was not exotic. It came from a completely ordinary sequence: merge the clinical table in,
derive a label from it, forget that the source columns are still sitting there. Notice that the
squamous run in section 3 had the same structure and did **not** blow up, because `Histology` was
dropped when the label was made. One line of difference.

Leaks that are worth watching for in imaging studies:

- **Outcome-derived columns**, as here.
- **Patient identifiers**, if they encode anything — a site prefix, a chronological ID.
- **Acquisition metadata** that correlates with the outcome: if the sick patients were scanned on the
  newer scanner, slice thickness predicts survival and the model learns the hospital's workflow.
- **Repeated patients** split across train and test — the model recognizes the person, not the disease.
- **Preprocessing fitted on all the data**, such as normalizing with a mean computed before the split.

The general defense is to ask one question of every column: *could this have been known only after
the outcome was?* If yes, it does not belong in the input.

## 6. What the honest result looks like

After the fix, the best model reaches roughly 0.65 out-of-fold on two-year mortality, and the spread
across models is wide. That is a weak signal in a small cohort — plausible, unremarkable, and worth
reporting exactly as it is.

It is also, unavoidably, a much less exciting sentence than the one available ten minutes earlier.
That asymmetry is the reason leaks survive to publication: the result that is wrong is the one that
looks like a finding.

## Exercises

1. Add `Overall.Stage` as a feature and re-run the mortality benchmark. The AUC will rise. Is that a
   leak, or is stage legitimately known before the outcome? Argue both sides.
2. Construct a leak on purpose: add a column equal to `dead_2y` plus a little noise, and find how
   much noise is needed before the AUC stops being suspicious.
3. Run the squamous benchmark with three different `--seed` values. How much does the winning model
   change? What does that say about reporting the best of eight?
4. Split the cohort by patient ID into two halves, fit on one and test on the other, and compare
   with the cross-validated number. Which would you put in a paper?

## References

- Kaufman S, Rosset S, Perlich C, Stitelman O. *Leakage in data mining: formulation, detection, and
  avoidance.* ACM Transactions on Knowledge Discovery from Data. 2012;6(4):1–21.
- Varoquaux G, Cheplygina V. *Machine learning for medical imaging: methodological failures and
  recommendations for the future.* npj Digital Medicine. 2022;5:48.
- Cawley GC, Talbot NLC. *On over-fitting in model selection and subsequent selection bias in
  performance evaluation.* Journal of Machine Learning Research. 2010;11:2079–2107.
- Aerts HJWL, et al. (2019). *Data From NSCLC-Radiomics.* The Cancer Imaging Archive.
  [doi:10.7937/K9/TCIA.2015.PF0M9REI](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI)
"""),
]

build(pathlib.Path(__file__).resolve().parent.parent / FN, cells)
