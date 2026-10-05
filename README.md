# Stunting target-leakage audit (E1–E5)

Code for the manuscript *When Accuracy Is an Artifact: A Target-Leakage Audit of Machine
Learning Models for Child Stunting Classification in Indonesia*.

## 1. Setup (once)
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q             # must show: 13 passed
```

## 2. Data (included in `data/`)
```
data/jeneponto/2021.xlsx ... 2024.xlsx      Mendeley v4, doi:10.17632/wzwpc9j5bx.4
data/lombok/overal data.xlsx                 Mendeley v2, doi:10.17632/z4mrz37dm3.2
data/lombok/pre-intervention/*.xlsx
data/lombok/post-intervention/*.xlsx
```
`Overall Data.xlsx` is identical to the four yearly files (checked row by row) and is not read.

## 3. Run (full settings: 5 x 5 CV, 1000 bootstraps)
```
python run_all.py --dataset jeneponto        # about 30-60 min on a laptop
python run_all.py --dataset lombok           # about 1-2 min
pip freeze > results/environment.txt
```

`--quick` (1 repeat) is only for checking that the code runs. Do not report quick results.

On a slow machine you can run the Jeneponto audit in chunks (each result is cached as a CSV):
```
python run_chunk.py all E12      # then: all E3, all E1z, all E5b_abl, all E5b_full
python run_chunk.py 2024 E12     # then: 2024 E3, 2024 E1z, 2024 E5b_abl, 2024 E5b_full
```
Overfitting check (Table X): `python run_chunk.py all OF_full`, `all OF_abl`, `2024 OF_full`, `2024 OF_abl`.
The manuscript tables were produced with this chunked run (full settings, 5 x 5 CV, 1000 bootstraps).

## 4. What the data checks found (handled automatically, all counts in data_report.json)
| Finding | Handling |
|---|---|
| Jeneponto label "Stunted" = provided "Z-Score H/A" < -2 in 100% of rows | reported as evidence that labels are rule-derived |
| Age is in completed months | +0.5 month (capped at 60); mean HAZ difference to provided z-score 0.14 -> 0.01 |
| 20,158 weights stored by Excel as dates (e.g. 13.5 -> 13 May 2025) | restored to day + month/10; validated against provided W/A z-score |
| 216 provided z-scores stored as dates | set to missing (used only for checks) |
| 2021-2023 files contain only stunted children | analyses run on all years (published setting) and on 2024 only |
| No child identifier in Jeneponto | E5a (group leakage) not run -> limitation |
| Lombok overall file has categories only (no height/weight) | E1-L categorical replication; WHO rule cannot be recomputed |
| Lombok pre/post files: only stunted children, no sex variable | E4 not feasible -> E4_feasibility.json |

## 5. Output files -> manuscript
| File | Use |
|---|---|
| `results/jeneponto/data_report.json`, `label_agreement.csv` | IV.A Data characteristics |
| `results/jeneponto/subset_*/table_V.csv`, `E1_E2_E3.csv` | Table V, IV.B, IV.C (DeLong/McNemar vs. rule) |
| `results/jeneponto/subset_*/fig1_ablation.png` | Fig. 1 |
| `results/jeneponto/subset_*/E5b_smote.csv` | IV.E Pipeline leakage |
| `results/lombok/E1L_categorical.csv` | Replication of the categorical (H/A-as-feature) setup |
| `results/lombok/E4_feasibility.json` | Why E4 was not possible |

## 6. Method details to report (TRIPOD+AI)
- HAZ/WAZ: WHO 2006 LMS tables (monthly, linear interpolation); length table < 24 months, height
  table >= 24 months; no position correction (position not recorded).
- Exclusions: missing age/sex/height, age outside 0–60 months, |HAZ| > 6, WAZ < -6 or > 5.
- Target: the original label in the file. The WHO rule uses -HAZ as its AUROC score.
- CV: stratified 5-fold, 5 repeats; metrics are means over repeats; 95% CI from 1000 bootstrap
  resamples of the first repeat's out-of-fold predictions.
- E5b "SMOTE before split" is evaluated on partly synthetic folds on purpose (the leaky practice).

## 7. Data source and citation
Both datasets are public on Mendeley Data and are redistributed here unchanged for reproducibility.
Please cite the original datasets and check their licence on the Mendeley pages:
- A. H. RS et al., Dataset Stunting and Nutritional Status of Toddler from Jeneponto Regency, South Sulawesi, Indonesia, Mendeley Data, v4, doi:10.17632/wzwpc9j5bx.4
- A. A. Firdaus, Child stunting status dataset from East Lombok district, West Nusa Tenggara, Indonesia, Mendeley Data, v2, doi:10.17632/z4mrz37dm3.2

WHO reference tables: WHO Child Growth Standards (2006), monthly LMS parameters for length/height-for-age and weight-for-age.
