"""Run one (subset, experiment) chunk of the Jeneponto audit; results are cached as CSV."""
import json, sys, time
from pathlib import Path
import pandas as pd
import config as cfg
from src import data as D, experiments as E
import run_all as R

sub, exp = sys.argv[1], sys.argv[2]
dcfg = cfg.DATASETS["jeneponto"]
df, report = R.load_jeneponto(dcfg)
root = Path("results/jeneponto"); root.mkdir(parents=True, exist_ok=True)
(root / "data_report.json").write_text(json.dumps(report, indent=2))
D.label_agreement_table(df).to_csv(root / "label_agreement.csv")
year = dcfg["subsets"][sub]
d = df if year is None else df[df["year"] == year].reset_index(drop=True)
out = root / f"subset_{sub}"; out.mkdir(parents=True, exist_ok=True)
full, ablated, t = ["age_month", "sex", "height", "weight"], ["age_month", "sex", "weight"], "status_raw"
t0 = time.time()
if exp == "E12":
    r = E.cross_sectional(d, full, t, cfg, "E1/E2 full inputs")
elif exp == "E3":
    r = E.cross_sectional(d, ablated, t, cfg, "E3 without height")
elif exp == "E1z":
    r = E.cross_sectional(d, ["haz_provided"], t, cfg, "E1 provided H/A z-score only", False, True)
    r = r[r.model != "WHO rule (no training)"]
elif exp == "E5b_abl":
    r = E.smote_leakage(d, ablated, t, cfg)
elif exp == "E5b_full":
    r = E.smote_leakage(d, full, t, cfg)
r.to_csv(out / f"part_{exp}.csv", index=False)
print(sub, exp, f"{time.time()-t0:.0f}s")
