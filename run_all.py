"""Run the audit and write tables/figures to results/.

  python run_all.py --dataset jeneponto      # E1, E2, E3, E5b on 'all years' and '2024 only'
  python run_all.py --dataset lombok         # E1-L (categorical replication) + E4 feasibility report
  python run_all.py --dataset jeneponto --quick   # 1 repeat, fewer trees (about 5-10 min)
  python run_all.py --synthetic --quick      # code check without real data
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
from src import data as D
from src import experiments as E
from src import lombok as LB


def load_jeneponto(dcfg) -> tuple[pd.DataFrame, dict]:
    raw = D.read_files(dcfg["files"], dcfg.get("sheet", 0))
    std = D.standardize(raw, dcfg["columns"], cfg.SEX_MAP, cfg.STUNTED_LABELS, cfg.NOT_STUNTED_LABELS)
    std["child_id"] = D.make_child_id(raw, dcfg.get("child_id", []))
    df, rep = D.clean_and_label(std)
    df["child_id"] = std.loc[df.index, "child_id"] if std["child_id"] is not None else None
    return df.reset_index(drop=True), rep


def ablation_figure(e13: pd.DataFrame, out: Path):
    t = e13[e13.model != "WHO rule (no training)"]
    piv = t.pivot_table(index="model", columns="experiment", values="auroc")
    ax = piv.plot.bar(figsize=(6.5, 4.0), rot=0, ylim=(0.4, 1.02))
    rule = e13.loc[e13.model == "WHO rule (no training)", "auroc"].iloc[0]
    ax.axhline(rule, ls="--", lw=1, color="black", label=f"WHO rule, no training ({rule:.3f})")
    ax.set_ylabel("AUROC (mean of repeats)")
    ax.set_xlabel("")
    ax.legend(title="", fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=300)   # no title: IJIMAI puts captions below figures
    plt.close()


def table_v(e12: pd.DataFrame, tag: str) -> pd.DataFrame:
    t = e12.copy()
    t["Dataset and method"] = tag + " \u2013 " + t["model"]
    t["AUROC (95% CI)"] = t.apply(lambda r: f"{r.auroc:.3f} ({r.auroc_ci_low:.3f}\u2013{r.auroc_ci_high:.3f})", axis=1)
    for c, n in [("balanced_acc", "Bal. acc."), ("f1", "F1"), ("accuracy", "Accuracy"), ("kappa_vs_rule", "kappa vs. rule")]:
        t[n] = t[c].map("{:.3f}".format)
    return t[["Dataset and method", "AUROC (95% CI)", "Accuracy", "Bal. acc.", "F1", "kappa vs. rule"]]


def run_jeneponto(args, dcfg, out_root: Path):
    df, report = load_jeneponto(dcfg)
    print(json.dumps(report, indent=2))
    (out_root / "data_report.json").write_text(json.dumps(report, indent=2))
    D.label_agreement_table(df).to_csv(out_root / "label_agreement.csv")
    target = "status_raw"
    for sub_name, year in dcfg.get("subsets", {"all": None}).items():
        d = df if year is None else df[df["year"] == year].reset_index(drop=True)
        out = out_root / f"subset_{sub_name}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== subset {sub_name}: n={len(d)}, prevalence={d[target].mean():.3f}")
        full = ["age_month", "sex", "height", "weight"]
        ablated = ["age_month", "sex", "weight"]
        print("E1/E2 ...")
        e12 = E.cross_sectional(d, full, target, cfg, "E1/E2 full inputs", args.tune, args.quick)
        print("E3 ...")
        e3 = E.cross_sectional(d, ablated, target, cfg, "E3 without height", args.tune, args.quick)
        e13 = pd.concat([e12, e3])
        if d["haz_provided"].notna().any():
            print("E1 variant: provided z-score as the only feature ...")
            ev = E.cross_sectional(d, ["haz_provided"], target, cfg, "E1 provided H/A z-score only", False, True)
            e13 = pd.concat([e13, ev[ev.model != "WHO rule (no training)"]])
        e13.to_csv(out / "E1_E2_E3.csv", index=False)
        table_v(e12, f"JP-{sub_name}").to_csv(out / "table_V.csv", index=False)
        ablation_figure(pd.concat([e12, e3]), out / "fig1_ablation.png")
        print("E5b ...")
        pd.concat([E.smote_leakage(d, ablated, target, cfg, args.quick),
                   E.smote_leakage(d, full, target, cfg, args.quick)]).to_csv(out / "E5b_smote.csv", index=False)
    if not dcfg.get("child_id"):
        (out_root / "E5a_NOT_RUN.txt").write_text(
            "E5a (group leakage) not run: the dataset has no child identifier, so repeated records of the same "
            "child across years cannot be detected. Report this as a limitation.\n")


def run_lombok(args, out_root: Path):
    d, rep = LB.load_overall(cfg.LOMBOK, cfg.SEX_MAP)
    print(json.dumps(rep, indent=2))
    (out_root / "data_report_overall.json").write_text(json.dumps(rep, indent=2))
    print("E1-L ...")
    LB.categorical_replication(d, cfg, args.quick).to_csv(out_root / "E1L_categorical.csv", index=False)
    print("E4 feasibility ...")
    fz = LB.e4_feasibility(cfg.LOMBOK)
    print(json.dumps(fz, indent=2, default=str))
    (out_root / "E4_feasibility.json").write_text(json.dumps(fz, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["jeneponto", "lombok"], default="jeneponto")
    ap.add_argument("--quick", action="store_true", help="1 repeat, 200 bootstraps, fewer trees")
    ap.add_argument("--tune", action="store_true", help="nested CV grid search for RF/XGBoost")
    ap.add_argument("--synthetic", action="store_true", help="synthetic Jeneponto-like data (code check only)")
    args = ap.parse_args()
    if args.quick:
        cfg.N_REPEATS, cfg.N_BOOTSTRAP = 1, 200
    t0 = time.time()
    if args.synthetic:
        import make_synthetic
        make_synthetic.write_all()
        out = Path("results") / "SYNTHETIC_DO_NOT_REPORT"
        out.mkdir(parents=True, exist_ok=True)
        run_jeneponto(args, make_synthetic.CONFIG_JENEPONTO, out)
    else:
        out = Path("results") / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_info.json").write_text(json.dumps({
            "python": platform.python_version(), "quick": args.quick, "tune": args.tune,
            "repeats": cfg.N_REPEATS, "splits": cfg.N_SPLITS, "seed": cfg.SEED}, indent=2))
        if args.dataset == "jeneponto":
            run_jeneponto(args, cfg.DATASETS["jeneponto"], out)
        else:
            run_lombok(args, out)
    print(f"\nDone in {time.time() - t0:.0f} s. Results in {out}/")


if __name__ == "__main__":
    main()
