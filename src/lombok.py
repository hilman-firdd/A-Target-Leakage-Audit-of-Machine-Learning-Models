"""East Lombok (Mendeley v2): categorical replication (E1-L) and E4 feasibility report.

The overall file has sex, dates and CATEGORICAL indices (BB/U, TB/U, BB/TB) but no height or
weight, so the WHO rule cannot be recomputed. Pre/post files contain only stunted children and
no sex, so a prospective outcome (recovered vs. still stunted) cannot be observed.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import metrics as M
from . import models as Z
from .experiments import _oof_proba
from . import who


def _txt(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\xa0", " ").str.strip().str.lower()


def load_overall(lcfg: dict, sex_map: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(lcfg["overall_file"])
    rep = {"n_raw": len(raw)}
    d = pd.DataFrame({
        "sex": _txt(raw["Gender"]).map(sex_map),
        "age_month": (pd.to_datetime(raw["Measurement Date"]) - pd.to_datetime(raw["Date of Birth"])).dt.days / who.DAYS_PER_MONTH,
        "bbu": _txt(raw["BB/U"]).map(lcfg["bbu_order"]),
        "bbtb": _txt(raw["BB/TB"]).map(lcfg["bbtb_order"]),
        "tbu": _txt(raw["TB/U"]).map(lcfg["tbu_order"]),
    })
    bad = d.isna().any(axis=1) | ~d["age_month"].between(0, 60)
    rep["dropped_missing_or_unusable_category"] = int(bad.sum())   # '-', 'outliers', missing
    d = d[~bad].copy()
    d["stunted"] = (d["tbu"] <= 1).astype(int)
    rep["n_clean"] = len(d)
    rep["prevalence"] = round(float(d["stunted"].mean()), 4)
    return d.reset_index(drop=True), rep


def categorical_replication(d: pd.DataFrame, cfg, fast=False) -> pd.DataFrame:
    """E1-L: replicate a Firdaus et al.-style setup with the categorical indices."""
    y = d["stunted"].values
    rows = [{"experiment": "E1-L", "features": "TB/U index only", "model": "Label index, no training",
             **M.classification_metrics(y, -d["tbu"].values, (d["tbu"].values <= 1).astype(int))}]
    variants = {
        "sex+age+BB/U+BB/TB+TB/U (label index included)": ["sex", "age_month", "bbu", "bbtb", "tbu"],
        "sex+age+BB/U+BB/TB (label index removed)": ["sex", "age_month", "bbu", "bbtb"],
        "sex+age only": ["sex", "age_month"],
    }
    for vname, feats in variants.items():
        X = d[feats].values
        for name, model in Z.classifiers(cfg.SEED, fast=fast).items():
            reps = []
            for rep in range(cfg.N_REPEATS):
                cv = StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED + rep)
                prob = _oof_proba(model, X, y, cv)
                reps.append(M.classification_metrics(y, prob, (prob >= 0.5).astype(int), prob))
            rows.append({"experiment": "E1-L", "features": vname, "model": name, "n": len(d),
                         "prevalence": y.mean(), **pd.DataFrame(reps).mean().to_dict()})
    return pd.DataFrame(rows)


def _load_visit(pattern: str, sheet: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(pattern)):
        df = pd.read_excel(p, sheet_name=sheet)
        df["region"] = re.sub(r"\s*village$", "", Path(p).stem.strip(), flags=re.I).lower()
        df["columns_present"] = "|".join(map(str, df.columns))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(pattern)
    df = pd.concat(frames, ignore_index=True)
    if "Date of Birth" in df:
        df["Date of Birth"] = pd.to_datetime(_txt(df["Date of Birth"]).replace("nan", None), errors="coerce")
    df["TB/U"] = _txt(df["TB/U"])
    return df


def e4_feasibility(lcfg: dict) -> dict:
    pre = _load_visit(lcfg["pre_glob"], lcfg["sheet"])
    post = _load_visit(lcfg["post_glob"], lcfg["sheet"])
    rep = {}
    for name, v in (("feb_pre", pre), ("aug_post", post)):
        rep[f"{name}_n"] = len(v)
        rep[f"{name}_tbu_counts"] = v["TB/U"].value_counts().to_dict()
        rep[f"{name}_has_sex_column"] = any(c.lower() in ("gender", "sex", "jk", "jenis kelamin") for c in v.columns)
        rep[f"{name}_has_measurement_date"] = "Measurement Date" in v.columns
        rep[f"{name}_missing_dob"] = int(v["Date of Birth"].isna().sum()) if "Date of Birth" in v else len(v)
        rep[f"{name}_n_by_region"] = v.groupby("region").size().to_dict()
    pre_k = pre.dropna(subset=["Date of Birth"])
    counts = post.groupby(["region", "Date of Birth"]).size().rename("n_post").reset_index()
    m = pre_k.merge(counts, on=["region", "Date of Birth"], how="left")
    rep["feb_records_with_dob"] = len(pre_k)
    rep["match_region_dob_unique"] = int((m["n_post"] == 1).sum())
    rep["match_region_dob_ambiguous"] = int((m["n_post"] > 1).sum())
    rep["match_region_dob_none"] = int(m["n_post"].isna().sum())
    rep["conclusion"] = ("Both files list only stunted children and contain no sex variable. Children absent "
                         "in August cannot be classified as recovered, moved or missed, so the prospective "
                         "outcome is not observable. E4 is not feasible with this dataset.")
    return rep
