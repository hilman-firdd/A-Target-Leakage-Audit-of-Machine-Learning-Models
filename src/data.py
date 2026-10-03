"""Load raw files, standardize columns, clean implausible values, reconstruct WHO labels."""
from __future__ import annotations

import datetime as _dt
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import who

STD_COLS = ["age_month", "sex", "height", "weight", "status_raw", "measured_lying", "year"]


def read_files(pattern, sheet=0) -> pd.DataFrame:
    """pattern: a glob string or an explicit list of file paths."""
    if isinstance(pattern, (list, tuple)):
        paths = list(pattern)
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}. Check config.py and the data/ folder.")
    else:
        paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match '{pattern}'. Check config.py.")
    frames = []
    for p in paths:
        if p.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(p, sheet_name=sheet)
        else:
            df = pd.read_csv(p)
        df["__source_file"] = Path(p).stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def map_sex(s: pd.Series, sex_map: dict) -> pd.Series:
    return _norm_text(s).map(sex_map)


def map_status(s: pd.Series, stunted: list[str], not_stunted: list[str]) -> pd.Series:
    """1 = stunted, 0 = not stunted, NaN = unrecognised. 'sangat pendek' counts as stunted."""
    txt = _norm_text(s)
    out = pd.Series(np.nan, index=s.index)
    # check not-stunted first so that e.g. "tidak stunting" is not caught by "stunting"
    for lab in sorted(not_stunted, key=len, reverse=True):
        out[txt.str.contains(lab, regex=False) & out.isna()] = 0
    for lab in sorted(stunted, key=len, reverse=True):
        out[txt.str.contains(lab, regex=False) & out.isna()] = 1
    return out


def recover_excel_date_weight(s: pd.Series) -> tuple[pd.Series, int]:
    """Excel silently turned many weights such as '13.5' into dates (13 May 2025).

    A date cell with day D and month M (1-9) is restored to D + M/10. Validated on the
    Jeneponto 2024 file against the provided weight-for-age z-scores (median |error| 0.027 z).
    Months 10-12 would be ambiguous (D.10 vs D.1) and are set to NaN.
    """
    is_date = s.map(lambda x: isinstance(x, (_dt.datetime, pd.Timestamp)))
    out = s.copy().astype(object)
    out[is_date] = s[is_date].map(lambda x: x.day + x.month / 10 if x.month < 10 else np.nan)
    return _to_number(out.astype(str).replace("nan", np.nan)), int(is_date.sum())


def _to_number(s: pd.Series) -> pd.Series:
    """Handle Indonesian decimal commas such as '85,5'.

    Cells that Excel turned into dates are returned as NaN (never parsed as the year).
    Use recover_excel_date_weight() for weights, where the original value can be restored.
    """
    if not pd.api.types.is_numeric_dtype(s):
        s = s.map(lambda x: np.nan if isinstance(x, (_dt.datetime, _dt.date, pd.Timestamp)) else x)
        s = s.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(s, errors="coerce")


def standardize(raw: pd.DataFrame, cols: dict, sex_map: dict, stunted: list, not_stunted: list) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    if cols.get("age_unit") == "dates":      # age from (measurement date - date of birth)
        dob = pd.to_datetime(raw[cols["dob"]], errors="coerce")
        meas = pd.to_datetime(raw[cols["measurement_date"]], errors="coerce")
        age = (meas - dob).dt.days / who.DAYS_PER_MONTH
    else:
        age = _to_number(raw[cols["age"]])
        if cols.get("age_unit") == "day":
            age = age / who.DAYS_PER_MONTH
        offset = cols.get("age_offset_months", 0.0)      # completed months -> mid-month
        age = np.where(age <= 60, np.minimum(age + offset, 60.0), age + offset)
        age = pd.Series(age, index=raw.index)
    df["age_month"] = age
    df["sex"] = map_sex(raw[cols["sex"]], sex_map)
    df["height"] = _to_number(raw[cols["height"]]) if cols.get("height") else np.nan
    df.attrs["weight_recovered_from_dates"] = 0
    if cols.get("weight"):
        df["weight"], df.attrs["weight_recovered_from_dates"] = recover_excel_date_weight(raw[cols["weight"]])
    else:
        df["weight"] = np.nan
    df["haz_provided"] = _to_number(raw[cols["haz_provided"]]) if cols.get("haz_provided") else np.nan
    df.attrs["haz_provided_stored_as_date"] = int(raw[cols["haz_provided"]].map(
        lambda x: isinstance(x, (_dt.datetime, pd.Timestamp))).sum()) if cols.get("haz_provided") else 0
    df["status_raw"] = map_status(raw[cols["status"]], stunted, not_stunted) if cols.get("status") else np.nan
    if cols.get("measured_lying"):
        df["measured_lying"] = _norm_text(raw[cols["measured_lying"]]).str.contains("terlentang|lying|recumbent|berbaring")
    else:
        df["measured_lying"] = np.nan
    if cols.get("year") == "__from_filename__":
        df["year"] = raw["__source_file"].str.extract(r"(20\d\d)")[0].astype(float)
    else:
        df["year"] = raw[cols["year"]] if cols.get("year") else np.nan
    df["source_file"] = raw.get("__source_file")
    return df


def make_child_id(raw: pd.DataFrame, keys: list[str], add_source: bool = False) -> pd.Series | None:
    if not keys:
        return None
    parts = [_norm_text(raw[k]) for k in keys]
    if add_source:
        parts.append(_norm_text(raw["__source_file"]))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def clean_and_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows, compute HAZ/WAZ, flag implausible values, reconstruct label.

    Returns the cleaned frame and a report dict for the Data Characteristics section.
    """
    report = {"n_raw": len(df), "weight_recovered_from_excel_dates": df.attrs.get("weight_recovered_from_dates", 0),
              "provided_haz_stored_as_excel_dates_set_missing": df.attrs.get("haz_provided_stored_as_date", 0)}
    lying = df["measured_lying"] if df["measured_lying"].notna().any() else None
    df = df.copy()
    df["haz"] = who.haz(df["height"], df["age_month"], df["sex"].fillna(0), lying)
    df["waz"] = who.waz(df["weight"], df["age_month"], df["sex"].fillna(0)) if df["weight"].notna().any() else np.nan

    missing = df[["age_month", "sex", "height"]].isna().any(axis=1)
    out_of_range = ~df["age_month"].between(0, 60) | df["haz"].isna()
    implausible_haz = df["haz"].abs() > 6
    implausible_waz = df["waz"].notna() & ((df["waz"] < -6) | (df["waz"] > 5))
    report.update({
        "dropped_missing_core": int(missing.sum()),
        "dropped_age_out_of_range": int((out_of_range & ~missing).sum()),
        "dropped_implausible_haz": int((implausible_haz & ~missing & ~out_of_range).sum()),
        "dropped_implausible_waz": int((implausible_waz & ~missing & ~out_of_range & ~implausible_haz).sum()),
    })
    keep = ~(missing | out_of_range | implausible_haz | implausible_waz)
    df = df[keep].copy()
    df["stunted_who"] = who.stunting_rule(df["haz"])
    report["n_clean"] = len(df)
    report["haz_mean"] = round(float(df["haz"].mean()), 3)
    report["haz_sd"] = round(float(df["haz"].std()), 3)  # WHO: SD ~1.35-1.95 expected (Mei et al.)
    report["prevalence_who_rule"] = round(float(df["stunted_who"].mean()), 4)
    if df["haz_provided"].notna().any():
        p = df.dropna(subset=["haz_provided"])
        diff = p["haz"] - p["haz_provided"]
        report["haz_recomputed_minus_provided_mean"] = round(float(diff.mean()), 4)
        report["haz_recomputed_minus_provided_mae"] = round(float(diff.abs().mean()), 4)
        if p["status_raw"].notna().any():
            q = p.dropna(subset=["status_raw"])
            report["original_label_equals_provided_haz_below_minus2_pct"] = round(
                100 * float(((q["haz_provided"] < -2).astype(int) == q["status_raw"]).mean()), 2)
    if df["year"].notna().any() and df["status_raw"].notna().any():
        report["original_prevalence_by_year"] = {str(int(k)): round(float(v), 4) for k, v in
                                                 df.groupby("year")["status_raw"].mean().items()}
    if df["status_raw"].notna().any():
        lab = df.dropna(subset=["status_raw"])
        report["n_with_original_label"] = len(lab)
        report["unrecognised_original_label"] = int(df["status_raw"].isna().sum())
        report["prevalence_original_label"] = round(float(lab["status_raw"].mean()), 4)
        report["label_disagreement_pct"] = round(100 * float((lab["status_raw"] != lab["stunted_who"]).mean()), 2)
    return df, report


def label_agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    lab = df.dropna(subset=["status_raw"])
    return pd.crosstab(lab["status_raw"].astype(int).rename("original_label"),
                       lab["stunted_who"].rename("who_rule"), margins=True)
