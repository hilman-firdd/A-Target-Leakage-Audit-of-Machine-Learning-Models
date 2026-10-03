"""WHO 2006 Child Growth Standards: z-score computation (LMS method).

Tables: WHO monthly LMS parameters for length/height-for-age (lhfa) and
weight-for-age (wfa), 0-60 months, both sexes (who_tables/who2006_lms_monthly.csv).
Values between whole months are linearly interpolated. WHO anthro uses daily
tables; the monthly interpolation differs by < 0.01 z in practice.

Conventions (WHO):
- Age < 24 months -> recumbent LENGTH table; age >= 24 months -> standing HEIGHT table.
- If the measurement position does not match the age (e.g. standing at < 24 months),
  add 0.7 cm (standing -> length) or subtract 0.7 cm (recumbent -> height).
- HAZ flagged as biologically implausible if |HAZ| > 6; WAZ if < -6 or > 5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLE_PATH = Path(__file__).resolve().parents[1] / "who_tables" / "who2006_lms_monthly.csv"
DAYS_PER_MONTH = 30.4375
MALE, FEMALE = 1, 2


def _load_tables() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH)
    return df.sort_values(["indicator", "sex", "measure", "age_month"]).reset_index(drop=True)


_TABLES = _load_tables()


def _lms_at(indicator: str, sex: np.ndarray, measure: np.ndarray, age_month: np.ndarray):
    """Interpolate L, M, S for each row. Returns three float arrays (NaN if out of range)."""
    n = len(age_month)
    L, M, S = (np.full(n, np.nan) for _ in range(3))
    for s in (MALE, FEMALE):
        for meas in np.unique(measure):
            sub = _TABLES[(_TABLES.indicator == indicator) & (_TABLES.sex == s) & (_TABLES.measure == meas)]
            if sub.empty:
                continue
            idx = (sex == s) & (measure == meas)
            a = age_month[idx]
            ok = (a >= sub.age_month.min()) & (a <= sub.age_month.max())
            for arr, col in ((L, "L"), (M, "M"), (S, "S")):
                vals = np.full(idx.sum(), np.nan)
                vals[ok] = np.interp(a[ok], sub.age_month.values, sub[col].values)
                arr[idx] = vals
    return L, M, S


def _z_from_lms(x: np.ndarray, L: np.ndarray, M: np.ndarray, S: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(L) < 1e-9, np.log(x / M) / S, ((x / M) ** L - 1.0) / (L * S))
    return z


def _wfa_restricted(x, L, M, S):
    """WHO restricted application of LMS for weight-for-age beyond +/-3 SD."""
    z = _z_from_lms(x, L, M, S)
    sd3p = M * (1 + L * S * 3) ** (1 / L)
    sd2p = M * (1 + L * S * 2) ** (1 / L)
    sd3n = M * (1 + L * S * -3) ** (1 / L)
    sd2n = M * (1 + L * S * -2) ** (1 / L)
    z = np.where(z > 3, 3 + (x - sd3p) / (sd3p - sd2p), z)
    z = np.where(z < -3, -3 + (x - sd3n) / (sd2n - sd3n), z)
    return z


def haz(height_cm, age_month, sex, measured_lying=None) -> np.ndarray:
    """Length/height-for-age z-score.

    measured_lying: optional boolean array (True = recumbent). If None, the position is
    assumed to match the age convention (no 0.7 cm adjustment).
    """
    h = np.asarray(height_cm, dtype=float).copy()
    a = np.asarray(age_month, dtype=float)
    s = np.asarray(sex, dtype=int)
    if measured_lying is not None:
        lying = np.asarray(measured_lying, dtype=bool)
        h = np.where((a < 24) & ~lying, h + 0.7, h)
        h = np.where((a >= 24) & lying, h - 0.7, h)
    measure = np.where(a < 24, "length", "height")
    L, M, S = _lms_at("lhfa", s, measure, a)
    return _z_from_lms(h, L, M, S)


def waz(weight_kg, age_month, sex) -> np.ndarray:
    w = np.asarray(weight_kg, dtype=float)
    a = np.asarray(age_month, dtype=float)
    s = np.asarray(sex, dtype=int)
    L, M, S = _lms_at("wfa", s, np.full(len(a), "na"), a)
    return _wfa_restricted(w, L, M, S)


def height_from_haz(z, age_month, sex) -> np.ndarray:
    """Inverse for lhfa (L = 1): used only for synthetic test data."""
    a = np.asarray(age_month, dtype=float)
    measure = np.where(a < 24, "length", "height")
    L, M, S = _lms_at("lhfa", np.asarray(sex, dtype=int), measure, a)
    return M * (1 + L * S * np.asarray(z, dtype=float)) ** (1 / L)


def weight_from_waz(z, age_month, sex) -> np.ndarray:
    a = np.asarray(age_month, dtype=float)
    L, M, S = _lms_at("wfa", np.asarray(sex, dtype=int), np.full(len(a), "na"), a)
    return M * (1 + L * S * np.asarray(z, dtype=float)) ** (1 / L)


def stunting_rule(haz_values, cutoff: float = -2.0) -> np.ndarray:
    """Deterministic WHO rule: stunted (1) if HAZ < -2."""
    return (np.asarray(haz_values) < cutoff).astype(int)
