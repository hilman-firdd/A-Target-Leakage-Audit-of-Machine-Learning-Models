"""Run with:  python -m pytest -q"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import metrics as M
from src import who

WHO_JSON = Path("/usr/local/lib/python3.12/dist-packages/pygrowup/tables")  # optional cross-check


def test_median_gives_zero_haz():
    t = pd.read_csv(who.TABLE_PATH)
    t = t[t.indicator == "lhfa"]
    for _, r in t.iterrows():
        if (r.measure == "length" and r.age_month >= 24) or (r.measure == "height" and r.age_month < 24):
            continue
        z = who.haz([r.M], [r.age_month], [r.sex])[0]
        assert abs(z) < 1e-9


def test_minus_two_sd_height_is_cutoff():
    # WHO published -2 SD length for boys at 12 months is 71.0 cm (rounded)
    z = who.haz([71.0], [12], [1])[0]
    assert -2.05 < z < -1.95
    assert who.stunting_rule([z - 0.01])[0] == 1


def test_length_height_switch_and_adjustment():
    # standing height at 18 months must be converted to length (+0.7 cm)
    z_lying = who.haz([80.0], [18], [2], measured_lying=[True])[0]
    z_standing = who.haz([79.3], [18], [2], measured_lying=[False])[0]
    assert abs(z_lying - z_standing) < 1e-9


def test_out_of_range_age_is_nan():
    assert np.isnan(who.haz([100.0], [70], [1])[0])


@pytest.mark.skipif(not WHO_JSON.exists(), reason="pygrowup tables not installed")
def test_against_published_sd2neg_values():
    for sex, code in [("boys", 1), ("girls", 2)]:
        for r in json.load(open(WHO_JSON / f"lhfa_{sex}_2_5_zscores.json")):
            z = who.haz([float(r["SD2neg"])], [float(r["Month"])], [code])[0]
            assert abs(z + 2) < 0.06  # published values are rounded to 0.1 cm


def test_status_mapping_handles_indonesian_labels():
    s = pd.Series(["Sangat Pendek", "pendek", "Normal", "Tinggi", "tidak stunting", "???"])
    out = D.map_status(s, ["sangat pendek", "pendek", "stunting"], ["normal", "tinggi", "tidak stunting"])
    assert out.tolist()[:5] == [1, 1, 0, 0, 0] and np.isnan(out.iloc[5])


def test_decimal_comma():
    assert D._to_number(pd.Series(["85,5", "90"])).tolist() == [85.5, 90.0]


def test_delong_identical_scores_p_is_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    _, _, p = M.delong_test(y, s, s)
    assert p == 1.0


def test_delong_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good = y + rng.normal(0, 0.5, 2000)
    bad = y + rng.normal(0, 3.0, 2000)
    a1, a2, p = M.delong_test(y, good, bad)
    assert a1 > a2 and p < 0.001


def test_excel_date_weight_recovery():
    import datetime as dt
    s = pd.Series([dt.datetime(2025, 5, 13), "12.8", 14, dt.datetime(2025, 11, 9)], dtype=object)
    w, n = D.recover_excel_date_weight(s)
    assert n == 2
    assert w.iloc[0] == pytest.approx(13.5) and w.iloc[1] == pytest.approx(12.8) and w.iloc[2] == 14
    assert np.isnan(w.iloc[3])  # month 10-12 is ambiguous -> NaN


def test_not_stunted_is_not_counted_as_stunted():
    import config as cfg
    s = pd.Series(["Stunted", "Not Stunted", "very short", "short", "normal", "Tinggi", "-", "outliers"])
    out = D.map_status(s, cfg.STUNTED_LABELS, cfg.NOT_STUNTED_LABELS)
    assert out.tolist()[:6] == [1, 0, 1, 1, 0, 0] and out.iloc[6:].isna().all()


def test_dates_are_never_parsed_as_numbers():
    import datetime as dt
    out = D._to_number(pd.Series([dt.datetime(2025, 12, 2), "-2.11", 1.5], dtype=object))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == -2.11 and out.iloc[2] == 1.5


def test_age_offset_never_exceeds_60_months():
    raw = pd.DataFrame({"a": [59, 60], "s": ["M", "F"], "h": [100.0, 101.0], "st": ["Not Stunted"] * 2})
    cols = {"age": "a", "age_unit": "month", "age_offset_months": 0.5, "sex": "s", "height": "h",
            "weight": None, "status": "st", "haz_provided": None, "measured_lying": None, "year": None}
    d = D.standardize(raw, cols, {"m": 1, "f": 2}, ["stunted"], ["not stunted"])
    assert d["age_month"].tolist() == [59.5, 60.0]
