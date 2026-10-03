"""Synthetic data that mimics the structure of the two Mendeley datasets.

FOR TESTING THE PIPELINE ONLY. Never report results from synthetic data.
Original labels are made from heights with realistic noise (rounding to 0.5 cm, +/-0.5 cm
measurement error, 2% recording errors) so that the WHO rule is not trivially perfect.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import who

ROOT = Path("data/synthetic")
COLS = {"age": "Usia (bulan)", "age_unit": "month", "sex": "Jenis Kelamin", "height": "Tinggi Badan (cm)",
        "weight": "Berat Badan (kg)", "status": "Status TB/U", "measured_lying": None, "year": "Tahun"}
CONFIG_JENEPONTO = {"files": str(ROOT / "jeneponto/*.csv"), "sheet": 0,
                    "columns": {**COLS, "age_offset_months": 0.0, "haz_provided": None},
                    "child_id": ["Nama Anak", "Tanggal Lahir"], "subsets": {"all": None}}


def _child_rows(rng, n, haz, age, sex):
    wz = 0.55 * haz + rng.normal(0, 0.8, n)
    h_true = who.height_from_haz(haz, age, sex)
    h_meas = np.round((h_true + rng.normal(0, 0.5, n)) * 2) / 2      # digit preference to 0.5 cm
    w = np.round(who.weight_from_waz(wz, age, sex), 1)
    lab = who.stunting_rule(who.haz(h_meas, age, sex))
    flip = rng.random(n) < 0.02
    lab = np.where(flip, 1 - lab, lab)
    status = np.where(lab == 1, rng.choice(["Pendek", "Sangat Pendek"], n, p=[0.7, 0.3]),
                      rng.choice(["Normal", "Tinggi"], n, p=[0.97, 0.03]))
    return h_meas, w, status


def write_all(n_children: int = 6000, seed: int = 7):
    rng = np.random.default_rng(seed)
    (ROOT / "jeneponto").mkdir(parents=True, exist_ok=True)
    # Jeneponto-like: children measured in 1-3 of the years 2021-2024
    rows = []
    sex0 = rng.integers(1, 3, n_children)
    dob = rng.integers(0, 4 * 365, n_children)
    haz0 = rng.normal(-1.1, 1.2, n_children)
    for year in range(2021, 2025):
        seen = rng.random(n_children) < 0.45
        age = (year - 2021) * 12 + 30 - dob / 30.4375 + rng.uniform(-2, 2, n_children)
        ok = seen & (age >= 0) & (age < 60)
        idx = np.where(ok)[0]
        hz = 0.9 * haz0[idx] + rng.normal(0, 0.35, len(idx))
        h, w, st = _child_rows(rng, len(idx), hz, age[idx], sex0[idx])
        rows.append(pd.DataFrame({"Nama Anak": [f"anak {i}" for i in idx], "Tanggal Lahir": dob[idx],
                                  "Usia (bulan)": np.round(age[idx]).astype(int).clip(0, 59),
                                  "Jenis Kelamin": np.where(sex0[idx] == 1, "L", "P"),
                                  "Tinggi Badan (cm)": h, "Berat Badan (kg)": w, "Status TB/U": st, "Tahun": year}))
    pd.concat(rows).to_csv(ROOT / "jeneponto/jeneponto_synthetic.csv", index=False)

    # Lombok-like: Feb and Aug files per region
    for region in ["denggen", "majidi", "rakam"]:
        n = 400
        sex = rng.integers(1, 3, n)
        age_f = rng.uniform(0, 53, n)
        hz_f = rng.normal(-1.4, 1.1, n)
        hz_a = 0.85 * hz_f + 0.1 + rng.normal(0, 0.35, n)
        names = [f"{region} balita {i}" for i in range(n)]
        dobs = [f"2020-01-{(i % 28) + 1:02d}" for i in range(n)]
        for visit, age, hz, folder in [("feb", age_f, hz_f, "lombok_pre"), ("aug", age_f + 6, hz_a, "lombok_post")]:
            (ROOT / folder).mkdir(parents=True, exist_ok=True)
            keep = rng.random(n) < (1.0 if visit == "feb" else 0.85)    # 15% lost to follow-up
            h, w, st = _child_rows(rng, n, hz, age, sex)
            pd.DataFrame({"Nama Anak": names, "Tanggal Lahir": dobs, "Usia (bulan)": np.round(age).astype(int),
                          "Jenis Kelamin": np.where(sex == 1, "L", "P"), "Tinggi Badan (cm)": h,
                          "Berat Badan (kg)": w, "Status TB/U": st, "Tahun": 2024})[keep] \
                .to_csv(ROOT / folder / f"{region}.csv", index=False)
    print(f"Synthetic data written to {ROOT}/")


if __name__ == "__main__":
    write_all()
