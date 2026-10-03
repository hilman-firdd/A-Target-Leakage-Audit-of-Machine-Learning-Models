"""Configuration, filled in after inspecting the real Mendeley files (v4 Jeneponto, v2 East Lombok).

Folder layout expected (copy the files exactly as downloaded):
  data/jeneponto/2021.xlsx, 2022.xlsx, 2023.xlsx, 2024.xlsx      (Overall Data.xlsx = same rows; not used)
  data/lombok/overal data.xlsx
  data/lombok/pre-intervention/*.xlsx
  data/lombok/post-intervention/*.xlsx
"""

SEED = 42
N_SPLITS = 5
N_REPEATS = 5
N_BOOTSTRAP = 1000

# Original status texts. "not stunted" is checked before "stunted".
STUNTED_LABELS = ["very short", "short", "severely stunted", "stunted", "sangat pendek", "pendek"]
NOT_STUNTED_LABELS = ["not stunted", "normal", "tall", "tinggi"]

# M/F in Jeneponto; M/W (wanita) in East Lombok.
SEX_MAP = {"m": 1, "male": 1, "l": 1, "laki-laki": 1,
           "f": 2, "female": 2, "w": 2, "p": 2, "perempuan": 2}

DATASETS = {
    "jeneponto": {
        # The four yearly files together are identical to 'Overall Data.xlsx' (verified row by row)
        # and give us the year. Listing them explicitly avoids reading Overall/Preprocessed twice.
        "files": [f"data/jeneponto/{y}.xlsx" for y in (2021, 2022, 2023, 2024)],
        "sheet": 0,
        "columns": {
            "age": "Age (Month)",
            "age_unit": "month",
            # Ages are completed months; the provider computed z-scores from exact age.
            # +0.5 month brings the mean difference to the provided HAZ from 0.14 to 0.01.
            "age_offset_months": 0.5,
            "sex": "Gender",
            "height": "Height",
            "weight": "Weight",               # ~50% stored as Excel dates; recovered automatically
            "status": "Height for Age",       # 'Stunted' / 'Not Stunted'
            "haz_provided": "Z-Score H/A",    # z-score supplied in the file
            "measured_lying": None,           # not recorded
            "year": "__from_filename__",
        },
        "child_id": [],                       # no child identifier -> E5a not possible
        # 2021-2023 files contain only stunted children; 2024 contains all children.
        # 'all' reproduces the published setting; '2024' is the population-like subset.
        "subsets": {"all": None, "2024": 2024},
    },
}

LOMBOK = {
    "overall_file": "data/lombok/overal data.xlsx",
    "pre_glob": "data/lombok/pre-intervention/*.xlsx",
    "post_glob": "data/lombok/post-intervention/*.xlsx",
    "sheet": "Application List",
    # ordinal coding of the categorical indices in the overall file
    "bbu_order": {"very less": 0, "less": 1, "normal": 2, "more risk": 3},
    "bbtb_order": {"malnutrition": 0, "undernutrition": 1, "good nutrition": 2,
                   "risk of overnutrition": 3, "overnutrition": 4, "obesity": 5},
    "tbu_order": {"very short": 0, "short": 1, "normal": 2, "tinggi": 3},
}
