"""Step 1: look at the raw Mendeley files before editing config.py.

Usage:  python inspect_data.py "data/jeneponto/*.xlsx"
Prints sheet names, columns, dtypes, example values and unique values of short text columns.
"""
import glob
import sys

import pandas as pd

pattern = sys.argv[1] if len(sys.argv) > 1 else "data/**/*.xlsx"
paths = sorted(glob.glob(pattern, recursive=True))
if not paths:
    sys.exit(f"No files match {pattern}")
for p in paths:
    print("=" * 80, "\n", p)
    if p.lower().endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(p)
        print("sheets:", xl.sheet_names)
        df = xl.parse(xl.sheet_names[0])
    else:
        df = pd.read_csv(p)
    print(f"rows={len(df)} cols={df.shape[1]}")
    for c in df.columns:
        s = df[c]
        uniq = s.dropna().unique()
        ex = list(uniq[:12]) if (s.dtype == object and len(uniq) <= 12) else list(s.dropna().head(3))
        print(f"  {str(c)[:40]:40s} {str(s.dtype):10s} missing={s.isna().sum():6d}  e.g. {ex}")
