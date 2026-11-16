"""Streaming loader for the annual NAVAREA CSV files.

Each annual file (``data/2017.csv`` …) is a few megabytes — small enough to
read one year at a time with pandas, but we deliberately *stream year by year*
rather than concatenating all 13 files, to keep memory flat and to make
partition selection (Section 4.2) explicit.

The ``text`` field is multi-line and quoted; pandas handles the embedded
newlines correctly. All columns are read as strings (``dtype=str``,
``keep_default_na=False``) so that empty cells remain empty strings rather than
NaN, and message numbers keep any leading structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd

EXPECTED_COLUMNS = [
    "msgYear", "msgNumber", "navArea", "subregion", "text", "status",
    "issueDate", "authority", "cancelDate", "cancelNavArea",
    "cancelMsgYear", "cancelMsgNumber",
]


def year_path(data_dir: Path, year: int) -> Path:
    return Path(data_dir) / f"{year}.csv"


def available_years(data_dir: Path) -> List[int]:
    """Return the sorted list of years for which a ``<year>.csv`` file exists."""
    years = []
    for p in Path(data_dir).glob("*.csv"):
        stem = p.stem
        if stem.isdigit():
            years.append(int(stem))
    return sorted(years)


def load_year(data_dir: Path, year: int) -> pd.DataFrame:
    """Load one annual CSV as a string DataFrame (empty cells -> '')."""
    path = year_path(data_dir, year)
    if not path.is_file():
        raise FileNotFoundError(f"No NAVAREA CSV for year {year}: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def iter_rows(
    data_dir: Path,
    years: Iterable[int],
    nav_area: Optional[int] = None,
) -> Iterator[Dict[str, str]]:
    """Yield raw row dicts across the requested ``years``.

    If ``nav_area`` is given, rows whose ``navArea`` does not match are skipped
    (the archive analysed here is NAVAREA IV, ``navArea == '4'``).
    """
    for year in years:
        path = year_path(data_dir, year)
        if not path.is_file():
            continue
        df = load_year(data_dir, year)
        for row in df.to_dict(orient="records"):
            if nav_area is not None and str(row.get("navArea", "")).strip() != str(nav_area):
                continue
            yield row
