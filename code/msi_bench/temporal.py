"""Parsing of NAVAREA date-time groups (DTG) into ISO 8601 datetimes.

The DTG format is ``DDHHMMZ MON [YYYY]`` (Zulu/UTC), e.g. ``312113Z DEC 2023``.
Variants seen in the archive:

* full year:        ``100049Z MAY 2024``  (issueDate / cancelDate columns)
* two-digit year:   ``212231Z MAY 24``    (in-text "CANCEL THIS MSG …")
* no year:          ``081715Z MAY``       (in-text observation times)

Parsing is deliberately tolerant: malformed or impossible values yield
``None`` together with a defect tag, so Experiment E0 can quantify the defect
frequency rather than crash on it.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# DD HH MM Z  MON  [YY or YYYY]
_DTG = re.compile(r"(\d{2})(\d{2})(\d{2})Z\s+([A-Z]{3})(?:\s+(\d{2,4}))?")


def parse_dtg(
    text: str,
    default_year: Optional[int] = None,
) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse the first DTG found in ``text``.

    Returns ``(datetime, None)`` on success, or ``(None, defect_tag)`` when no
    DTG is present or the fields are out of range. ``default_year`` supplies the
    year when the DTG omits it (e.g. in-text observation times).
    """
    m = _DTG.search(text or "")
    if not m:
        return None, "no_dtg"
    day, hour, minute, mon, year = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None, "bad_month"
    if year is None:
        if default_year is None:
            return None, "missing_year"
        yr = default_year
    else:
        yr = int(year)
        if yr < 100:  # two-digit year
            yr += 2000
    try:
        return datetime(yr, month, int(day), int(hour), int(minute)), None
    except ValueError:
        return None, "out_of_range"


def all_dtgs(text: str, default_year: Optional[int] = None) -> List[datetime]:
    """Return every parseable DTG in ``text`` (in document order)."""
    out: List[datetime] = []
    for day, hour, minute, mon, year in _DTG.findall(text or ""):
        month = _MONTHS.get(mon)
        if month is None:
            continue
        if year:
            yr = int(year)
            yr += 2000 if yr < 100 else 0
        elif default_year is not None:
            yr = default_year
        else:
            continue
        try:
            out.append(datetime(yr, month, int(day), int(hour), int(minute)))
        except ValueError:
            continue
    return out
