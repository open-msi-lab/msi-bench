"""Orchestration of the preprocessing pipeline.

:func:`parse_message` turns one raw CSV row into a :class:`ParsedMessage`,
running coordinate extraction, temporal parsing, message-function and
event-type classification, authority normalisation, and data-quality flagging.
:func:`parse_corpus` streams the requested year partitions and returns the list
of parsed messages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from ..config import Config
from ..geo import parse_geometry
from ..schema import MessageFunction, ParsedMessage
from ..temporal import parse_dtg
from . import authority as authority_mod
from . import event_type as event_type_mod
from . import loader
from . import message_function as mf

# Capture the DTG that follows "CANCEL THIS MSG" (a message's own expiry).
_RE_SELF_EXPIRY_DTG = re.compile(
    r"CANCEL\s+THIS\s+MSG\s+(\d{6}Z\s+[A-Z]{3}(?:\s+\d{2,4})?)", re.IGNORECASE
)


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    return int(value) if value.isdigit() else None


def parse_message(row: Dict[str, str]) -> ParsedMessage:
    """Parse a single raw NAVAREA row into a :class:`ParsedMessage`."""
    msg_year = _to_int(row.get("msgYear")) or 0
    text = row.get("text", "") or ""
    # Canonicalise newlines at ingestion so the parsed text — and everything
    # derived from it (geometry/function parsing, the event's rep_text, the
    # official record, the LLM prompts and therefore the response-cache keys) —
    # is byte-identical across platforms. A Windows checkout/export carries CRLF
    # while macOS/Linux carry LF; without this, a cache (or benchmark) built on
    # one OS fails to replay on the other even though the source message is the
    # same, breaking the "same raw data ⇒ same cache key" guarantee.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    defects: List[str] = []

    # --- geometry ---
    coords, geom_kind, center = parse_geometry(text)
    has_coords = bool(coords)

    # --- temporal: issue, cancel-column, in-text self-expiry ---
    issue_time, issue_defect = parse_dtg(row.get("issueDate", ""), default_year=msg_year)
    if issue_defect:
        defects.append(f"issue_{issue_defect}")

    cancel_time, _ = parse_dtg(row.get("cancelDate", ""), default_year=msg_year)

    self_expiry_time = None
    m = _RE_SELF_EXPIRY_DTG.search(text)
    if m:
        self_expiry_time, _ = parse_dtg(m.group(1), default_year=msg_year)

    # --- lifecycle references ---
    supersessions = mf.extract_supersessions(text)
    function = mf.classify_function(text, supersessions, has_coords)

    # --- data-quality flags (surfaced for E0) ---
    if cancel_time and issue_time and cancel_time < issue_time:
        defects.append("cancel_before_issue")

    msg = ParsedMessage(
        msg_year=msg_year,
        msg_number=_to_int(row.get("msgNumber")) or 0,
        nav_area=_to_int(row.get("navArea")) or 0,
        subregion=(row.get("subregion", "") or "").strip(),
        raw_text=text,
        status=(row.get("status", "") or "").strip(),
        authority_raw=(row.get("authority", "") or "").strip(),
        issue_time=issue_time,
        cancel_time=cancel_time,
        self_expiry_time=self_expiry_time,
        cancel_nav_area=_to_int(row.get("cancelNavArea")),
        cancel_msg_year=_to_int(row.get("cancelMsgYear")),
        cancel_msg_number=_to_int(row.get("cancelMsgNumber")),
        in_text_cancels=supersessions,
        function=function,
        event_type=event_type_mod.classify_event_type(text),
        authority_norm=authority_mod.normalize_authority(row.get("authority", "")),
        coordinates=coords,
        geometry_kind=geom_kind,
        centroid=center,
        defects=defects,
    )

    # Sub-hour self-cancellation (rapid retraction) — an official-error signal.
    dur = msg.active_duration_hours
    if dur is not None and 0 <= dur < 1.0:
        msg.defects.append("sub_hour_cancel")
    # A discrete-event warning that yielded no parseable coordinates.
    if function.is_discrete_event and not has_coords:
        msg.defects.append("no_coords_for_event")

    return msg


def parse_corpus(
    cfg: Config,
    years: Iterable[int],
    nav_area: Optional[int] = None,
    functions: Optional[Iterable[MessageFunction]] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> List[ParsedMessage]:
    """Parse all messages in the given ``years`` (streamed per year).

    Parameters
    ----------
    years:
        Iterable of years to include (e.g. ``cfg.years('benchmark')``).
    nav_area:
        Restrict to this NAVAREA (defaults to ``cfg.nav_area``).
    functions:
        If given, keep only messages with these message functions (e.g.
        ``[NEW_WARNING, UPDATE]`` for the discrete-event benchmark).
    progress:
        Optional callback invoked with the running message count.
    """
    nav_area = cfg.nav_area if nav_area is None else nav_area
    keep = set(functions) if functions is not None else None
    out: List[ParsedMessage] = []
    count = 0
    for row in loader.iter_rows(cfg.data_dir, years, nav_area=nav_area):
        msg = parse_message(row)
        count += 1
        if progress is not None and count % 1000 == 0:
            progress(count)
        if keep is not None and msg.function not in keep:
            continue
        out.append(msg)
    return out
