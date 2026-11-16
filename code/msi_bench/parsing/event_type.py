"""Keyword-based event-type classification (Section 4.2).

Maps a warning's text to one of the MSI-Model event types. The rule set is
ordered by specificity: distress/rescue and security categories take
precedence over the broad navigational-hazard category, which in turn takes
precedence over the meteorological and port categories only where the more
specific cues are absent.

This rule-based classifier handles the large majority of NAVAREA IV messages;
the framework reserves an optional LLM assist for the residual
ambiguous fraction, exposed here via :func:`classify_event_type` returning
``EventType.OTHER`` so callers can route those messages to the LLM if desired.
"""

from __future__ import annotations

import re

from ..schema import EventType

# Ordered (EventType, compiled-pattern) rules; first match wins.
_RULES = [
    (
        EventType.SAR_INFO,
        re.compile(
            r"SEARCH\s+AND\s+RESCUE|DISTRESS|\bSAR\b|PERSON\s+IN\s+WATER|"
            r"MISSING\s+VESSEL|OVERDUE\s+VESSEL|EPIRB",
            re.IGNORECASE,
        ),
    ),
    (
        EventType.PIRACY_ALERT,
        re.compile(r"PIRAC|ARMED\s+ROBBERY|\bPIRATE", re.IGNORECASE),
    ),
    (
        EventType.MET_WARNING,
        re.compile(
            r"HURRICANE|TROPICAL\s+(STORM|CYCLONE|DEPRESSION)|\bCYCLONE\b|"
            r"\bGALE\b|\bSTORM\s+WARNING|TSUNAMI|SPACE\s+WEATHER|SOLAR\s+FLARE|"
            r"DENSE\s+FOG",
            re.IGNORECASE,
        ),
    ),
    (
        EventType.PORT_INFO,
        re.compile(
            r"PORT\s+(CLOSED|CLOSURE|OPEN|REOPEN)|HARBOU?R\s+CLOSED|"
            r"TERMINAL\s+CLOSED",
            re.IGNORECASE,
        ),
    ),
    (
        EventType.NAV_WARNING,
        re.compile(
            r"WRECK|OBSTRUCTION|SHOAL|\bCABLE\b|PIPELINE|\bBUOY\b|DERELICT|"
            r"ADRIFT|HAZARDOUS\s+OPERATIONS|ROCKET|LAUNCH|FIRING|EXERCISE|"
            r"\bSURVEY\b|DRILLING|ICEBERG|ICE\s+LIMIT|SEA\s+ICE|UNLIT|"
            r"UNRELIABLE|LIGHT\s+(OUT|UNRELIABLE)|NAVTEX\s+STATION|RESTRICTED|"
            r"\bMINE\b|UNCHARTED|SEISMIC|DREDG",
            re.IGNORECASE,
        ),
    ),
]


def classify_event_type(text: str) -> EventType:
    """Return the best-matching :class:`EventType` (``OTHER`` if none match)."""
    text = text or ""
    for event_type, pattern in _RULES:
        if pattern.search(text):
            return event_type
    return EventType.OTHER
