"""Originating-authority normalisation (Section 4.2).

The ``authority`` column packs the originator together with a serial number and
a DTG, e.g. ``"AVURNAV LOCAL CAYENNE 1016/24 090903Z MAY 24."``. We strip the
serial/DTG tail to recover the originator name (``"AVURNAV LOCAL CAYENNE"``),
collapse whitespace, and apply a small alias map for known spelling variants.
The result is the *real multi-origin signal* that E0 quantifies and that the
synthetic overlay's provenance realism draws on.
"""

from __future__ import annotations

import re

# Originator = leading text up to the first " <serial>/<serial>" group.
_RE_SERIAL = re.compile(r"^\s*(.*?)\s+\d+\s*/\s*\d+\b")
# Fallback: leading text up to the first DTG group.
_RE_DTG_TAIL = re.compile(r"^\s*(.*?)\s+\d{6}Z\b")

# Canonical aliases for known spelling variants.
_ALIASES = {
    "COGARD COMMS COMMAND": "COGARD COMMUNICATIONS COMMAND",
    "COMMANDER COGARD COMMUNICATIONS COMMAND": "COGARD COMMUNICATIONS COMMAND",
    "NORTH AMERICAN ICE SERVICE": "NAIS",
    "INTERNATIONAL ICE PATROL": "IIP",
}


def normalize_authority(raw: str) -> str:
    """Return the normalised originator name for a raw ``authority`` value."""
    if not raw:
        return "UNKNOWN"
    text = raw.strip()
    m = _RE_SERIAL.match(text) or _RE_DTG_TAIL.match(text)
    name = m.group(1) if m else text
    name = name.strip().rstrip(".").strip()
    name = re.sub(r"\s+", " ", name).upper()
    if not name:
        return "UNKNOWN"
    return _ALIASES.get(name, name)
