"""Message-function classification (Section 4.2).

Each NAVAREA message is assigned one of five functions:

* ``ADMIN_ERROR``        — "MSG NOT PROMULGATED" / "PROMULGATED IN ERROR".
* ``RECURRING_BULLETIN`` — NAIS iceberg bulletins, MODU position lists, etc.
* ``CANCEL_ONLY``        — a pure lifecycle message ("CANCEL NAVAREA IV n/yy
  AND THIS MSG, OPERATIONS COMPLETED") with no hazard body.
* ``UPDATE``             — supersedes a prior NAVAREA message *and* carries new
  hazard content.
* ``NEW_WARNING``        — a fresh hazard with no supersession of a prior message.

A crucial distinction: ``CANCEL THIS MSG <dtg>`` is a message's *own* expiry
instruction (present on nearly every warning) and is **not** a supersession;
only ``CANCEL NAVAREA IV n/yy`` references *another* message.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..schema import MessageFunction

# Supersession reference to another message: "CANCEL NAVAREA IV 496/24".
_RE_SUPERSEDE = re.compile(r"CANCEL\s+NAVAREA\s+IV\s+(\d+)\s*/\s*(\d{2,4})", re.IGNORECASE)
# A message's own expiry instruction.
_RE_SELF_EXPIRY = re.compile(r"CANCEL\s+THIS\s+MSG", re.IGNORECASE)
# Administrative error / non-promulgation.
_RE_ADMIN = re.compile(r"NOT\s+PROMULGATED|PROMULGATED\s+IN\s+ERROR", re.IGNORECASE)
# A cancel-only body: supersedes another message "AND THIS MSG".
_RE_CANCEL_ONLY = re.compile(
    r"CANCEL\s+NAVAREA\s+IV\s+\d+\s*/\s*\d{2,4}\s+AND\s+THIS\s+MSG", re.IGNORECASE
)
# Recurring periodic series: NAIS iceberg bulletins, MODU position lists, and
# the weekly "NAVAREA IV WARNINGS IN FORCE AS OF ..." re-broadcast summaries.
# The in-force summaries enumerate many active warning numbers and would
# otherwise be mis-typed as UPDATE and collapse unrelated warnings into one
# spurious lifecycle chain (they are periodic re-broadcasts, not discrete events).
_RE_RECURRING = re.compile(
    r"ICEBERG\s+BULLETIN|NORTH\s+AMERICAN\s+ICE\s+SERVICE|"
    r"DRILLING\s+UNITS\s+POSITIONS|MODU\s+POSITIONS|"
    r"WARNINGS\s+IN\s+FORCE",
    re.IGNORECASE,
)


def extract_supersessions(text: str) -> List[Tuple[int, int]]:
    """Return list of (msg_number, msg_year) that this message supersedes.

    Two-digit years are normalised to four digits (``24`` -> ``2024``).
    """
    out: List[Tuple[int, int]] = []
    for num, year in _RE_SUPERSEDE.findall(text or ""):
        y = int(year)
        y += 2000 if y < 100 else 0
        out.append((int(num), y))
    return out


def has_self_expiry(text: str) -> bool:
    return bool(_RE_SELF_EXPIRY.search(text or ""))


def classify_function(
    text: str,
    supersessions: List[Tuple[int, int]],
    has_coords: bool,
) -> MessageFunction:
    """Assign the message-function class.

    Precedence: ADMIN_ERROR > RECURRING_BULLETIN > CANCEL_ONLY > UPDATE >
    NEW_WARNING. ``has_coords`` (whether the coordinate parser found any
    position) helps separate cancel-only bodies from substantive warnings.
    """
    text = text or ""
    if _RE_ADMIN.search(text):
        return MessageFunction.ADMIN_ERROR
    if _RE_RECURRING.search(text):
        return MessageFunction.RECURRING_BULLETIN
    # Pure cancellation: "CANCEL NAVAREA IV n/yy AND THIS MSG" with no positions.
    if _RE_CANCEL_ONLY.search(text) and not has_coords:
        return MessageFunction.CANCEL_ONLY
    if supersessions:
        return MessageFunction.UPDATE
    return MessageFunction.NEW_WARNING
