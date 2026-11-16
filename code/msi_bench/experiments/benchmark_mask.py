"""Lifecycle-reference masking (Section 4.3 leakage rule).

Shared by E3-R and embedding-corpus preparation: removes in-text
"CANCEL NAVAREA IV n/yy" references so text similarity cannot link chains by
literally reading the cross-reference.
"""

from __future__ import annotations

import re

_RE = re.compile(r"CANCEL\s+NAVAREA\s+IV\s+\d+\s*/\s*\d+", re.IGNORECASE)


def mask_lifecycle(text: str) -> str:
    return _RE.sub("[REF MASKED]", text or "")
