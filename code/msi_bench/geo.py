"""Coordinate parsing and geodesy for NAVAREA warning text.

NAVAREA warnings encode positions in the mariner format ``DD-MM.MMN DDD-MM.MMW``
(degrees-minutes with a hemisphere letter), with degenerate variants such as
``DD-MMN`` (no decimal minutes). Positions appear as points, tracklines, or
bounded areas, signalled by cue phrases ("IN AREAS BOUND BY", "ALONG TRACKLINE
JOINING", "IN VICINITY").

This module extracts coordinates robustly (tolerant of line wrapping and
varying whitespace), classifies the geometry kind from cue phrases, and
provides geodesic distance. Geodesic distance uses :mod:`geographiclib` when
available and falls back to the Haversine formula otherwise.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from .schema import GeometryKind, LatLon

# Mean Earth radius in nautical miles (1 nm = 1852 m; R = 6371.0088 km).
_EARTH_RADIUS_NM = 6371008.8 / 1852.0

try:  # high-accuracy geodesics if available
    from geographiclib.geodesic import Geodesic

    _GEOD = Geodesic.WGS84
except Exception:  # pragma: no cover - optional dependency
    _GEOD = None


# A single coordinate token: degrees-"-"-minutes(.decimals)?-hemisphere.
# Examples matched: "28-47.00N", "080-15.00W", "47-08N", "052-55W".
_COORD_TOKEN = re.compile(r"(\d{1,3})-(\d{1,2}(?:\.\d+)?)\s*([NSEW])")

# Cue phrases that determine the geometry kind.
_RE_AREA = re.compile(r"BOUND\s+BY|BOUNDED\s+BY|AREA[S]?\s+BOUND", re.IGNORECASE)
_RE_LINE = re.compile(r"TRACKLINE|ALONG\s+TRACK|WITHIN\s+.*MILES?\s+OF", re.IGNORECASE)
_RE_VICINITY = re.compile(r"IN\s+VICINITY|VICINITY\s+OF", re.IGNORECASE)


def _token_to_decimal(deg: str, minutes: str, hemi: str) -> Optional[float]:
    """Convert one ``deg-min<hemi>`` token to signed decimal degrees."""
    value = int(deg) + float(minutes) / 60.0
    if hemi in ("S", "W"):
        value = -value
    # Range validation distinguishes real coordinates from spurious matches.
    if hemi in ("N", "S") and abs(value) > 90.0:
        return None
    if hemi in ("E", "W") and abs(value) > 180.0:
        return None
    return value


def extract_coordinates(text: str) -> List[LatLon]:
    """Return the ordered list of (lat, lon) pairs found in ``text``.

    Tokens are scanned in document order; a latitude token (N/S) immediately
    followed by a longitude token (E/W) forms one coordinate. Unpaired or
    out-of-range tokens are ignored. This pairing rule reflects the NAVAREA
    convention of writing latitude before longitude.
    """
    tokens = []
    for deg, minutes, hemi in _COORD_TOKEN.findall(text or ""):
        val = _token_to_decimal(deg, minutes, hemi)
        if val is not None:
            tokens.append((hemi, val))

    coords: List[LatLon] = []
    i = 0
    while i < len(tokens) - 1:
        hemi_a, val_a = tokens[i]
        hemi_b, val_b = tokens[i + 1]
        if hemi_a in ("N", "S") and hemi_b in ("E", "W"):
            coords.append((val_a, val_b))
            i += 2
        else:
            i += 1  # skip an unpaired token and resync
    return coords


def classify_geometry(text: str, coords: List[LatLon]) -> GeometryKind:
    """Classify geometry from cue phrases and coordinate count."""
    if not coords:
        return GeometryKind.UNKNOWN
    if _RE_AREA.search(text or ""):
        return GeometryKind.POLYGON
    if _RE_LINE.search(text or ""):
        return GeometryKind.LINE
    if _RE_VICINITY.search(text or "") or len(coords) == 1:
        return GeometryKind.POINT
    return GeometryKind.MULTIPOINT


def centroid(coords: List[LatLon]) -> Optional[LatLon]:
    """Mean-of-vertices centroid (adequate for NAVAREA-scale extents)."""
    if not coords:
        return None
    lat = sum(c[0] for c in coords) / len(coords)
    lon = sum(c[1] for c in coords) / len(coords)
    return (lat, lon)


# Lettered sub-area markers ("A.", "B.", …) used in multi-area warnings.
_RE_AREA_MARKER = re.compile(r"(?m)^\s*([A-Z])\.\s")


def extract_coordinate_groups(text: str) -> List[List[LatLon]]:
    """Split coordinates by lettered sub-area markers (A./B./C.).

    Multi-area warnings (e.g. rocket launches with separate boxes A, B, C) are
    segmented so each sub-area is a distinct coordinate group; this avoids
    averaging vertices across disjoint boxes into a meaningless centroid.
    Returns one group per sub-area, or a single group when no markers are found.
    """
    if not text:
        return []
    parts = _RE_AREA_MARKER.split(text)
    if len(parts) <= 1:
        coords = extract_coordinates(text)
        return [coords] if coords else []
    segments = []
    rest = parts[1:]  # [letter, segment, letter, segment, ...]
    for k in range(0, len(rest) - 1, 2):
        segments.append(rest[k + 1])
    groups = [g for g in (extract_coordinates(s) for s in segments) if g]
    if not groups:
        coords = extract_coordinates(text)
        return [coords] if coords else []
    return groups


def haversine_nm(a: LatLon, b: LatLon) -> float:
    """Great-circle distance between two points in nautical miles.

    Uses :mod:`geographiclib` (WGS-84 ellipsoid) when available, otherwise the
    spherical Haversine formula.
    """
    lat1, lon1 = a
    lat2, lon2 = b
    if _GEOD is not None:
        meters = _GEOD.Inverse(lat1, lon1, lat2, lon2)["s12"]
        return meters / 1852.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_NM * math.asin(min(1.0, math.sqrt(h)))


def parse_geometry(text: str) -> Tuple[List[LatLon], GeometryKind, Optional[LatLon]]:
    """Convenience: extract coordinates, classify kind, and compute centroid.

    For multi-area warnings, the representative centroid is taken from the
    dominant (vertex-richest) sub-area rather than averaged across all sub-areas,
    so it lies within a real declared area instead of between disjoint boxes.
    """
    coords = extract_coordinates(text)
    kind = classify_geometry(text, coords)
    if kind == GeometryKind.POLYGON:
        groups = extract_coordinate_groups(text)
        if len(groups) >= 2:
            dominant = max(groups, key=len)
            return coords, kind, centroid(dominant)
    return coords, kind, centroid(coords)
