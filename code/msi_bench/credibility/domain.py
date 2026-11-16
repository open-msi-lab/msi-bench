"""Domain knowledge for the MCAM (Section 3.3).

Device/authentication/observation scales, per-type field
definitions for completeness, critical attributes and contradiction predicates
(Section 3.3), plausible numeric ranges, and a data-driven
navigability model (the empirical NAVAREA IV operational envelope, used in
place of an external coastline dataset for the spatial-descriptive check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..geo import haversine_nm
from ..schema import EventType, LatLon

# --- device type base trust scores --------------------------------
DEVICE_TYPE_SCORE: Dict[str, float] = {
    "PROFESSIONAL": 0.95,
    "MARINE_CERTIFIED": 0.85,
    "CONSUMER_GPS": 0.65,
    "MOBILE_DEVICE": 0.50,
    "UNKNOWN": 0.30,
}

# --- authentication level certification scores --------------------
CERT_SCORE: Dict[str, float] = {
    "PKI_FULL": 1.00,
    "PKI_BASIC": 0.85,
    "REGISTERED": 0.65,
    "ANONYMOUS": 0.30,
}

# --- spatial scale σs by observation type (nm) --------------------
OBSERVATION_SIGMA_S: Dict[str, float] = {
    "visual": 5.0,
    "radar": 15.0,
    "ais": 25.0,
    "shore": 50.0,
}

# --- event-type-specific temporal scale σt (hours), by hazard timescale -------
# Mapped from the discrete event types; refined per-record by hazard keywords
# in the generator. These are *currency* decay scales, not validity windows.
EVENT_SIGMA_T_H: Dict[EventType, float] = {
    EventType.NAV_WARNING: 7 * 24.0,
    EventType.MET_WARNING: 24.0,
    EventType.SAR_INFO: 6.0,
    EventType.PIRACY_ALERT: 24.0,
    EventType.PORT_INFO: 7 * 24.0,
    EventType.OTHER: 24.0,
}

# --- completeness field definitions (Section 3.3) ------------------------
REQUIRED_FIELDS: Dict[EventType, Set[str]] = {
    EventType.NAV_WARNING: {"hazard_type", "position"},
    EventType.MET_WARNING: {"phenomenon_type", "position"},
    EventType.SAR_INFO: {"incident_type", "position"},
    EventType.PIRACY_ALERT: {"attack_type", "position"},
    EventType.PORT_INFO: {"port_name", "status"},
    EventType.OTHER: {"description"},
}
OPTIONAL_FIELDS: Dict[EventType, Set[str]] = {
    EventType.NAV_WARNING: {"severity_level", "restriction_type", "valid_end"},
    EventType.MET_WARNING: {"intensity_level", "affected_area_category", "valid_end"},
    EventType.SAR_INFO: {"status", "vessel_identity"},
    EventType.PIRACY_ALERT: {"status", "vessel_identity"},
    EventType.PORT_INFO: {"restriction_type", "valid_end"},
    EventType.OTHER: set(),
}

# --- critical attributes Acrit per type (Section 3.3) --------------------
CRITICAL_ATTRS: Dict[EventType, Set[str]] = {
    EventType.NAV_WARNING: {"hazard_type", "existence_status", "severity_level"},
    EventType.MET_WARNING: {"phenomenon_type", "intensity_level", "affected_area_category"},
    EventType.SAR_INFO: {"incident_type", "status", "vessel_identity"},
    EventType.PIRACY_ALERT: {"attack_type", "status", "vessel_identity"},
    EventType.PORT_INFO: {"status", "restriction_type"},
    EventType.OTHER: set(),
}

# --- plausible numeric ranges for the numerical-range consistency check ----
NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "wind_speed_kn": (0.0, 200.0),
    "wave_height_m": (0.0, 30.0),
    "depth_m": (0.0, 11000.0),
    "vessel_length_m": (0.0, 500.0),
    "duration_hours": (0.0, 3 * 365 * 24.0),
    "severity_score": (0.0, 1.0),
}

# Mutually-exclusive status groups for contradiction detection.
_EXISTENCE_TRUE = {"confirmed", "in_force", "active", "ongoing"}
_EXISTENCE_FALSE = {"cancelled", "withdrawn", "resolved", "closed"}

# Critical attributes whose differing values do NOT constitute a contradiction
# (ordinal discrepancies, or naturally multi-valued identifiers).
_NON_CONTRADICTORY = {"severity_level", "intensity_level", "vessel_identity"}

# Extractor-uncertainty wildcards. A categorical contradiction requires two
# mutually exclusive LEAF categories; the keyword extractor emits
# "generic"/"other"/"unknown" when a report's free text names no specific
# category — the ABSENCE of a categorical claim, which cannot contradict a
# specific one. This is the extraction-confidence scoping of Section 3.4
# applied to conflict semantics: on the shipped benchmark, 69%
# of same-event genuine hazard_type "contradictions" involved the wildcard.
WILDCARD_ATTR_VALUES = {"generic", "other", "unknown", "unspecified"}

_EXISTENCE_ATTR_NAMES = ("existence_status", "status", "incident_status")


def attr_contradicts(name: str, a: object, b: object) -> bool:
    """Return True iff attribute values ``a`` and ``b`` are contradictory."""
    if a is None or b is None:
        return False
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    if not sa or not sb:
        return False
    if name in _EXISTENCE_ATTR_NAMES:
        return (sa in _EXISTENCE_TRUE and sb in _EXISTENCE_FALSE) or (
            sa in _EXISTENCE_FALSE and sb in _EXISTENCE_TRUE
        )
    if name in _NON_CONTRADICTORY:
        return False
    # hazard_type / phenomenon_type / restriction_type / attack_type / …:
    # mutually exclusive leaf categories — differing SPECIFIC values contradict;
    # an unspecified (wildcard) label contradicts nothing.
    if sa in WILDCARD_ATTR_VALUES or sb in WILDCARD_ATTR_VALUES:
        return False
    return sa != sb


def is_denial(structured_data: Dict) -> bool:
    """True iff the record's existence/status assertion DENIES the hazard
    (cancelled / withdrawn / resolved / closed) — the assertion-polarity used by
    fused-record interpretation (Section 3.6 / E4d)."""
    for key in _EXISTENCE_ATTR_NAMES:
        v = structured_data.get(key)
        if v is not None and str(v).strip().lower() in _EXISTENCE_FALSE:
            return True
    return False


def attr_compatible(name: str, a: object, b: object) -> bool:
    """Compatible = shared, non-contradictory, and (for categoricals) equal."""
    if a is None or b is None:
        return False
    if attr_contradicts(name, a, b):
        return False
    return True


@dataclass
class NavigabilityModel:
    """Data-driven 'navigable water' check (replaces an external coastline set).

    Built from the centroids of real NAVAREA IV events, it defines the empirical
    operational envelope. Positions inside the envelope score 1.0; positions
    outside decay with distance — so a fabricated report placed on land or far
    outside the operational region receives a low spatial-descriptive score,
    without requiring a GIS dependency. For deployment, swap in a GEBCO/Natural
    Earth landmask behind the same ``navigability`` interface.
    """

    centroids: Sequence[LatLon]
    decay_nm: float = 60.0  # score decays over this distance outside the hull

    def __post_init__(self) -> None:
        lats = [c[0] for c in self.centroids] or [0.0]
        lons = [c[1] for c in self.centroids] or [0.0]
        # Padded bounding box of the operational region.
        self.lat_min, self.lat_max = min(lats), max(lats)
        self.lon_min, self.lon_max = min(lons), max(lons)
        self._hull = None
        try:
            from shapely.geometry import MultiPoint

            if len(self.centroids) >= 3:
                self._hull = MultiPoint([(lon, lat) for lat, lon in self.centroids]).convex_hull
        except Exception:
            self._hull = None

    def navigability(self, pos: Optional[LatLon]) -> float:
        """Continuous navigability score in [0, 1] for a position."""
        if pos is None:
            return 0.5  # unknown position → neutral
        lat, lon = pos
        if self._hull is not None:
            from shapely.geometry import Point

            p = Point(lon, lat)
            if self._hull.contains(p) or self._hull.touches(p):
                return 1.0
            # distance in degrees → approximate nm at this latitude
            deg = self._hull.distance(p)
            nm = deg * 60.0 * max(0.2, math.cos(math.radians(lat)))
            return float(max(0.0, math.exp(-nm / self.decay_nm)))
        # Fallback: padded bounding-box membership.
        pad = 2.0
        if (self.lat_min - pad <= lat <= self.lat_max + pad
                and self.lon_min - pad <= lon <= self.lon_max + pad):
            return 1.0
        dlat = max(0.0, self.lat_min - lat, lat - self.lat_max)
        dlon = max(0.0, self.lon_min - lon, lon - self.lon_max)
        nm = math.hypot(dlat, dlon) * 60.0
        return float(max(0.0, math.exp(-nm / self.decay_nm)))
