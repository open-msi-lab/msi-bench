"""The MSI-Model record type and enumerations (Section 3.2).

Two record granularities are used:

* :class:`ParsedMessage` — the structured result of running the pipeline on
  a single raw NAVAREA CSV row. It retains the original archive fields plus the
  derived attributes (message function, geometry, normalised authority, …).
* :class:`MSIRecord` — the general multi-source report consumed by the
  credibility/clustering/fusion framework (official, crowdsourced, sensor, or
  adversarial). It is the eight-tuple ⟨ID, Type, Source, Content, Spatial,
  Temporal, Meta, Trust⟩ of Eq. (1), flattened for practical computation.

These are plain dataclasses (not Pydantic) to keep the core dependency-free.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ProviderType(str, enum.Enum):
    OFFICIAL = "OFFICIAL"
    CROWDSOURCE = "CROWDSOURCE"
    SENSOR = "SENSOR"


class MessageFunction(str, enum.Enum):
    """Message-function class assigned by the classifier (Section 4.3)."""

    NEW_WARNING = "NEW_WARNING"
    UPDATE = "UPDATE"
    CANCEL_ONLY = "CANCEL_ONLY"
    RECURRING_BULLETIN = "RECURRING_BULLETIN"
    ADMIN_ERROR = "ADMIN_ERROR"

    @property
    def is_discrete_event(self) -> bool:
        """Only NEW_WARNING / UPDATE enter the discrete-event benchmark."""
        return self in (MessageFunction.NEW_WARNING, MessageFunction.UPDATE)


class EventType(str, enum.Enum):
    NAV_WARNING = "NAV_WARNING"
    MET_WARNING = "MET_WARNING"
    SAR_INFO = "SAR_INFO"
    PIRACY_ALERT = "PIRACY_ALERT"
    PORT_INFO = "PORT_INFO"
    OTHER = "OTHER"


class GeometryKind(str, enum.Enum):
    POINT = "POINT"
    LINE = "LINE"
    POLYGON = "POLYGON"
    MULTIPOINT = "MULTIPOINT"
    UNKNOWN = "UNKNOWN"


class VerificationStatus(str, enum.Enum):
    UNVERIFIED = "UNVERIFIED"
    AUTO_VERIFIED = "AUTO_VERIFIED"
    CROSS_VERIFIED = "CROSS_VERIFIED"
    MANUAL_VERIFIED = "MANUAL_VERIFIED"


class Label(str, enum.Enum):
    """Binary ground-truth credibility label (Section 4.3)."""

    CREDIBLE = "CREDIBLE"
    NON_CREDIBLE = "NON_CREDIBLE"


class Category(str, enum.Enum):
    """Provenance category — the finest ground-truth label in MSI-Bench."""

    OFFICIAL = "OFFICIAL"
    GENUINE_CROWD = "GENUINE_CROWD"
    GENUINE_SENSOR = "GENUINE_SENSOR"
    ATTACK_FABRICATION = "ATTACK_FABRICATION"  # T1
    ATTACK_REPLAY = "ATTACK_REPLAY"            # T2
    ATTACK_TAMPER = "ATTACK_TAMPER"            # T3
    ATTACK_IMPERSONATION = "ATTACK_IMPERSONATION"  # T4
    ATTACK_SYBIL = "ATTACK_SYBIL"              # T5

    @property
    def label(self) -> Label:
        return Label.CREDIBLE if not self.name.startswith("ATTACK") else Label.NON_CREDIBLE


# Coordinate alias: (latitude, longitude) in signed decimal degrees, WGS 84.
LatLon = Tuple[float, float]


@dataclass
class ParsedMessage:
    """Structured output of the pipeline for one raw NAVAREA message."""

    # --- original archive fields ---
    msg_year: int
    msg_number: int
    nav_area: int
    subregion: str
    raw_text: str
    status: str  # 'A' active, 'C' cancelled
    authority_raw: str

    # --- derived temporal fields ---
    issue_time: Optional[datetime] = None
    cancel_time: Optional[datetime] = None          # from the cancelDate column
    self_expiry_time: Optional[datetime] = None      # in-text "CANCEL THIS MSG <dtg>"

    # --- lifecycle / cross-reference fields ---
    cancel_nav_area: Optional[int] = None
    cancel_msg_year: Optional[int] = None
    cancel_msg_number: Optional[int] = None
    # In-text supersession references: list of (msg_number, msg_year).
    in_text_cancels: List[Tuple[int, int]] = field(default_factory=list)

    # --- derived classifications ---
    function: MessageFunction = MessageFunction.NEW_WARNING
    event_type: EventType = EventType.OTHER
    authority_norm: str = ""

    # --- derived geometry ---
    coordinates: List[LatLon] = field(default_factory=list)
    geometry_kind: GeometryKind = GeometryKind.UNKNOWN
    centroid: Optional[LatLon] = None

    # --- data-quality flags surfaced for E0 ---
    defects: List[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        """Stable identifier: ``<year>-<number>`` (unique within a NAVAREA)."""
        return f"{self.msg_year}-{self.msg_number}"

    @property
    def active_duration_hours(self) -> Optional[float]:
        """Active duration = (cancel/expiry time) − issue time, in hours."""
        end = self.cancel_time or self.self_expiry_time
        if self.issue_time is None or end is None:
            return None
        return (end - self.issue_time).total_seconds() / 3600.0


@dataclass
class MSIRecord:
    """A multi-source report — the unit consumed by the framework (Eq. (1))."""

    record_id: str
    event_type: EventType
    provider_type: ProviderType

    # Source provenance
    provider_id: str = "unknown"
    collection_method: str = "unknown"
    authority_norm: str = ""

    # Content
    raw_text: str = ""
    structured_data: Dict[str, object] = field(default_factory=dict)

    # Spatial
    geometry_kind: GeometryKind = GeometryKind.UNKNOWN
    coordinates: List[LatLon] = field(default_factory=list)
    centroid: Optional[LatLon] = None
    spatial_uncertainty_nm: float = 0.0

    # Temporal (datetimes are timezone-naive UTC / Zulu)
    issue_time: Optional[datetime] = None
    valid_start: Optional[datetime] = None
    valid_end: Optional[datetime] = None

    # Meta / device & network metadata (for device credibility + Sybil checks)
    device_class: str = "UNKNOWN"
    auth_level: str = "ANONYMOUS"
    network_origin: Optional[str] = None
    submission_position: Optional[LatLon] = None
    observation_type: str = "visual"  # visual | radar | ais | shore
    vessel_max_speed_kn: Optional[float] = None
    trajectory: List[Tuple[datetime, LatLon]] = field(default_factory=list)
    # Lifecycle references: record_ids of earlier reports that this report
    # updates or cancels (the S-124 reference carried in Meta). Two reports
    # linked by such a reference are one lifecycle, so their disjoint validity
    # windows are succession and not a temporal contradiction (rule R2).
    references: List[str] = field(default_factory=list)

    # Trust (filled by the MCAM)
    credibility_score: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    # --- ground-truth annotations (evaluation only; never read by the model) ---
    gt_category: Optional[Category] = None
    gt_event_id: Optional[str] = None
    gt_quality_tier: Optional[str] = None  # good | moderate | poor (crowd only)
    provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def gt_label(self) -> Optional[Label]:
        return self.gt_category.label if self.gt_category is not None else None
