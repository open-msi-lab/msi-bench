"""Credibility-weighted Dempster–Shafer evidence fusion (Section 3.6).

The frame of discernment is Θ = {Htrue, Hfalse}; focal elements are ``{T}``,
``{F}``, and Θ. Each report's BPA is built directly from its MCAM credibility
(Eq. (12)). Evidence is combined with a conflict-aware strategy: standard
Dempster combination is blended with a Murphy-style credibility-weighted average
via the sigmoid ``w(K̄)`` of the order-invariant mean pairwise conflict K̄
(Section 3.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .credibility import domain
from .geo import centroid as geo_centroid
from .schema import LatLon, MSIRecord, ProviderType, VerificationStatus
from .similarity import word_ngrams

# A BPA is a dict with keys 'T' (={Htrue}), 'F' (={Hfalse}), 'Θ' (=frame).
BPA = Dict[str, float]


def build_bpa(cr: float, q: float = 0.6, deny: bool = False) -> BPA:
    """Construct a report's BPA from its credibility (Eq. (12)).

    ``deny=True`` builds the assertion-signed variant: a report whose
    existence/status assertion DENIES the hypothesis lends its credibility mass
    to {H_false}. Within the designed pipeline this is never needed — Section
    3.6 scopes the frame to a conflict-free subcluster whose reports agree,
    so m({H_true}) = Cr is semantically appropriate. It exists for the
    no-partition ablation (E4d): when contradictory reports are force-fused,
    representing a denial as support would mis-specify the fusion input, so the
    ablation is evaluated in both the historical 'blind' idiom and this signed
    idiom (the Zadeh high-conflict regime).
    """
    cr = min(1.0, max(0.0, cr))
    m_committed = cr
    m_opposed = (1.0 - cr) * q
    m_theta = (1.0 - cr) * (1.0 - q)
    if deny:
        return {"T": m_opposed, "F": m_committed, "Θ": m_theta}
    return {"T": m_committed, "F": m_opposed, "Θ": m_theta}


def conflict_factor(m1: BPA, m2: BPA) -> float:
    """K = Σ_{B∩C=∅} m1(B)·m2(C) for the binary frame (Eq. (13))."""
    return m1["T"] * m2["F"] + m1["F"] * m2["T"]


def dempster_combine(m1: BPA, m2: BPA) -> Tuple[BPA, float]:
    """Dempster's rule over the binary frame (Eq. (13)). Returns (combined, K)."""
    k = conflict_factor(m1, m2)
    denom = 1.0 - k
    if denom < 1e-9:  # near-total conflict; fall back to averaging
        return ({"T": 0.5 * (m1["T"] + m2["T"]),
                 "F": 0.5 * (m1["F"] + m2["F"]),
                 "Θ": 0.5 * (m1["Θ"] + m2["Θ"])}, k)
    t = (m1["T"] * m2["T"] + m1["T"] * m2["Θ"] + m1["Θ"] * m2["T"]) / denom
    f = (m1["F"] * m2["F"] + m1["F"] * m2["Θ"] + m1["Θ"] * m2["F"]) / denom
    th = (m1["Θ"] * m2["Θ"]) / denom
    return {"T": t, "F": f, "Θ": th}, k


def effective_source_groups(records: Sequence[MSIRecord],
                            theta_ndup: float) -> List[List[int]]:
    """Group near-verbatim subcluster members into effective sources (Definition 4).

    Fusion-stage independence discount: Section 3.6 assumes the evidence
    entering Eq. (13) comes from independent sources, and Section 3.3's
    check (d) already *identifies* when it does not — near-verbatim texts
    (word-3-gram containment ≥ θ_ndup) are one effective source regardless of
    the identities attached. This function applies exactly that predicate —
    same containment measure, same θ, same scoping (the pair qualifies only
    when at least one report is free-text CROWDSOURCE and both carry ≥ 4
    shingles) — and returns the connected components of the linkage relation.
    Each component contributes ONE item of evidence to the BPA combination,
    so a k-member scripted campaign can no longer manufacture corroboration
    sharpening out of one authored text (the effective-source collapse of
    Definition 4). Members whose texts genuinely differ (independent
    observers, or a paraphrasing adversary — the paraphrased-tier residual of
    Section 5.1) remain separate sources.
    """
    n = len(records)
    ngrams = [word_ngrams(r.raw_text) for r in records]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if len(ngrams[i]) < 4:
            continue
        for j in range(i + 1, n):
            if len(ngrams[j]) < 4:
                continue
            if (records[i].provider_type != ProviderType.CROWDSOURCE
                    and records[j].provider_type != ProviderType.CROWDSOURCE):
                continue
            if len(ngrams[i] & ngrams[j]) >= theta_ndup * min(len(ngrams[i]),
                                                              len(ngrams[j])):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups)]


def mean_pairwise_conflict(bpas: Sequence[BPA]) -> float:
    """K̄ = mean pairwise conflict (Section 3.6); order-invariant, in [0, 1]."""
    n = len(bpas)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += conflict_factor(bpas[i], bpas[j])
    return total * 2.0 / (n * (n - 1))


def _iterative_dempster(bpas: Sequence[BPA]) -> BPA:
    acc = dict(bpas[0])
    for m in bpas[1:]:
        acc, _ = dempster_combine(acc, m)
    return acc


def _iterative_yager(bpas: Sequence[BPA]) -> BPA:
    """Yager's rule (baseline of Section 5.3): conflict mass goes to ignorance.

    Implemented associatively as the unnormalised conjunctive (TBM) combination
    with an explicit empty-set accumulator, transferred to Θ once at the end —
    equivalent to Yager's quasi-associative formulation over the ground masses,
    and order-invariant by construction.
    """
    t, f, th = bpas[0]["T"], bpas[0]["F"], bpas[0]["Θ"]
    for m in bpas[1:]:
        t2, f2, th2 = m["T"], m["F"], m["Θ"]
        nt = t * t2 + t * th2 + th * t2
        nf = f * f2 + f * th2 + th * f2
        nth = th * th2
        t, f, th = nt, nf, nth  # remainder (incl. prior ∅ mass) is implicit
    return {"T": t, "F": f, "Θ": th + max(0.0, 1.0 - t - f - th)}


def _murphy_average(bpas: Sequence[BPA], weights: Sequence[float]) -> BPA:
    """Credibility-weighted average, then (n-1) self-combinations (Section 3.6)."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(bpas))
    w = w / w.sum()
    avg = {k: float(sum(wi * m[k] for wi, m in zip(w, bpas))) for k in ("T", "F", "Θ")}
    sharp = dict(avg)
    for _ in range(len(bpas) - 1):
        sharp, _ = dempster_combine(sharp, avg)
    return sharp


@dataclass
class FusedRecord:
    belief: float
    plausibility: float
    mass: BPA
    k_bar: float
    provenance: List[str]
    centroid: Optional[LatLon] = None
    union_centroid: Optional[LatLon] = None
    union_area_nm2: float = 0.0
    valid_start: Optional[datetime] = None
    valid_end: Optional[datetime] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    record_id: str = ""
    # cross-references to parallel fused records from sibling subclusters of the
    # same (partitioned) cluster — the competing interpretations (Section 3.6)
    conflict_siblings: List[str] = field(default_factory=list)
    # fused content (E4d): the subcluster's existence interpretation ("affirm" |
    # "deny", weight-majority assertion of the members) and the weight-majority
    # specific hazard category (extractor wildcards abstain), evaluated against
    # event ground truth by the fused-record accuracy metrics (Section 5.3).
    interpretation: str = "affirm"
    fused_hazard_type: Optional[str] = None
    # Section 3.6 independence discount bookkeeping: number of effective sources
    # that actually entered the BPA combination (== len(provenance) when no
    # members were near-verbatim-linked, or when the discount is disabled).
    n_effective_sources: int = 0


def _geometry_union(records: Sequence[MSIRecord]):
    """Credibility-weighted geometric union of contributing geometries.

    Returns ``(union_centroid_latlon, area_nm2)``. Implements the precautionary
    union of Section 3.6: points are buffered by their uncertainty,
    lines to corridors, polygons used directly, on a local tangent plane (nm).
    Falls back to ``(None, 0.0)`` when shapely is unavailable.
    """
    try:
        from shapely.geometry import LineString, Point, Polygon
        from shapely.ops import unary_union
    except Exception:
        return None, 0.0
    cents = [r.centroid for r in records if r.centroid]
    if not cents:
        return None, 0.0
    lat0, lon0 = cents[0]
    coslat = max(0.2, math.cos(math.radians(lat0)))

    def proj(latlon):
        return ((latlon[1] - lon0) * 60.0 * coslat, (latlon[0] - lat0) * 60.0)

    geoms = []
    for r in records:
        if not r.coordinates:
            continue
        pts = [proj(c) for c in r.coordinates]
        try:
            if len(pts) >= 3 and r.geometry_kind.value == "POLYGON":
                g = Polygon(pts)
                geoms.append(g if g.is_valid else g.buffer(0))
            elif len(pts) >= 2 and r.geometry_kind.value == "LINE":
                geoms.append(LineString(pts).buffer(max(r.spatial_uncertainty_nm, 1.0)))
            else:
                geoms.append(Point(pts[0]).buffer(max(r.spatial_uncertainty_nm, 1.0)))
        except Exception:
            continue
    if not geoms:
        return None, 0.0
    u = unary_union(geoms)
    c = u.centroid
    union_latlon = (lat0 + c.y / 60.0, lon0 + c.x / (60.0 * coslat))
    return union_latlon, float(u.area)


def fuse_subcluster(
    records: Sequence[MSIRecord],
    credibility: Sequence[float],
    q: float = 0.6,
    beta: float = 10.0,
    k0: float = 0.5,
    method: str = "blend",
    record_id: str = "",
    signed: bool = False,
    theta_ndup: Optional[float] = None,
) -> FusedRecord:
    """Fuse a conflict-free subcluster into one confidence-scored record.

    ``method`` selects the combination rule (for baselines):

    * ``"blend"``  — proposed conflict-aware blend of Dempster and Murphy.
    * ``"dempster"`` — standard Dempster only (credibility-informed BPA).
    * ``"uniform"`` — uniform BPA m({T}) = 0.5 AND uniform content weights —
      the credibility-blind baseline end to end (fused geometry/attributes are
      unweighted means/votes, so credibility informs nothing).
    * ``"murphy"`` — Murphy credibility-weighted average + self-combination.
    * ``"yager"`` — Yager's rule (conflict mass to ignorance).

    ``signed=True`` uses assertion-signed BPAs (denials support {H_false}) —
    only meaningful when fusing an UNpartitioned cluster that may contain
    contradictory assertions (see :func:`build_bpa`).

    ``theta_ndup`` enables the Section 3.6 fusion-stage independence discount:
    near-verbatim members (see :func:`effective_source_groups`) collapse to
    one effective source — the group's highest-credibility member carries the
    group's BPA and content-vote weight; the remaining members contribute
    zero marginal evidence. Provenance, the precautionary geometry union, and
    the conservative temporal window still cover ALL members (a verbatim
    relay confirms the broadcast's reach, not the hazard — it is bookkept,
    not corroborating). ``None`` disables the discount (used for the E4 ablation).
    """
    denies = [domain.is_denial(r.structured_data) for r in records]

    # Section 3.6 independence discount: one BPA and one content vote per
    # effective source. Representative = highest credibility, ties broken by
    # position (deterministic).
    if theta_ndup is not None and theta_ndup <= 1.0 and len(records) > 1:
        groups = effective_source_groups(records, theta_ndup)
    else:
        groups = [[i] for i in range(len(records))]
    reps = sorted(max(g, key=lambda i: (credibility[i], -i)) for g in groups)

    eff_cred = [credibility[i] for i in reps]
    eff_denies = [denies[i] for i in reps]
    if method == "uniform":
        bpas = [build_bpa(0.5, q, deny=(d and signed)) for d in eff_denies]
    else:
        bpas = [build_bpa(cr, q, deny=(d and signed))
                for cr, d in zip(eff_cred, eff_denies)]

    k_bar = mean_pairwise_conflict(bpas)
    if method in ("dempster", "uniform"):
        fused = _iterative_dempster(bpas)
    elif method == "murphy":
        fused = _murphy_average(bpas, eff_cred)
    elif method == "yager":
        fused = _iterative_yager(bpas)
    else:  # blend (proposed)
        m_std = _iterative_dempster(bpas)
        m_avg = _murphy_average(bpas, eff_cred)
        w = 1.0 / (1.0 + math.exp(-beta * (k_bar - k0)))
        fused = {k: (1 - w) * m_std[k] + w * m_avg[k] for k in ("T", "F", "Θ")}

    belief = fused["T"]
    plausibility = 1.0 - fused["F"]

    # content weights: credibility, except for the credibility-blind baseline;
    # non-representative members of a linked group carry zero vote weight
    # (one voice per effective source).
    rep_set = set(reps)
    cw = [(1.0 if method == "uniform" else credibility[i]) if i in rep_set
          else 0.0 for i in range(len(records))]

    # fused geometry: weighted centroid of contributing centroids
    pts = [(r.centroid, c) for r, c in zip(records, cw) if r.centroid]
    fused_centroid = None
    if pts:
        wsum = sum(c for _, c in pts) or 1.0
        lat = sum(p[0] * c for p, c in pts) / wsum
        lon = sum(p[1] * c for p, c in pts) / wsum
        fused_centroid = (lat, lon)

    # fused spatial extent: credibility-weighted geometric union (Section 3.6)
    union_centroid, union_area = _geometry_union(records)

    # fused temporal validity: widest (most conservative) period
    starts = [r.valid_start or r.issue_time for r in records if (r.valid_start or r.issue_time)]
    ends = [r.valid_end for r in records if r.valid_end]
    status = VerificationStatus.CROSS_VERIFIED if belief > 0.75 else VerificationStatus.UNVERIFIED

    # fused content (E4d): weight-majority existence interpretation and
    # specific hazard category (extractor wildcards abstain from the vote)
    deny_w = sum(c for c, d in zip(cw, denies) if d)
    affirm_w = sum(c for c, d in zip(cw, denies) if not d)
    interpretation = "deny" if deny_w > affirm_w else "affirm"
    votes: Dict[str, float] = {}
    for r, c in zip(records, cw):
        ht = r.structured_data.get("hazard_type")
        if ht is None:
            continue
        key = str(ht).strip().lower()
        if not key or key in domain.WILDCARD_ATTR_VALUES:
            continue
        votes[key] = votes.get(key, 0.0) + c
    fused_hazard = max(votes, key=votes.get) if votes else None

    return FusedRecord(
        belief=belief,
        plausibility=plausibility,
        mass=fused,
        k_bar=k_bar,
        provenance=[r.record_id for r in records],
        centroid=fused_centroid,
        union_centroid=union_centroid,
        union_area_nm2=union_area,
        valid_start=min(starts) if starts else None,
        valid_end=max(ends) if ends else None,
        verification_status=status,
        record_id=record_id,
        interpretation=interpretation,
        fused_hazard_type=fused_hazard,
        n_effective_sources=len(reps),
    )
