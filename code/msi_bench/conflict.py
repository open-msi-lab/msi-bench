"""Conflict detection, classification, and partitioning (Section 3.5).

Implements the four-type taxonomy (CONF_SPATIAL / TEMPORAL / SEMANTIC / SOURCE),
the three-level severity classification (CRITICAL / MAJOR / MINOR), and the
partitioning of a cluster into conflict-free subclusters prior to D-S fusion
(Section 3.5). Partitioning uses a greedy maximal-independent-set scheme over
the irreconcilable-conflict graph, which yields conflict-free subclusters and
reduces to the typical affirm/deny bipartition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .credibility import domain
from .schema import MSIRecord, ProviderType
from .similarity import (
    SimilarityModel,
    _validity_interval,
    spatial_similarity,
    type_similarity,
)

CONF_SPATIAL = "CONF_SPATIAL"
CONF_TEMPORAL = "CONF_TEMPORAL"
CONF_SEMANTIC = "CONF_SEMANTIC"
CONF_SOURCE = "CONF_SOURCE"

CRITICAL = "CRITICAL"
MAJOR = "MAJOR"
MINOR = "MINOR"


@dataclass
class Conflict:
    i: int  # local index within the cluster
    j: int
    type: str
    severity: str


def detect_conflicts(
    records: Sequence[MSIRecord],
    sim_model: SimilarityModel,
    credibility: Sequence[float],
    theta_sem: float = 0.75,
    theta_sconf: float = 0.10,
    high_cred: float = 0.75,
    theta_rel: float = 0.50,
) -> List[Conflict]:
    """Detect and classify conflicts among the reports of one cluster.

    Definition 2 makes relatedness — ``Sim(ri, rj) ≥ θ_rel`` — a
    precondition of every conflict; the evidence of relatedness must come from
    dimensions other than the contradicted one. R1/R2 instantiate it on the
    semantic axis (``Sim_m ≥ θ_sem``, stricter: the pair must purport to
    describe the same hazard, while space/time carry the contradiction);
    R3/R4 contradict an *attribute*, so they instantiate it on the composite,
    so that weakly related cluster co-members are not counted as semantic
    contradictions.
    """
    conflicts: List[Conflict] = []
    n = len(records)
    chain = _reference_chains(records)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            sem = sim_model.semantic_similarity(a, b)
            spa = spatial_similarity(a, b, sim_model.sigma_p)
            ctype = None

            conflict_attr = None
            # R1 — spatial conflict: same purported hazard but disjoint location.
            if sem >= theta_sem and spa < theta_sconf:
                ctype = CONF_SPATIAL
            # R2 — temporal conflict: validity periods do not overlap at all
            # (Overlap(Pi,Pj)=0), but the reports are otherwise related.
            # Tested on the interval Jaccard directly — NOT on the blended
            # temporal_similarity, whose issuance-proximity term would mask a
            # genuine no-overlap conflict between reports issued close in time.
            # A pair linked by update or cancellation references, directly or
            # through intermediate messages, is one lifecycle (a warning and
            # the messages that supersede it): its disjoint windows are
            # succession, so R2 does not apply to it.
            elif (sem >= theta_sem and chain[i] != chain[j]
                  and _no_validity_overlap(a, b)):
                ctype = CONF_TEMPORAL
            elif sim_model.sim(a, b) >= theta_rel:  # relatedness floor
                # R4 — source conflict: official vs crowd on existence/status.
                conflict_attr = _source_conflict(a, b)
                if conflict_attr is not None:
                    ctype = CONF_SOURCE
                else:
                    # R3 — semantic conflict: contradictory critical attribute.
                    conflict_attr = _semantic_conflict(a, b)
                    if conflict_attr is not None:
                        ctype = CONF_SEMANTIC

            if ctype is None:
                continue
            sev = _severity(ctype, conflict_attr, credibility[i], credibility[j], high_cred)
            conflicts.append(Conflict(i=i, j=j, type=ctype, severity=sev))
    return conflicts


def _reference_chains(records: Sequence[MSIRecord]) -> List[int]:
    """Lifecycle-chain id of each report in the cluster.

    Two reports share an id iff they are connected by ``references`` (one
    updates or cancels the other), directly or through intermediate reports
    of the same cluster: A <- B <- C puts A and C in one chain although
    neither references the other. A report without references to or from
    the cluster is a chain of its own.
    """
    index = {r.record_id: i for i, r in enumerate(records)}
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, r in enumerate(records):
        for ref in r.references:
            j = index.get(ref)
            if j is not None:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(records))]


def _no_validity_overlap(a: MSIRecord, b: MSIRecord) -> bool:
    """True iff the two reports' validity intervals do not overlap (interval Jaccard = 0).

    Uses the same validity-interval construction as the temporal similarity
    (filling event-type default durations when an endpoint is missing). Returns
    False when either interval is undefined, so a conflict is only asserted on
    positive evidence of disjoint validity periods.
    """
    ia, ib = _validity_interval(a), _validity_interval(b)
    if ia is None or ib is None:
        return False
    inter = min(ia[1], ib[1]) - max(ia[0], ib[0])
    return inter <= 0.0


# Existence/status attributes whose contradiction is IRRECONCILABLE
# (affirm vs deny, e.g. "confirmed" vs "cancelled") and must trigger partition.
_EXISTENCE_ATTRS = {"existence_status", "status", "incident_status"}


def _semantic_conflict(a: MSIRecord, b: MSIRecord):
    """Return the offending critical attribute name, or None."""
    acrit = domain.CRITICAL_ATTRS.get(a.event_type, set()) & domain.CRITICAL_ATTRS.get(
        b.event_type, set())
    for attr in acrit:
        if attr in a.structured_data and attr in b.structured_data:
            if domain.attr_contradicts(attr, a.structured_data[attr], b.structured_data[attr]):
                return attr
    return None


def _source_conflict(a: MSIRecord, b: MSIRecord):
    """Return the offending existence/status attribute name, or None."""
    types = {a.provider_type, b.provider_type}
    if ProviderType.OFFICIAL not in types or ProviderType.CROWDSOURCE not in types:
        return None
    for attr in _EXISTENCE_ATTRS:
        if attr in a.structured_data and attr in b.structured_data:
            if domain.attr_contradicts(attr, a.structured_data[attr], b.structured_data[attr]):
                return attr
    return None


def _severity(ctype: str, attr, cr_i: float, cr_j: float, high: float) -> str:
    # Affirm-vs-deny on existence/status is irreconcilable → CRITICAL, so the
    # cluster is partitioned before D-S fusion (Section 3.5).
    if attr in _EXISTENCE_ATTRS:
        return CRITICAL
    if ctype in (CONF_SPATIAL, CONF_TEMPORAL) and cr_i >= high and cr_j >= high:
        return CRITICAL
    if ctype in (CONF_SEMANTIC, CONF_SOURCE):
        return MAJOR
    return MINOR


def partition_cluster(
    n: int, conflicts: Sequence[Conflict]
) -> List[List[int]]:
    """Partition cluster indices ``0..n-1`` into conflict-free subclusters.

    Edges are the *irreconcilable* (CRITICAL-severity) conflicts. A greedy
    maximal-independent-set extraction guarantees no subcluster contains an
    irreconcilable pair; in the common affirm/deny case this yields two
    subclusters.
    """
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for c in conflicts:
        if c.severity == CRITICAL:
            adj[c.i].add(c.j)
            adj[c.j].add(c.i)
    if all(len(v) == 0 for v in adj.values()):
        return [list(range(n))]

    remaining = set(range(n))
    subclusters: List[List[int]] = []
    while remaining:
        group: List[int] = []
        blocked: set = set()
        # deterministic order for reproducibility
        for node in sorted(remaining):
            if node in blocked:
                continue
            group.append(node)
            blocked |= adj[node]
        subclusters.append(sorted(group))
        remaining -= set(group)
    return subclusters
