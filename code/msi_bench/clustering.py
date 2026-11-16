"""STS-DBSCAN: spatiotemporal–semantic clustering (Section 3.4).

Extends DBSCAN with (1) the composite similarity measure in place of a single
distance metric, and (2) a semantic-compatibility constraint that forbids
grouping reports of incompatible event types. Includes the adaptive-ε strategy
(Kneedle on the sorted k-similarity curve, with a calibrated fallback) and the
medoid-based post-processing merge.

Cluster labels are integers ``0..k-1``; ``-1`` denotes noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import MSIRecord, ProviderType
from .similarity import type_similarity

NOISE = -1


def adaptive_epsilon(
    sim: np.ndarray, min_pts: int = 2, fallback: float = 0.45
) -> Tuple[float, float, bool]:
    """Select ε at the knee of the sorted k-similarity curve (Section 3.4).

    Returns ``(epsilon, confidence, used_fallback)``. ``confidence`` is the
    maximum perpendicular distance from the sorted curve to its endpoint chord;
    when it falls below ``0.05 · range(f)`` the calibrated ``fallback`` is used.
    """
    n = sim.shape[0]
    if n <= min_pts:
        return fallback, 0.0, True
    # k-th highest similarity to OTHER points for each report.
    simk = []
    for i in range(n):
        row = np.delete(sim[i], i)
        row_sorted = np.sort(row)[::-1]
        k = min(min_pts, len(row_sorted))
        simk.append(row_sorted[k - 1])
    f = np.sort(np.asarray(simk))  # ascending curve
    rng = float(f[-1] - f[0])
    if rng <= 1e-9:
        return fallback, 0.0, True
    # perpendicular distance of each point to the chord (0,f0)->(n-1,f_{n-1})
    x = np.arange(n, dtype=float)
    x0, x1, y0, y1 = x[0], x[-1], f[0], f[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    dist = np.abs((y1 - y0) * x - (x1 - x0) * f + x1 * y0 - y1 * x0) / denom
    knee = int(np.argmax(dist))
    confidence = float(dist[knee])
    if confidence < 0.05 * rng:
        return fallback, confidence, True
    return float(f[knee]), confidence, False


@dataclass
class STSDBSCAN:
    """Spatiotemporal–semantic DBSCAN."""

    min_pts: int = 2
    eps: Optional[float] = None  # None → adaptive
    theta_compat: float = 0.50
    eps_fallback: float = 0.45
    merge: bool = True

    used_fallback: bool = False
    chosen_eps: float = 0.0

    def _compatible(self, a: MSIRecord, b: MSIRecord) -> bool:
        """Semantic-compatibility veto, scoped to extraction confidence.

        A type parsed out of free-text crowd prose is *advisory*: 30% of
        multi-source events carry type disagreement purely from crowd
        extraction noise (e.g. OTHER for a paraphrased NAV warning), and
        absence or uncertainty of type information is not evidence of a
        different event. Advisory types still shape the semantic similarity
        score; only *authoritative* assignments (official metadata, sensor
        channels) can veto identity. On single-provenance official streams
        (E3-R) this reduces to the original predicate.
        """
        if (a.provider_type == ProviderType.CROWDSOURCE
                or b.provider_type == ProviderType.CROWDSOURCE):
            return True
        return type_similarity(a.event_type, b.event_type) >= self.theta_compat

    def fit(self, records: Sequence[MSIRecord], sim: np.ndarray) -> np.ndarray:
        n = len(records)
        if n == 0:
            return np.array([], dtype=int)
        if self.eps is None:
            self.chosen_eps, _, self.used_fallback = adaptive_epsilon(
                sim, self.min_pts, self.eps_fallback)
        else:
            self.chosen_eps = self.eps
        eps = self.chosen_eps

        # Semantically constrained neighbourhoods (include self).
        neighbors: List[List[int]] = []
        for i in range(n):
            nb = [i]
            for j in range(n):
                if j != i and sim[i, j] >= eps and self._compatible(records[i], records[j]):
                    nb.append(j)
            neighbors.append(nb)

        labels = np.full(n, NOISE, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cid = -1
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if len(neighbors[i]) < self.min_pts:
                continue  # not core (may be claimed as border later)
            cid += 1
            labels[i] = cid
            seeds = list(neighbors[i])
            in_seeds = set(seeds)  # O(1) membership → keeps cluster expansion at O(n²)
            qi = 0
            while qi < len(seeds):
                p = seeds[qi]
                qi += 1
                if not visited[p]:
                    visited[p] = True
                    if len(neighbors[p]) >= self.min_pts:  # p is core → expand
                        for q in neighbors[p]:
                            if q not in in_seeds:
                                seeds.append(q)
                                in_seeds.add(q)
                if labels[p] == NOISE:
                    labels[p] = cid
        if self.merge and cid >= 1:
            labels = self._merge_clusters(records, sim, labels, eps)
        return labels

    def _merge_clusters(
        self, records: Sequence[MSIRecord], sim: np.ndarray, labels: np.ndarray, eps: float
    ) -> np.ndarray:
        """Medoid-based post-processing merge."""
        clusters: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            if lab != NOISE:
                clusters.setdefault(lab, []).append(idx)
        # medoid = member with highest mean similarity to the others
        medoid: Dict[int, int] = {}
        for lab, members in clusters.items():
            if len(members) == 1:
                medoid[lab] = members[0]
                continue
            best, best_score = members[0], -1.0
            for m in members:
                score = float(np.mean([sim[m, o] for o in members if o != m]))
                if score > best_score:
                    best, best_score = m, score
            medoid[lab] = best

        labs = sorted(clusters)
        parent = {lab: lab for lab in labs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(labs)):
            for b in range(a + 1, len(labs)):
                la, lb = labs[a], labs[b]
                ma, mb = medoid[la], medoid[lb]
                if sim[ma, mb] >= 0.8 * eps and self._compatible(records[ma], records[mb]):
                    parent[find(la)] = find(lb)

        # relabel to contiguous ids
        remap: Dict[int, int] = {}
        out = np.full(len(labels), NOISE, dtype=int)
        for idx, lab in enumerate(labels):
            if lab == NOISE:
                continue
            root = find(lab)
            if root not in remap:
                remap[root] = len(remap)
            out[idx] = remap[root]
        return out


def sim_to_distance(sim: np.ndarray) -> np.ndarray:
    """Convert a similarity matrix to a distance matrix (1 - sim, clipped)."""
    return np.clip(1.0 - sim, 0.0, 1.0)
