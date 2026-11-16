"""Composite multi-dimensional similarity (Section 3.4).

``Sim(rᵢ, rⱼ) = ωs·Sims + ωt·Simt + ωm·Simm`` integrates spatial, temporal, and
semantic similarity. The composite depends only on report attributes (never on
credibility), so it is computed once and reused by both the social-credibility
neighbour identification (Section 3.3) and STS-DBSCAN (Section 3.4).

Design notes
------------
* **Spatial** — polygon/polygon pairs use the geodesic Jaccard index on a local
  tangent-plane projection (areas in nm²); all other pairs use the exponential
  centroid-distance kernel ``exp(-dH/σp)``, σp denoting the point-similarity
  scale (distinct from the credibility scale σs).
* **Temporal** — interval Jaccard of validity periods, an issuance-proximity
  kernel, and a lifecycle-succession kernel (Eq. (9)) over the validity-window
  adjacency gap; missing validity windows fall back to event-type default
  durations. In the default ``gate`` mode the three enter as a
  temporal-compatibility gate (Eq. (10)) multiplying the spatial/semantic vote;
  ``additive`` retains the original weighted sum for baseline variants.
* **Semantic** — Wu–Palmer type similarity over a small MSI taxonomy,
  attribute Jaccard/agreement (with the renormalised fallback when no
  attributes are shared), and text cosine similarity over sentence embeddings
  with a TF-IDF fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geo import haversine_nm
from .schema import EventType, GeometryKind, LatLon, MSIRecord

try:
    from shapely.geometry import LineString, Point, Polygon

    _HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _HAS_SHAPELY = False


# Event-type default validity durations (hours), by hazard timescale.
_DEFAULT_DURATION_H: Dict[EventType, float] = {
    EventType.NAV_WARNING: 7 * 24.0,
    EventType.MET_WARNING: 24.0,
    EventType.SAR_INFO: 24.0,
    EventType.PIRACY_ALERT: 7 * 24.0,
    EventType.PORT_INFO: 7 * 24.0,
    EventType.OTHER: 24.0,
}


# ---------------------------------------------------------------------------
# Wu–Palmer type similarity over a small MSI type taxonomy.
# Each type sits under its own depth-2 super-class so that *different* hazard
# types (e.g. NAV vs MET) have their LCS at the root and score below θcompat,
# while genuinely related incident types share a parent and score above it.
# ---------------------------------------------------------------------------
_TAX_PARENT: Dict[str, Optional[str]] = {
    "ROOT": None,
    "ENV": "ROOT",
    "ATMOS": "ROOT",
    "INCIDENT": "ROOT",
    "PORTGRP": "ROOT",
    "NAV_WARNING": "ENV",
    "MET_WARNING": "ATMOS",
    "SAR_INFO": "INCIDENT",
    "PIRACY_ALERT": "INCIDENT",
    "PORT_INFO": "PORTGRP",
    "OTHER": "ROOT",
}


def _depth(node: str) -> int:
    d = 1
    while _TAX_PARENT.get(node):
        node = _TAX_PARENT[node]
        d += 1
    return d


def _lcs_depth(a: str, b: str) -> int:
    ancestors = set()
    n = a
    while n is not None:
        ancestors.add(n)
        n = _TAX_PARENT.get(n)
    n = b
    while n is not None:
        if n in ancestors:
            return _depth(n)
        n = _TAX_PARENT.get(n)
    return 1


def type_similarity(a: EventType, b: EventType) -> float:
    """Wu–Palmer similarity in [0, 1].

    OTHER vs OTHER is fully compatible; OTHER vs a specific type scores 0.4 —
    below the default θcompat (0.5) — so an unknown type does not silently pass
    the semantic-compatibility constraint and cluster with any specific type.
    """
    if a == b:
        return 1.0
    if a == EventType.OTHER or b == EventType.OTHER:
        return 0.4
    na, nb = a.value, b.value
    return 2.0 * _lcs_depth(na, nb) / (_depth(na) + _depth(nb))


# ---------------------------------------------------------------------------
# Near-verbatim text duplication (Section 3.3 independence check (d)).
# ---------------------------------------------------------------------------
def word_ngrams(text: Optional[str], n: int = 3) -> frozenset:
    """Word n-gram shingle set of a text (lowercased whitespace tokens)."""
    toks = (text or "").lower().split()
    if not toks:
        return frozenset()
    if len(toks) < n:
        return frozenset([tuple(toks)])
    return frozenset(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def text_duplication(a: Optional[str], b: Optional[str], n: int = 3) -> float:
    """Near-verbatim duplication score: n-gram containment |A∩B| / min(|A|,|B|).

    An *authorship* signal, distinct from semantic similarity: two independent
    observers of the same event paraphrase (low shingle overlap — dev p99 0.57
    on the benchmark's same-event crowd pairs), while copy-paste relay,
    campaign-scripted variants (a shared base text plus a token suffix), and
    tampered resubmissions contain their source near-verbatim (≥ 0.92).
    Containment rather than Jaccard so that extending a copied text with a
    suffix cannot dilute the score.
    """
    A, B = word_ngrams(a, n), word_ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


# ---------------------------------------------------------------------------
# Text encoder: transformer embeddings if configured, else TF-IDF fallback.
# ---------------------------------------------------------------------------
class TextEncoder:
    """Encodes raw text into L2-normalised vectors for cosine similarity.

    Uses a sentence-transformers model when one is configured *and* the library
    is installed; otherwise fits a TF-IDF vectoriser on the supplied corpus
    (Section 3.4 lightweight alternative).
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu") -> None:
        self.model = None
        self._tfidf = None
        self.kind = "tfidf"  # recorded into experiment results (a silent fallback must never
        # masquerade as an encoder run)
        if model_name:
            try:  # pragma: no cover - optional heavy dependency
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name, device=device)
                self.kind = f"sbert:{model_name}"
            except Exception as exc:
                print(f"[similarity] sentence-transformers unavailable ({exc}); "
                      "using TF-IDF fallback.")

    def fit_tfidf(self, corpus: Sequence[str]) -> None:
        """Fit the TF-IDF fallback on a corpus (no-op if a model is loaded)."""
        if self.model is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000,
        )
        # Guard against a degenerate corpus.
        corpus = [c for c in corpus if c and c.strip()]
        if len(corpus) < 2:
            corpus = (corpus or ["maritime warning"]) + ["maritime warning hazard"]
        self._tfidf.fit(corpus)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) matrix of L2-normalised embeddings."""
        if self.model is not None:  # pragma: no cover
            vecs = np.asarray(self.model.encode(list(texts), show_progress_bar=False))
        else:
            if self._tfidf is None:
                self.fit_tfidf(list(texts))
            vecs = self._tfidf.transform(list(texts)).toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# ---------------------------------------------------------------------------
# Spatial similarity helpers.
# ---------------------------------------------------------------------------
def _project_nm(coords: Sequence[LatLon], origin: LatLon) -> List[Tuple[float, float]]:
    """Local equirectangular projection to nautical miles about ``origin``."""
    lat0 = math.radians(origin[0])
    out = []
    for lat, lon in coords:
        x = (lon - origin[1]) * 60.0 * math.cos(lat0)  # 1° lon ≈ 60·cos(lat) nm
        y = (lat - origin[0]) * 60.0                    # 1° lat ≈ 60 nm
        out.append((x, y))
    return out


def _shapely_geom(rec: MSIRecord, origin: LatLon):
    """Build a projected shapely geometry (nm units) for a record, or None."""
    if not _HAS_SHAPELY or not rec.coordinates:
        return None
    pts = _project_nm(rec.coordinates, origin)
    kind = rec.geometry_kind
    try:
        if kind == GeometryKind.POLYGON and len(pts) >= 3:
            g = Polygon(pts)
            return g if g.is_valid else g.buffer(0)
        if kind == GeometryKind.LINE and len(pts) >= 2:
            buf = max(rec.spatial_uncertainty_nm, 1.0)
            return LineString(pts).buffer(buf)
        # point / multipoint: buffer by uncertainty (default 1 nm)
        buf = max(rec.spatial_uncertainty_nm, 1.0)
        return Point(pts[0]).buffer(buf)
    except Exception:
        return None


def spatial_similarity(a: MSIRecord, b: MSIRecord, sigma_p: float) -> float:
    """Spatial similarity in [0, 1]."""
    if a.centroid is None or b.centroid is None:
        return 0.0
    # Polygon/polygon → geodesic-projected Jaccard.
    if (
        _HAS_SHAPELY
        and a.geometry_kind == GeometryKind.POLYGON
        and b.geometry_kind == GeometryKind.POLYGON
    ):
        origin = a.centroid
        ga = _shapely_geom(a, origin)
        gb = _shapely_geom(b, origin)
        if ga is not None and gb is not None and ga.area > 0 and gb.area > 0:
            inter = ga.intersection(gb).area
            union = ga.union(gb).area
            if union > 0:
                return float(inter / union)
    # Otherwise → exponential centroid-distance kernel.
    d = haversine_nm(a.centroid, b.centroid)
    return math.exp(-d / sigma_p)


# ---------------------------------------------------------------------------
# Temporal similarity.
# ---------------------------------------------------------------------------
def _validity_interval(rec: MSIRecord) -> Optional[Tuple[float, float]]:
    """Return (start, end) as POSIX-like seconds, filling defaults if needed."""
    start = rec.valid_start or rec.issue_time
    if start is None:
        return None
    end = rec.valid_end
    if end is None:
        dur_h = _DEFAULT_DURATION_H.get(rec.event_type, 24.0)
        end = start + timedelta(hours=dur_h)
    return start.timestamp(), end.timestamp()


def _succession_factor(a: MSIRecord, b: MSIRecord, tau_succ_h: float) -> float:
    """Lifecycle-succession kernel exp(−gap_adj/τ_succ) in [0, 1] (Eq. (9)).

    ``gap_adj = min(|start_B − end_A|, |start_A − end_B|)`` measures how close
    one report's validity window *begins* to where the other's *ends* — the
    signature of a lifecycle handoff (an UPDATE supersedes its predecessor at
    the predecessor's cancellation, so a true chain pair has gap_adj ≈ 0 by
    archive construction). Recurring but distinct events at the same site are
    separated by weeks–months of dead time and score ≈ 0. Interval overlap is
    deliberately NOT credited here: near-identical windows (same-event reports)
    have gap_adj ≈ the full duration and must earn their temporal affinity from
    overlap/proximity instead — succession is a *complementary* relation.
    """
    ia, ib = _validity_interval(a), _validity_interval(b)
    if not ia or not ib:
        return 0.0
    gap_adj_h = min(abs(ib[0] - ia[1]), abs(ia[0] - ib[1])) / 3600.0
    return math.exp(-gap_adj_h / tau_succ_h)


def temporal_similarity(
    a: MSIRecord, b: MSIRecord, tau_h: float = 48.0, mu1: float = 0.6, mu2: float = 0.4,
    tau_succ_h: Optional[float] = None,
) -> float:
    """Temporal similarity in [0, 1] (overlap, proximity, optionally Eq. (9)).

    Graded overlap/proximity blend; when ``tau_succ_h`` is given, a lifecycle
    succession pair (adjacent validity windows) is scored by the succession
    kernel instead when that is higher — chains stop being suppressed by the
    overlap term's blindness to handoffs (Section 3.4).
    """
    ia, ib = _validity_interval(a), _validity_interval(b)
    overlap = 0.0
    if ia and ib:
        inter = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
        union = max(ia[1], ib[1]) - min(ia[0], ib[0])
        overlap = inter / union if union > 0 else 0.0
    prox = 0.0
    if a.issue_time and b.issue_time:
        dt_h = abs((a.issue_time - b.issue_time).total_seconds()) / 3600.0
        prox = math.exp(-dt_h / tau_h)
    graded = mu1 * overlap + mu2 * prox
    if tau_succ_h is not None:
        return max(graded, _succession_factor(a, b, tau_succ_h))
    return graded


def temporal_compatibility(
    a: MSIRecord, b: MSIRecord, tau_prox_h: float = 48.0, tau_succ_h: float = 72.0
) -> float:
    """Temporal-compatibility gate g_t ∈ [0, 1] (Section 3.4).

    ``g_t = max(J_interval, exp(−Δissue/τ_prox), exp(−gap_adj/τ_succ))`` — two
    reports are temporally compatible when their validity windows agree
    (same-event corroboration), their issuance is close (co-observation), or
    one window begins where the other ends (lifecycle succession). Reports of
    *distinct* recurrences at the same site fail all three and gate to ≈ 0, so
    spatial/semantic affinity alone can no longer merge them — the failure mode
    that let months-apart co-located events fuse under the additive composite.
    """
    ia, ib = _validity_interval(a), _validity_interval(b)
    overlap = 0.0
    if ia and ib:
        inter = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
        union = max(ia[1], ib[1]) - min(ia[0], ib[0])
        overlap = inter / union if union > 0 else 0.0
    prox = 0.0
    if a.issue_time and b.issue_time:
        dt_h = abs((a.issue_time - b.issue_time).total_seconds()) / 3600.0
        prox = math.exp(-dt_h / tau_prox_h)
    return max(overlap, prox, _succession_factor(a, b, tau_succ_h))


# ---------------------------------------------------------------------------
# Attribute similarity.
# ---------------------------------------------------------------------------
def attribute_similarity(a: MSIRecord, b: MSIRecord) -> Optional[float]:
    """Mean per-attribute agreement over shared structured attributes.

    Returns ``None`` when the two records share no structured attributes (the
    caller then renormalises over type + text only).
    """
    common = set(a.structured_data) & set(b.structured_data)
    if not common:
        return None
    scores = []
    for key in common:
        va, vb = a.structured_data[key], b.structured_data[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            denom = max(abs(va), abs(vb), 1e-9)
            scores.append(max(0.0, 1.0 - abs(va - vb) / denom))
        else:
            scores.append(1.0 if str(va).lower() == str(vb).lower() else 0.0)
    return float(np.mean(scores))


@dataclass
class SimilarityModel:
    """Holds weights/scales and a fitted text encoder; computes Sim(·,·).

    ``temporal_mode`` selects how time enters the composite (Section 3.4):

    * ``"gate"`` — Sim = [(ωs·Sims + ωm·Simm)/(ωs+ωm)] · g_t, with the
      temporal-compatibility gate g_t = max(overlap, issue-proximity,
      succession). Temporal evidence acts as a *constraint* on event identity:
      pairs that are neither co-present, co-observed, nor lifecycle-successive
      cannot be merged by spatial/semantic affinity alone.
    * ``"additive"`` — the original weighted vote ωs·Sims + ωt·Simt + ωm·Simm
      (baselines are evaluated in this, their published, form).
    """

    encoder: TextEncoder
    sigma_p: float = 5.0
    w_spatial: float = 0.40
    w_temporal: float = 0.25
    w_semantic: float = 0.35
    nu_type: float = 0.35
    nu_attr: float = 0.35
    nu_text: float = 0.30
    temporal_mode: str = "additive"  # "gate" | "additive"
    tau_prox_h: float = 48.0
    tau_succ_h: Optional[float] = None  # None → succession kernel off
    mu1: float = 0.6
    mu2: float = 0.4

    def __post_init__(self) -> None:
        self._text_cache: Dict[str, np.ndarray] = {}

    def prepare(self, records: Sequence[MSIRecord]) -> None:
        """Pre-compute and cache text embeddings for a record set."""
        texts = [r.raw_text or "" for r in records]
        vecs = self.encoder.encode(texts)
        for rec, v in zip(records, vecs):
            self._text_cache[rec.record_id] = v

    def _text_sim(self, a: MSIRecord, b: MSIRecord) -> float:
        va = self._text_cache.get(a.record_id)
        vb = self._text_cache.get(b.record_id)
        if va is None or vb is None:
            return 0.0
        return float(max(0.0, np.dot(va, vb)))

    def semantic_similarity(self, a: MSIRecord, b: MSIRecord) -> float:
        s_type = type_similarity(a.event_type, b.event_type)
        s_text = self._text_sim(a, b)
        s_attr = attribute_similarity(a, b)
        if s_attr is None:  # renormalised fallback (no shared attributes)
            denom = self.nu_type + self.nu_text
            return (self.nu_type * s_type + self.nu_text * s_text) / denom
        return self.nu_type * s_type + self.nu_attr * s_attr + self.nu_text * s_text

    def sim(self, a: MSIRecord, b: MSIRecord) -> float:
        s = spatial_similarity(a, b, self.sigma_p)
        m = self.semantic_similarity(a, b)
        if self.temporal_mode == "gate":
            g = temporal_compatibility(a, b, self.tau_prox_h, self.tau_succ_h or 72.0)
            wv = self.w_spatial + self.w_semantic
            if wv <= 0:
                return g
            return ((self.w_spatial * s + self.w_semantic * m) / wv) * g
        t = temporal_similarity(a, b, self.tau_prox_h, self.mu1, self.mu2,
                                tau_succ_h=self.tau_succ_h)
        return self.w_spatial * s + self.w_temporal * t + self.w_semantic * m

    def pairwise_matrix(self, records: Sequence[MSIRecord]) -> np.ndarray:
        """Symmetric (n, n) composite-similarity matrix (diagonal = 1).

        Exact per-pair computation (uses polygon Jaccard where applicable).
        Suitable for small/medium sets; for large sets use
        :meth:`pairwise_matrix_fast`.
        """
        self.prepare(records)
        n = len(records)
        S = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                v = self.sim(records[i], records[j])
                S[i, j] = S[j, i] = v
        return S

    def pairwise_matrix_fast(self, records: Sequence[MSIRecord]) -> np.ndarray:
        """Vectorised (n, n) composite-similarity matrix for large sets.

        Approximations vs. :meth:`pairwise_matrix` (documented): spatial uses the
        centroid-distance kernel for *all* geometry types (no polygon Jaccard in
        the matrix), and the semantic term uses the no-shared-attribute
        renormalised form (Simtype + Simtext). Polygon Jaccard and attribute
        agreement are still applied exactly where it matters (conflict detection
        on small clusters). The clustering and neighbour structures are
        insensitive to these approximations in practice.
        """
        self.prepare(records)
        n = len(records)
        if n == 0:
            return np.zeros((0, 0))

        # --- text cosine via matrix product ---
        V = np.nan_to_num(np.vstack([self._text_cache[r.record_id] for r in records]))
        with np.errstate(all="ignore"):
            text = np.clip(np.nan_to_num(V @ V.T), 0.0, 1.0)

        # --- spatial: vectorised haversine on centroids ---
        R_nm = 6371008.8 / 1852.0
        lat = np.radians([(r.centroid[0] if r.centroid else 0.0) for r in records])
        lon = np.radians([(r.centroid[1] if r.centroid else 0.0) for r in records])
        has_c = np.array([r.centroid is not None for r in records])
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
        dist = 2 * R_nm * np.arcsin(np.clip(np.sqrt(a), 0, 1))
        spatial = np.exp(-dist / self.sigma_p)
        mask = has_c[:, None] & has_c[None, :]
        spatial = np.where(mask, spatial, 0.0)

        # --- temporal: interval Jaccard + issuance proximity (+ succession) ---
        starts, ends, issue = [], [], []
        for r in records:
            iv = _validity_interval(r)
            starts.append(iv[0] if iv else np.nan)
            ends.append(iv[1] if iv else np.nan)
            issue.append(r.issue_time.timestamp() if r.issue_time else np.nan)
        s = np.array(starts); e = np.array(ends); it = np.array(issue)
        inter = np.minimum(e[:, None], e[None, :]) - np.maximum(s[:, None], s[None, :])
        inter = np.clip(inter, 0, None)
        union = np.maximum(e[:, None], e[None, :]) - np.minimum(s[:, None], s[None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            overlap = np.where(union > 0, inter / union, 0.0)
        overlap = np.nan_to_num(overlap)
        dt_h = np.abs(it[:, None] - it[None, :]) / 3600.0
        with np.errstate(invalid="ignore", over="ignore"):
            prox = np.nan_to_num(np.exp(-dt_h / self.tau_prox_h))
        succ = None
        if self.tau_succ_h is not None or self.temporal_mode == "gate":
            tau_succ = self.tau_succ_h or 72.0
            # gap_adj(i, j) = min(|start_i − end_j|, |start_j − end_i|) — Eq. (9)
            adj = np.abs(s[:, None] - e[None, :]) / 3600.0
            gap_adj = np.minimum(adj, adj.T)
            with np.errstate(invalid="ignore", over="ignore"):
                succ = np.nan_to_num(np.exp(-gap_adj / tau_succ))
        temporal = self.mu1 * overlap + self.mu2 * prox
        if succ is not None and self.temporal_mode != "gate":
            temporal = np.maximum(temporal, succ)

        # --- semantic: type (lookup) + attribute agreement + text ---
        types = list(EventType)
        tindex = {t: i for i, t in enumerate(types)}
        T = np.array([[type_similarity(a_, b_) for b_ in types] for a_ in types])
        ids = np.array([tindex[r.event_type] for r in records])
        type_sim = T[np.ix_(ids, ids)]

        # Attribute agreement on the dominant categorical attribute (hazard_type
        # / phenomenon_type / attack_type / incident_type), vectorised. Pairs
        # sharing such an attribute use the full Simtype+Simattr+Simtext form;
        # pairs that do not fall back to the renormalised Simtype+Simtext form
        # (the |Acommon| = 0 case).
        _ATTR_KEYS = ("hazard_type", "phenomenon_type", "attack_type", "incident_type")

        def _attr_val(rec):
            for k in _ATTR_KEYS:
                if k in rec.structured_data:
                    return str(rec.structured_data[k]).lower()
            return None

        avals = [_attr_val(r) for r in records]
        has = np.array([v is not None for v in avals])
        codes = {}
        cidx = np.array([codes.setdefault(v, len(codes)) if v is not None else -1
                         for v in avals])
        both = has[:, None] & has[None, :]
        attr_match = ((cidx[:, None] == cidx[None, :]) & both).astype(float)

        full = self.nu_type * type_sim + self.nu_attr * attr_match + self.nu_text * text
        denom = self.nu_type + self.nu_text
        fallback = (self.nu_type * type_sim + self.nu_text * text) / denom
        semantic = np.where(both, full, fallback)

        if self.temporal_mode == "gate":
            g = np.maximum(np.maximum(overlap, prox), succ)
            wv = self.w_spatial + self.w_semantic
            if wv <= 0:
                S = g
            else:
                S = ((self.w_spatial * spatial + self.w_semantic * semantic) / wv) * g
        else:
            S = (self.w_spatial * spatial + self.w_temporal * temporal
                 + self.w_semantic * semantic)
        np.fill_diagonal(S, 1.0)
        return S
