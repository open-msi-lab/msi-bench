"""Shared helpers for the experiment runners (E1–E7).

Constructs the similarity model, navigability model, and MCAM from a record set;
provides clustering-baseline runners and label/metric utilities; and a small
JSON-writer that records results under ``<output_dir>/<experiment>/``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..clustering import NOISE, STSDBSCAN
from ..config import Config
from ..credibility.domain import NavigabilityModel
from ..credibility.mcam import MCAM, MCAMParams
from ..schema import MSIRecord
from ..similarity import SimilarityModel, TextEncoder


# Process-level caches so the real-archive corpus / envelope are parsed once.
_EMB_CORPUS: Optional[List[str]] = None
_NAV_CENTROIDS: Optional[List[Tuple[float, float]]] = None


def embedding_corpus(cfg: Config) -> List[str]:
    """Official-warning text from the 2012–2021 partition (split discipline).

    The TF-IDF / embedding vocabulary is fitted on this training corpus only,
    never on the evaluation records, and lifecycle-reference lines are masked.
    Cached to ``cache_dir`` for reuse across experiments.
    """
    global _EMB_CORPUS
    if _EMB_CORPUS is not None:
        return _EMB_CORPUS
    cache = cfg.cache_dir / "embedding_corpus.json"
    if cache.is_file():
        _EMB_CORPUS = json.loads(cache.read_text("utf-8"))
        return _EMB_CORPUS
    from ..parsing.pipeline import parse_corpus
    from ..schema import MessageFunction
    from .benchmark_mask import mask_lifecycle
    msgs = parse_corpus(cfg, cfg.years("embedding"), nav_area=cfg.nav_area,
                        functions=[MessageFunction.NEW_WARNING, MessageFunction.UPDATE])
    _EMB_CORPUS = [mask_lifecycle(m.raw_text) for m in msgs if (m.raw_text or "").strip()]
    cache.write_text(json.dumps(_EMB_CORPUS), "utf-8")
    return _EMB_CORPUS


def real_operational_centroids(cfg: Config) -> List[Tuple[float, float]]:
    """Centroids of real NAVAREA IV discrete events (domain prior).

    Used to build the navigable-water envelope. This is the *real* archive
    geography — independent of the synthetic genuine/attack labels — so the
    spatial-descriptive check does not see evaluation labels.
    """
    global _NAV_CENTROIDS
    if _NAV_CENTROIDS is not None:
        return _NAV_CENTROIDS
    cache = cfg.cache_dir / "operational_centroids.json"
    if cache.is_file():
        _NAV_CENTROIDS = [tuple(c) for c in json.loads(cache.read_text("utf-8"))]
        return _NAV_CENTROIDS
    from ..parsing.pipeline import parse_corpus
    from ..schema import MessageFunction
    years = list(cfg.years("embedding")) + list(cfg.years("benchmark"))
    msgs = parse_corpus(cfg, years, nav_area=cfg.nav_area,
                        functions=[MessageFunction.NEW_WARNING, MessageFunction.UPDATE])
    _NAV_CENTROIDS = [m.centroid for m in msgs if m.centroid]
    cache.write_text(json.dumps(_NAV_CENTROIDS), "utf-8")
    return _NAV_CENTROIDS


_TAU_FROM_CFG = object()  # sentinel: read tau_succ_h from config


def make_similarity(cfg: Config, sigma_p: float = 5.0, weights=None,
                    temporal_mode: Optional[str] = None,
                    tau_succ_h=_TAU_FROM_CFG) -> SimilarityModel:
    """A SimilarityModel whose encoder is fitted on the 2012–2021 corpus.

    ``temporal_mode`` overrides ``clustering.temporal_mode`` — clustering
    baseline variants pass ``"additive"`` (with ``tau_succ_h=None``) so they
    are evaluated in their published form, while the proposed composite uses
    the succession-aware temporal-compatibility gate. τ_succ
    defaults to 72 h, grounded in the 2012–2021 chain-handoff gap distribution
    (adjacency gap p50–p99 = 0 h, 75% within 72 h) against recurrence dead
    times (p50 ≈ 55 d).
    """
    enc = TextEncoder(model_name=cfg.get("embedding", "model"),
                      device=cfg.get("embedding", "device", default="cpu"))
    enc.fit_tfidf(embedding_corpus(cfg))
    sw = weights or cfg.raw["clustering"]["sim_weights"]
    if tau_succ_h is _TAU_FROM_CFG:
        tau_succ_h = cfg.get("clustering", "tau_succ_h", default=72.0)
        tau_succ_h = float(tau_succ_h) if tau_succ_h is not None else None
    return SimilarityModel(
        encoder=enc, sigma_p=sigma_p,
        w_spatial=sw["spatial"], w_temporal=sw["temporal"], w_semantic=sw["semantic"],
        temporal_mode=temporal_mode or cfg.get("clustering", "temporal_mode",
                                               default="gate"),
        tau_prox_h=float(cfg.get("clustering", "tau_prox_h", default=48.0)),
        tau_succ_h=tau_succ_h,
    )


def build_similarity(cfg: Config, records: Sequence[MSIRecord]) -> SimilarityModel:
    return make_similarity(cfg)


def build_navigability(cfg: Config) -> NavigabilityModel:
    cents = real_operational_centroids(cfg)
    return NavigabilityModel(centroids=cents or [(28.0, -80.0)])


def mcam_params_from_cfg(cfg: Config, **overrides) -> MCAMParams:
    dw = cfg.raw["mcam"]["dim_weights"]
    p = MCAMParams(
        w_device=dw["device"], w_content=dw["content"],
        w_spatiotemporal=dw["spatiotemporal"], w_social=dw["social"],
        delta_ind_nm=cfg.get("mcam", "delta_ind_nm", default=0.5),
        delta_disp_nm=cfg.get("mcam", "delta_disp_nm", default=2.0),
        theta_ndup=cfg.get("mcam", "theta_ndup", default=0.75),
        lambda_history_days=cfg.get("mcam", "lambda_history_days", default=90),
        n_min_history=cfg.get("mcam", "n_min_history", default=10),
        assessment_mode=cfg.get("mcam", "assessment_mode", default="submission"),
        assessment_lag_h=cfg.get("mcam", "assessment_lag_h", default=1.0),
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def assess(
    cfg: Config,
    records: Sequence[MSIRecord],
    histories: Dict[str, List[Tuple[datetime, str]]],
    sim: np.ndarray,
    params: Optional[MCAMParams] = None,
    now: Optional[datetime] = None,
) -> List:
    nav = build_navigability(cfg)
    mcam = MCAM(params or mcam_params_from_cfg(cfg), navigability=nav,
               histories=histories, now=now)
    return mcam.assess(records, sim)


# --- clustering helpers ----------------------------------------------------
def relabel_noise(labels: np.ndarray) -> np.ndarray:
    """Map each noise point (-1) to a unique singleton label."""
    out = labels.copy()
    nxt = int(out.max()) + 1 if len(out) and out.max() >= 0 else 0
    for i in range(len(out)):
        if out[i] == NOISE:
            out[i] = nxt
            nxt += 1
    return out


def cluster_metrics(gt: Sequence, pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 v_measure_score)
    pred = relabel_noise(np.asarray(pred))
    gt_codes = _codes(gt)
    return {
        "ARI": round(float(adjusted_rand_score(gt_codes, pred)), 4),
        "NMI": round(float(normalized_mutual_info_score(gt_codes, pred)), 4),
        "V": round(float(v_measure_score(gt_codes, pred)), 4),
        "n_clusters": int(len(set(pred))),
        "noise_rate": round(float(np.mean(np.asarray(pred) < 0)), 4) if (np.asarray(pred) < 0).any() else 0.0,
    }


def _codes(labels: Sequence) -> np.ndarray:
    mapping: Dict = {}
    out = []
    for x in labels:
        out.append(mapping.setdefault(x, len(mapping)))
    return np.asarray(out)


def dbscan_on_sim(sim: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Run DBSCAN on a precomputed distance = 1 - sim, threshold 1 - eps."""
    from sklearn.cluster import DBSCAN
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    return DBSCAN(eps=1.0 - eps, min_samples=min_samples, metric="precomputed").fit_predict(dist)


def dev_calibrated_eps(records: Sequence[MSIRecord], sim: np.ndarray,
                       min_pts: int = 2, eps_fallback: float = 0.45):
    """Adaptive ε calibrated on the dev-genuine submatrix (split discipline).

    Mirrors E3's split discipline for the pipeline experiments (E4/E6): the
    knee is located on the similarity submatrix of dev-split genuine records
    only, never on the evaluation records. Returns ``(eps, used_fallback)``.
    """
    from ..clustering import adaptive_epsilon
    dev_idx = [i for i, r in enumerate(records)
               if r.gt_event_id is not None and r.provenance.get("split") == "dev"]
    if not dev_idx:
        return eps_fallback, True
    eps, _, fb = adaptive_epsilon(sim[np.ix_(dev_idx, dev_idx)], min_pts, eps_fallback)
    return eps, fb


# --- data-driven dimension weighting (Section 3.3) --------------------------------
_DD_DIMS = ("cd", "cc", "cst", "cso")


def _dd_label(rec) -> int:
    return 1 if rec.gt_category and rec.gt_category.label.value == "CREDIBLE" else 0


def data_driven_weighting(scores_map, dev_records, eval_records):
    """Data-driven weighting (Section 3.3): a standardized logistic regression over the
    four MCAM dimension scores (Cd, Cc, Cst, Cso), fitted on ``dev_records`` and
    applied to ``eval_records``.

    This is the "labels available" weighting method offered alongside
    the expert default; the optimizer is free to down-weight near-chance
    dimensions instead of the fixed a-priori vector. Event-level separation
    (no leakage) is the caller's responsibility — pass disjoint dev/eval event
    splits.

    Returns ``(eval_scores, info)`` where ``eval_scores`` maps record_id → learned
    credibility ∈ [0, 1] and ``info`` holds the relative dimension weights and raw
    coefficients (for reporting).
    """
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def _feat(recs):
        X = np.array([[float(getattr(scores_map[r.record_id], d)) for d in _DD_DIMS]
                      for r in recs], dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    Xd = _feat(dev_records)
    yd = np.array([_dd_label(r) for r in dev_records])
    # The device dimension nearly separates the classes, pushing the logistic
    # weights toward divergence — harmless overflow warnings in the solver's
    # intermediate matmuls, but the fit still converges to a stable optimum.
    # Standardise and silence those cosmetic warnings so publication runs are clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            scaler = StandardScaler().fit(Xd)
            lr = LogisticRegression(max_iter=5000, C=1.0).fit(scaler.transform(Xd), yd)
            pe = lr.predict_proba(scaler.transform(_feat(eval_records)))[:, 1]
    eval_scores = {r.record_id: float(p) for r, p in zip(eval_records, pe)}
    coef = lr.coef_[0]
    pos = np.clip(coef, 0.0, None)
    rel = pos / pos.sum() if pos.sum() > 0 else np.abs(coef) / max(np.abs(coef).sum(), 1e-9)
    info = {"weights_relative": {d: round(float(w), 3) for d, w in zip(_DD_DIMS, rel)},
            "raw_coef": [round(float(c), 3) for c in coef]}
    return eval_scores, info


# --- shared build specs for the regeneration-dependent sub-experiments -------
# E1's multi-seed variance study (Section 4.5) and E6's 2024 holdout (Section 5.4) both
# REBUILD the synthetic overlay (at new seeds / on the 2024 data), so they need
# LLM generations the base build never made. The ``generate`` phase warms exactly
# these builds; to guarantee the warmed cache keys match what the experiments
# later rebuild offline, both sides source the build parameters from these
# helpers — there is then no way for the specs to drift apart.
HOLDOUT_EVENT_LIMIT = 120  # 2024 holdout events built/evaluated


def multiseed_seeds(cfg: Config) -> List[int]:
    """The seeds for the Section 4.5 variance study: base seed + the next N-1."""
    n = int(cfg.raw.get("n_pipeline_seeds", 5))
    return [int(cfg.seed) + k for k in range(n)]


def multiseed_build_kwargs(n_events: Optional[int], seed: int,
                           degraded: bool = False) -> Dict:
    """``build_benchmark`` kwargs for one variance-study seed (matches E1).

    ``limit_events=n_events`` reselects the SAME event set as the base build
    (``sorted(events)[:n_events]`` with ``n_events == |events|``), so only the
    seed-scoped salts change — exactly the intent of the variance study.
    ``reserve_multiplier=0``: the variance study never runs the E6 pressure
    sweep, so generating a per-seed attack reserve would be pure LLM waste.
    """
    return {"degraded": degraded, "limit_events": n_events, "seed": seed,
            "reserve_multiplier": 0}


def holdout_build_kwargs(cfg: Config) -> Dict:
    """``build_benchmark`` kwargs for the 2024 temporal holdout (matches E6).

    No reserve: the holdout is scored once at native prevalence (Section 5.4);
    the pressure sweep runs on the primary benchmark only.
    """
    return {"limit_events": HOLDOUT_EVENT_LIMIT, "years": list(cfg.years("holdout")),
            "reserve_multiplier": 0}


def assess_benchmark(cfg: Config, bench, params=None):
    """Build similarity + assess credibility over all benchmark records.

    Returns ``(records, sim_model, sim_matrix, score_map)`` where ``score_map``
    maps record_id → :class:`DimensionScores`.
    """
    records = bench.records
    sim_model = build_similarity(cfg, records)
    sim = sim_model.pairwise_matrix_fast(records)
    now = datetime.fromisoformat(bench.meta["now"]) if bench.meta.get("now") else None
    scores = assess(cfg, records, bench.histories, sim, params=params, now=now)
    return records, sim_model, sim, {s.record_id: s for s in scores}


def run_pipeline(
    cfg: Config, bench, *, use_semantic=True, use_conflict=True,
    bpa_method="blend", uniform_credibility=False, params=None,
    use_independence_discount=None,
):
    """End-to-end pipeline: assess → cluster → conflict → fuse.

    Returns a dict with cluster labels, the per-record score map, and the list
    of fused records (each a ``msi_bench.fusion.FusedRecord``). Pipeline
    ablations are controlled by the keyword flags (Section 5.4, E6b).
    ``use_independence_discount`` overrides ``fusion.independence_discount``
    (Section 3.6; default on — the proposed pipeline); the discount reuses the
    check-(d) θ_ndup so the two stages stay in lockstep.
    """
    from ..clustering import STSDBSCAN
    from ..conflict import detect_conflicts, partition_cluster
    from ..fusion import fuse_subcluster

    records, sim_model, sim, smap = assess_benchmark(cfg, bench, params=params)
    cr = {rid: (0.5 if uniform_credibility else s.cr) for rid, s in smap.items()}

    # ε calibrated on the dev-genuine submatrix (split discipline)
    # for both the proposed clustering and the no-semantic ablation.
    min_pts = cfg.get("clustering", "min_pts", default=2)
    eps_fb = cfg.get("clustering", "eps_fallback", default=0.45)
    eps, _ = dev_calibrated_eps(records, sim, min_pts, eps_fb)
    if use_semantic:
        labels = STSDBSCAN(min_pts=min_pts, eps=eps).fit(records, sim)
    else:
        labels = dbscan_on_sim(sim, eps)

    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab >= 0:
            clusters.setdefault(int(lab), []).append(idx)

    fused = []
    q = cfg.get("fusion", "q_discount", default=0.6)
    discount_on = (cfg.get("fusion", "independence_discount", default=True)
                   if use_independence_discount is None else use_independence_discount)
    theta_nd = (cfg.get("mcam", "theta_ndup", default=0.75) if discount_on else None)
    for cidx, members in enumerate(clusters.values()):
        recs = [records[i] for i in members]
        crs = [cr[records[i].record_id] for i in members]
        if use_conflict and len(recs) > 1:
            conflicts = detect_conflicts(recs, sim_model, crs)
            parts = partition_cluster(len(recs), conflicts)
        else:
            parts = [list(range(len(recs)))]
        sub_fused = []
        for pi, part in enumerate(parts):
            sub = [recs[i] for i in part]
            subcr = [crs[i] for i in part]
            sub_fused.append(fuse_subcluster(
                sub, subcr, q=q, method=bpa_method, record_id=f"fused-{cidx}-{pi}",
                theta_ndup=theta_nd))
        # A partitioned cluster yields parallel records (competing
        # interpretations) cross-referenced to one another (Section 3.6).
        if len(sub_fused) > 1:
            for fr in sub_fused:
                fr.conflict_siblings = [o.record_id for o in sub_fused if o is not fr]
        fused.extend(sub_fused)
    return {"records": records, "labels": labels, "score_map": smap,
            "credibility": cr, "fused": fused, "sim_model": sim_model}


def write_results(cfg: Config, name: str, payload: Dict) -> Path:
    out = cfg.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_results.json"
    path.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    print(f"[{name}] results written to {path}")
    return path
