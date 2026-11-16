"""Experiment E5 — Sensitivity analysis (Section 5.5).

Tier-1: dimension-weight profiles (5a, doubling as the scenario-adaptive
profiles) and the temporal-scale multiplier (5b), reported as primary AUC-ROC.
Tier-2: σp point-similarity scale (5c) and similarity weights (5f) as clustering
ARI, historical decay λ (5d) as primary AUC, and MinPts (5e) as ARI/noise. All
clustering sweeps calibrate ε on the dev split and report ARI on the test split
(Section 4.5); credibility sweeps report on the test split.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..clustering import STSDBSCAN, adaptive_epsilon
from ..config import Config, get_config
from ..generation.benchmark import Benchmark
from ..schema import Category
from ..similarity import SimilarityModel, TextEncoder
from .. import stats
from . import common
from .e1_credibility import _scores_for, _subset

# (αd, αc, αst, αso) profiles of the sensitivity sweep (Section 5.5).
_WEIGHT_PROFILES = {
    "Default": (0.25, 0.30, 0.20, 0.25),
    "Content-heavy": (0.15, 0.45, 0.15, 0.25),
    "Social-heavy": (0.15, 0.25, 0.15, 0.45),
    "Spatiotemporal-heavy": (0.15, 0.25, 0.40, 0.20),
    "Equal": (0.25, 0.25, 0.25, 0.25),
}

# (ωs, ωt, ωm) similarity-weight profiles for 5f.
_SIM_WEIGHT_PROFILES = {
    "Default": (0.40, 0.25, 0.35),
    "Spatial-dominant": (0.70, 0.15, 0.15),
    "Semantic-dominant": (0.20, 0.15, 0.65),
    "Equal": (1 / 3, 1 / 3, 1 / 3),
}

# Scenario-adaptive matched threats (Section 5.5): each "heavy" profile is paired
# with the attack type it is meant to counter, so we can check whether
# emphasising a dimension helps on-target (vs Default) — the off-target cost is
# read from each profile's overall AUC in 5a.
_SCENARIO_MATCH = {
    "Content-heavy": ("ATTACK_FABRICATION", "content-forgery (T1)"),
    "Social-heavy": ("ATTACK_SYBIL", "Sybil (T5)"),
    "Spatiotemporal-heavy": ("ATTACK_REPLAY", "replay (T2)"),
}


def _scenario_adaptive(profile_obj: Dict, test: Sequence) -> Dict:
    """On-target AUC of each heavy profile vs Default on its matched-threat
    subset (genuine-primary vs the matched attack)."""
    out: Dict = {}
    default = profile_obj["Default"]
    genuine = [r for r in test if r.gt_category and r.gt_category.name.startswith("GENUINE")]
    for prof, (atk_name, desc) in _SCENARIO_MATCH.items():
        if prof not in profile_obj:
            continue
        atk = Category[atk_name]
        sub = genuine + [r for r in test if r.gt_category == atk]
        y = np.array([1 if r.gt_category.name.startswith("GENUINE") else 0 for r in sub])
        if len(np.unique(y)) < 2:
            continue
        s_prof = np.array([profile_obj[prof][r.record_id].cr for r in sub])
        s_def = np.array([default[r.record_id].cr for r in sub])
        out[prof] = {
            "matched_threat": desc,
            "auc_on_target_profile": round(stats.roc_auc(y, s_prof), 4),
            "auc_on_target_default": round(stats.roc_auc(y, s_def), 4),
        }
    return out


def _gen_split_indices(records) -> Tuple[List[int], List[int]]:
    """Indices of genuine, event-bearing reports in the dev / test splits."""
    dev, test = [], []
    for i, r in enumerate(records):
        if r.gt_event_id is None:
            continue
        (dev if r.provenance.get("split") == "dev" else test).append(i)
    return dev, test


def _ari_dev_calibrated(cfg, records, sim, dev_idx, test_idx, min_pts) -> float:
    """Calibrate ε on the dev genuine subset, cluster + score on the test subset."""
    if not dev_idx or not test_idx:
        return float("nan")
    eps_dev, _, _ = adaptive_epsilon(
        sim[np.ix_(dev_idx, dev_idx)], min_pts,
        cfg.get("clustering", "eps_fallback", default=0.45))
    test_recs = [records[i] for i in test_idx]
    test_sim = sim[np.ix_(test_idx, test_idx)]
    labels = STSDBSCAN(min_pts=min_pts, eps=eps_dev).fit(test_recs, test_sim)
    gt = [records[i].gt_event_id for i in test_idx]
    return common.cluster_metrics(gt, labels)["ARI"]


def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    test = [r for r in bench.records if r.provenance.get("split") == "test"]
    records = bench.records
    dev_idx, test_idx = _gen_split_indices(records)
    min_pts = cfg.get("clustering", "min_pts", default=2)
    results = {"dimension_weights_auc_primary": {}, "sigma_t_multiplier_auc_primary": {},
               "sigma_p_clustering_ARI": {}, "history_lambda_auc_primary": {},
               "minpts_clustering": {}, "sim_weights_clustering_ARI": {}}

    # --- 5a dimension-weight profiles (expert presets) ---
    dev = [r for r in records if r.provenance.get("split") == "dev"]
    test_p, dev_p = _subset(test, False), _subset(dev, False)
    profile_obj: Dict[str, Dict] = {}  # name -> {rid: DimensionScores}
    for name, (wd, wc, wst, wso) in _WEIGHT_PROFILES.items():
        params = common.mcam_params_from_cfg(
            cfg, w_device=wd, w_content=wc, w_spatiotemporal=wst, w_social=wso)
        obj = common.assess_benchmark(cfg, bench, params=params)[3]
        profile_obj[name] = obj
        smap = {rid: {"cr": s.cr} for rid, s in obj.items()}
        y, s = _scores_for(test_p, smap, "cr")
        results["dimension_weights_auc_primary"][name] = round(stats.roc_auc(y, s), 4)
        print(f"[E5] weights {name:<22} AUC={results['dimension_weights_auc_primary'][name]}")

    # --- 5a+ data-driven weighting (Section 3.3): learned on dev, evaluated on test ---
    learned, dd_info = common.data_driven_weighting(profile_obj["Default"], dev_p, test_p)
    y_p = np.array([1 if r.gt_category and r.gt_category.label.value == "CREDIBLE" else 0
                    for r in test_p])
    s_dd = np.array([learned[r.record_id] for r in test_p])
    results["dimension_weights_auc_primary"]["Data-driven"] = round(stats.roc_auc(y_p, s_dd), 4)
    results["data_driven_weights_relative"] = dd_info["weights_relative"]
    print(f"[E5] weights {'Data-driven':<22} AUC={results['dimension_weights_auc_primary']['Data-driven']} "
          f"(rel w={dd_info['weights_relative']})")

    # --- 5a++ scenario-adaptive: on-target gain vs off-target cost ---
    results["scenario_adaptive"] = _scenario_adaptive(profile_obj, test)
    print(f"[E5] scenario-adaptive: {results['scenario_adaptive']}")

    # --- 5b temporal-scale multiplier (scales σt via event sigma map) ---
    import msi_bench.credibility.domain as dom
    base = dict(dom.EVENT_SIGMA_T_H)
    try:
        for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
            for k in base:
                dom.EVENT_SIGMA_T_H[k] = base[k] * mult
            smap = {rid: {"cr": s.cr} for rid, s in common.assess_benchmark(cfg, bench)[3].items()}
            y, s = _scores_for(_subset(test, False), smap, "cr")
            results["sigma_t_multiplier_auc_primary"][f"{mult}x"] = round(stats.roc_auc(y, s), 4)
    finally:
        dom.EVENT_SIGMA_T_H.clear()
        dom.EVENT_SIGMA_T_H.update(base)  # always restore the module global

    # --- 5d historical-decay λ → primary AUC ---
    for lam in (30, 60, 90, 180, 365):
        params = common.mcam_params_from_cfg(cfg, lambda_history_days=lam)
        smap = {rid: {"cr": s.cr} for rid, s in
                common.assess_benchmark(cfg, bench, params=params)[3].items()}
        y, s = _scores_for(_subset(test, False), smap, "cr")
        results["history_lambda_auc_primary"][f"{lam}d"] = round(stats.roc_auc(y, s), 4)

    # --- 5h+ θ_ndup: near-verbatim duplication threshold (Section 3.3 check (d)) ---
    # Tier-3 parameter of the Sybil defence. Reported per threshold:
    # primary AUC, genuine-vs-Sybil AUC, Sybil detection and FAR at the
    # dev-Youden operating point; 1.01 disables the check. The dev-grounded
    # default is 0.75 (genuine-paraphrase p99 0.57 vs within-campaign p1 0.92 —
    # expect a wide plateau).
    from datetime import datetime as _dt
    from .e1_credibility import _youden_threshold
    model = common.make_similarity(cfg)
    sim_full = model.pairwise_matrix_fast(records)
    now = _dt.fromisoformat(bench.meta["now"]) if bench.meta.get("now") else None
    results["theta_ndup_sweep"] = {}
    gen_test = [r for r in test if r.gt_category and r.gt_category.label.value == "CREDIBLE"]
    syb_test = [r for r in test if r.gt_category == Category.ATTACK_SYBIL]
    for th in (0.60, 0.70, 0.75, 0.80, 0.90, 1.01):
        params = common.mcam_params_from_cfg(cfg, theta_ndup=th)
        scores = common.assess(cfg, records, bench.histories, sim_full,
                               params=params, now=now)
        smap = {s.record_id: {"cr": s.cr} for s in scores}
        y, s = _scores_for(_subset(test, False), smap, "cr")
        sub = [r for r in test if r.gt_category and (
            r.gt_category.name.startswith("GENUINE") or r.gt_category == Category.ATTACK_SYBIL)]
        ys, ss = _scores_for(sub, smap, "cr")
        yd, sd = _scores_for(_subset(dev, False), smap, "cr")
        tau = _youden_threshold(yd, sd)
        results["theta_ndup_sweep"][f"{th:g}"] = {
            "auc_primary": round(stats.roc_auc(y, s), 4),
            "auc_vs_sybil": round(stats.roc_auc(ys, ss), 4) if len(np.unique(ys)) > 1 else None,
            "sybil_detection": round(float(np.mean(
                [smap[r.record_id]["cr"] < tau for r in syb_test])), 4) if syb_test else None,
            "false_alarm": round(float(np.mean(
                [smap[r.record_id]["cr"] < tau for r in gen_test])), 4) if gen_test else None,
        }
        print(f"[E5] θ_ndup={th:g}: {results['theta_ndup_sweep'][f'{th:g}']}")

    # --- 5c σp point-similarity scale → clustering ARI (dev-calibrated ε, test) ---
    for sigma_p in (2, 3, 5, 8, 12):
        model = common.make_similarity(cfg, sigma_p=float(sigma_p))
        sim = model.pairwise_matrix_fast(records)
        results["sigma_p_clustering_ARI"][f"{sigma_p}nm"] = round(
            _ari_dev_calibrated(cfg, records, sim, dev_idx, test_idx, min_pts), 4)

    # --- 5f similarity-weight profiles → clustering ARI (dev-calibrated ε, test) ---
    for name, (ws, wt, wm) in _SIM_WEIGHT_PROFILES.items():
        model = common.make_similarity(
            cfg, weights={"spatial": ws, "temporal": wt, "semantic": wm})
        sim = model.pairwise_matrix_fast(records)
        results["sim_weights_clustering_ARI"][name] = round(
            _ari_dev_calibrated(cfg, records, sim, dev_idx, test_idx, min_pts), 4)

    # --- 5e MinPts → clustering ARI + noise rate (dev-calibrated ε, test) ---
    model = common.make_similarity(cfg)
    sim = model.pairwise_matrix_fast(records)
    for mp in (1, 2, 3, 4, 5):
        if not dev_idx or not test_idx:
            continue
        eps_dev, _, _ = adaptive_epsilon(sim[np.ix_(dev_idx, dev_idx)], mp,
                                         cfg.get("clustering", "eps_fallback", default=0.45))
        test_recs = [records[i] for i in test_idx]
        labels = STSDBSCAN(min_pts=mp, eps=eps_dev).fit(test_recs, sim[np.ix_(test_idx, test_idx)])
        gt = [records[i].gt_event_id for i in test_idx]
        m = common.cluster_metrics(gt, labels)
        results["minpts_clustering"][f"minpts={mp}"] = {"ARI": m["ARI"], "noise_rate": m["noise_rate"]}

    print(f"[E5] σt sweep: {results['sigma_t_multiplier_auc_primary']}")
    print(f"[E5] σp sweep (ARI): {results['sigma_p_clustering_ARI']}")
    print(f"[E5] λ sweep (AUC): {results['history_lambda_auc_primary']}")
    print(f"[E5] sim-weights (ARI): {results['sim_weights_clustering_ARI']}")
    common.write_results(cfg, "e5", results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E5 (sensitivity).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
