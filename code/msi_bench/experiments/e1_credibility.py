"""Experiment E1 — MCAM credibility assessment (Section 5.1; Tables 6–8, Figs. 4–5).

Loads a built benchmark and reports: overall discrimination (AUC-ROC/PR with
bootstrap CIs, in primary and secondary positive-class forms, Section 4.5);
the macro (attack-type-balanced) AUC co-headline with per-attack AUC;
per-dimension AUC; per-attack detection at a dev-calibrated
threshold plus the detection/false-alarm cost curve and per-attack operating
points; the quality-tier Spearman ranking and per-generator stratification
controls; credibility-level distribution by category; calibration (Brier +
reliability bins); the negative-control label-shuffle (AUC ≈ 0.5); and the
social-credibility formulation comparison (polarity vs naive vs binary).

Run with::

    python -m msi_bench.experiments.e1_credibility --benchmark results/benchmark
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config, get_config
from ..credibility.mcam import credibility_level
from ..generation.benchmark import Benchmark
from ..schema import Category, MSIRecord
from .. import stats
from . import common

ATTACKS = [c for c in Category if c.name.startswith("ATTACK")]
GENUINE_PRIMARY = (Category.GENUINE_CROWD, Category.GENUINE_SENSOR)


def _scores_for(records, score_map, attr) -> Tuple[np.ndarray, np.ndarray]:
    labels, vals = [], []
    for r in records:
        labels.append(1 if r.gt_category and r.gt_category.label.value == "CREDIBLE" else 0)
        vals.append(score_map[r.record_id][attr])
    return np.array(labels), np.array(vals)


def _groups_for(records):
    """Bootstrap group id per record: event id for genuine, own id for attacks."""
    return [r.gt_event_id if r.gt_event_id is not None else r.record_id for r in records]


def _subset(records, include_official: bool):
    if include_official:
        return [r for r in records if r.gt_category]
    return [r for r in records if r.gt_category and (
        r.gt_category in GENUINE_PRIMARY or r.gt_category.name.startswith("ATTACK"))]


def _assess_all(cfg, bench, params=None):
    records = bench.records
    sim_model = common.build_similarity(cfg, records)
    sim = sim_model.pairwise_matrix_fast(records)
    now = datetime.fromisoformat(bench.meta["now"])
    scores = common.assess(cfg, records, bench.histories, sim, params=params, now=now)
    return {s.record_id: s for s in scores}


def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    print(f"[E1] loaded benchmark: {bench.meta['total_records']} records, "
          f"{bench.meta['category_counts']}")
    smap_obj = _assess_all(cfg, bench)
    smap = {rid: {"cd": s.cd, "cc": s.cc, "cst": s.cst, "cso": s.cso, "cr": s.cr}
            for rid, s in smap_obj.items()}

    dev = [r for r in bench.records if r.provenance.get("split") == "dev"]
    test = [r for r in bench.records if r.provenance.get("split") == "test"]
    bootstrap = cfg.raw["bootstrap_resamples"]

    # --- (1a) overall discrimination, primary & secondary ---
    results: Dict = {"benchmark_meta": bench.meta, "n_dev": len(dev), "n_test": len(test)}
    for form, incl in (("primary", False), ("secondary", True)):
        subset = _subset(test, incl)
        y, s = _scores_for(subset, smap, "cr")
        groups = _groups_for(subset)
        auc, lo, hi = stats.bootstrap_auc_ci_grouped(y, s, groups, bootstrap, seed=cfg.seed)
        results[f"auc_roc_{form}"] = {"auc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                      "bootstrap": "event-level"}
        results[f"auc_pr_{form}"] = round(stats.pr_auc(y, s), 4)

    # --- (1b) per-dimension AUC (primary, test) ---
    results["per_dimension_auc_primary"] = {}
    yt, _ = _scores_for(_subset(test, False), smap, "cr")
    for dim in ("cd", "cc", "cst", "cso", "cr"):
        _, sv = _scores_for(_subset(test, False), smap, dim)
        results["per_dimension_auc_primary"][dim] = round(stats.roc_auc(yt, sv), 4)

    # --- (1b+) data-driven weighting (Section 3.3) vs expert vs device-only ------
    # The headline credibility configuration: with labels available, the four
    # dimension weights are LEARNED on dev (logistic regression), free to
    # down-weight near-chance dimensions the fixed a-priori vector cannot.
    # Reported beside the expert default and the strongest single dimension, with
    # an event-level CI and DeLong tests. Event-level dev/test split = no leakage.
    dev_p, test_p = _subset(dev, False), _subset(test, False)
    learned, dd_info = common.data_driven_weighting(smap_obj, dev_p, test_p)
    y_p, s_expert = _scores_for(test_p, smap, "cr")
    _, s_device = _scores_for(test_p, smap, "cd")
    s_dd = np.array([learned[r.record_id] for r in test_p])
    auc_dd, lo_dd, hi_dd = stats.bootstrap_auc_ci_grouped(
        y_p, s_dd, _groups_for(test_p), bootstrap, seed=cfg.seed)
    _, _, p_vs_dev = stats.delong_test(y_p, s_dd, s_device)
    _, _, p_vs_exp = stats.delong_test(y_p, s_dd, s_expert)
    results["weighting_methods_primary"] = {
        "expert": {"auc": round(stats.roc_auc(y_p, s_expert), 4)},
        "device_only": {"auc": round(stats.roc_auc(y_p, s_device), 4)},
        "data_driven": {"auc": round(auc_dd, 4), "ci95": [round(lo_dd, 4), round(hi_dd, 4)],
                        "weights_relative": dd_info["weights_relative"],
                        "raw_coef": dd_info["raw_coef"],
                        "delong_p_vs_device_only": round(p_vs_dev, 4),
                        "delong_p_vs_expert": round(p_vs_exp, 4)},
    }
    print(f"  weighting (primary AUC): expert={results['weighting_methods_primary']['expert']['auc']} "
          f"device-only={results['weighting_methods_primary']['device_only']['auc']} "
          f"data-driven={results['weighting_methods_primary']['data_driven']['auc']} "
          f"(rel w={dd_info['weights_relative']})")

    # --- (1c) per-attack detection at a DEV-calibrated threshold ---
    yd, sd = _scores_for(_subset(dev, False), smap, "cr")
    tau = _youden_threshold(yd, sd)
    results["operating_threshold_dev"] = round(tau, 4)
    results["per_attack_detection_test"] = {}
    for atk in ATTACKS:
        recs = [r for r in test if r.gt_category == atk]
        if not recs:
            continue
        flagged = np.mean([smap[r.record_id]["cr"] < tau for r in recs])
        results["per_attack_detection_test"][atk.value] = round(float(flagged), 4)
    genuine_test = [r for r in test if r.gt_category and r.gt_category.label.value == "CREDIBLE"]
    if genuine_test:
        far = np.mean([smap[r.record_id]["cr"] < tau for r in genuine_test])
        results["false_alarm_rate_test"] = round(float(far), 4)

    # --- (1c+) detection by adversary identity tier / sophistication --------
    # The adaptive-adversary axis (throwaway vs farmed identities; Sybil
    # low/medium/high): reports how much of each attack's detection survives an
    # attacker whose identity metadata matches the genuine population.
    tier_flags: Dict[str, List[bool]] = {}
    for r in test:
        if not (r.gt_category and r.gt_category.name.startswith("ATTACK")):
            continue
        tier = r.provenance.get("identity_tier") or r.provenance.get("sophistication")
        if tier is None:
            continue
        key = f"{r.gt_category.value}:{tier}"
        tier_flags.setdefault(key, []).append(smap[r.record_id]["cr"] < tau)
    results["per_attack_detection_by_tier_test"] = {
        k: round(float(np.mean(v)), 4) for k, v in sorted(tier_flags.items())}

    # --- (1c-2) per-attack AUC + macro-AUC co-headline -------
    # The claim is BREADTH of coverage; pooled AUC lets the two most
    # numerous attack families dominate the average. Macro-AUC weights each
    # attack family equally (genuine-vs-family AUC, averaged over families)
    # and is reported as a co-headline beside pooled AUC — both, always.
    test_p = _subset(test, False)
    genuine_test_p = [r for r in test_p if r.gt_category in GENUINE_PRIMARY]
    results["per_attack_auc_test"] = {}
    for atk in ATTACKS:
        fam = [r for r in test_p if r.gt_category == atk]
        if not fam:
            continue
        y3, s3 = _scores_for(genuine_test_p + fam, smap, "cr")
        results["per_attack_auc_test"][atk.value] = round(stats.roc_auc(y3, s3), 4)
    macro, mlo, mhi = _macro_auc_ci(genuine_test_p, test_p, smap, bootstrap,
                                    seed=cfg.seed)
    results["macro_auc_test"] = {"auc": round(macro, 4),
                                 "ci95": [round(mlo, 4), round(mhi, 4)],
                                 "bootstrap": "event-level"}

    # --- (1c-3) operating points: detection/false-alarm cost curve +
    #     per-attack dev-calibrated thresholds ----------------
    # The single pooled-Youden point lands in a high-recall corner no operator
    # is forced to choose (a threshold-placement effect, not genuine-score
    # damage). The curve prices detection at each false-alarm
    # budget — τ_b is the dev-genuine score quantile at budget b, so budgets
    # are set without touching test — and the per-attack Youden row shows each
    # family's own optimum under a family-specific alarm policy.
    gen_dev = [r for r in _subset(dev, False) if r.gt_category in GENUINE_PRIMARY]
    gen_dev_scores = np.sort([smap[r.record_id]["cr"] for r in gen_dev])
    gen_test_scores = np.array([smap[r.record_id]["cr"] for r in genuine_test_p])
    atk_by_fam = {atk.value: [smap[r.record_id]["cr"] for r in test_p
                              if r.gt_category == atk] for atk in ATTACKS}
    atk_all = [v for vals in atk_by_fam.values() for v in vals]
    curve = []
    for b in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        k = int(np.floor(b * len(gen_dev_scores)))
        tau_b = float(gen_dev_scores[min(k, len(gen_dev_scores) - 1)])
        det_fam = {f: round(float(np.mean([s < tau_b for s in vals])), 4)
                   for f, vals in atk_by_fam.items() if vals}
        curve.append({
            "far_budget_dev": b,
            "tau": round(tau_b, 4),
            "far_test": round(float(np.mean(gen_test_scores < tau_b)), 4),
            "detection_pooled_test": round(float(np.mean([s < tau_b for s in atk_all])), 4),
            "detection_macro_test": round(float(np.mean(list(det_fam.values()))), 4),
            "detection_by_attack_test": det_fam,
        })
    results["detection_far_curve_test"] = curve
    results["per_attack_operating_test"] = {}
    for atk in ATTACKS:
        fam_dev = [r for r in _subset(dev, False) if r.gt_category == atk]
        fam_test = atk_by_fam.get(atk.value) or []
        if not fam_dev or not fam_test:
            continue
        y4, s4 = _scores_for(gen_dev + fam_dev, smap, "cr")
        tau_a = _youden_threshold(y4, s4)
        results["per_attack_operating_test"][atk.value] = {
            "tau_dev": round(tau_a, 4),
            "detection_test": round(float(np.mean([s < tau_a for s in fam_test])), 4),
            "far_test": round(float(np.mean(gen_test_scores < tau_a)), 4),
        }

    # --- (1d-q) quality-tier ranking --------------
    # Spearman ρ between the MCAM score and the generation quality tier of
    # genuine crowd reports (poor < moderate < good): the score should rank
    # report quality, not just separate attacks.
    _TIER_ORD = {"poor": 0, "moderate": 1, "good": 2}
    qx, qy = [], []
    for r in test:
        if (r.gt_category == Category.GENUINE_CROWD
                and r.gt_quality_tier in _TIER_ORD):
            qx.append(smap[r.record_id]["cr"])
            qy.append(_TIER_ORD[r.gt_quality_tier])
    rho, rho_p = stats.spearman(qx, qy)
    results["quality_tier_spearman_test"] = {
        "rho": round(rho, 4), "p": (round(rho_p, 6) if rho_p is not None else None),
        "n": len(qx), "tiers": "poor < moderate < good"}

    # --- (1d-g) per-generator stratification ------------
    # Attack text is generated by family A only (channel separation), so the
    # generator-artifact control is matched-family vs cross-family genuine:
    # if AUC(genuine-A crowd vs attacks) ≈ AUC(genuine-B crowd vs attacks),
    # detection is not keying on family-A style fingerprints.
    atk_test_recs = [r for r in test_p
                     if r.gt_category and r.gt_category.name.startswith("ATTACK")]
    results["per_generator_auc_test"] = {}
    for famlab in ("A", "B"):
        gen_f = [r for r in test if r.gt_category == Category.GENUINE_CROWD
                 and r.provenance.get("generator_family") == famlab]
        if not gen_f:
            continue
        y5, s5 = _scores_for(gen_f + atk_test_recs, smap, "cr")
        results["per_generator_auc_test"][f"genuine_crowd_{famlab}_vs_attacks"] = {
            "auc": round(stats.roc_auc(y5, s5), 4), "n_genuine": len(gen_f)}

    # --- (1e) credibility-level distribution by category (test) ---
    levels: Dict[str, Dict[str, int]] = {}
    for r in test:
        if not r.gt_category:
            continue
        lvl = credibility_level(smap[r.record_id]["cr"])
        levels.setdefault(r.gt_category.value, {})
        levels[r.gt_category.value][lvl] = levels[r.gt_category.value].get(lvl, 0) + 1
    results["credibility_levels_by_category_test"] = levels

    # --- (1g) calibration (Brier + reliability bins) on test ---
    y_all, s_all = _scores_for(_subset(test, True), smap, "cr")
    results["brier_score"] = round(float(np.mean((s_all - y_all) ** 2)), 4)
    results["reliability_bins"] = _reliability(y_all, s_all)

    # --- (1h) negative control (shuffle test labels → AUC ≈ 0.5) ---
    rng = np.random.default_rng(cfg.seed)
    y_shuf = y_all.copy(); rng.shuffle(y_shuf)
    results["negative_control_auc"] = round(stats.roc_auc(y_shuf, s_all), 4)

    # --- (1f) social-credibility formulation comparison --------------------
    # Beyond overall AUC (on which polarity and naive are near-tied here),
    # report where polarity is *designed* to matter: detecting fabrications that
    # genuine neighbours contradict (the genuine-vs-T1 subset) and the
    # false-alarm rate on genuine mixed-support events. Each formulation is
    # thresholded on its own dev split (Section 4.5).
    results["social_formulation_primary"] = {}
    fab = Category.ATTACK_FABRICATION
    for mode in ("polarity", "naive", "binary"):
        params = common.mcam_params_from_cfg(cfg, social_mode=mode)
        smap2 = _assess_all(cfg, bench, params=params)
        sd2 = {rid: {"cr": s.cr} for rid, s in smap2.items()}
        y2, s2 = _scores_for(_subset(test, False), sd2, "cr")
        auc_overall = round(stats.roc_auc(y2, s2), 4)
        t1_sub = [r for r in _subset(test, False)
                  if r.gt_category in GENUINE_PRIMARY or r.gt_category == fab]
        yt1, st1 = _scores_for(t1_sub, sd2, "cr")
        yd2, sd2d = _scores_for(_subset(dev, False), sd2, "cr")
        tau2 = _youden_threshold(yd2, sd2d)
        fab_recs = [r for r in test if r.gt_category == fab]
        gen_recs = [r for r in test if r.gt_category and r.gt_category.label.value == "CREDIBLE"]
        results["social_formulation_primary"][mode] = {
            "auc_overall": auc_overall,
            "auc_vs_fabrication": round(stats.roc_auc(yt1, st1), 4),
            "t1_detection": round(float(np.mean([sd2[r.record_id]["cr"] < tau2 for r in fab_recs])), 4) if fab_recs else None,
            "false_alarm": round(float(np.mean([sd2[r.record_id]["cr"] < tau2 for r in gen_recs])), 4) if gen_recs else None,
        }
    # back-compatible overall-AUC view (used by the summary print)
    results["social_formulation_auc_primary"] = {
        m: v["auc_overall"] for m, v in results["social_formulation_primary"].items()}

    # --- (1i) multi-seed generation-variance stability of the primary AUC ---
    # Regenerate the synthetic overlay under n_pipeline_seeds distinct seeds at the
    # same scale and recompute the headline AUC, reporting mean ± std (Section 4.5). A
    # conclusion that survives regeneration is robust to generator stochasticity.
    results["primary_auc_multiseed"] = _multiseed_auc(cfg, bench)

    _print_summary(results)
    common.write_results(cfg, "e1", results)
    _figure(cfg, test, smap)
    _figure_far_curve(cfg, results)
    return results


def _multiseed_auc(cfg: Config, bench) -> dict:
    """Primary AUC across n_pipeline_seeds regenerated overlays (mean ± std).

    The build specs come from ``common`` so that the ``generate`` phase warms the
    exact same builds (no cache-key drift); seed 0 is the base build (cache hit),
    the rest regenerate the overlay under fresh generator stochasticity.
    """
    from ..generation.benchmark import build_benchmark
    seeds = common.multiseed_seeds(cfg)
    n_events = bench.meta.get("n_events")
    degraded = bool(bench.meta.get("degraded", False))
    aucs = []
    for seed in seeds:
        try:
            b = build_benchmark(cfg, verbose=False,
                                **common.multiseed_build_kwargs(n_events, seed, degraded))
        except Exception as exc:  # offline cache miss → report, don't crash E1
            return {"error": str(exc), "n_seeds": len(seeds)}
        test_k = [r for r in b.records if r.provenance.get("split") == "test"]
        smap_k = _assess_all(cfg, b)
        sd_k = {rid: {"cr": s.cr} for rid, s in smap_k.items()}
        y, s = _scores_for(_subset(test_k, False), sd_k, "cr")
        aucs.append(stats.roc_auc(y, s))
    arr = np.asarray([a for a in aucs if a == a])  # drop NaN
    if arr.size == 0:
        return {"n_seeds": len(seeds), "values": aucs}
    return {"n_seeds": len(seeds), "seeds": seeds,
            "values": [round(float(a), 4) for a in aucs],
            "mean": round(float(arr.mean()), 4), "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 4)}


def _macro_auc_ci(genuine, test_p, smap, n_resamples: int, seed: int = 0):
    """Macro (attack-type-balanced) AUC with an event-level bootstrap CI.

    Each replicate resamples genuine EVENTS and attack records (their own
    groups, as in ``stats.bootstrap_auc_ci_grouped``), recomputes each
    genuine-vs-family AUC, and averages over the families present.
    """
    fams = {}
    for r in test_p:
        if r.gt_category and r.gt_category.name.startswith("ATTACK"):
            fams.setdefault(r.gt_category.value, []).append(r)

    def _macro(gen_recs, fam_map):
        gy = [1] * len(gen_recs)
        gs = [smap[r.record_id]["cr"] for r in gen_recs]
        vals = []
        for recs in fam_map.values():
            if not recs:
                continue
            y = np.array(gy + [0] * len(recs))
            s = np.array(gs + [smap[r.record_id]["cr"] for r in recs])
            a = stats.roc_auc(y, s)
            if a == a:
                vals.append(a)
        return float(np.mean(vals)) if vals else float("nan")

    point = _macro(genuine, fams)
    rng = np.random.default_rng(seed)
    gen_by_ev = {}
    for r in genuine:
        gen_by_ev.setdefault(r.gt_event_id if r.gt_event_id is not None
                             else r.record_id, []).append(r)
    ev_keys = list(gen_by_ev)
    boot = []
    for _ in range(n_resamples):
        chosen = rng.choice(len(ev_keys), size=len(ev_keys), replace=True)
        g = [r for i in chosen for r in gen_by_ev[ev_keys[i]]]
        fm = {}
        for fam, recs in fams.items():
            idx = rng.choice(len(recs), size=len(recs), replace=True)
            fm[fam] = [recs[i] for i in idx]
        m = _macro(g, fm)
        if m == m:
            boot.append(m)
    if not boot:
        return point, point, point
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(s)
    thresholds = np.unique(s[order])
    best_t, best_j = 0.5, -1.0
    P, N = (y == 1).sum(), (y == 0).sum()
    for t in thresholds:
        pred_pos = s >= t
        tpr = np.sum(pred_pos & (y == 1)) / P if P else 0
        fpr = np.sum(pred_pos & (y == 0)) / N if N else 0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def _reliability(y: np.ndarray, s: np.ndarray, bins: int = 10) -> List[Dict]:
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for k in range(bins):
        m = (s >= edges[k]) & (s < edges[k + 1] if k < bins - 1 else s <= edges[k + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": [round(edges[k], 2), round(edges[k + 1], 2)],
                    "n": int(m.sum()), "mean_score": round(float(s[m].mean()), 3),
                    "empirical_credible": round(float(y[m].mean()), 3)})
    return out


def _print_summary(r: Dict) -> None:
    print("\n[E1] Summary (test split):")
    print(f"  AUC-ROC primary  : {r['auc_roc_primary']['auc']} CI {r['auc_roc_primary']['ci95']}")
    print(f"  macro-AUC        : {r['macro_auc_test']['auc']} CI {r['macro_auc_test']['ci95']} "
          f"(per-attack: {r['per_attack_auc_test']})")
    print(f"  AUC-ROC secondary: {r['auc_roc_secondary']['auc']}")
    print(f"  per-dimension AUC: {r['per_dimension_auc_primary']}")
    print(f"  detection @τ={r['operating_threshold_dev']}: {r['per_attack_detection_test']}")
    print(f"  false-alarm rate : {r.get('false_alarm_rate_test')}")
    print("  detection/FAR    : " + " | ".join(
        f"b={row['far_budget_dev']:.0%}: far={row['far_test']} "
        f"pool={row['detection_pooled_test']} macro={row['detection_macro_test']}"
        for row in r.get("detection_far_curve_test", [])
        if row["far_budget_dev"] in (0.05, 0.10, 0.20)))
    print(f"  quality Spearman : {r.get('quality_tier_spearman_test')}")
    print(f"  per-generator    : {r.get('per_generator_auc_test')}")
    print(f"  Brier            : {r['brier_score']}   negative-control AUC: {r['negative_control_auc']}")
    print(f"  social forms     : {r['social_formulation_auc_primary']}")


# Display labels for the attack categories (T1–T5 naming of the paper).
_CAT_LABEL = {
    "OFFICIAL": "Official",
    "GENUINE_CROWD": "Genuine crowd",
    "GENUINE_SENSOR": "Genuine sensor",
    "ATTACK_FABRICATION": "T1 Fabrication",
    "ATTACK_REPLAY": "T2 Replay",
    "ATTACK_TAMPER": "T3 Tamper",
    "ATTACK_IMPERSONATION": "T4 Impersonation",
    "ATTACK_SYBIL": "T5 Sybil",
}


def _save_fig(cfg, fig, stem: str) -> None:
    """Publication figures: PNG at 300 dpi plus a vector PDF, no in-image title."""
    fig.tight_layout()
    out = cfg.output_dir / "e1"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.pdf")


def _figure(cfg, test, smap) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cats, data = [], []
    for cat in Category:
        vals = [smap[r.record_id]["cr"] for r in test if r.gt_category == cat]
        if vals:
            cats.append(_CAT_LABEL.get(cat.value, cat.value))
            data.append(vals)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MCAM credibility $C_r(r)$")
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(cfg, fig, "e1_score_distributions")
    plt.close(fig)


def _figure_far_curve(cfg, results) -> None:
    """Detection/false-alarm cost curve (Fig. 4): pooled and macro
    detection vs realized test FAR at each dev-side budget, with each attack
    family's own dev-Youden operating point overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    curve = results.get("detection_far_curve_test") or []
    if not curve:
        return
    far = [p["far_test"] for p in curve]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(far, [p["detection_pooled_test"] for p in curve],
            marker="o", ms=4, color="#4C72B0", label="Pooled detection")
    ax.plot(far, [p["detection_macro_test"] for p in curve],
            marker="s", ms=4, color="#DD8452", ls="--", label="Macro detection")
    for fam, op in (results.get("per_attack_operating_test") or {}).items():
        ax.scatter(op["far_test"], op["detection_test"], marker="x", s=45,
                   color="#55A868", zorder=3)
        ax.annotate(_CAT_LABEL.get(fam, fam).split()[0], fontsize=8,
                    xy=(op["far_test"], op["detection_test"]),
                    xytext=(4, -9), textcoords="offset points", color="#3d7a4a")
    tau = results.get("operating_threshold_dev")
    far_tau = results.get("false_alarm_rate_test")
    if tau is not None and far_tau is not None:
        ax.axvline(far_tau, color="grey", lw=0.8, ls=":")
        ax.annotate(f"pooled Youden $\\tau$ = {tau:.3f}", fontsize=8, color="grey",
                    xy=(far_tau, 0.05), xytext=(-5, 0), ha="right",
                    textcoords="offset points")
    ax.set_xlabel("False-alarm rate on genuine reports (test)")
    ax.set_ylabel("Attack detection rate (test)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(cfg, fig, "e1_far_cost_curve")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E1 (credibility).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark",
                    help="Path to a built benchmark directory.")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    bd = args.benchmark
    from pathlib import Path
    if not Path(bd).is_absolute():
        bd = cfg.project_root / bd
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
