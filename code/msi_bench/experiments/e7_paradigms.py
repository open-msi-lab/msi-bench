"""Experiment E7 — Alternative paradigms (Section 5.1).

Positions the MCAM against a rule-based filter and an LLM-as-judge baseline
(generator family C ∉ {A, B}; the judge sees only pipeline-observable fields —
no ground-truth leakage). When the judge client is the offline mock, a
transparent heuristic stands in for the LLM rating so the comparison still runs
without an API key. Reports primary AUC-ROC plus, per paradigm, the macro
(attack-type-balanced) AUC and per-attack AUC (the coverage
claim is breadth, so the co-headline must weight every attack family equally
for every paradigm, not just the proposed one); the LLM-as-judge hypothesis is
phrased neutrally (strong on content plausibility, blind to social/spatiotemporal
context), and a judge win is reported as a finding.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..clustering import STSDBSCAN
from ..config import Config, get_config
from ..credibility.domain import NavigabilityModel
from ..generation.benchmark import Benchmark, parallel_map
from ..llm.client import OfflineCacheMiss, get_client
from ..schema import MSIRecord
from .. import stats
from . import common
from .e1_credibility import _groups_for, _scores_for, _subset

# The judge is asked to reply with ONLY a number (~4 tokens). A small ceiling
# protects against a non-compliant model padding the answer and being cut off
# mid-text — which would make the score regex miss and silently fall back to
# 0.5. Override via `llm.judge_max_tokens` in config. NB: reasoning models on
# the Responses proxy floor this internally (min_output_tokens) so reasoning
# tokens don't crowd out the digits.
JUDGE_MAX_TOKENS = 32


def _rule_based_score(rec: MSIRecord, nav: NavigabilityModel) -> float:
    score = 1.0
    if rec.centroid is None:
        score -= 0.4
    else:
        score *= max(0.2, nav.navigability(rec.centroid))
    if len((rec.raw_text or "").strip()) < 15:
        score -= 0.3
    if rec.issue_time and rec.valid_end and rec.issue_time > rec.valid_end:
        score -= 0.5  # temporal inconsistency (e.g., replay)
    return float(min(1.0, max(0.0, score)))


def _judge_score(rec: MSIRecord, client, max_tokens: int = JUDGE_MAX_TOKENS,
                 miss_log: Optional[list] = None) -> float:
    if client.is_mock:
        # transparent heuristic stand-in: content plausibility only
        s = 0.5
        if rec.centroid is not None:
            s += 0.2
        if len((rec.raw_text or "")) > 40:
            s += 0.15
        if "duration_hours" in rec.structured_data:
            d = rec.structured_data["duration_hours"]
            if isinstance(d, (int, float)) and (d < 0 or d > 1e5):
                s -= 0.4
        return float(min(1.0, max(0.0, s)))
    system = ("You are a maritime safety analyst. Rate the credibility of the "
              "following report from 0.00 (fabricated) to 1.00 (credible). "
              "Reply with ONLY the number.")
    user = (f"Text: {rec.raw_text}\nType: {rec.event_type.value}\n"
            f"Has position: {rec.centroid is not None}\n"
            f"Device class: {rec.device_class}\nAuth: {rec.auth_level}")
    try:
        resp = client.complete(system, user, max_tokens=max_tokens, temperature=0.0)
        import re
        m = re.search(r"[01]?\.\d+|[01]", resp)
        return float(m.group(0)) if m else 0.5
    except OfflineCacheMiss:
        # The judge prompt carries pipeline-observable fields; a benchmark change
        # (e.g. the adaptive-adversary metadata) invalidates those cache entries.
        # Record the miss so the LLM-as-judge row is flagged as provisional
        # instead of being silently biased toward the 0.5 default.
        if miss_log is not None:
            miss_log.append(rec.record_id)
        return 0.5
    except Exception:
        return 0.5


def _features(rec) -> List[float]:
    """Hand-crafted features for the supervised ML baseline."""
    dev_rank = {"PROFESSIONAL": 4, "MARINE_CERTIFIED": 3, "CONSUMER_GPS": 2,
                "MOBILE_DEVICE": 1, "UNKNOWN": 0}.get(rec.device_class, 0)
    auth_rank = {"PKI_FULL": 3, "PKI_BASIC": 2, "REGISTERED": 1, "ANONYMOUS": 0}.get(
        rec.auth_level, 0)
    return [
        float(len(rec.raw_text or "")),
        1.0 if rec.centroid else 0.0,
        float(len(rec.structured_data)),
        float(dev_rank), float(auth_rank),
        1.0 if rec.provider_type.value == "OFFICIAL" else 0.0,
    ]


def _ml_classifier_scores(dev, test) -> Dict[str, Dict[str, float]]:
    """Supervised gradient-boosted classifier on hand-crafted features.

    Trained on the dev split's binary credible/non-credible labels; XGBoost if
    available, else sklearn GradientBoosting. Returns test-record credibility.
    """
    Xtr = np.array([_features(r) for r in dev])
    ytr = np.array([1 if r.gt_category.label.value == "CREDIBLE" else 0 for r in dev])
    if len(np.unique(ytr)) < 2:
        return {r.record_id: {"cr": 0.5} for r in test}
    model = None
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=80, max_depth=3, use_label_encoder=False,
                              eval_metric="logloss", verbosity=0)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=80, max_depth=3)
    model.fit(Xtr, ytr)
    Xte = np.array([_features(r) for r in test])
    proba = model.predict_proba(Xte)[:, 1]
    return {r.record_id: {"cr": float(p)} for r, p in zip(test, proba)}


def _dawid_skene_scores(records, group_labels) -> Dict[str, Dict[str, float]]:
    """Dawid–Skene-style source-reliability EM (truth inference [11]).

    Items are the PIPELINE'S predicted clusters (not ground-truth events: a baseline
    must consume the same observable grouping MCAM does, never the GT event ids).
    Each source's reliability is its agreement with the per-cluster
    consensus hazard_type, refined over a few EM iterations; a report's
    credibility is its source's estimated reliability. Noise points form
    singleton clusters, so isolated fabrications stay at the 0.5 prior — the known
    blind spot of consensus methods, reported as a finding.
    """
    from collections import defaultdict
    by_group = defaultdict(list)
    for r, g in zip(records, group_labels):
        key = int(g) if g >= 0 else f"noise::{r.record_id}"  # noise → singleton
        by_group[key].append(r)
    rel = defaultdict(lambda: 0.5)
    for _ in range(5):
        # E-step: weighted consensus hazard_type per predicted cluster
        consensus = {}
        for grp, recs in by_group.items():
            votes = defaultdict(float)
            for r in recs:
                ht = str(r.structured_data.get("hazard_type", "?"))
                votes[ht] += rel[r.provider_id]
            consensus[grp] = max(votes, key=votes.get) if votes else "?"
        # M-step: source reliability = agreement with consensus (smoothed)
        agree, total = defaultdict(float), defaultdict(float)
        for grp, recs in by_group.items():
            for r in recs:
                ht = str(r.structured_data.get("hazard_type", "?"))
                agree[r.provider_id] += 1.0 if ht == consensus[grp] else 0.0
                total[r.provider_id] += 1.0
        for s in total:
            rel[s] = (agree[s] + 1.0) / (total[s] + 2.0)  # Beta(1,1) smoothing
    return {r.record_id: {"cr": float(rel[r.provider_id])} for r in records}


def warm_judge_cache(cfg: Config, benchmark_dir: str) -> int:
    """Run only the LLM-as-judge calls (the sole networked step of any
    experiment) so the response cache holds them for an offline ``compute`` run.

    Deliberately skips the heavy scoring/clustering/bootstrap that :func:`run`
    performs — those are pure computation the offline machine redoes. Returns the
    number of test records judged. The judge is deterministic (temperature 0.0)
    and keyed by record id, so the warmed entries replay :func:`run` exactly.
    """
    bench = Benchmark.load(benchmark_dir)
    test = [r for r in bench.records if r.provenance.get("split") == "test"]
    judge = get_client(cfg, "judge_c")
    if judge.is_mock:
        print("[generate] judge is the offline mock; nothing to warm.")
        return 0
    judge_max_tokens = int(cfg.get("llm", "judge_max_tokens", default=JUDGE_MAX_TOKENS))
    concurrency = int(cfg.get("llm", "concurrency", default=1))
    parallel_map(lambda r: _judge_score(r, judge, judge_max_tokens),
                 test, concurrency, label="E7 judge (warm)")
    return len(test)


def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    dev = [r for r in bench.records if r.provenance.get("split") == "dev"]
    test = [r for r in bench.records if r.provenance.get("split") == "test"]
    nav = common.build_navigability(cfg)

    # MCAM (proposed)
    records, _, sim, smap = common.assess_benchmark(cfg, bench)
    mcam_scores = {rid: {"cr": s.cr} for rid, s in smap.items()}

    # Predicted clusters (pipeline-observable grouping) for the Dawid–Skene
    # baseline — never the GT event ids.
    ds_labels = STSDBSCAN(min_pts=cfg.get("clustering", "min_pts", default=2), eps=None,
                          eps_fallback=cfg.get("clustering", "eps_fallback", default=0.45)
                          ).fit(records, sim)

    judge = get_client(cfg, "judge_c")
    judge_max_tokens = int(cfg.get("llm", "judge_max_tokens", default=JUDGE_MAX_TOKENS))
    concurrency = int(cfg.get("llm", "concurrency", default=1))
    rule_scores = {r.record_id: {"cr": _rule_based_score(r, nav)} for r in test}
    # The judge calls are independent and network-bound, so run them on the same
    # thread pool as generation (llm.concurrency). Scores are keyed by record_id
    # and deterministic at temperature 0.0, so the result is identical to the
    # sequential path — only the wall-clock of a cache-cold run changes.
    judge_misses: List[str] = []
    judge_vals = parallel_map(
        lambda r: _judge_score(r, judge, judge_max_tokens, miss_log=judge_misses),
        test, concurrency, label="E7 judge")
    judge_scores = {r.record_id: {"cr": v} for r, v in zip(test, judge_vals)}
    ml_scores = _ml_classifier_scores(dev, test)
    ds_scores = _dawid_skene_scores(records, ds_labels)

    results_judge_note = None
    if judge_misses:
        results_judge_note = (
            f"{len(judge_misses)} of {len(test)} judge prompts were absent from the "
            "offline response cache (their pipeline-observable metadata changed with "
            "the adaptive-adversary benchmark) and defaulted to 0.5. The LLM-as-judge "
            "row is PROVISIONAL until `run generate` re-warms the judge on a "
            "networked machine.")
        print(f"[E7] WARNING: {results_judge_note}")

    subset = _subset(test, False)
    y, mcam_s = _scores_for(subset, mcam_scores, "cr")
    groups = _groups_for(subset)
    # MCAM with data-driven weights (Section 3.3): the label-using configuration, for a
    # fair comparison against the supervised GBDT (both fit on the dev labels).
    dd_eval, dd_info = common.data_driven_weighting(smap, _subset(dev, False), subset)
    dd_scores = {rid: {"cr": v} for rid, v in dd_eval.items()}
    results = {"judge_model": judge.tag, "judge_cache_misses": len(judge_misses),
               "methods": {}}
    if results_judge_note:
        results["judge_note"] = results_judge_note
    baselines = (("Rule-based filter", rule_scores),
                 ("ML classifier (GBDT/XGBoost)", ml_scores),
                 ("Dawid-Skene EM", ds_scores),
                 ("LLM-as-judge", judge_scores))
    # per-family AUC scaffolding for the macro co-headline:
    # genuine-vs-family for each attack family, averaged with equal weight.
    genuine_sub = [r for r in subset
                   if r.gt_category and not r.gt_category.name.startswith("ATTACK")]
    fam_subs = {}
    for r in subset:
        if r.gt_category and r.gt_category.name.startswith("ATTACK"):
            fam_subs.setdefault(r.gt_category.value, []).append(r)

    def _macro_and_per_attack(smp):
        per = {}
        for fam, recs in sorted(fam_subs.items()):
            yf, sf = _scores_for(genuine_sub + recs, smp, "cr")
            a = stats.roc_auc(yf, sf)
            if a == a:
                per[fam] = round(a, 4)
        return (round(float(np.mean(list(per.values()))), 4) if per else None), per

    for name, smp in (("MCAM (proposed)", mcam_scores),
                      ("MCAM (data-driven, §3.5.2)", dd_scores), *baselines):
        _, s = _scores_for(subset, smp, "cr")
        auc, lo, hi = stats.bootstrap_auc_ci_grouped(
            y, s, groups, cfg.raw["bootstrap_resamples"], seed=cfg.seed)
        macro, per_attack = _macro_and_per_attack(smp)
        results["methods"][name] = {"auc_primary": round(auc, 4),
                                    "ci95": [round(lo, 4), round(hi, 4)],
                                    "auc_macro": macro,
                                    "per_attack_auc": per_attack}
        print(f"[E7] {name:<30} AUC={auc:.4f} CI=[{lo:.3f},{hi:.3f}] macro={macro}")

    # --- significance: MCAM vs each baseline family (DeLong + Holm–Bonferroni) ---
    names, raw_p, diffs = [], [], []
    for name, smp in baselines:
        _, s = _scores_for(subset, smp, "cr")
        auc_a, auc_b, p = stats.delong_test(y, mcam_s, s)
        names.append(name); raw_p.append(p); diffs.append(auc_a - auc_b)
    adjusted, reject = stats.holm_bonferroni(raw_p)
    results["mcam_vs_baselines"] = {
        name: {"auc_delta": round(d, 4), "p_delong": round(p, 4),
               "p_holm": round(a, 4), "significant_at_0.05": bool(r),
               "mcam_better": bool(d > 0)}
        for name, d, p, a, r in zip(names, diffs, raw_p, adjusted, reject)}
    print("[E7] MCAM vs baselines (DeLong, Holm–Bonferroni):")
    for name in names:
        m = results["mcam_vs_baselines"][name]
        print(f"     {name:<30} ΔAUC={m['auc_delta']:+.4f} p_holm={m['p_holm']:.4f} "
              f"{'*' if m['significant_at_0.05'] else ''}")

    # Fair, label-matched contrast: data-driven MCAM vs the supervised GBDT (both
    # fit on the dev labels). The headline expert MCAM needs no labels at all.
    _, dd_s = _scores_for(subset, dd_scores, "cr")
    _, ml_s = _scores_for(subset, ml_scores, "cr")
    a_dd, a_ml, p_dd_ml = stats.delong_test(y, dd_s, ml_s)
    results["data_driven_mcam_vs_ml"] = {
        "auc_data_driven_mcam": round(a_dd, 4), "auc_ml_gbdt": round(a_ml, 4),
        "auc_delta": round(a_dd - a_ml, 4), "p_delong": round(p_dd_ml, 4),
        "weights_relative": dd_info["weights_relative"]}
    print(f"[E7] data-driven MCAM {a_dd:.4f} vs GBDT {a_ml:.4f} "
          f"(Δ={a_dd - a_ml:+.4f}, p_delong={p_dd_ml:.4f}); learned w={dd_info['weights_relative']}")

    common.write_results(cfg, "e7", results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E7 (alternative paradigms).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
