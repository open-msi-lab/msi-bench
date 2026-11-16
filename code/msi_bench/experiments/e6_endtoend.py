"""Experiment E6 — End-to-end pipeline evaluation (Section 5.4).

Runs the complete pipeline (MCAM → STS-DBSCAN → conflict resolution → D-S
fusion) and reports event-detection, attack-rejection, and genuine-retention
rates; the pipeline ablation (no-credibility / no-semantic-clustering /
no-conflict-management); and a coarse adversarial-pressure sweep.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config, get_config
from ..generation.benchmark import Benchmark, build_benchmark
from . import common


def _pipeline_metrics(cfg, bench, with_test_view: bool = True, **flags) -> Dict[str, float]:
    """Run the pipeline once; report benchmark-wide metrics plus the held-out
    TEST view (headline numbers are reported on the test split, Section 4.5 —
    the pipeline still ingests the full operational record stream, but events
    and records are SCORED only where their event-level split matches).
    Test-view keys carry a ``_test`` suffix.
    """
    out = common.run_pipeline(cfg, bench, **flags)
    records = out["records"]
    rid_event = {r.record_id: r.gt_event_id for r in records}
    is_attack = {r.record_id: bool(r.gt_category and r.gt_category.name.startswith("ATTACK"))
                 for r in records}
    rid_split = {r.record_id: r.provenance.get("split", "dev") for r in records}
    ev_split = {eid: (meta or {}).get("split") for eid, meta in bench.events.items()}

    believed = [fr for fr in out["fused"] if fr.belief > 0.5]
    believed_provenance = set()
    attack_led_provenance = set()   # members of believed records that are attack-MAJORITY
    covered_events = set()
    n_attack_led = 0
    for fr in believed:
        believed_provenance.update(fr.provenance)
        evs = [rid_event.get(rid) for rid in fr.provenance if rid_event.get(rid) is not None]
        if evs:
            covered_events.add(Counter(evs).most_common(1)[0][0])
        # attack-majority believed records are the REAL end-to-end failures
        # (a spurious hazard published); attacks absorbed as a minority into a
        # genuine-majority record ride along at low credibility weight instead.
        if sum(1 for rid in fr.provenance if is_attack.get(rid)) * 2 > len(fr.provenance):
            n_attack_led += 1
            attack_led_provenance.update(fr.provenance)
    # CROSS_VERIFIED bar (belief > 0.75 — the Section 3.6 status threshold): the bar
    # at which the fusion stage asserts independent corroboration, and where
    # the Section 3.6 independence discount is designed to bite (a collapsed
    # campaign still carries its best member's credibility past 0.5, but can
    # no longer manufacture the corroboration sharpening that 0.75 demands).
    cv_provenance = set()
    for fr in believed:
        if fr.belief > 0.75:
            cv_provenance.update(fr.provenance)
    clustered = {records[i].record_id for i, lab in enumerate(out["labels"]) if lab >= 0}
    cr = out.get("credibility", {})
    rid_cat = {r.record_id: (r.gt_category.value if r.gt_category else "?") for r in records}

    # credibility-weighted attack influence inside believed records
    w_att = w_all = 0.0
    for fr in believed:
        for rid in fr.provenance:
            w = float(cr.get(rid, 0.0))
            w_all += w
            if is_attack.get(rid):
                w_att += w

    def view(split):
        events_present = {r.gt_event_id for r in records if r.gt_event_id is not None
                          and (split is None or ev_split.get(r.gt_event_id) == split)}
        attack_ids = [rid for rid, a in is_attack.items()
                      if a and (split is None or rid_split.get(rid) == split)]
        genuine_ids = [rid for rid, a in is_attack.items()
                       if not a and (split is None or rid_split.get(rid) == split)]
        by_type: Dict[str, List[bool]] = {}
        by_type_cv: Dict[str, List[bool]] = {}
        for rid in attack_ids:
            by_type.setdefault(rid_cat[rid], []).append(rid in believed_provenance)
            by_type_cv.setdefault(rid_cat[rid], []).append(rid in cv_provenance)
        return {
            "event_detection_rate": round(
                len(covered_events & events_present) / max(1, len(events_present)), 4),
            "attack_rejection_rate": round(
                np.mean([rid not in believed_provenance for rid in attack_ids]), 4)
            if attack_ids else None,
            # decomposition of acceptance: absorbed-as-minority vs attack-led
            "attack_acceptance_by_type": {
                k: round(float(np.mean(v)), 4) for k, v in sorted(by_type.items())},
            # same decomposition at the CROSS_VERIFIED bar (belief > 0.75)
            "attack_acceptance_by_type_cross_verified": {
                k: round(float(np.mean(v)), 4) for k, v in sorted(by_type_cv.items())},
            "attack_in_attack_led_share": round(
                np.mean([rid in attack_led_provenance for rid in attack_ids]), 4)
            if attack_ids else None,
            "genuine_retention_rate": round(
                np.mean([rid in clustered for rid in genuine_ids]), 4)
            if genuine_ids else None,
        }

    m = view(None)
    m["n_fused_believed"] = len(believed)
    m["n_attack_led_believed"] = n_attack_led
    m["attack_credibility_weight_share_in_believed"] = round(
        w_att / w_all, 4) if w_all > 0 else None
    # Section 3.6 campaign-amplification instrument: fused records whose provenance
    # is Sybil-majority. Without the independence discount their belief
    # sharpens with campaign size k; with it, belief is capped at the best
    # member's single-source level (scripted tiers), while the paraphrased
    # tier — whose members evade the containment predicate by construction —
    # keeps its sources separate (the paraphrased-tier residual of Section 5.1).
    syb = [fr for fr in out["fused"]
           if sum(1 for rid in fr.provenance
                  if rid_cat.get(rid) == "ATTACK_SYBIL") * 2 > len(fr.provenance)]
    m["sybil_majority_fused"] = {
        "n": len(syb),
        "belief_p50": round(float(np.median([fr.belief for fr in syb])), 4) if syb else None,
        "belief_max": round(max(fr.belief for fr in syb), 4) if syb else None,
        "n_believed": sum(1 for fr in syb if fr.belief > 0.5),
        "n_cross_verified": sum(1 for fr in syb if fr.belief > 0.75),
        "effective_over_member_sources": (
            round(sum(fr.n_effective_sources for fr in syb)
                  / max(1, sum(len(fr.provenance) for fr in syb)), 4)) if syb else None,
    }
    if with_test_view:
        m.update({f"{k}_test": v for k, v in view("test").items()})
    return m


def _view_at_prevalence(bench: Benchmark, target_p: float, seed: int):
    """Return ``(view, info)``: a benchmark view at attack prevalence ≈
    ``target_p`` with the genuine corpus held FIXED (Section 5.4).

    The genuine overlay is never thinned. The attack overlay is scaled to the
    target by stratified subsampling — preserving the native attack-type mix —
    from the base attacks plus the attack reserve (an enlarged pool generated
    with the benchmark precisely so above-native prevalence is reachable
    without manufacturing records). Thinning the genuine corpus instead would
    measure corroboration starvation, not adversarial pressure: detection would
    'collapse' because events no longer had enough reports to form clusters. If even
    base + reserve cannot reach the target, the view saturates and is flagged
    (``pool_exhausted``) rather than silently distorted.
    """
    rng = random.Random(seed)
    genuine = [r for r in bench.records
               if r.gt_category and not r.gt_category.name.startswith("ATTACK")]
    pool: Dict[str, List] = {}
    for r in list(bench.records) + list(bench.reserve):
        if r.gt_category and r.gt_category.name.startswith("ATTACK"):
            pool.setdefault(r.gt_category.value, []).append(r)
    G = len(genuine)
    a_pool = sum(len(v) for v in pool.values())
    need = 0 if target_p <= 0.0 else int(round(target_p * G / (1.0 - target_p)))
    keep = min(need, a_pool)
    attacks: List = []
    if keep and a_pool:
        for cat in sorted(pool):
            recs = list(pool[cat])
            rng.shuffle(recs)
            attacks.extend(recs[:int(round(keep * len(recs) / a_pool))])
    recs = genuine + attacks
    info = {"n_genuine": G, "n_attacks": len(attacks), "attack_pool_size": a_pool,
            "pool_exhausted": bool(need > a_pool),
            "attack_mix": {cat: sum(1 for r in attacks
                                    if r.gt_category.value == cat) for cat in sorted(pool)}}
    view = Benchmark(records=recs, events=bench.events, histories=bench.histories,
                     meta=bench.meta)
    return view, info


def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    results = {"pipeline_ablation": {}, "adversarial_pressure": {}, "temporal_holdout": None}

    # (6a/6b) end-to-end + pipeline ablation, with full-pipeline latency.
    variants = {
        "Full pipeline": {},
        "No credibility (uniform BPA)": {"uniform_credibility": True},
        "No semantic clustering": {"use_semantic": False},
        "No conflict management": {"use_conflict": False},
        "No independence discount (§4.4.2)": {"use_independence_discount": False},
    }
    for name, flags in variants.items():
        t0 = time.perf_counter()
        m = _pipeline_metrics(cfg, bench, **flags)
        m["latency_s"] = round(time.perf_counter() - t0, 2)
        results["pipeline_ablation"][name] = m
        print(f"[E6] {name:<32} detect={m['event_detection_rate_test']} "
              f"reject={m['attack_rejection_rate_test']} retain={m['genuine_retention_rate_test']} "
              f"(test; full={m['event_detection_rate']}) ({m['latency_s']}s)")

    # (6c) robustness across the prevalence grid {0,5,10,20,30,50}%,
    # with the genuine corpus HELD FIXED: attacks are scaled by
    # stratified subsampling from base + reserve, so every row measures
    # adversarial pressure under constant genuine corroboration. Benchmark-wide
    # view: prevalence subsampling deliberately distorts the record population,
    # so per-split scoring is not meaningful here.
    if not bench.reserve:
        print("[E6] NOTE: no attack reserve found (attack_reserve.json) — "
              "above-native pressure rows will saturate at the base pool "
              "(~14%) and be flagged pool_exhausted. Rebuild with "
              "benchmark.reserve_attack_multiplier > 0 to enable the full sweep.")
    for target in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        view, vinfo = _view_at_prevalence(bench, target, cfg.seed)
        n_att = vinfo["n_attacks"]
        prevalence = round(100.0 * n_att / max(1, len(view.records)), 1)
        m = _pipeline_metrics(cfg, view, with_test_view=False)
        results["adversarial_pressure"][f"target_{int(target * 100)}pct"] = {
            "target_prevalence_pct": round(100.0 * target, 1),
            "achieved_prevalence_pct": prevalence, **vinfo, **m}
        print(f"[E6] pressure target={int(target * 100):>3}%  achieved={prevalence:>5}%  "
              f"(genuine={vinfo['n_genuine']} attacks={n_att}"
              f"{' POOL-EXHAUSTED' if vinfo['pool_exhausted'] else ''})  "
              f"detect={m['event_detection_rate']} reject={m['attack_rejection_rate']}")

    # (6d) temporal generalization on the 2024 holdout (built once, run once).
    # Build spec comes from ``common`` so the ``generate`` phase warms the same
    # build (no cache-key drift); offline, a cache miss is reported not fatal.
    try:
        kw = common.holdout_build_kwargs(cfg)
        hy = kw["years"]
        holdout = build_benchmark(cfg, verbose=False, **kw)
        m = _pipeline_metrics(cfg, holdout)
        results["temporal_holdout"] = {"years": [hy[0], hy[-1]],
                                       "n_events": holdout.meta["n_events"], **m}
        print(f"[E6] 2024 holdout: detect={m['event_detection_rate_test']} (test; "
              f"full={m['event_detection_rate']}) reject={m['attack_rejection_rate_test']}")
    except Exception as exc:
        results["temporal_holdout"] = {"error": str(exc)}
        print(f"[E6] holdout skipped: {exc}")

    common.write_results(cfg, "e6", results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E6 (end-to-end).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
