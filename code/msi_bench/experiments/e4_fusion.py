"""Experiment E4 — Credibility-weighted D-S fusion (Section 5.3, Tables 10–11).

Pipeline: assess → cluster (ε calibrated on the dev-genuine submatrix)
→ detect conflicts → partition → fuse. Reports four blocks:

* **4a** Coverage / Confidence / Redundancy / J per combination rule — on the
  held-out TEST split (headline, Section 4.3) and benchmark-wide (whole-benchmark view).
* **4b** Conflict-detection validation: the T3 perturbation logs
  (``provenance.perturbation`` — the exact tamper mode) are ground truth. For
  every tamper co-clustered with genuine reports of its source event, did a
  conflict of the expected type fire against them? Per-mode recall + type
  agreement + severity check, plus the false-fire rate on genuine same-event
  pairs — turning "we counted conflicts" into a validated detector.
* **4d** Fused-record accuracy:
  fused position error (nm) and temporal IoU vs the true event, fused
  existence/type accuracy, belief calibration (Brier/ECE + reliability bins),
  and attack contamination of believed records. Compared across combination
  rules (incl. the designed Yager baseline) and across conflict management:
  partition ON vs OFF, the latter in both the historical credibility-only BPA
  idiom ("blind") and the assertion-signed idiom ("signed") — the Zadeh
  high-conflict regime, reported on the contested-cluster stress subset.
* **Social-mode comparison**: the E4d table under polarity vs
  naive Cso — where contradiction-aware trust reduction must surface if the
  polarity formulation has fused-content value.
* **Independence-discount value** (Section 3.6): the E4d
  table with the fusion-stage independence discount ON (proposed) vs OFF,
  plus per-tier belief on Sybil-majority fused records — the instrument for
  the T5 fusion re-admission finding: scripted campaigns
  collapse to one effective source; the paraphrased tier measurably evades.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..clustering import STSDBSCAN
from ..conflict import CRITICAL, detect_conflicts, partition_cluster
from ..config import Config, get_config
from ..credibility.domain import WILDCARD_ATTR_VALUES
from ..fusion import FusedRecord, fuse_subcluster
from ..generation.benchmark import Benchmark
from ..geo import haversine_nm
from . import common

METHODS = {
    "Proposed (conflict-aware blend)": "blend",
    "Standard D-S (credibility BPA)": "dempster",
    "Standard D-S (uniform BPA)": "uniform",
    "Murphy average": "murphy",
    "Yager rule": "yager",
}

# Expected conflict signature per T3 perturbation mode (generation rules →
# conflict taxonomy of Section 3.5). ``severity_downgrade`` is a designed
# NEGATIVE control: ordinal severity discrepancies are classified as
# non-contradictory, so a validated detector must NOT fire a contradiction
# for it.
_EXPECTED_CONFLICT = {
    "coord_shift": {"CONF_SPATIAL"},
    "type_swap": {"CONF_SEMANTIC"},
    "existence_flip": {"CONF_SEMANTIC", "CONF_SOURCE"},
    "severity_downgrade": set(),
}


# --------------------------------------------------------------------------
# ground truth helpers
# --------------------------------------------------------------------------
def _event_truth(bench: Benchmark, records) -> Dict[str, Dict]:
    """Per-event ground truth for the fused-record accuracy metrics (4d).

    The official backbone record ``off-<event_id>`` is BY CONSTRUCTION the
    event's authoritative snapshot (Section 4.3): its centroid is the event
    centroid, its validity window the event's operational window, and its
    ``hazard_type`` the event's hazard key. Every fusion variant is scored
    against the same truth, so shared inputs cannot favour a method.
    """
    truth: Dict[str, Dict] = {}
    for r in records:
        if r.record_id.startswith("off-") and r.gt_event_id:
            truth[r.gt_event_id] = {
                "centroid": r.centroid,
                "valid_start": r.valid_start,
                "valid_end": r.valid_end,
                "hazard_key": str(r.structured_data.get("hazard_type") or "").lower(),
                "split": bench.events.get(r.gt_event_id, {}).get("split"),
            }
    return truth


def _provenance_majority(fr: FusedRecord, rid_event) -> Optional[str]:
    """Majority event over ALL provenance reports (Section 4.3 rule).

    Attack reports (gt_event_id = None) are counted via a sentinel, so a fused
    record whose provenance is attack-majority matches no event.
    """
    keys = [rid_event.get(rid) or "__NONE__" for rid in fr.provenance]
    if not keys:
        return None
    top = Counter(keys).most_common(1)[0][0]
    return None if top == "__NONE__" else top


def _majority_split(fr: FusedRecord, rid_split) -> str:
    return Counter(rid_split.get(rid, "dev") for rid in fr.provenance).most_common(1)[0][0]


def _interval_iou(a_start, a_end, b_start, b_end) -> Optional[float]:
    if not (a_start and a_end and b_start and b_end):
        return None
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    inter = max(0.0, (e - s).total_seconds())
    union = (max(a_end, b_end) - min(a_start, b_start)).total_seconds()
    return inter / union if union > 0 else None


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _quant(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    v = sorted(vals)
    return float(v[min(len(v) - 1, int(p * len(v)))])


def _reliability_bins(y: np.ndarray, s: np.ndarray, bins: int = 10):
    out, ece = [], 0.0
    edges = np.linspace(0, 1, bins + 1)
    n_tot = len(y)
    for k in range(bins):
        m = (s >= edges[k]) & ((s < edges[k + 1]) if k < bins - 1 else (s <= edges[k + 1]))
        if m.sum() == 0:
            continue
        conf, acc = float(s[m].mean()), float(y[m].mean())
        ece += (m.sum() / n_tot) * abs(conf - acc)
        out.append({"bin": [round(edges[k], 2), round(edges[k + 1], 2)],
                    "n": int(m.sum()), "mean_belief": round(conf, 3),
                    "empirical_genuine": round(acc, 3)})
    return out, round(float(ece), 4)


def _fusion_metrics(
    fused: List[FusedRecord],
    rid_event: Dict[str, Optional[str]],
    rid_split: Dict[str, str],
    is_attack: Dict[str, bool],
    event_truth: Dict[str, Dict],
    split: Optional[str],
    with_bins: bool = False,
) -> Dict:
    """4a + 4d metrics over the fused records of one variant.

    ``split`` restricts scoring to the held-out test population: coverage is
    over events whose event-level split matches; record-level metrics are over
    fused records whose provenance-majority split matches. ``None`` = all.
    """
    events_all = {e for e, t in event_truth.items() if split is None or t["split"] == split}
    sel = [fr for fr in fused if split is None or _majority_split(fr, rid_split) == split]

    covered = set()
    # Believed-provenance is collected over ALL fused records (any majority
    # split, either interpretation): a test attack absorbed into a dev-majority
    # believed record is still an accepted record — matching E6's semantics.
    believed_provenance: set = set()
    for fr in fused:  # coverage credit from any fused record (event-keyed)
        if fr.belief <= 0.5:
            continue
        believed_provenance.update(fr.provenance)
        ev = _provenance_majority(fr, rid_event)
        if ev is not None and ev in events_all:
            covered.add(ev)
    coverage = len(covered) / max(1, len(events_all))

    if not sel:
        return {"coverage": round(coverage, 4), "n_fused": 0}
    confidence = float(np.mean([fr.belief for fr in sel]))
    redundancy = _redundancy_eq9(sel)
    J = 0.4 * coverage + 0.4 * confidence - 0.2 * redundancy

    # ---- 4d fused-record accuracy ----
    pos_err, t_iou, type_ok = [], [], []
    y_cal, s_cal = [], []          # calibration over affirm-interpretation records
    n_exist_ok = 0
    n_believed_affirm = n_believed_with_attack = 0
    attack_ids_in_believed = {rid for rid in believed_provenance if is_attack.get(rid)}
    for fr in sel:
        ev = _provenance_majority(fr, rid_event)
        genuine = ev is not None
        emitted_active = fr.interpretation == "affirm" and fr.belief > 0.5
        if emitted_active == genuine:
            n_exist_ok += 1
        if fr.interpretation == "affirm":
            y_cal.append(1.0 if genuine else 0.0)
            s_cal.append(fr.belief)
            if fr.belief > 0.5:
                n_believed_affirm += 1
                if any(is_attack.get(rid) for rid in fr.provenance):
                    n_believed_with_attack += 1
        if genuine:
            t = event_truth.get(ev)
            if t:
                if fr.centroid and t["centroid"]:
                    pos_err.append(haversine_nm(fr.centroid, t["centroid"]))
                iou = _interval_iou(fr.valid_start, fr.valid_end,
                                    t["valid_start"], t["valid_end"])
                if iou is not None:
                    t_iou.append(iou)
                if fr.fused_hazard_type and t["hazard_key"] and (
                        t["hazard_key"] not in WILDCARD_ATTR_VALUES):
                    type_ok.append(1.0 if fr.fused_hazard_type == t["hazard_key"] else 0.0)

    attacks_in_split = {rid for rid, a in is_attack.items()
                        if a and (split is None or rid_split.get(rid) == split)}
    y = np.asarray(y_cal)
    s = np.asarray(s_cal)
    brier = float(np.mean((s - y) ** 2)) if len(y) else None
    bins, ece = _reliability_bins(y, s) if len(y) else ([], None)

    out = {
        "coverage": round(coverage, 4),
        "confidence": round(confidence, 4),
        "redundancy": round(redundancy, 4),
        "J": round(J, 4),
        "n_fused": len(sel),
        "position_error_nm": {
            "median": round(_quant(pos_err, 0.5), 3) if pos_err else None,
            "mean": round(float(np.mean(pos_err)), 3) if pos_err else None,
            "p90": round(_quant(pos_err, 0.9), 3) if pos_err else None,
            "n": len(pos_err)},
        "temporal_iou": {
            "mean": round(float(np.mean(t_iou)), 4) if t_iou else None,
            "median": round(_quant(t_iou, 0.5), 4) if t_iou else None,
            "n": len(t_iou)},
        "existence_accuracy": round(n_exist_ok / len(sel), 4),
        "type_accuracy": {
            "acc": round(float(np.mean(type_ok)), 4) if type_ok else None,
            "n": len(type_ok)},
        "belief_brier": round(brier, 4) if brier is not None else None,
        "belief_ece": ece,
        "contamination": {
            "believed_fused_with_attack_share": round(
                n_believed_with_attack / n_believed_affirm, 4) if n_believed_affirm else None,
            "attack_records_in_believed_share": round(
                len(attack_ids_in_believed & attacks_in_split) / len(attacks_in_split), 4)
            if attacks_in_split else None},
    }
    if with_bins:
        out["reliability_bins"] = bins
    return out


def _redundancy_eq9(fused) -> float:
    """Redundancy (Section 4.5): mean pairwise similarity over fused records.

    Fused records carry geometry and a validity window but no text, so the
    composite similarity is instantiated over its available spatial and
    temporal components (renormalised), matching the intent of penalising
    near-duplicate fused outputs.
    """
    recs = [fr for fr in fused if fr.centroid is not None]
    n = len(recs)
    if n < 2:
        return 0.0
    total, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            spa = math.exp(-haversine_nm(recs[i].centroid, recs[j].centroid) / 5.0)
            tmp = _interval_iou(recs[i].valid_start, recs[i].valid_end,
                                recs[j].valid_start, recs[j].valid_end) or 0.0
            total += 0.6 * spa + 0.4 * tmp
            cnt += 1
    return total / cnt if cnt else 0.0


# --------------------------------------------------------------------------
# conflict detection + validation (4b)
# --------------------------------------------------------------------------
def _detect_all(clusters, records, sim_model, cr) -> List[Tuple[List[int], list]]:
    """Detect conflicts once per cluster; returns [(member_idxs, conflicts)]."""
    out = []
    for members in clusters.values():
        recs = [records[i] for i in members]
        crs = [cr[records[i].record_id] for i in members]
        conflicts = detect_conflicts(recs, sim_model, crs) if len(recs) > 1 else []
        out.append((members, conflicts))
    return out


def _validate_conflicts(records, cluster_conflicts) -> Dict:
    """4b: conflict detection validated against the T3 perturbation logs."""
    rid_of = {r.record_id: i for i, r in enumerate(records)}
    # index conflicts per cluster by unordered global pair
    per_mode = {m: {"co_clustered": 0, "detected_any": 0, "detected_expected": 0,
                    "critical_severity": 0, "false_type_counts": Counter()}
                for m in _EXPECTED_CONFLICT}
    n_tamper_total = sum(1 for r in records
                         if r.gt_category and r.gt_category.name == "ATTACK_TAMPER")
    n_tamper_coclustered = 0

    genuine_pairs = 0
    genuine_pairs_conflicted = 0
    genuine_conflict_types = Counter()
    edges_total = edges_attack = 0

    for members, conflicts in cluster_conflicts:
        edge_map: Dict[Tuple[int, int], list] = defaultdict(list)
        for c in conflicts:
            gi, gj = members[c.i], members[c.j]
            edge_map[(min(gi, gj), max(gi, gj))].append(c)
        edges_total += len(edge_map)
        for (gi, gj) in edge_map:
            a, b = records[gi], records[gj]
            if (a.gt_category and a.gt_category.name.startswith("ATTACK")) or (
                    b.gt_category and b.gt_category.name.startswith("ATTACK")):
                edges_attack += 1

        # genuine same-event co-clustered pairs → false-fire rate
        by_event: Dict[str, List[int]] = defaultdict(list)
        for gi in members:
            r = records[gi]
            if r.gt_event_id is not None:
                by_event[r.gt_event_id].append(gi)
        for ev, idxs in by_event.items():
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    genuine_pairs += 1
                    key = (min(idxs[x], idxs[y]), max(idxs[x], idxs[y]))
                    if key in edge_map:
                        genuine_pairs_conflicted += 1
                        for c in edge_map[key]:
                            genuine_conflict_types[c.type] += 1

        # tamper opportunities: T3 in-cluster with genuine records of its event
        member_set = set(members)
        for gi in members:
            t = records[gi]
            if not (t.gt_category and t.gt_category.name == "ATTACK_TAMPER"):
                continue
            pert = t.provenance.get("perturbation") or {}
            mode = pert.get("mode")
            if mode not in per_mode:
                continue
            if mode == "type_swap":
                src = rid_of.get(t.provenance.get("tampered_from"))
                if src is not None and str(records[src].structured_data.get(
                        "hazard_type", "")).lower() == str(pert.get("to", "")).lower():
                    continue  # no-op swap (drew the source's own value) — nothing to detect
            orig_ev = t.provenance.get("orig_event")
            partners = [gj for gj in member_set
                        if gj != gi and records[gj].gt_event_id == orig_ev]
            if not partners:
                continue
            n_tamper_coclustered += 1
            st = per_mode[mode]
            st["co_clustered"] += 1
            hit_types, hit_crit = set(), False
            for gj in partners:
                key = (min(gi, gj), max(gi, gj))
                for c in edge_map.get(key, []):
                    hit_types.add(c.type)
                    if c.severity == CRITICAL:
                        hit_crit = True
            if hit_types:
                st["detected_any"] += 1
                if hit_types & _EXPECTED_CONFLICT[mode]:
                    st["detected_expected"] += 1
                else:
                    st["false_type_counts"].update(hit_types)
                if hit_crit:
                    st["critical_severity"] += 1

    result = {"n_tamper_total": n_tamper_total,
              "n_tamper_coclustered_with_source_event": n_tamper_coclustered,
              "per_mode": {}, "genuine_pair_false_fire": {
                  "co_clustered_same_event_pairs": genuine_pairs,
                  "pairs_with_conflict": genuine_pairs_conflicted,
                  "rate": round(genuine_pairs_conflicted / genuine_pairs, 4)
                  if genuine_pairs else None,
                  "types": dict(genuine_conflict_types)},
              "edges_total": edges_total,
              "edges_attack_involved": edges_attack,
              "edges_attack_involved_share": round(edges_attack / edges_total, 4)
              if edges_total else None}
    for mode, st in per_mode.items():
        n = st["co_clustered"]
        result["per_mode"][mode] = {
            "co_clustered": n,
            "detected_any": st["detected_any"],
            "detected_expected_type": st["detected_expected"],
            "recall_any": round(st["detected_any"] / n, 4) if n else None,
            "recall_expected_type": round(st["detected_expected"] / n, 4) if n else None,
            "critical_severity_share": round(st["critical_severity"] / max(1, st["detected_any"]), 4)
            if st["detected_any"] else None,
            "unexpected_types": dict(st["false_type_counts"]),
            "expected": sorted(_EXPECTED_CONFLICT[mode]) or ["(none — negative control)"],
        }
    return result


# --------------------------------------------------------------------------
# fusion variants
# --------------------------------------------------------------------------
def _fuse_variant(cluster_conflicts, records, cr, q, method, partition=True,
                  signed=False, theta_ndup=None):
    """Fuse all clusters under one (rule, conflict-management) variant.

    Returns ``(fused_records, cluster_index_per_fused)``. ``theta_ndup``
    enables the Section 3.6 fusion-stage independence discount (the proposed
    pipeline); ``None`` fuses every member as an independent source (the
    pre-discount ablation).
    """
    fused: List[FusedRecord] = []
    cluster_of: List[int] = []
    for cidx, (members, conflicts) in enumerate(cluster_conflicts):
        recs = [records[i] for i in members]
        crs = [cr[records[i].record_id] for i in members]
        parts = (partition_cluster(len(recs), conflicts) if partition
                 else [list(range(len(recs)))])
        sub = []
        for pi, part in enumerate(parts):
            sub.append(fuse_subcluster([recs[i] for i in part],
                                       [crs[i] for i in part], q=q, method=method,
                                       record_id=f"fused-{cidx}-{pi}",
                                       signed=signed, theta_ndup=theta_ndup))
        if len(sub) > 1:
            for fr in sub:
                fr.conflict_siblings = [o.record_id for o in sub if o is not fr]
        fused.extend(sub)
        cluster_of.extend([cidx] * len(sub))
    return fused, cluster_of


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    records, sim_model, sim, smap = common.assess_benchmark(cfg, bench)
    cr = {rid: s.cr for rid, s in smap.items()}
    min_pts = cfg.get("clustering", "min_pts", default=2)
    eps_fb = cfg.get("clustering", "eps_fallback", default=0.45)
    q = cfg.get("fusion", "q_discount", default=0.6)
    # Section 3.6 independence discount (the proposed pipeline; same θ_ndup as
    # check (d)). ``None`` only if explicitly disabled in config.
    theta_nd = (cfg.get("mcam", "theta_ndup", default=0.75)
                if cfg.get("fusion", "independence_discount", default=True) else None)

    # ε on the dev-genuine submatrix; clustering over the full operational set
    eps, used_fb = common.dev_calibrated_eps(records, sim, min_pts, eps_fb)
    labels = STSDBSCAN(min_pts=min_pts, eps=eps).fit(records, sim)
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab >= 0:
            clusters.setdefault(int(lab), []).append(idx)

    cluster_conflicts = _detect_all(clusters, records, sim_model, cr)
    conflict_types = Counter()
    severity = Counter()
    n_partitions = 0
    for members, conflicts in cluster_conflicts:
        for c in conflicts:
            conflict_types[c.type] += 1
            severity[c.severity] += 1
        if len(partition_cluster(len(members), conflicts)) > 1:
            n_partitions += 1

    validation = _validate_conflicts(records, cluster_conflicts)

    rid_event = {r.record_id: r.gt_event_id for r in records}
    rid_split = {r.record_id: r.provenance.get("split", "dev") for r in records}
    is_attack = {r.record_id: bool(r.gt_category and r.gt_category.name.startswith("ATTACK"))
                 for r in records}
    event_truth = _event_truth(bench, records)
    # scoring universe: events actually present in the record set
    events_present = {r.gt_event_id for r in records if r.gt_event_id is not None}
    event_truth = {e: t for e, t in event_truth.items() if e in events_present}

    results: Dict = {
        "chosen_eps": round(eps, 4), "eps_used_fallback": used_fb,
        "eps_calibration": "dev-genuine submatrix knee (§5.6-3c)",
        "encoder": sim_model.encoder.kind,
        "n_clusters": len(clusters), "n_partitions_triggered": n_partitions,
        "conflict_type_counts": dict(conflict_types),
        "severity_counts": dict(severity),
        "conflict_validation": validation,
        "methods": {}, "methods_test": {},
        "conflict_management": {}, "zadeh_stress_test": {},
        "social_mode_e4d_test": {},
    }

    # ---- 4a + 4d across combination rules (conflict management ON) ----
    for label, method in METHODS.items():
        fused, _ = _fuse_variant(cluster_conflicts, records, cr, q, method,
                                 theta_ndup=theta_nd)
        results["methods"][label] = _fusion_metrics(
            fused, rid_event, rid_split, is_attack, event_truth, split=None)
        results["methods_test"][label] = _fusion_metrics(
            fused, rid_event, rid_split, is_attack, event_truth, split="test",
            with_bins=(method == "blend"))

    # ---- conflict-management value: partition ON vs OFF (blind / signed) ----
    # The stress condition where the rules genuinely differ is the CONTESTED
    # clusters (a CRITICAL existence conflict between substantially credible
    # parties — the Zadeh pathology). Metrics are reported on the test split
    # for the full population and for the stress subset.
    stress_clusters, mincr_dist = set(), []
    for cidx, (members, conflicts) in enumerate(cluster_conflicts):
        for c in conflicts:
            if c.severity != CRITICAL:
                continue
            a, b = records[members[c.i]], records[members[c.j]]
            pair_min = min(cr[a.record_id], cr[b.record_id])
            mincr_dist.append(round(pair_min, 3))
            if pair_min >= 0.65:
                stress_clusters.add(cidx)
    results["zadeh_stress_test"]["n_stress_clusters"] = len(stress_clusters)
    results["zadeh_stress_test"]["critical_edge_min_cr_quantiles"] = {
        "p10": _quant(mincr_dist, 0.10), "p50": _quant(mincr_dist, 0.50),
        "p90": _quant(mincr_dist, 0.90), "n_edges": len(mincr_dist)}

    cm_variants = [("partition", True, False),
                   ("no_partition_blind", False, False),
                   ("no_partition_signed", False, True)]
    for method in ("blend", "dempster", "murphy", "yager"):
        for name, part, sgn in cm_variants:
            fused, cluster_of = _fuse_variant(
                cluster_conflicts, records, cr, q, method, partition=part,
                signed=sgn, theta_ndup=theta_nd)
            key = f"{method}/{name}"
            results["conflict_management"][key] = _fusion_metrics(
                fused, rid_event, rid_split, is_attack, event_truth, split="test")
            stress_fused = [fr for fr, ci in zip(fused, cluster_of)
                            if ci in stress_clusters]
            results["zadeh_stress_test"][key] = _fusion_metrics(
                stress_fused, rid_event, rid_split, is_attack, event_truth,
                split=None) if stress_fused else {"n_fused": 0}

    # ---- social-mode comparison on fused-record accuracy ----
    now = datetime.fromisoformat(bench.meta["now"]) if bench.meta.get("now") else None
    for mode in ("polarity", "naive"):
        if mode == "polarity":
            cr_m, cc_m = cr, cluster_conflicts
        else:
            params = common.mcam_params_from_cfg(cfg, social_mode=mode)
            scores_m = common.assess(cfg, records, bench.histories, sim,
                                     params=params, now=now)
            cr_m = {s.record_id: s.cr for s in scores_m}
            cc_m = _detect_all(clusters, records, sim_model, cr_m)
        row = {}
        for name, part, sgn in (("partition", True, False),
                                ("no_partition_blind", False, False)):
            fused, _ = _fuse_variant(cc_m, records, cr_m, q, "blend",
                                     partition=part, signed=sgn,
                                     theta_ndup=theta_nd)
            row[name] = _fusion_metrics(fused, rid_event, rid_split, is_attack,
                                        event_truth, split="test")
        results["social_mode_e4d_test"][mode] = row

    # ---- Section 3.6 independence-discount value (ON = proposed vs OFF) ----
    # Full-population fused-record metrics (blend/partition, test split) plus
    # the campaign instrument: belief on Sybil-majority fused records by
    # sophistication tier. The scripted tiers are near-verbatim by
    # construction and collapse to one effective source; the paraphrased tier
    # evades the containment predicate (the paraphrased-tier residual of Section 5.1) and keeps its
    # members as separate sources — measured, not assumed.
    rid_tier = {r.record_id: (r.provenance.get("identity_tier") or "?")
                for r in records}
    rid_cat = {r.record_id: (r.gt_category.value if r.gt_category else "?")
               for r in records}
    results["independence_discount_test"] = {}
    for dlabel, th in (("discount_on", theta_nd if theta_nd is not None
                        else cfg.get("mcam", "theta_ndup", default=0.75)),
                       ("discount_off", None)):
        fused, _ = _fuse_variant(cluster_conflicts, records, cr, q, "blend",
                                 partition=True, theta_ndup=th)
        block = _fusion_metrics(fused, rid_event, rid_split, is_attack,
                                event_truth, split="test")
        tiers: Dict[str, Dict[str, List[float]]] = {}
        for fr in fused:
            syb = [rid for rid in fr.provenance
                   if rid_cat.get(rid) == "ATTACK_SYBIL"]
            if len(syb) * 2 <= len(fr.provenance):
                continue
            tier = Counter(rid_tier[rid] for rid in syb).most_common(1)[0][0]
            t = tiers.setdefault(tier, {"belief": [], "eff_ratio": []})
            t["belief"].append(fr.belief)
            t["eff_ratio"].append(fr.n_effective_sources
                                  / max(1, len(fr.provenance)))
        block["sybil_majority_by_tier"] = {
            tier: {"n": len(v["belief"]),
                   "belief_p50": round(float(np.median(v["belief"])), 4),
                   "belief_max": round(max(v["belief"]), 4),
                   "n_cross_verified": sum(1 for b in v["belief"] if b > 0.75),
                   "effective_over_member_sources": round(
                       float(np.mean(v["eff_ratio"])), 4)}
            for tier, v in sorted(tiers.items())}
        results["independence_discount_test"][dlabel] = block

    # ---- E4e: BPA discounting-factor q sweep --
    # q moderates the mass a non-credible residual commits to {H_false}
    # (Eq. (12)); q = 1.0 is the degenerate Bayesian binary assignment with
    # no explicit uncertainty mass (Section 3.6). Blend/partition, test split.
    results["q_sweep_test"] = {}
    for qv in (0.2, 0.4, 0.6, 0.8, 1.0):
        fused, _ = _fuse_variant(cluster_conflicts, records, cr, qv, "blend",
                                 partition=True, theta_ndup=theta_nd)
        mq = _fusion_metrics(fused, rid_event, rid_split, is_attack,
                             event_truth, split="test")
        results["q_sweep_test"][f"q={qv}"] = {
            k: mq.get(k) for k in ("coverage", "confidence", "J",
                                   "existence_accuracy", "belief_ece",
                                   "belief_brier")}

    common.write_results(cfg, "e4", results)
    _print_summary(results)
    return results


def _print_summary(r: Dict) -> None:
    print(f"\n[E4] ε={r['chosen_eps']} (dev-calibrated), {r['n_clusters']} clusters, "
          f"{r['n_partitions_triggered']} partitions; conflicts: {r['conflict_type_counts']}")
    v = r["conflict_validation"]
    print("[E4] 4b conflict validation (T3 perturbation logs):")
    for mode, st in v["per_mode"].items():
        print(f"     {mode:<20} co-clustered={st['co_clustered']:>3} "
              f"recall_any={st['recall_any']} expected-type={st['recall_expected_type']} "
              f"(expects {st['expected']})")
    g = v["genuine_pair_false_fire"]
    print(f"     genuine-pair false fire: {g['pairs_with_conflict']}/{g['co_clustered_same_event_pairs']}"
          f" = {g['rate']}; attack-involved edge share: {v['edges_attack_involved_share']}")
    print("\n[E4] 4a/4d (TEST split):")
    hdr = f"  {'method':<34} {'Cov':>6} {'Conf':>6} {'J':>6} {'posE_med':>8} {'tIoU':>6} {'existAcc':>8} {'ECE':>6}"
    print(hdr)
    def _fmt(v, width):
        return f"{'—' if v is None else v:>{width}}"
    for label, m in r["methods_test"].items():
        pe = (m.get("position_error_nm") or {}).get("median")
        ti = (m.get("temporal_iou") or {}).get("mean")
        print(f"  {label:<34} {m.get('coverage', 0):>6.3f} {m.get('confidence', 0):>6.3f} "
              f"{m.get('J', 0):>6.3f} {_fmt(pe, 8)} {_fmt(ti, 6)} "
              f"{_fmt(m.get('existence_accuracy'), 8)} {_fmt(m.get('belief_ece'), 6)}")
    print("\n[E4] conflict management (blend, TEST): "
          + " | ".join(f"{k.split('/', 1)[1]}: exist={m.get('existence_accuracy')} "
                       f"contam={((m.get('contamination') or {}).get('believed_fused_with_attack_share'))}"
                       for k, m in r["conflict_management"].items() if k.startswith("blend/")))
    z = r["zadeh_stress_test"]
    print(f"[E4] Zadeh stress subset: {z['n_stress_clusters']} contested clusters "
          f"(critical-edge min-cr p50={z['critical_edge_min_cr_quantiles']['p50']})")
    idt = r.get("independence_discount_test") or {}
    for dlabel, m in idt.items():
        tiers = m.get("sybil_majority_by_tier") or {}
        tier_s = ", ".join(
            f"{t}: p50={v['belief_p50']} xver={v['n_cross_verified']}/{v['n']}"
            for t, v in tiers.items())
        print(f"[E4] §4.4.2 {dlabel:<12} ECE={m.get('belief_ece')} "
              f"exist={m.get('existence_accuracy')} | Sybil-majority fused: {tier_s}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E4 (fusion + E4d).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
