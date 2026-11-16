"""Experiment E3-R — Real-Data Chain Recovery (Section 5.2, Table 9).

Clusters official 2022–2023 NEW_WARNING/UPDATE messages on 100% real data, with
lifecycle-reference lines masked from the text and chain metadata hidden from
the algorithm. Ground truth = Layer-1 lifecycle chains (built without any
similarity model, so they cannot inflate the score). Reports ARI/NMI/V-measure
for STS-DBSCAN against spatial-only, spatiotemporal, composite-without-semantic-
constraint, and agglomerative baselines, all sharing the chosen ε.

This experiment needs NO LLM/API access. Run with::

    python -m msi_bench.experiments.e3r_real_clustering
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

import numpy as np

from ..clustering import STSDBSCAN
from ..config import Config, get_config
from ..parsing.chains import build_chains
from ..parsing.pipeline import parse_corpus
from ..schema import GeometryKind, MSIRecord, MessageFunction, ProviderType
from ..similarity import SimilarityModel, TextEncoder
from .benchmark_mask import mask_lifecycle  # local helper
from . import common


def _records_and_gt(cfg: Config, limit: Optional[int]):
    msgs_all = parse_corpus(cfg, cfg.years("benchmark"), nav_area=cfg.nav_area,
                            functions=[MessageFunction.NEW_WARNING, MessageFunction.UPDATE])
    # Build Layer-1 chains over the FULL discrete set (identical to E0), so the
    # ground-truth event partition matches |E|; then cluster only the
    # coordinate-bearing messages, carrying their true chain id as the label.
    chains = build_chains(msgs_all)
    msgs = [m for m in msgs_all if m.centroid is not None]
    if limit:
        msgs = msgs[:limit]
    records: List[MSIRecord] = []
    gt: List[str] = []
    for m in msgs:
        records.append(MSIRecord(
            record_id=m.uid, event_type=m.event_type, provider_type=ProviderType.OFFICIAL,
            provider_id=m.authority_norm, authority_norm=m.authority_norm,
            raw_text=mask_lifecycle(m.raw_text),
            structured_data={}, geometry_kind=m.geometry_kind,
            coordinates=m.coordinates, centroid=m.centroid,
            issue_time=m.issue_time, valid_start=m.issue_time,
            valid_end=m.cancel_time or m.self_expiry_time,
            observation_type="shore", submission_position=m.centroid,
        ))
        gt.append(chains.chain_id_of[m.uid])
    return records, gt


def _variant_matrix(cfg, records, *, w_s, w_t, w_m, ours: bool = False):
    # Encoder fitted on the 2012–2021 official corpus (split discipline), not
    # on the evaluation messages. NAVAREA warnings are predominantly areal, so
    # the headline real-data result uses the EXACT pairwise matrix (geodesic
    # polygon–polygon Jaccard) rather than the vectorised centroid-kernel
    # approximation — the areal similarity component is thus exercised here.
    # The proposed composite uses the succession-aware gate (Section 3.4);
    # external baselines run in their published additive forms.
    kwargs = {} if ours else {"temporal_mode": "additive", "tau_succ_h": None}
    model = common.make_similarity(
        cfg, sigma_p=5.0,
        weights={"spatial": w_s, "temporal": w_t, "semantic": w_m}, **kwargs)
    return model, model.pairwise_matrix(records)


def run(cfg: Config, limit: Optional[int] = None) -> dict:
    from ..clustering import adaptive_epsilon
    print("[E3-R] building real-data record set (masked text, hidden chains) …")
    records, gt = _records_and_gt(cfg, limit)
    n = len(records)
    n_events = len(set(gt))
    print(f"[E3-R] {n} messages; {n_events} ground-truth chains.")

    # Composite similarity (full) and component variants for baselines.
    model, composite = _variant_matrix(cfg, records, w_s=0.40, w_t=0.25, w_m=0.35,
                                       ours=True)
    _, spatial = _variant_matrix(cfg, records, w_s=1.0, w_t=0.0, w_m=0.0)
    _, st = _variant_matrix(cfg, records, w_s=0.6, w_t=0.4, w_m=0.0)

    # Split discipline: chains are split
    # CHAIN-level into dev/test with the benchmark dev fraction; each method
    # calibrates its ε at the knee of its own dev submatrix and is scored on
    # the held-out test chains only. Chain-level (not message-level) splitting
    # keeps every ground-truth lifecycle intact on one side.
    import random as _random
    dev_frac = float(cfg.get("benchmark", "dev_test_split", default=0.70))
    chain_ids = sorted(set(gt))
    _random.Random(int(cfg.get("seed", default=20260613))).shuffle(chain_ids)
    dev_chains = set(chain_ids[: int(round(dev_frac * len(chain_ids)))])
    dev_idx = [i for i, g in enumerate(gt) if g in dev_chains]
    test_idx = [i for i, g in enumerate(gt) if g not in dev_chains]
    test_recs = [records[i] for i in test_idx]
    gt_test = [gt[i] for i in test_idx]
    min_pts = cfg.get("clustering", "min_pts", default=2)
    eps_fb = cfg.get("clustering", "eps_fallback", default=0.45)

    def dev_eps(mat):
        e, _, fb = adaptive_epsilon(mat[np.ix_(dev_idx, dev_idx)], min_pts, eps_fb)
        return e, fb

    def sub(mat):
        return mat[np.ix_(test_idx, test_idx)]

    eps, used_fallback = dev_eps(composite)
    eps_spatial, fb_spatial = dev_eps(spatial)
    eps_st, fb_st = dev_eps(st)
    print(f"[E3-R] dev-calibrated ε: ours {eps:.3f} (fb {used_fallback}) | "
          f"spatial {eps_spatial:.3f} | st {eps_st:.3f}")

    sts = STSDBSCAN(min_pts=min_pts, eps=eps)
    sts_labels = sts.fit(test_recs, sub(composite))

    results = {
        "n_messages": n, "n_ground_truth_chains": n_events,
        "n_test_messages": len(test_idx), "n_test_chains": len(chain_ids) - len(dev_chains),
        "split": {"dev_fraction": dev_frac, "level": "chain",
                  "seed": int(cfg.get("seed", default=20260613))},
        "temporal_mode": model.temporal_mode,
        "encoder": model.encoder.kind,
        "chosen_eps": round(eps, 4), "used_fallback": used_fallback,
        "eps_by_method": {
            "STS-DBSCAN (ours)": round(eps, 4),
            "DBSCAN-Spatial": round(eps_spatial, 4),
            "ST-DBSCAN": round(eps_st, 4),
            "DBSCAN-Composite (no semantic constraint)": round(eps, 4),
            "Agglomerative": round(eps, 4)},
        "methods": {},
    }
    nc_labels = common.dbscan_on_sim(sub(composite), eps)
    results["methods"]["STS-DBSCAN (ours)"] = common.cluster_metrics(gt_test, sts_labels)
    results["methods"]["DBSCAN-Spatial"] = common.cluster_metrics(
        gt_test, common.dbscan_on_sim(sub(spatial), eps_spatial))
    results["methods"]["ST-DBSCAN"] = common.cluster_metrics(
        gt_test, common.dbscan_on_sim(sub(st), eps_st))
    results["methods"]["DBSCAN-Composite (no semantic constraint)"] = common.cluster_metrics(
        gt_test, nc_labels)
    results["methods"]["Agglomerative"] = common.cluster_metrics(
        gt_test, _agglomerative(sub(composite), eps))
    results["constraint_effect"] = _constraint_effect(test_recs, gt_test,
                                                      sts_labels, nc_labels)

    print("\n[E3-R] Clustering quality on real NAVAREA IV lifecycle chains (test):")
    print(f"  {'method':<42} {'ARI':>7} {'NMI':>7} {'V':>7}")
    for name, m in results["methods"].items():
        print(f"  {name:<42} {m['ARI']:>7.3f} {m['NMI']:>7.3f} {m['V']:>7.3f}")
    ce = results["constraint_effect"]
    print(f"[E3-R] semantic constraint: {ce['type_incompatible_pairs_prevented']} "
          f"type-incompatible pair-merges prevented "
          f"({ce['prevented_correctly_pct']}% correctly — different true chains); "
          f"{ce['true_chain_pairs_lost_to_constraint']} true chain pairs lost.")
    common.write_results(cfg, "e3r", results)
    return results


def _constraint_effect(records, gt, ours_labels, nc_labels) -> dict:
    """Quantify what the semantic-compatibility constraint DOES: the
    type-incompatible merges it prevents explicitly.

    A *prevented pair* is co-clustered by the unconstrained composite but
    separated by STS-DBSCAN and type-incompatible (Wu–Palmer < θ_compat).
    Each is 'correct' when the two messages truly belong to different chains,
    a 'loss' when the constraint split a genuine chain pair.
    """
    from ..similarity import type_similarity
    prevented = correct = 0
    lost_true_pairs = 0
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            same_nc = nc_labels[i] == nc_labels[j] and nc_labels[i] != -1
            same_ours = ours_labels[i] == ours_labels[j] and ours_labels[i] != -1
            if not same_nc or same_ours:
                continue
            if type_similarity(records[i].event_type, records[j].event_type) < 0.5:
                prevented += 1
                if gt[i] != gt[j]:
                    correct += 1
                else:
                    lost_true_pairs += 1
    return {
        "type_incompatible_pairs_prevented": prevented,
        "prevented_correctly": correct,
        "prevented_correctly_pct": round(100.0 * correct / prevented, 1) if prevented else None,
        "true_chain_pairs_lost_to_constraint": lost_true_pairs,
    }


def _agglomerative(sim: np.ndarray, eps: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - eps,
        metric="precomputed", linkage="average")
    return model.fit_predict(dist)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E3-R (real-data clustering).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap messages (debug).")
    args = ap.parse_args(argv)
    run(get_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
