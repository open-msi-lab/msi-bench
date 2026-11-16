"""Experiment E3 — STS-DBSCAN clustering on the multi-source overlay (Section 5.2).

Complements the fully-real E3-R: here STS-DBSCAN groups the heterogeneous
multi-source genuine reports (evaluated against their ground-truth event id) and
is compared to spatial-only, spatiotemporal, composite-without-constraint, and
agglomerative baselines. The proposed composite uses the succession-aware
temporal-compatibility gate (Section 3.4); the external baselines run
in their published additive forms. Each method selects its own adaptive ε at the
knee of its OWN dev-subset similarity curve (a shared ε is miscalibrated for
matrices with different value scales). Also reports the noise-classification
split (attacks should fall to noise more than genuine reports).
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np

from ..clustering import NOISE, STSDBSCAN
from ..config import Config, get_config
from ..generation.benchmark import Benchmark
from ..schema import Category
from ..similarity import SimilarityModel, TextEncoder
from . import common


def _model(cfg, *, w_s, w_t, w_m, mode=None, succession=True):
    kwargs = {} if succession else {"tau_succ_h": None}
    return common.make_similarity(
        cfg, sigma_p=5.0, weights={"spatial": w_s, "temporal": w_t, "semantic": w_m},
        temporal_mode=mode, **kwargs)


def run(cfg: Config, benchmark_dir: str) -> dict:
    from ..clustering import adaptive_epsilon
    bench = Benchmark.load(benchmark_dir)
    records = bench.records
    min_pts = cfg.get("clustering", "min_pts", default=2)
    eps_fb = cfg.get("clustering", "eps_fallback", default=0.45)
    # Proposed composite: config temporal mode (gate). External baselines: their
    # published additive forms, without the succession kernel.
    model = _model(cfg, w_s=0.40, w_t=0.25, w_m=0.35)
    composite = model.pairwise_matrix_fast(records)
    spatial = _model(cfg, w_s=1.0, w_t=0.0, w_m=0.0, mode="additive",
                     succession=False).pairwise_matrix_fast(records)
    st = _model(cfg, w_s=0.6, w_t=0.4, w_m=0.0, mode="additive",
                succession=False).pairwise_matrix_fast(records)

    # Split discipline (Section 4.5): every method calibrates its ε on the dev
    # genuine subset of its OWN matrix and is scored on the held-out TEST subset.
    dev_idx = [i for i, r in enumerate(records)
               if r.gt_event_id is not None and r.provenance.get("split") == "dev"]
    test_idx = [i for i, r in enumerate(records)
                if r.gt_event_id is not None and r.provenance.get("split") == "test"]

    def dev_eps(mat):
        if not dev_idx:
            return eps_fb, True
        e, _, fb = adaptive_epsilon(mat[np.ix_(dev_idx, dev_idx)], min_pts, eps_fb)
        return e, fb

    eps, used_fallback = dev_eps(composite)
    eps_spatial, fb_spatial = dev_eps(spatial)
    eps_st, fb_st = dev_eps(st)

    test_recs = [records[i] for i in test_idx]
    gt = [records[i].gt_event_id for i in test_idx]

    def sub(mat):
        return mat[np.ix_(test_idx, test_idx)]

    def eval_sts():
        labels = STSDBSCAN(min_pts=min_pts, eps=eps).fit(test_recs, sub(composite))
        return common.cluster_metrics(gt, labels), labels

    sts_metrics, sts_labels = eval_sts()
    results = {"chosen_eps": round(eps, 4), "used_fallback": used_fallback,
               "temporal_mode": model.temporal_mode,
               "encoder": model.encoder.kind,
               "eps_by_method": {
                   "STS-DBSCAN (ours)": round(eps, 4),
                   "DBSCAN-Spatial": round(eps_spatial, 4),
                   "ST-DBSCAN": round(eps_st, 4),
                   "DBSCAN-Composite (no constraint)": round(eps, 4)},
               "eps_fallback_by_method": {
                   "STS-DBSCAN (ours)": used_fallback,
                   "DBSCAN-Spatial": fb_spatial,
                   "ST-DBSCAN": fb_st,
                   "DBSCAN-Composite (no constraint)": used_fallback},
               "n_test_genuine": len(test_idx), "methods": {}}
    results["methods"]["STS-DBSCAN (ours)"] = sts_metrics
    results["methods"]["DBSCAN-Spatial"] = common.cluster_metrics(
        gt, common.dbscan_on_sim(sub(spatial), eps_spatial))
    results["methods"]["ST-DBSCAN"] = common.cluster_metrics(
        gt, common.dbscan_on_sim(sub(st), eps_st))
    results["methods"]["DBSCAN-Composite (no constraint)"] = common.cluster_metrics(
        gt, common.dbscan_on_sim(sub(composite), eps))

    # noise classification: attacks vs genuine landing in noise. Run on the
    # full record set at the dev-calibrated ε (a security view, not an ARI score).
    full_labels = STSDBSCAN(min_pts=min_pts, eps=eps).fit(records, composite)
    attack_idx = [i for i, r in enumerate(records)
                  if r.gt_category and r.gt_category.name.startswith("ATTACK")]
    genuine_idx = [i for i, r in enumerate(records)
                   if r.gt_category and not r.gt_category.name.startswith("ATTACK")]
    results["noise_classification"] = {
        "attack_noise_rate": round(float(np.mean([full_labels[i] == NOISE for i in attack_idx])), 4)
        if attack_idx else None,
        "genuine_noise_rate": round(float(np.mean([full_labels[i] == NOISE for i in genuine_idx])), 4)
        if genuine_idx else None,
    }

    print("\n[E3] Clustering quality (genuine multi-source reports):")
    for name, m in results["methods"].items():
        print(f"  {name:<38} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} V={m['V']:.3f}")
    print(f"[E3] noise: attacks {results['noise_classification']['attack_noise_rate']} "
          f"vs genuine {results['noise_classification']['genuine_noise_rate']}")
    common.write_results(cfg, "e3", results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E3 (clustering).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
