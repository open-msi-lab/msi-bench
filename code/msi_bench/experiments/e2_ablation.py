"""Experiment E2 — MCAM dimension ablation (Section 5.1, Table 8).

Reports primary AUC-ROC for the full model and the four leave-one-dimension-out
variants (remaining weights renormalised by construction in :class:`MCAMParams`)
in the main-text table, with bootstrap CIs and paired-bootstrap significance
(full vs each ablation, Holm–Bonferroni). The complete 15-variant table — full,
4 leave-one-out, 6 pairwise (two active dims), 4 single-dimension — is
emitted under ``supplement_all15``.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from ..config import Config, get_config
from ..generation.benchmark import Benchmark
from ..schema import Category
from .. import stats
from . import common
from .e1_credibility import _groups_for, _scores_for, _subset

_DIMS = ("device", "content", "spatiotemporal", "social")


def _all_15_variants() -> dict:
    """name → active-dimension tuple for the full 15-variant ablation grid."""
    v = {"full": _DIMS}
    for d in _DIMS:                                   # 4 leave-one-out
        v[f"-{d}"] = tuple(x for x in _DIMS if x != d)
    for pair in itertools.combinations(_DIMS, 2):     # 6 pairwise (two active)
        v["+".join(p[:4] for p in pair)] = pair
    for d in _DIMS:                                   # 4 single-dimension
        v[f"only-{d}"] = (d,)
    return v


def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    test = [r for r in bench.records if r.provenance.get("split") == "test"]
    all_variants = _all_15_variants()
    main_names = ["full"] + [f"-{d}" for d in _DIMS]  # 5 main-text variants

    subset = _subset(test, False)
    groups = _groups_for(subset)
    results = {"variants": {}, "supplement_all15": {}}
    score_vecs: dict = {}
    y_ref = None
    for name, dims in all_variants.items():
        params = common.mcam_params_from_cfg(cfg, active_dims=dims)
        smap = {rid: {"cr": s.cr} for rid, s in
                common.assess_benchmark(cfg, bench, params=params)[3].items()}
        y, s = _scores_for(subset, smap, "cr")
        y_ref = y
        score_vecs[name] = s
        auc, lo, hi = stats.bootstrap_auc_ci_grouped(
            y, s, groups, cfg.raw["bootstrap_resamples"], seed=cfg.seed)
        entry = {"auc_primary": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                 "active_dims": list(dims)}
        results["supplement_all15"][name] = entry
        if name in main_names:
            results["variants"][name] = {k: entry[k] for k in ("auc_primary", "ci95")}
            print(f"[E2] {name:<16} AUC={auc:.4f}  CI=[{lo:.3f},{hi:.3f}]")

    full = results["variants"]["full"]["auc_primary"]
    results["delta_from_full"] = {k: round(full - v["auc_primary"], 4)
                                  for k, v in results["variants"].items() if k != "full"}

    # --- significance: full vs each main ablation (paired bootstrap + Holm–Bonferroni) ---
    abl_names = main_names[1:]
    raw_p, diffs = [], []
    for k in abl_names:
        d, p = stats.paired_bootstrap_auc_diff(
            y_ref, score_vecs["full"], score_vecs[k], cfg.raw["bootstrap_resamples"], seed=cfg.seed)
        diffs.append(d); raw_p.append(p)
    adjusted, reject = stats.holm_bonferroni(raw_p)
    results["full_vs_ablation_significance"] = {
        k: {"auc_drop": round(d, 4), "p_paired": round(p, 4), "p_holm": round(a, 4),
            "significant_at_0.05": bool(r)}
        for k, d, p, a, r in zip(abl_names, diffs, raw_p, adjusted, reject)}
    print("[E2] full vs ablation (paired bootstrap, Holm–Bonferroni):")
    for k in abl_names:
        m = results["full_vs_ablation_significance"][k]
        print(f"     {k:<16} ΔAUC={m['auc_drop']:+.4f} p_holm={m['p_holm']:.4f} "
              f"{'*' if m['significant_at_0.05'] else ''}")
    common.write_results(cfg, "e2", results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run Experiment E2 (ablation).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
