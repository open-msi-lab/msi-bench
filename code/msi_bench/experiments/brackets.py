"""Schema-aligned vs schema-degraded brackets (Section 4.6).

The central mitigation for the "crowd reports inherit official structure" objection
is to report BOTH brackets: an optimistic schema-aligned variant (structured
attributes extracted post-generation) and a pessimistic schema-degraded variant
(raw text only; the pipeline re-derives Type/Spatial/Temporal). This runner builds
both benchmarks at a common scale and reports the headline-metric gap:

    * E1 — primary AUC-ROC (credibility discrimination)
    * E3 — STS-DBSCAN ARI (clustering quality, dev-calibrated ε, test split)
    * E4 — composite fusion J (Proposed conflict-aware blend)

It reuses the actual E1/E3/E4 runners, so the bracket numbers are identical to
those experiments. The degraded run is executed first and the aligned run last,
so the canonical ``results/e1|e3|e4`` directories retain the aligned (primary)
numbers while ``results/brackets`` holds both plus the gap.

Run with::

    python -m msi_bench.experiments.brackets --limit-events 150
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ..config import Config, get_config
from ..generation.benchmark import build_benchmark
from . import common, e1_credibility, e3_clustering, e4_fusion


def _headline(e1_res: dict, e3_res: dict, e4_res: dict) -> dict:
    e4_methods = e4_res.get("methods", {})
    proposed = next((m for k, m in e4_methods.items() if "Proposed" in k), {})
    return {
        "e1_auc_primary": e1_res.get("auc_roc_primary", {}).get("auc"),
        "e3_sts_ARI": e3_res.get("methods", {}).get("STS-DBSCAN (ours)", {}).get("ARI"),
        "e4_proposed_J": proposed.get("J"),
    }


def _run_one(cfg: Config, bench_dir: str) -> dict:
    return _headline(e1_credibility.run(cfg, bench_dir),
                     e3_clustering.run(cfg, bench_dir),
                     e4_fusion.run(cfg, bench_dir))


def run(cfg: Config, limit_events: Optional[int] = None) -> dict:
    aligned_dir = str(cfg.output_dir / "benchmark")
    degraded_dir = str(cfg.output_dir / "benchmark_degraded")

    print("[brackets] building schema-degraded benchmark …")
    build_benchmark(cfg, degraded=True, limit_events=limit_events,
                    verbose=False).save(Path(degraded_dir))
    print("[brackets] building schema-aligned benchmark …")
    build_benchmark(cfg, degraded=False, limit_events=limit_events,
                    verbose=False).save(Path(aligned_dir))

    print("\n===== brackets: SCHEMA-DEGRADED =====")
    degraded = _run_one(cfg, degraded_dir)
    print("\n===== brackets: SCHEMA-ALIGNED =====")
    aligned = _run_one(cfg, aligned_dir)

    def gap(key):
        a, d = aligned.get(key), degraded.get(key)
        return round(a - d, 4) if (a is not None and d is not None) else None

    results = {
        "schema_aligned": aligned,
        "schema_degraded": degraded,
        "aligned_minus_degraded": {k: gap(k) for k in aligned},
    }
    print("\n[brackets] headline gap (aligned − degraded):")
    for k in aligned:
        print(f"  {k:<18} aligned={aligned[k]}  degraded={degraded[k]}  "
              f"gap={results['aligned_minus_degraded'][k]}")
    common.write_results(cfg, "brackets", results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run the schema-aligned vs degraded brackets.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-events", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    run(cfg, limit_events=args.limit_events)


if __name__ == "__main__":
    main()
