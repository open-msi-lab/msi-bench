"""Unified CLI dispatcher for MSI-Bench experiments.

Examples
--------
    python -m msi_bench.experiments.run e0
    python -m msi_bench.experiments.run e3r
    python -m msi_bench.experiments.run build --limit-events 150
    python -m msi_bench.experiments.run e1 --benchmark results/benchmark
    python -m msi_bench.experiments.run all --limit-events 150   # build + E0..E7
    python -m msi_bench.experiments.run all --resume             # skip finished stages (crash resume)
    python -m msi_bench.experiments.run generate                 # PHASE 1 (online PC): all LLM calls -> cache
    python -m msi_bench.experiments.run compute                  # PHASE 2 (offline): replay cache, heavy compute
    python -m msi_bench.experiments.run freeze                   # pin build -> benchmark_release/

``e0`` and ``e3r`` run on the real archive and need no benchmark or API key.
The benchmark-dependent experiments (e1, e2, e3, e4, e5, e6, e7) require a built
benchmark; ``all`` builds one first.

Generation vs computation are also offered as two explicit phases for the
network-split workflow: ``generate`` performs every LLM call once
on a networked machine (builds both benchmark variants and warms the judge),
populating ``llm_cache.jsonl``; ``compute`` then replays that cache fully offline
and runs all the heavy experiments. Copy ``results/`` + the cache between them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import get_config
from . import (brackets, common, e0_characterize, e1_credibility, e2_ablation,
               e3_clustering, e3r_real_clustering, e4_fusion, e5_sensitivity,
               e6_endtoend, e7_paradigms, freeze, validate_benchmark)
from .build import main as build_main

_BENCH_EXPS = {
    "e1": e1_credibility, "e2": e2_ablation, "e3": e3_clustering,
    "e4": e4_fusion, "e5": e5_sensitivity, "e6": e6_endtoend, "e7": e7_paradigms,
    "fidelity": validate_benchmark,
}


def _bench_dir(cfg, name="benchmark") -> str:
    return str(cfg.output_dir / name)


def _run_all(cfg, args) -> None:
    """Build + E0..E7 + fidelity + brackets. With ``--resume``, skip stages whose
    up-to-date output already exists (crash resume); LLM generation always resumes
    from the response cache regardless."""
    bdir = _bench_dir(cfg)
    records_json = Path(bdir) / "records.json"

    def _done(sentinel: Path, fresh_against: Path = None) -> bool:
        """Under --resume, is this stage already complete and up to date?

        A stage is skippable when its output exists and (for benchmark-derived
        stages) is newer than the benchmark, so a rebuilt benchmark forces a
        recompute. Without --resume, nothing is skipped (unchanged behaviour).
        """
        if not (args.resume and sentinel.is_file()):
            return False
        if (fresh_against and fresh_against.is_file()
                and sentinel.stat().st_mtime < fresh_against.stat().st_mtime):
            return False
        return True

    if _done(cfg.output_dir / "e0" / "e0_summary.json"):
        print("[all] skip E0 (cached)")
    else:
        e0_characterize.run(cfg)
    if _done(cfg.output_dir / "e3r" / "e3r_results.json"):
        print("[all] skip E3-R (cached)")
    else:
        e3r_real_clustering.run(cfg)

    if _done(records_json):
        print(f"[all] skip build (benchmark present at {bdir})")
    else:
        from ..generation.benchmark import build_benchmark
        bench = build_benchmark(cfg, degraded=args.degraded, limit_events=args.limit_events)
        bench.save(Path(bdir))

    for name, mod in _BENCH_EXPS.items():
        if _done(cfg.output_dir / name / f"{name}_results.json", fresh_against=records_json):
            print(f"\n===== {name.upper()} (skip, cached) =====")
            continue
        print(f"\n===== {name.upper()} =====")
        mod.run(cfg, bdir)
    # Both schema brackets: rebuilds aligned + degraded and
    # reports the headline gap. Leaves the canonical dirs holding the aligned
    # (primary) numbers.
    if _done(cfg.output_dir / "brackets" / "brackets_results.json", fresh_against=records_json):
        print("\n===== BRACKETS (skip, cached) =====")
    else:
        print("\n===== BRACKETS (schema-aligned vs degraded) =====")
        brackets.run(cfg, limit_events=args.limit_events)


def _run_generate(cfg, args) -> None:
    """PHASE 1 — networked machine: perform every LLM call the pipeline will ever
    need (build both benchmark variants + warm the LLM-as-judge), populating the
    response cache. No heavy experiment computation; pair with ``compute``."""
    cfg.raw.setdefault("llm", {})["offline"] = False  # generation must reach the model
    if not bool(cfg.get("llm", "enabled", default=False)):
        print("[generate] NOTE: llm.enabled=false → mock generator; the cache will "
              "hold templated text, not real model output.")
    from ..generation.benchmark import build_benchmark
    bdir, ddir = _bench_dir(cfg), _bench_dir(cfg, "benchmark_degraded")
    print("[generate] (1/5) schema-aligned benchmark …")
    base = build_benchmark(cfg, degraded=False, limit_events=args.limit_events)
    base.save(Path(bdir))
    print("[generate] (2/5) schema-degraded benchmark …")
    build_benchmark(cfg, degraded=True, limit_events=args.limit_events).save(Path(ddir))
    print("[generate] (3/5) warming LLM-as-judge cache …")
    n = e7_paradigms.warm_judge_cache(cfg, bdir)
    # (4/5) + (5/5): warm the caches the compute phase's *rebuilding* stages need
    # so `compute` runs fully offline — E1's multi-seed variance study (Section 4.5) and
    # E6's 2024 temporal holdout regenerate the overlay (fresh seeds /
    # the 2024 data) and so need LLM calls the base build never made. The build
    # specs come from common.* — the same helpers E1/E6 use — so the warmed cache
    # keys match what those experiments rebuild (no drift). Seed 0 of the sweep is
    # the base build and replays as cache hits.
    n_events = base.meta["n_events"]
    seeds = common.multiseed_seeds(cfg)
    print(f"[generate] (4/5) multi-seed overlays for the E1 variance study "
          f"(seeds {seeds[0]}…{seeds[-1]}) …")
    for seed in seeds:
        build_benchmark(cfg, verbose=False, **common.multiseed_build_kwargs(n_events, seed))
    print("[generate] (5/5) 2024 temporal holdout for E6 …")
    build_benchmark(cfg, verbose=False, **common.holdout_build_kwargs(cfg))
    cache_file = cfg.cache_dir / "llm_cache.jsonl"
    print(f"\n[generate] done — judge warmed over {n} records; multi-seed + holdout "
          f"overlays generated. The cache now holds every LLM response the full "
          f"compute phase needs (E0–E7 + brackets + multi-seed + holdout).\n"
          f"  Next: copy results/ and {cache_file} to the compute machine, then run\n"
          f"        python -m msi_bench.experiments.run compute")


def _run_compute(cfg, args) -> None:
    """PHASE 2 — offline machine: replay the cache (no key/SDK/network, clear error
    on a miss) and run all heavy computation (E0..E7 + fidelity + brackets).

    Implies ``--resume``: the ``generate``-built benchmark is reused and finished
    stages are skipped; delete a ``results/<stage>`` dir to force its recompute."""
    cfg.raw.setdefault("llm", {})["offline"] = True
    if not (Path(_bench_dir(cfg)) / "records.json").is_file():
        print("[compute] NOTE: no prebuilt benchmark found; it will be rebuilt by "
              "replaying the cache. If this errors on a cache miss, run `generate` "
              "first on a networked machine and copy the cache + results over.")
    args.resume = True
    _run_all(cfg, args)


def main() -> None:
    ap = argparse.ArgumentParser(description="MSI-Bench experiment dispatcher.")
    ap.add_argument("experiment", choices=["e0", "e3r", "build", "all", "generate",
                                           "compute", "brackets", "freeze",
                                           *list(_BENCH_EXPS)])
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default=None, help="Built benchmark directory.")
    ap.add_argument("--limit-events", type=int, default=None)
    ap.add_argument("--degraded", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="In `all`: skip stages whose up-to-date output already "
                         "exists, to resume a crashed pipeline. (LLM generation "
                         "always resumes from the response cache regardless.)")
    ap.add_argument("--offline", action="store_true",
                    help="Replay LLM generation/judging from the response cache "
                         "only (no key/SDK/network); error on a cache miss. For a "
                         "machine without network access.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Parallel generation workers (overrides llm.concurrency). "
                         "Speeds up a cold real build; output is unchanged.")
    args = ap.parse_args()
    cfg = get_config(args.config)
    if args.offline:
        cfg.raw.setdefault("llm", {})["offline"] = True
    if args.concurrency is not None:
        cfg.raw.setdefault("llm", {})["concurrency"] = args.concurrency
    exp = args.experiment

    if exp == "e0":
        e0_characterize.run(cfg)
        return
    if exp == "e3r":
        e3r_real_clustering.run(cfg)
        return
    if exp == "brackets":
        brackets.run(cfg, limit_events=args.limit_events)
        return
    if exp == "freeze":
        freeze.run(cfg)
        return
    if exp == "build":
        argv = []
        if args.limit_events:
            argv += ["--limit-events", str(args.limit_events)]
        if args.degraded:
            argv += ["--degraded"]
        if args.offline:
            argv += ["--offline"]
        if args.concurrency is not None:
            argv += ["--concurrency", str(args.concurrency)]
        build_main(argv)
        return

    if exp == "generate":
        _run_generate(cfg, args)
        return
    if exp == "compute":
        _run_compute(cfg, args)
        return
    if exp == "all":
        _run_all(cfg, args)
        return

    bdir = args.benchmark or _bench_dir(cfg)
    if not Path(bdir).exists():
        # Fall back to a frozen release so a fresh clone can run
        # the experiments without rebuilding (and without an API key).
        frozen = cfg.project_root / "benchmark_release"
        if not args.benchmark and (frozen / "records.json").is_file():
            print(f"[run] {bdir} not built; using frozen benchmark at {frozen}")
            bdir = str(frozen)
        else:
            sys.exit(f"Benchmark not found at {bdir}. Run `build` first "
                     f"(python -m msi_bench.experiments.run build), or `freeze` a build.")
    _BENCH_EXPS[exp].run(cfg, bdir)


if __name__ == "__main__":
    main()
