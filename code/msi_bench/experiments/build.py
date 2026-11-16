"""Build and persist MSI-Bench (Section 4.3).

Constructs the benchmark from the real archive + synthetic overlay and writes it
to ``<output_dir>/benchmark/`` (records, events, histories, meta, DATASHEET).
Uses the configured LLM generators, or the deterministic mock when LLM use is
disabled or no API key is present.

Run with::

    python -m msi_bench.experiments.build                 # full benchmark
    python -m msi_bench.experiments.build --limit-events 150   # quick subset
    python -m msi_bench.experiments.build --degraded      # schema-degraded variant
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from ..config import get_config
from ..generation.benchmark import build_benchmark


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build MSI-Bench.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="benchmark", help="Subdir under output_dir.")
    ap.add_argument("--limit-events", type=int, default=None)
    ap.add_argument("--degraded", action="store_true",
                    help="Schema-degraded crowd variant (raw text only).")
    ap.add_argument("--offline", action="store_true",
                    help="Replay generation from the response cache only "
                         "(no key/SDK/network); error on a cache miss.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Parallel generation workers (overrides llm.concurrency).")
    args = ap.parse_args(argv)

    cfg = get_config(args.config)
    if args.offline:
        cfg.raw.setdefault("llm", {})["offline"] = True
    if args.concurrency is not None:
        cfg.raw.setdefault("llm", {})["concurrency"] = args.concurrency
    bench = build_benchmark(cfg, degraded=args.degraded, limit_events=args.limit_events)
    out = cfg.output_dir / args.out
    bench.save(out)
    print(f"[build] saved benchmark to {out}")
    print(f"[build] meta: {bench.meta}")


if __name__ == "__main__":
    main()
