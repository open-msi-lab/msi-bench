"""Benchmark fidelity validation (Section 4.6, reported as results).

Compares the synthetic GENUINE_CROWD text against the real official-warning
corpus on quantitative fidelity axes — length distribution (KS test),
type–token ratio, and out-of-vocabulary rate against the official lexicon —
plus a simple artifact screen (AI-disclosure phrases, exact duplicates). The
synthetic crowd text should be measurably *less* formal (shorter, higher OOV)
than official text; if it is not, generation has failed.

Run with::

    python3 -m msi_bench.experiments.validate_benchmark --benchmark results/benchmark
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..config import Config, get_config
from ..generation.benchmark import Benchmark
from ..schema import Category
from . import common

_AI_ARTIFACTS = re.compile(
    r"as an ai|language model|i cannot|i'm sorry|here is a|certainly[,!]", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _stats(texts: Sequence[str]) -> Dict[str, float]:
    lengths = [len(_tokens(t)) for t in texts if t]
    all_tok = [w for t in texts for w in _tokens(t)]
    ttr = len(set(all_tok)) / len(all_tok) if all_tok else 0.0
    import numpy as np
    return {"n": len(lengths),
            "mean_words": round(float(np.mean(lengths)), 2) if lengths else 0.0,
            "median_words": round(float(np.median(lengths)), 2) if lengths else 0.0,
            "type_token_ratio": round(ttr, 4)}


def run(cfg: Config, benchmark_dir: str) -> dict:
    bench = Benchmark.load(benchmark_dir)
    crowd = [r.raw_text for r in bench.records if r.gt_category == Category.GENUINE_CROWD]
    official = common.embedding_corpus(cfg)  # real 2012–2021 official text

    crowd_stats = _stats(crowd)
    official_stats = _stats(official)

    # KS test on word-length distributions (scipy if available)
    ks = None
    try:
        from scipy.stats import ks_2samp
        cl = [len(_tokens(t)) for t in crowd if t]
        ol = [len(_tokens(t)) for t in official if t]
        if cl and ol:
            stat, p = ks_2samp(cl, ol)
            ks = {"statistic": round(float(stat), 4), "p_value": round(float(p), 6)}
    except Exception:
        pass

    # OOV rate of synthetic crowd against the official lexicon
    off_vocab = set(w for t in official for w in _tokens(t))
    crowd_tok = [w for t in crowd for w in _tokens(t)]
    oov = sum(1 for w in crowd_tok if w not in off_vocab) / len(crowd_tok) if crowd_tok else 0.0

    # artifact screen
    n_ai = sum(1 for t in crowd if _AI_ARTIFACTS.search(t or ""))
    dup = len(crowd) - len(set(crowd))

    results = {
        "crowd": crowd_stats,
        "official": official_stats,
        "length_ks_test": ks,
        "crowd_oov_rate_vs_official": round(oov, 4),
        "less_formal_than_official":
            crowd_stats["mean_words"] < official_stats["mean_words"],
        "artifact_screen": {
            "ai_disclosure_hits": n_ai,
            "exact_duplicates": dup,
            "ai_disclosure_rate": round(n_ai / max(1, len(crowd)), 4),
        },
        "attack_content_shape_parity": _content_shape_parity(bench),
    }
    print("[fidelity] crowd:", crowd_stats)
    print("[fidelity] official:", official_stats)
    print("[fidelity] length KS:", ks, "| crowd OOV vs official:", round(oov, 4))
    print("[fidelity] less formal than official:", results["less_formal_than_official"],
          "| AI-artifact hits:", n_ai, "| duplicates:", dup)
    for k, v in results["attack_content_shape_parity"].items():
        print(f"[fidelity] shape parity {k}: {v}")
    common.write_results(cfg, "fidelity", results)
    return results


def _content_shape_parity(bench) -> Dict[str, Dict]:
    """Content-shape parity of the LLM-text attacks against genuine crowd —
    word-length distribution and structured field counts. The length-anchored
    prompts and the platform-extractor parity are accepted when the
    sophisticated/farmed tiers' medians sit inside the genuine interquartile
    band and the field counts spread over the genuine 1–3 range instead of a
    fixed {2,3}. Reported per population; the NAIVE tier is expected to stay
    short — low effort is its designed signature, so it is listed for
    contrast, not gated.
    """
    from collections import Counter
    import numpy as np

    def _shape(recs):
        words = sorted(len(_tokens(r.raw_text)) for r in recs)
        if not words:
            return None
        fc = Counter(len(r.structured_data) for r in recs)
        return {"n": len(recs),
                "words_q25_q50_q75": [int(np.percentile(words, q)) for q in (25, 50, 75)],
                "n_structured_dist": {str(k): v for k, v in sorted(fc.items())}}

    crowd = [r for r in bench.records if r.gt_category == Category.GENUINE_CROWD]
    pops = {"genuine_crowd": crowd}
    fab = [r for r in bench.records if r.gt_category == Category.ATTACK_FABRICATION]
    pops["t1_sophisticated"] = [r for r in fab
                                if r.provenance.get("sophistication") == "sophisticated"]
    pops["t1_naive_(short_by_design)"] = [r for r in fab
                                          if r.provenance.get("sophistication") == "naive"]
    syb = [r for r in bench.records if r.gt_category == Category.ATTACK_SYBIL]
    for tier in ("throwaway", "registered", "farmed", "paraphrased"):
        sel = [r for r in syb if r.provenance.get("identity_tier") == tier]
        if sel:
            pops[f"t5_{tier}"] = sel
    out = {}
    for name, recs in pops.items():
        s = _shape(recs)
        if s:
            out[name] = s
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark fidelity validation.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", default="results/benchmark")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    from pathlib import Path
    bd = args.benchmark if Path(args.benchmark).is_absolute() else cfg.project_root / args.benchmark
    run(cfg, str(bd))


if __name__ == "__main__":
    main()
