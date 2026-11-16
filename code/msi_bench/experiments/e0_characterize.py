"""Experiment E0 — Real-Data Characterisation (Section 4.2).

Establishes the empirical facts of the NAVAREA IV archive on 100% real data
(no synthetic data, no LLM/API access). Outputs:

* messages/year by message-function class (verifies volume);
* event-chain statistics and the |E| estimate (Layer-1 chains over the
  benchmark years);
* empirical active-duration distributions by hazard timescale class, which
  ground the σt defaults (Section 3.3);
* official error/correction rates (ADMIN_ERROR, sub-hour self-cancellations,
  internal date inconsistencies), cited in Section 4.2;
* originating-authority distribution;
* coordinate parse rate and geometry-type mix;
* text-length statistics.

Run with::

    python -m msi_bench.experiments.e0_characterize          # full archive
    python -m msi_bench.experiments.e0_characterize --years 2022 2023

Results are written to ``<output_dir>/e0/``: a JSON summary, a Markdown report,
and two PNG figures. The JSON additionally records, under ``manuscript_backfill``,
the headline values quoted in the paper.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import Config, get_config
from ..parsing.chains import build_chains
from ..parsing.pipeline import parse_corpus
from ..schema import EventType, GeometryKind, MessageFunction, ParsedMessage

# --- hazard timescale classes (for grounding σt) -----------------
# Maps free text to the four timescale categories. Applied only to
# discrete-event NAV warnings; the median active duration per class is the
# empirical anchor for σt.
_TIMESCALE_PATTERNS = [
    ("transient", re.compile(
        r"DERELICT|ADRIFT|DRIFTING|OIL\s+SPILL|PERSON\s+IN\s+WATER|"
        r"CAPSIZED|PARTIALLY\s+SUBMERGED", re.IGNORECASE)),
    ("scheduled", re.compile(
        r"EXERCISE|ROCKET|LAUNCH|FIRING|\bSURVEY\b|SEISMIC|OPERATIONS|"
        r"CABLE\s+OPERATIONS|HAZARDOUS\s+OPERATIONS|DRILLING|DREDG", re.IGNORECASE)),
    ("semi_permanent", re.compile(
        r"\bBUOY\b|UNLIT|LIGHT\s+(OUT|UNRELIABLE)|CONSTRUCTION|UNRELIABLE|"
        r"OFF\s+STATION|OUT\s+OF\s+SERVICE", re.IGNORECASE)),
    ("permanent", re.compile(
        r"WRECK|SHOAL|\bROCK\b|OBSTRUCTION|UNCHARTED|PINNACLE|REEF", re.IGNORECASE)),
]

_MAX_DURATION_H = 3 * 365 * 24  # filter out implausible/erroneous durations


def _timescale_class(text: str) -> Optional[str]:
    for name, pat in _TIMESCALE_PATTERNS:
        if pat.search(text or ""):
            return name
    return None


def _quartiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    q = statistics.quantiles(s, n=4) if len(s) >= 2 else [s[0], s[0], s[0]]
    return {
        "n": len(s),
        "min": round(s[0], 2),
        "q1": round(q[0], 2),
        "median": round(statistics.median(s), 2),
        "q3": round(q[2], 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.fmean(s), 2),
    }


def _active_durations_hours(messages: Sequence[ParsedMessage]) -> List[float]:
    out = []
    for m in messages:
        d = m.active_duration_hours
        if d is not None and 0 < d <= _MAX_DURATION_H:
            out.append(d)
    return out


def run(cfg: Config, years: Optional[Sequence[int]] = None) -> Dict:
    """Run E0 over the requested years (default: the full available archive)."""
    from ..parsing.loader import available_years

    if years is None:
        years = available_years(cfg.data_dir)
    years = sorted(int(y) for y in years)
    if not years:
        raise SystemExit(
            f"No NAVAREA CSV files found in {cfg.data_dir}. "
            "Place the annual files (e.g. 2017.csv) there and retry."
        )

    print(f"[E0] Parsing NAVAREA {cfg.nav_area} archive for years "
          f"{years[0]}–{years[-1]} …")
    messages = parse_corpus(
        cfg, years, nav_area=cfg.nav_area,
        progress=lambda n: print(f"      parsed {n} messages…"),
    )
    total = len(messages)
    print(f"[E0] Parsed {total} messages.")
    if total == 0:
        raise SystemExit("No messages parsed — check navArea and data files.")

    # --- (a) messages/year by function class -------------------------------
    by_year_func: Dict[int, Counter] = defaultdict(Counter)
    func_totals: Counter = Counter()
    for m in messages:
        by_year_func[m.msg_year][m.function.value] += 1
        func_totals[m.function.value] += 1

    discrete = [m for m in messages if m.function.is_discrete_event]

    # --- (d) official error/correction rates -------------------------------
    n_admin = func_totals.get(MessageFunction.ADMIN_ERROR.value, 0)
    n_sub_hour = sum(1 for m in messages if "sub_hour_cancel" in m.defects)
    n_cancel_before_issue = sum(1 for m in messages if "cancel_before_issue" in m.defects)
    # "Explicit error corrections or non-promulgations" = ADMIN_ERROR class.
    err_pct = round(100.0 * n_admin / total, 2)

    # --- (f) coordinate parse rate + geometry mix (discrete events) --------
    n_disc = len(discrete)
    n_disc_with_coords = sum(1 for m in discrete if m.coordinates)
    parse_rate = round(100.0 * n_disc_with_coords / n_disc, 2) if n_disc else 0.0
    geom_mix = Counter(
        m.geometry_kind.value for m in discrete if m.coordinates
    )

    # --- (e) authority distribution ----------------------------------------
    authority_counts = Counter(m.authority_norm for m in messages)

    # --- event-type mix (discrete events) ----------------------------------
    event_type_mix = Counter(m.event_type.value for m in discrete)

    # --- (g) text-length statistics ----------------------------------------
    char_lengths = [len(m.raw_text) for m in messages]
    line_counts = [m.raw_text.count("\n") + 1 for m in messages]

    # --- (c) active-duration distributions by hazard timescale -------------
    by_class_durations: Dict[str, List[float]] = defaultdict(list)
    for m in discrete:
        d = m.active_duration_hours
        if d is None or not (0 < d <= _MAX_DURATION_H):
            continue
        cls = _timescale_class(m.raw_text)
        if cls:
            by_class_durations[cls].append(d / 24.0)  # report in DAYS
    duration_by_class = {
        cls: _quartiles(vals) for cls, vals in sorted(by_class_durations.items())
    }
    overall_duration_days = _quartiles([d / 24.0 for d in _active_durations_hours(discrete)])

    # --- (b) Layer-1 chains over the benchmark-year discrete events --------
    bench_years = list(cfg.years("benchmark"))
    bench_discrete = [
        m for m in discrete if m.msg_year in bench_years
    ]
    chains = build_chains(bench_discrete)
    n_events = chains.n_chains  # each connected component == one event
    length_dist = chains.length_distribution()
    n_in_multi = sum(len(v) for v in chains.chains.values() if len(v) > 1)
    multi_frac = round(100.0 * n_in_multi / len(bench_discrete), 2) if bench_discrete else 0.0

    # --- headline values quoted in the paper (keys E0-*) --------------------
    perm = duration_by_class.get("permanent", {})
    trans = duration_by_class.get("transient", {})
    backfill = {
        "E0-ERR": err_pct,  # % messages that are error/non-promulgation
        "E0-DUR-PERM-days": perm.get("median"),   # permanent hazards, days
        "E0-DUR-TRANS-days": trans.get("median"),  # transient hazards, days
        "E0-EVENT-COUNT": n_events,  # |E| over benchmark years
    }

    summary = {
        "archive": {
            "nav_area": cfg.nav_area,
            "years": [years[0], years[-1]],
            "total_messages": total,
            "discrete_event_messages": n_disc,
        },
        "messages_by_year_function": {
            str(y): dict(by_year_func[y]) for y in sorted(by_year_func)
        },
        "function_totals": dict(func_totals),
        "function_percent": {
            k: round(100.0 * v / total, 2) for k, v in func_totals.items()
        },
        "event_type_mix_discrete": dict(event_type_mix),
        "error_rates": {
            "admin_error_count": n_admin,
            "admin_error_percent": err_pct,
            "sub_hour_self_cancel_count": n_sub_hour,
            "sub_hour_self_cancel_percent": round(100.0 * n_sub_hour / total, 2),
            "internal_date_inconsistency_count": n_cancel_before_issue,
            "internal_date_inconsistency_percent": round(
                100.0 * n_cancel_before_issue / total, 2),
        },
        "coordinate_parsing": {
            "discrete_with_coords": n_disc_with_coords,
            "parse_rate_percent": parse_rate,
            "geometry_mix": dict(geom_mix),
        },
        "authority_distribution": {
            "distinct_originators": len(authority_counts),
            "top_20": authority_counts.most_common(20),
        },
        "text_length": {
            "chars": _quartiles([float(x) for x in char_lengths]),
            "lines": _quartiles([float(x) for x in line_counts]),
        },
        "active_duration_days": {
            "overall": overall_duration_days,
            "by_timescale_class": duration_by_class,
        },
        "event_chains_benchmark_years": {
            "benchmark_years": [bench_years[0], bench_years[-1]] if bench_years else [],
            "discrete_messages": len(bench_discrete),
            "n_events_estimate": n_events,
            "chain_length_distribution": {str(k): v for k, v in sorted(length_dist.items())},
            "messages_in_multi_message_chains_percent": multi_frac,
        },
        "manuscript_backfill": backfill,
    }

    out_dir = cfg.output_dir / "e0"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e0_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    _write_report(out_dir / "e0_report.md", summary)
    _write_figures(out_dir, summary)
    print(f"[E0] Wrote summary, report, and figures to {out_dir}")
    print(f"[E0] |E| (benchmark-year events) ≈ {n_events}; "
          f"ADMIN_ERROR = {err_pct}% of messages")
    return summary


def _write_report(path: Path, s: Dict) -> None:
    """Render a human-readable Markdown report from the summary dict."""
    a = s["archive"]
    lines: List[str] = []
    lines.append(f"# Experiment E0 — Real-Data Characterisation\n")
    lines.append(
        f"NAVAREA {a['nav_area']} archive, {a['years'][0]}–{a['years'][1]}: "
        f"**{a['total_messages']:,} messages** "
        f"({a['discrete_event_messages']:,} discrete-event).\n"
    )

    lines.append("## Message-function composition\n")
    lines.append("| Function | Count | % |")
    lines.append("|---|---:|---:|")
    for k, v in sorted(s["function_totals"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:,} | {s['function_percent'][k]} |")
    lines.append("")

    er = s["error_rates"]
    lines.append("## Official error / correction rates\n")
    lines.append(f"- Explicit errors / non-promulgations (ADMIN_ERROR): "
                 f"**{er['admin_error_count']:,} ({er['admin_error_percent']}%)**")
    lines.append(f"- Sub-hour self-cancellations: "
                 f"{er['sub_hour_self_cancel_count']:,} ({er['sub_hour_self_cancel_percent']}%)")
    lines.append(f"- Internal date inconsistencies (cancel-before-issue): "
                 f"{er['internal_date_inconsistency_count']:,} "
                 f"({er['internal_date_inconsistency_percent']}%)\n")

    cp = s["coordinate_parsing"]
    lines.append("## Coordinate parsing (discrete-event messages)\n")
    lines.append(f"- Parse rate: **{cp['parse_rate_percent']}%** "
                 f"({cp['discrete_with_coords']:,} messages with ≥1 coordinate)")
    lines.append(f"- Geometry mix: {cp['geometry_mix']}\n")

    dc = s["active_duration_days"]
    lines.append("## Active-duration distributions by hazard timescale (days) — grounds the σt scales\n")
    lines.append("| Class | n | median | Q1 | Q3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls, q in dc["by_timescale_class"].items():
        lines.append(f"| {cls} | {q.get('n',0)} | {q.get('median','–')} "
                     f"| {q.get('q1','–')} | {q.get('q3','–')} |")
    lines.append("")

    ec = s["event_chains_benchmark_years"]
    if ec.get("benchmark_years"):
        lines.append("## Event chains (benchmark years) — the |E| estimate\n")
        lines.append(f"- Benchmark years: {ec['benchmark_years'][0]}–{ec['benchmark_years'][1]}")
        lines.append(f"- Discrete-event messages: {ec['discrete_messages']:,}")
        lines.append(f"- **Estimated distinct events |E| ≈ {ec['n_events_estimate']:,}**")
        lines.append(f"- Messages in multi-message chains: "
                     f"{ec['messages_in_multi_message_chains_percent']}%\n")

    ad = s["authority_distribution"]
    lines.append("## Originating authorities\n")
    lines.append(f"- Distinct normalised originators: **{ad['distinct_originators']}**")
    lines.append("\n| Originator | Count |")
    lines.append("|---|---:|")
    for name, cnt in ad["top_20"]:
        lines.append(f"| {name} | {cnt:,} |")
    lines.append("")

    bf = s["manuscript_backfill"]
    lines.append("## Manuscript back-fill values\n")
    lines.append("```json")
    lines.append(json.dumps(bf, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), "utf-8")


def _write_figures(out_dir: Path, s: Dict) -> None:
    """Render the two E0 figures (messages/year stacked bars; duration bars).

    Publication figures: no in-image titles (captions live in the text), PNG at
    300 dpi plus a vector PDF for the journal template.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[E0] matplotlib unavailable ({exc}); skipping figures.")
        return

    def _save(fig, stem: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)

    # Messages/year stacked by function class.
    mbyf = s["messages_by_year_function"]
    years = sorted(int(y) for y in mbyf)
    funcs = [f.value for f in MessageFunction]
    bottoms = [0] * len(years)
    fig, ax = plt.subplots(figsize=(9, 5))
    for f in funcs:
        vals = [mbyf[str(y)].get(f, 0) for y in years]
        ax.bar(years, vals, bottom=bottoms, label=f.replace("_", " "))
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xlabel("Year")
    ax.set_ylabel("Messages")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "e0_messages_by_year")

    # Active-duration medians by hazard timescale class, in
    # timescale order so the monotone rise (Fig. 3) reads directly;
    # log axis keeps the 3-day and 153-day bars simultaneously legible.
    dc = s["active_duration_days"]["by_timescale_class"]
    order = ["transient", "scheduled", "semi_permanent", "permanent"]
    classes = [c for c in order if c in dc] + sorted(c for c in dc if c not in order)
    medians = [dc[c].get("median", 0) for c in classes]
    if classes:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([c.replace("_", "-") for c in classes], medians, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_ylabel("Median active duration (days, log scale)")
        for i, v in enumerate(medians):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, "e0_duration_by_class")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E0.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="Years to analyse (default: all available).")
    args = parser.parse_args(argv)
    cfg = get_config(args.config)
    run(cfg, years=args.years)


if __name__ == "__main__":
    main()
