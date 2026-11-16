"""MSI-Bench assembly (Section 4.3).

Combines the real official backbone (events from Layer-1 chains) with the
channel-separated synthetic overlay (GENUINE_CROWD, GENUINE_SENSOR, T1–T5),
attaches ground-truth labels / event identities / full provenance, performs the
event-level dev/test split, and serialises everything to JSON together with a
datasheet. The genuine-text channel uses the configured LLM clients (or the
deterministic mock); every coordinate, timestamp, and identifier is code-owned.
"""

from __future__ import annotations

import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..config import Config
from ..geo import centroid as geo_centroid, classify_geometry, extract_coordinates
from ..llm.client import get_generator_clients
from ..parsing.event_type import classify_event_type
from ..parsing.chains import build_chains
from ..parsing.pipeline import parse_corpus
from ..schema import (Category, EventType, GeometryKind, MSIRecord,
                      MessageFunction, ParsedMessage, ProviderType)
from .contributors import DEVICE_NOISE_NM, ContributorPopulation
from . import generators as gen

_HAZARD_KEYS = [
    ("derelict", re.compile(r"DERELICT|ADRIFT|DRIFTING|CAPSIZED", re.I)),
    ("wreck", re.compile(r"WRECK|OBSTRUCTION|SHOAL|\bROCK\b|UNCHARTED", re.I)),
    ("rocket", re.compile(r"ROCKET|LAUNCH|FIRING|HAZARDOUS\s+OPERATIONS", re.I)),
    ("survey", re.compile(r"SURVEY|SEISMIC|CABLE|DRILLING|DREDG", re.I)),
    ("iceberg", re.compile(r"ICEBERG|ICE\s+LIMIT|SEA\s+ICE", re.I)),
    ("buoy", re.compile(r"BUOY|UNLIT|LIGHT|UNRELIABLE|OFF\s+STATION", re.I)),
]


def _hazard_key(text: str) -> str:
    for key, pat in _HAZARD_KEYS:
        if pat.search(text or ""):
            return key
    return "generic"


# Synthetic validity duration (hours) per hazard class — used for the overlay's
# temporal fields. (The overlay preserves the archive's real inter-event
# spacing, shifted so the newest event sits at the reference `now`; each report
# is assessed at its own submission time — see build and MCAMParams.)
_HAZARD_DURATION_H = {
    "derelict": 72.0, "iceberg": 168.0, "buoy": 168.0,
    "rocket": 72.0, "survey": 240.0, "wreck": 2160.0, "generic": 72.0,
}


def _event_age_days(rng: random.Random) -> float:
    """Original per-event/per-attack uniform draw in [0.2, 6.0] days.

    Retained at every original call site so the shared-rng draw sequence — and
    therefore every downstream record id and LLM-cache salt — stays
    unchanged. Its value is reused as (i) sub-day
    epoch jitter for events, (ii) the fabrication/Sybil offset after the
    anchored real event, and (iii) the naive-fabrication position within the
    operational window.
    """
    return rng.uniform(0.2, 6.0)


# Sensor modality → declared event type (code-owned, from the sensor's own
# observation rather than the seed event's ground truth). Radar
# contacts and AIS anomalies are navigational; met stations are meteorological.
_SENSOR_EVENT_TYPE = {
    "radar": EventType.NAV_WARNING,
    "ais": EventType.NAV_WARNING,
    "met": EventType.MET_WARNING,
}


@dataclass
class Event:
    event_id: str
    event_type: EventType
    centroid: Tuple[float, float]
    issue_time: datetime
    valid_end: Optional[datetime]
    hazard_key: str
    cancelled: bool
    rep_text: str
    split: str = "dev"


@dataclass
class _CrowdJob:
    """A crowd report's metadata, with all RNG already drawn (text deferred).

    Separating the drawn metadata from the network-bound text generation lets the
    text be produced concurrently (``llm.concurrency`` > 1) while the shared-rng
    draw sequence stays strictly sequential — so the benchmark is a pure function
    of (seed, config), identical for any worker count. ``fb_rng`` is a per-report
    RNG for the ROUGE-fallback path so text generation never touches the shared
    rng (it is unused on the sequential path).
    """
    ev: Event
    produced: int
    quality: str
    contrib: object
    sigma: float
    obs: Tuple[float, float]
    gen_fam: str
    in_paired: bool
    r_issue: datetime
    dur_h: float
    submission_pos: Tuple[float, float]
    e0: datetime
    salt: str
    fb_rng: Optional[random.Random]


@dataclass
class _FabJob:
    """A T1-fabrication record's metadata, with all RNG drawn (text deferred).

    Mirrors :class:`_CrowdJob` so fabrication text can also be generated
    concurrently while keeping the shared-rng draw sequence sequential.
    ``duration`` holds the naive tier's implausible form-field value (its
    designed content-plausibility signal); the final ``structured_data`` is
    built from the generated text itself (extractor parity).
    ``target_words``/``quality`` come from derived RNGs so the shared draw
    sequence — and with it the genuine-crowd cache replay — is untouched.
    """
    i: int
    sophisticated: bool
    adv: object
    pos: Tuple[float, float]
    hk: str
    duration: Optional[int]
    quality: str
    target_words: Optional[int]
    atk_epoch: datetime
    issue_offset_h: float
    submission_pos: Tuple[float, float]
    split: str
    salt: str
    fb_rng: Optional[random.Random]
    id_prefix: str = "t1"


@dataclass
class Benchmark:
    records: List[MSIRecord]
    events: Dict[str, Dict]
    histories: Dict[str, List[Tuple[datetime, str]]]
    meta: Dict
    # Attack reserve: extra attack records, disjoint from
    # `records`, used ONLY by E6's fixed-genuine adversarial-pressure sweep.
    # Every other experiment sees the base overlay at native prevalence.
    reserve: List[MSIRecord] = field(default_factory=list)

    # ------------------------------------------------------------------ I/O
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "records.json").write_text(
            json.dumps([_record_to_dict(r) for r in self.records], indent=1), "utf-8")
        (out_dir / "events.json").write_text(json.dumps(self.events, indent=1), "utf-8")
        hist = {k: [(t.isoformat(), o) for t, o in v] for k, v in self.histories.items()}
        (out_dir / "histories.json").write_text(json.dumps(hist), "utf-8")
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2), "utf-8")
        (out_dir / "DATASHEET.md").write_text(_datasheet(self.meta), "utf-8")
        if self.reserve:
            (out_dir / "attack_reserve.json").write_text(
                json.dumps([_record_to_dict(r) for r in self.reserve], indent=1), "utf-8")

    @staticmethod
    def load(out_dir: Path) -> "Benchmark":
        out_dir = Path(out_dir)
        records = [_record_from_dict(d) for d in
                   json.loads((out_dir / "records.json").read_text("utf-8"))]
        events = json.loads((out_dir / "events.json").read_text("utf-8"))
        hist_raw = json.loads((out_dir / "histories.json").read_text("utf-8"))
        histories = {k: [(datetime.fromisoformat(t), o) for t, o in v]
                     for k, v in hist_raw.items()}
        meta = json.loads((out_dir / "meta.json").read_text("utf-8"))
        rpath = out_dir / "attack_reserve.json"
        reserve = ([_record_from_dict(d) for d in json.loads(rpath.read_text("utf-8"))]
                   if rpath.is_file() else [])
        return Benchmark(records, events, histories, meta, reserve)

    def split_records(self, split: str) -> List[MSIRecord]:
        return [r for r in self.records if r.provenance.get("split") == split]


# --------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------
def _dt(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _record_to_dict(r: MSIRecord) -> Dict:
    return {
        "record_id": r.record_id,
        "event_type": r.event_type.value,
        "provider_type": r.provider_type.value,
        "provider_id": r.provider_id,
        "authority_norm": r.authority_norm,
        "raw_text": r.raw_text,
        "structured_data": r.structured_data,
        "geometry_kind": r.geometry_kind.value,
        "coordinates": r.coordinates,
        "centroid": r.centroid,
        "spatial_uncertainty_nm": r.spatial_uncertainty_nm,
        "issue_time": _dt(r.issue_time),
        "valid_start": _dt(r.valid_start),
        "valid_end": _dt(r.valid_end),
        "device_class": r.device_class,
        "auth_level": r.auth_level,
        "network_origin": r.network_origin,
        "submission_position": r.submission_position,
        "observation_type": r.observation_type,
        "gt_category": r.gt_category.value if r.gt_category else None,
        "gt_event_id": r.gt_event_id,
        "gt_quality_tier": r.gt_quality_tier,
        "provenance": r.provenance,
        # written only when present, so record sets without lifecycle
        # references serialise byte-identically to earlier builds
        **({"references": list(r.references)} if r.references else {}),
    }


def _record_from_dict(d: Dict) -> MSIRecord:
    def parse_dt(s):
        return datetime.fromisoformat(s) if s else None

    return MSIRecord(
        record_id=d["record_id"],
        event_type=EventType(d["event_type"]),
        provider_type=ProviderType(d["provider_type"]),
        provider_id=d["provider_id"],
        authority_norm=d.get("authority_norm", ""),
        raw_text=d["raw_text"],
        structured_data=d.get("structured_data", {}),
        geometry_kind=GeometryKind(d["geometry_kind"]),
        coordinates=[tuple(c) for c in d.get("coordinates", [])],
        centroid=tuple(d["centroid"]) if d.get("centroid") else None,
        spatial_uncertainty_nm=d.get("spatial_uncertainty_nm", 0.0),
        issue_time=parse_dt(d.get("issue_time")),
        valid_start=parse_dt(d.get("valid_start")),
        valid_end=parse_dt(d.get("valid_end")),
        device_class=d.get("device_class", "UNKNOWN"),
        auth_level=d.get("auth_level", "ANONYMOUS"),
        network_origin=d.get("network_origin"),
        submission_position=tuple(d["submission_position"]) if d.get("submission_position") else None,
        observation_type=d.get("observation_type", "visual"),
        gt_category=Category(d["gt_category"]) if d.get("gt_category") else None,
        gt_event_id=d.get("gt_event_id"),
        gt_quality_tier=d.get("gt_quality_tier"),
        provenance=d.get("provenance", {}),
        references=list(d.get("references", [])),
    )


# --------------------------------------------------------------------------
# Event extraction from Layer-1 chains
# --------------------------------------------------------------------------
def _extract_events(messages: Sequence[ParsedMessage]) -> Dict[str, Event]:
    chains = build_chains(messages)
    by_uid = {m.uid: m for m in messages}
    events: Dict[str, Event] = {}
    for root, members in chains.chains.items():
        msgs = [by_uid[u] for u in members if u in by_uid]
        # representative: earliest message that has a centroid (else earliest)
        with_coords = [m for m in msgs if m.centroid]
        rep = min(with_coords or msgs, key=lambda m: m.issue_time or datetime.max)
        if rep.centroid is None:
            continue  # no usable geometry for this event
        cancelled = any(m.status == "C" for m in msgs)
        valid_end = max((m.cancel_time or m.self_expiry_time for m in msgs
                         if (m.cancel_time or m.self_expiry_time)), default=None)
        events[root] = Event(
            event_id=root,
            event_type=rep.event_type,
            centroid=rep.centroid,
            issue_time=rep.issue_time or datetime(2023, 1, 1),
            valid_end=valid_end,
            hazard_key=_hazard_key(rep.raw_text),
            cancelled=cancelled,
            rep_text=rep.raw_text,
        )
    return events


# --------------------------------------------------------------------------
# Concurrent crowd-text generation (order- and seed-stable)
# --------------------------------------------------------------------------
def _fmt_dur(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class _Progress:
    """Dependency-free live progress indicator written to stderr.

    On a TTY it rewrites one line (count, %, rate, ETA) throttled to
    ``min_interval`` seconds; when stderr is redirected it prints a
    newline-terminated update at ~10% milestones so logs stay readable. Thread-
    safe, so the parallel and sequential paths share it. Goes to stderr, leaving
    stdout (results, logs) clean.
    """

    def __init__(self, total: int, label: str, enabled: bool = True,
                 min_interval: float = 0.2) -> None:
        self.total = int(total)
        self.label = label or "progress"
        self.enabled = bool(enabled) and self.total > 0
        self.min_interval = min_interval
        self.count = 0
        self.start = time.monotonic()
        self.last_t = 0.0
        self.next_pct = 0
        self.tty = sys.stderr.isatty()
        self._lock = threading.Lock()

    def tick(self, k: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.count += k
            now = time.monotonic()
            complete = self.count >= self.total
            if self.tty:
                if not complete and now - self.last_t < self.min_interval:
                    return
                self.last_t = now
                self._render(now)
            else:
                pct = 100 * self.count // self.total
                if complete or pct >= self.next_pct:
                    self.next_pct = (pct // 10 + 1) * 10
                    self._render(now)

    def done(self) -> None:
        if not self.enabled or not self.tty:
            return
        with self._lock:
            self._render(time.monotonic())
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _render(self, now: float) -> None:
        elapsed = now - self.start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * self.count / self.total
        eta = (self.total - self.count) / rate if rate > 0 else 0.0
        body = (f"[{self.label}] {self.count}/{self.total} ({pct:3.0f}%) "
                f"{rate:.1f}/s eta {_fmt_dur(eta)}")
        sys.stderr.write(("\r" + body + "    ") if self.tty else (body + "\n"))
        sys.stderr.flush()


def parallel_map(fn: Callable, items: Sequence, concurrency: int,
                 verbose: bool = True, label: str = "") -> List:
    """Apply ``fn`` to ``items`` preserving input order; thread-pooled if >1 worker.

    Threads (not processes) because the work is network-bound — Python releases
    the GIL during the HTTP call, so workers overlap their waits — and they share
    the in-process response cache directly. Results are returned in order, so the
    caller's assembly stays deterministic regardless of completion order. A live
    progress indicator (stderr) is shown while ``verbose``.
    """
    n = len(items)
    prog = _Progress(n, label, enabled=verbose)
    if concurrency <= 1 or n <= 1:
        out = []
        for x in items:
            out.append(fn(x))
            prog.tick()
        prog.done()
        return out
    results: List = [None] * n

    def _work(pair):
        i, x = pair
        results[i] = fn(x)
        prog.tick()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(_work, enumerate(items)))
    prog.done()
    return results


def _draw_crowd_job(ev, produced, rng, pop, qt, paired_slice, fam_of,
                    event_epoch, salt_prefix, seed, clients) -> _CrowdJob:
    """Draw one crowd report's metadata from the shared rng (text deferred).

    Draw order matches the sequential path exactly (quality, contributor,
    position, issue offset, submission position); only the text generation —
    which consumes no shared rng on the no-fallback path — is deferred.
    """
    quality = _weighted(rng, qt)
    contrib = pop.sample(rng)
    sigma = DEVICE_NOISE_NM[contrib.device_class]
    obs = gen.noise_position(ev.centroid, sigma, rng)
    in_paired = ev.event_id in paired_slice
    gen_fam = ("A" if produced % 2 == 0 else "B") if in_paired else fam_of[ev.event_id]
    r_issue = event_epoch[ev.event_id] + timedelta(hours=rng.uniform(0, 24))
    submission_pos = gen.noise_position(ev.centroid, rng.uniform(1.0, 8.0), rng)
    salt = f"{salt_prefix}:crowd-{ev.event_id}-{produced}"
    return _CrowdJob(
        ev=ev, produced=produced, quality=quality, contrib=contrib, sigma=sigma,
        obs=obs, gen_fam=gen_fam, in_paired=in_paired, r_issue=r_issue,
        dur_h=_HAZARD_DURATION_H.get(ev.hazard_key, 72.0), submission_pos=submission_pos,
        e0=event_epoch[ev.event_id], salt=salt, fb_rng=random.Random(f"{seed}|{salt}"))


def _crowd_text(job: _CrowdJob, clients, degraded, bcfg, gen_max_tokens) -> str:
    """Generate + coordinate-substitute a crowd report's text (parallel path).

    Uses the job's dedicated fallback RNG so the ROUGE-fallback path consumes no
    shared rng, keeping output independent of concurrent execution order.
    """
    text = gen.generate_crowd_text(
        clients[job.gen_fam], job.ev.hazard_key, _expertise(job.contrib),
        job.quality, None if degraded else job.ev.rep_text, job.fb_rng,
        max_rouge=bcfg.get("rouge_l_max", 0.6), max_tokens=gen_max_tokens,
        cache_salt=job.salt)
    return gen.substitute_coord(text, job.obs[0], job.obs[1])


def _build_crowd_record(job: _CrowdJob, text: str, degraded: bool, clients) -> MSIRecord:
    """Assemble the MSIRecord for a crowd report (shared by both build paths)."""
    ev = job.ev
    if degraded:
        coords = extract_coordinates(text)
        etype = classify_event_type(text)
        kind = classify_geometry(text, coords)
        centroid, valid_start, valid_end, structured = geo_centroid(coords), None, None, {}
    else:
        etype = classify_event_type(text)
        kind = GeometryKind.POINT
        coords, centroid = [job.obs], job.obs
        valid_start, valid_end = job.r_issue, job.e0 + timedelta(hours=job.dur_h)
        structured = _crowd_structured(text, job.quality)
    return MSIRecord(
        record_id=f"crowd-{ev.event_id}-{job.produced}",
        event_type=etype, provider_type=ProviderType.CROWDSOURCE,
        provider_id=job.contrib.provider_id, raw_text=text, structured_data=structured,
        geometry_kind=kind, coordinates=coords, centroid=centroid,
        spatial_uncertainty_nm=job.sigma,
        issue_time=job.r_issue, valid_start=valid_start, valid_end=valid_end,
        device_class=job.contrib.device_class, auth_level=job.contrib.auth_level,
        network_origin=job.contrib.network_origin,
        submission_position=job.submission_pos, observation_type="visual",
        gt_category=Category.GENUINE_CROWD, gt_event_id=ev.event_id,
        gt_quality_tier=job.quality,
        provenance={"split": ev.split, "generator": clients[job.gen_fam].tag,
                    "generator_family": job.gen_fam, "paired_slice": job.in_paired,
                    "degraded": degraded})


def _draw_fab_job(i, rng, pop, event_list, now, dev_frac, salt_prefix, seed,
                  event_epoch, window_start, qt, id_prefix="t1",
                  adv_tag="adv") -> _FabJob:
    """Draw one T1-fabrication record's metadata from the shared rng (text deferred).

    Sophisticated fabrications use a *farmed* identity (metadata matched to the
    genuine population) and are timestamped alongside the real event they mimic;
    naive ones keep the throwaway signature and land anywhere in the operational
    window. All original shared-rng draws are preserved in order; the additional draws
    (field-shape quality, length target) come from derived per-record RNGs.
    ``id_prefix``/``adv_tag`` namespace the attack-reserve pool ("t1r"/"advr")
    so reserve records and farmed identities never collide with the base build.
    """
    sophisticated = i % 2 == 1
    adv = pop.adversary(rng, i, farmed=sophisticated, tag=adv_tag)
    if sophisticated:  # plausible in-envelope location near a real event
        anchor = rng.choice(event_list)
        pos = gen.noise_position(anchor.centroid, 3.0, rng)
    else:              # implausible: place on land (continental interior)
        anchor = None
        pos = (rng.uniform(33, 45), rng.uniform(-100, -92))
    hk = rng.choice(["derelict", "wreck", "rocket", "generic"])
    duration = rng.choice([-5, 99999]) if not sophisticated else None  # implausible form field
    age_days = _event_age_days(rng)  # original draw, reused for the attack epoch
    atk_epoch = _attack_epoch(age_days, anchor, event_epoch, window_start, now)
    issue_offset_h = rng.uniform(0, 24)
    submission_pos = gen.noise_position(pos, rng.uniform(1.0, 8.0), rng)
    split = _attack_split(rng, dev_frac)
    quality = _weighted(random.Random(f"{seed}|{id_prefix}q-{i}"), qt)
    target_words = (gen.sample_target_words(random.Random(f"{seed}|{id_prefix}len-{i}"))
                    if sophisticated else None)
    return _FabJob(i=i, sophisticated=sophisticated, adv=adv, pos=pos, hk=hk,
                   duration=duration, quality=quality, target_words=target_words,
                   atk_epoch=atk_epoch, issue_offset_h=issue_offset_h,
                   submission_pos=submission_pos, split=split,
                   salt=f"{salt_prefix}:{id_prefix}-{i}",
                   fb_rng=random.Random(f"{seed}|{id_prefix}-{i}"), id_prefix=id_prefix)


def _attack_epoch(age_days, anchor, event_epoch, window_start, now) -> datetime:
    """Attack timing in the operational window.

    An anchored attack (sophisticated T1, Sybil) strikes 0.2–6 days after its
    real anchor event begins — a fabricator exploits genuine ongoing activity.
    An unanchored attack maps the same original draw onto a uniform position over
    the whole window, matching the genuine temporal density (a recent-only
    timestamp would otherwise fingerprint attacks under the widened frame).
    """
    if anchor is not None:
        return event_epoch[anchor.event_id] + timedelta(days=age_days)
    frac = (age_days - 0.2) / 5.8
    span_s = max(0.0, (now - window_start).total_seconds())
    return window_start + timedelta(seconds=frac * span_s)


def _fab_text(job: _FabJob, clients, gen_max_tokens) -> str:
    """Generate + coordinate-substitute a fabrication's text (per-job fallback rng)."""
    text = gen.generate_fabrication_text(
        clients["A"], job.hk, job.sophisticated, job.fb_rng,
        max_tokens=gen_max_tokens, cache_salt=job.salt,
        target_words=job.target_words)
    return gen.substitute_coord(text, job.pos[0], job.pos[1])


def _attack_structured(text: str, quality: str, degraded: bool,
                       duration: Optional[int] = None) -> Dict:
    """Structured fields for a crowdsource-channel attack.

    The platform's extraction pipeline is uniform: attacks submitted through the
    crowdsource channel pass through the SAME quality-conditioned extractor as
    genuine reports (`_crowd_structured`), so field counts and hazard_type
    provenance carry no attack fingerprint. In the schema-degraded bracket no
    crowdsource record carries extracted fields — attacks included. The naive
    fabricator's implausible
    ``duration_hours`` form value rides on top in the aligned bracket: filling
    the form carelessly is that tier's designed content-plausibility signal.
    """
    s = {} if degraded else _crowd_structured(text, quality)
    if duration is not None and not degraded:
        s["duration_hours"] = duration
    return s


def _build_fab_record(job: _FabJob, text: str, degraded: bool = False) -> MSIRecord:
    """Assemble the MSIRecord for a T1 fabrication (shared by both build paths)."""
    return MSIRecord(
        record_id=f"{job.id_prefix}-{job.i}", event_type=EventType.NAV_WARNING,
        provider_type=ProviderType.CROWDSOURCE, provider_id=job.adv.provider_id,
        raw_text=text,
        structured_data=_attack_structured(text, job.quality, degraded, job.duration),
        geometry_kind=GeometryKind.POINT, coordinates=[job.pos], centroid=job.pos,
        issue_time=job.atk_epoch + timedelta(hours=job.issue_offset_h),
        device_class=job.adv.device_class, auth_level=job.adv.auth_level,
        network_origin=job.adv.network_origin,
        submission_position=job.submission_pos, observation_type="visual",
        gt_category=Category.ATTACK_FABRICATION,
        provenance={"split": job.split,
                    "sophistication": "sophisticated" if job.sophisticated else "naive",
                    "identity_tier": "farmed" if job.sophisticated else "throwaway"})


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------
def build_benchmark(
    cfg: Config,
    degraded: bool = False,
    limit_events: Optional[int] = None,
    verbose: bool = True,
    years: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    reserve_multiplier: Optional[float] = None,
) -> Benchmark:
    """Construct MSI-Bench from the real archive + synthetic overlay.

    ``seed`` overrides ``cfg.seed`` for the synthetic overlay's RNG and
    contributor population, enabling the multi-seed generation-variance study
    (Section 4.5). The real backbone (events, chains) is seed-independent.
    ``reserve_multiplier`` overrides ``benchmark.reserve_attack_multiplier``:
    the extra attack pool for E6's fixed-genuine pressure
    sweep. Pass 0 to skip it (the multi-seed and holdout builds do — they
    never run the sweep, so warming a reserve for them would be waste).
    """
    seed = cfg.seed if seed is None else seed
    rng = random.Random(seed)
    bcfg = cfg.raw["benchmark"]

    bench_years = list(years) if years is not None else list(cfg.years("benchmark"))
    if verbose:
        print(f"[build] parsing discrete events for years {bench_years[0]}–{bench_years[-1]} …")
    bench_msgs = parse_corpus(cfg, bench_years, nav_area=cfg.nav_area,
                              functions=[MessageFunction.NEW_WARNING, MessageFunction.UPDATE])
    events = _extract_events(bench_msgs)
    # Canonical, limit-INDEPENDENT event order: always sort by id, then optionally
    # take the first `limit_events`. The unconditional sort keeps the per-type
    # dev/test shuffle and the A/B family assignment (both enumerate event_list
    # below) identical between a full build and a same-seed `limit=n_events`
    # rebuild (the Section 4.5 multi-seed study), which makes the build a pure
    # function of the seed.
    keys = sorted(events)
    if limit_events:
        keys = keys[:limit_events]
    events = {k: events[k] for k in keys}
    n_events = len(events)
    if verbose:
        print(f"[build] {n_events} events extracted; assigning event-level split …")

    # --- event-level dev/test split, stratified by type ---
    by_type: Dict[EventType, List[str]] = {}
    for eid, ev in events.items():
        by_type.setdefault(ev.event_type, []).append(eid)
    dev_frac = bcfg["dev_test_split"]
    for ids in by_type.values():
        rng.shuffle(ids)
        cut = int(round(len(ids) * dev_frac))
        for eid in ids[:cut]:
            events[eid].split = "dev"
        for eid in ids[cut:]:
            events[eid].split = "test"

    # Operational frame: the overlay PRESERVES the archive's real
    # inter-event temporal structure — each event's epoch is its real issue time
    # shifted by a constant so the newest event sits at the reference `now`.
    # Compressing all events into a single recent week would make co-located
    # recurring events temporally inseparable and over-merge clusters; real
    # spacing is the more faithful choice.
    # Freshness is handled downstream by per-record assessment time
    # (MCAMParams.assessment_mode="submission"), so genuine reports of older
    # events are scored as of their own arrival, not a later global snapshot.
    # The original per-event uniform draw is retained (draw-sequence/cache
    # compatibility) and reused as ±~3 h epoch jitter, which also breaks the
    # exact-tie epochs of same-DTG batch issues in the archive.
    now = datetime(2024, 6, 13, 0, 0)
    event_list = list(events.values())
    t_max = max(ev.issue_time for ev in event_list)
    frame_shift = now - t_max
    event_epoch: Dict[str, datetime] = {}
    for ev in event_list:
        jitter_h = _event_age_days(rng) - 3.1  # original draw → jitter in [-2.9, +2.9] h
        event_epoch[ev.event_id] = ev.issue_time + frame_shift + timedelta(hours=jitter_h)
    window_start = min(event_epoch.values())
    pop = ContributorPopulation(n=bcfg["n_contributors"], seed=seed, now=now)
    clients = get_generator_clients(cfg)
    gen_max_tokens = int(cfg.get("llm", "gen_max_tokens", default=gen.GEN_MAX_TOKENS))
    histories = dict(pop.histories)
    # Per-build salt prefix for the LLM response cache: the seed scopes every
    # generation slot so a same-seed rebuild replays cached text verbatim, while
    # a different seed (the Section 4.5 generation-variance study) regenerates fresh.
    salt_prefix = f"s{seed}"

    records: List[MSIRecord] = []
    counts: Dict[str, int] = {}

    def add(rec: MSIRecord):
        records.append(rec)
        key = rec.gt_category.value if rec.gt_category else "?"
        counts[key] = counts.get(key, 0) + 1

    # alternate generator families A/B across events; a 10% slice is generated by
    # BOTH families for the paired-generator comparison (Section 4.6).
    fam_of = {ev.event_id: ("A" if i % 2 == 0 else "B") for i, ev in enumerate(event_list)}
    paired_slice = {ev.event_id for i, ev in enumerate(event_list) if i % 10 == 0}

    # --- OFFICIAL backbone ---
    for ev in event_list:
        auth = f"official-{ev.event_id}"
        histories.setdefault(auth, [(now - timedelta(days=d), "CORRECT") for d in range(1, 20)])
        e0 = event_epoch[ev.event_id]
        dur_h = _HAZARD_DURATION_H.get(ev.hazard_key, 72.0)
        add(MSIRecord(
            record_id=f"off-{ev.event_id}",
            event_type=ev.event_type,
            provider_type=ProviderType.OFFICIAL,
            provider_id=auth,
            authority_norm=auth,
            raw_text=_mask_lifecycle(ev.rep_text),
            # The overlay rebases each event as a fresh, active snapshot, so the
            # official record is in force; the historical chain's cancellation
            # is not part of the operational frame (avoids spurious official-vs-
            # crowd source conflicts).
            structured_data={"hazard_type": ev.hazard_key, "existence_status": "in_force"},
            geometry_kind=GeometryKind.POINT,
            coordinates=[ev.centroid], centroid=ev.centroid,
            issue_time=e0, valid_start=e0, valid_end=e0 + timedelta(hours=dur_h),
            device_class="PROFESSIONAL", auth_level="PKI_FULL",
            observation_type="shore", submission_position=ev.centroid,
            gt_category=Category.OFFICIAL, gt_event_id=ev.event_id,
            provenance={"split": ev.split, "generator": "real"},
        ))

    # --- GENUINE_CROWD ---
    # The narrative-text LLM call is the only network-bound step here, so it can
    # run concurrently (set llm.concurrency > 1) to accelerate real generation.
    # To keep the benchmark a pure function of (seed, config) — identical for any
    # worker count — all RNG draws stay sequential (phase 1) and the text is
    # produced separately (phase 2) with a per-report fallback RNG, so text
    # generation never touches the shared rng. The default/mock path keeps the
    # original interleaved order, so concurrency=1 builds are byte-identical.
    target_crowd = int(round(bcfg["crowd_multiplier"] * n_events))
    qt = bcfg["quality_tiers"]
    lo, hi = bcfg["crowd_per_event"]
    concurrency = int(cfg.get("llm", "concurrency", default=1))
    parallel = concurrency > 1 and not all(c.is_mock for c in clients.values())

    if parallel:
        crowd_jobs: List[_CrowdJob] = []
        produced = 0
        ci = 0
        while produced < target_crowd:
            ev = event_list[ci % n_events]
            ci += 1
            for _ in range(rng.randint(lo, hi)):
                if produced >= target_crowd:
                    break
                crowd_jobs.append(_draw_crowd_job(
                    ev, produced, rng, pop, qt, paired_slice, fam_of,
                    event_epoch, salt_prefix, seed, clients))
                produced += 1
        if verbose:
            print(f"[build]   crowd: generating {len(crowd_jobs)} reports over "
                  f"{concurrency} workers …", flush=True)
        crowd_texts = parallel_map(
            lambda j: _crowd_text(j, clients, degraded, bcfg, gen_max_tokens),
            crowd_jobs, concurrency, verbose, "crowd")
        for job, text in zip(crowd_jobs, crowd_texts):
            add(_build_crowd_record(job, text, degraded, clients))
    else:
        prog = _Progress(target_crowd, "crowd", enabled=verbose)
        produced = 0
        ci = 0
        while produced < target_crowd:
            ev = event_list[ci % n_events]
            ci += 1
            for _ in range(rng.randint(lo, hi)):
                if produced >= target_crowd:
                    break
                quality = _weighted(rng, qt)
                contrib = pop.sample(rng)
                sigma = DEVICE_NOISE_NM[contrib.device_class]
                obs = gen.noise_position(ev.centroid, sigma, rng)
                # paired-slice events draw from both families (alternating per report)
                in_paired = ev.event_id in paired_slice
                gen_fam = ("A" if produced % 2 == 0 else "B") if in_paired else fam_of[ev.event_id]
                salt = f"{salt_prefix}:crowd-{ev.event_id}-{produced}"
                # Original interleaved order: the text (on the shared rng for the
                # mock/ROUGE-fallback paths) is drawn between the position draws,
                # so default/mock builds are reproducible byte for byte.
                text = gen.substitute_coord(gen.generate_crowd_text(
                    clients[gen_fam], ev.hazard_key, _expertise(contrib),
                    quality, None if degraded else ev.rep_text, rng,
                    max_rouge=bcfg.get("rouge_l_max", 0.6), max_tokens=gen_max_tokens,
                    cache_salt=salt), obs[0], obs[1])
                r_issue = event_epoch[ev.event_id] + timedelta(hours=rng.uniform(0, 24))
                submission_pos = gen.noise_position(ev.centroid, rng.uniform(1.0, 8.0), rng)
                job = _CrowdJob(
                    ev=ev, produced=produced, quality=quality, contrib=contrib,
                    sigma=sigma, obs=obs, gen_fam=gen_fam, in_paired=in_paired,
                    r_issue=r_issue, dur_h=_HAZARD_DURATION_H.get(ev.hazard_key, 72.0),
                    submission_pos=submission_pos, e0=event_epoch[ev.event_id],
                    salt=salt, fb_rng=None)
                add(_build_crowd_record(job, text, degraded, clients))
                produced += 1
                prog.tick()
        prog.done()

    # --- GENUINE_SENSOR (fully programmatic) ---
    target_sensor = int(round(bcfg["sensor_multiplier"] * n_events))
    for s in range(target_sensor):
        ev = event_list[s % n_events]
        stype = rng.choice(["radar", "ais", "met"])
        obs = gen.noise_position(ev.centroid, rng.uniform(0.1, 0.5), rng)
        e0 = event_epoch[ev.event_id]
        dur_h = _HAZARD_DURATION_H.get(ev.hazard_key, 72.0)
        s_issue = e0 + timedelta(hours=rng.uniform(0, 6))
        # A sensor's type/hazard come from its OWN modality and output, not the
        # seed event's ground truth: a radar reports an obstruction, a
        # met station a weather observation — neither knows the true hazard class.
        s_text = gen.sensor_text(stype, obs[0], obs[1], rng)
        s_etype = _SENSOR_EVENT_TYPE.get(stype, EventType.NAV_WARNING)
        add(MSIRecord(
            record_id=f"sensor-{s}", event_type=s_etype,
            provider_type=ProviderType.SENSOR, provider_id=f"sensor-{stype}-{s % 50}",
            raw_text=s_text,
            structured_data={"hazard_type": _hazard_key(s_text)},
            geometry_kind=GeometryKind.POINT, coordinates=[obs], centroid=obs,
            spatial_uncertainty_nm=0.3, issue_time=s_issue,
            valid_start=s_issue, valid_end=e0 + timedelta(hours=dur_h),
            device_class="MARINE_CERTIFIED", auth_level="PKI_BASIC",
            observation_type=stype if stype in ("radar", "ais") else "shore",
            submission_position=obs,
            gt_category=Category.GENUINE_SENSOR, gt_event_id=ev.event_id,
            provenance={"split": ev.split, "generator": "programmatic"},
        ))

    # --- T1 fabrication ---
    # Two-phase like GENUINE_CROWD: metadata drawn sequentially, text generated in
    # parallel when llm.concurrency > 1. Freshness-neutral timing — a
    # fabrication is drawn from the SAME issue-time-freshness distribution as
    # genuine reports, so Cst's freshness term cannot back-leak the attack label.
    n_fab = int(round(bcfg["attack_multipliers"]["fabrication"] * n_events))
    if parallel:
        fab_jobs = [_draw_fab_job(i, rng, pop, event_list, now, dev_frac, salt_prefix,
                                  seed, event_epoch, window_start, qt)
                    for i in range(n_fab)]
        fab_texts = parallel_map(lambda j: _fab_text(j, clients, gen_max_tokens),
                                  fab_jobs, concurrency, verbose, "fabrication")
        for job, text in zip(fab_jobs, fab_texts):
            add(_build_fab_record(job, text, degraded))
    else:
        prog = _Progress(n_fab, "fabrication", enabled=verbose)
        for i in range(n_fab):
            sophisticated = i % 2 == 1
            adv = pop.adversary(rng, i, farmed=sophisticated)
            if sophisticated:  # plausible in-envelope location near a real event
                anchor = rng.choice(event_list)
                pos = gen.noise_position(anchor.centroid, 3.0, rng)
            else:              # implausible: place on land (continental interior)
                anchor = None
                pos = (rng.uniform(33, 45), rng.uniform(-100, -92))
            hk = rng.choice(["derelict", "wreck", "rocket", "generic"])
            target_words = (gen.sample_target_words(random.Random(f"{seed}|t1len-{i}"))
                            if sophisticated else None)  # derived rng: no shared draw
            # original order: text (on the shared rng for mock) between hk and duration
            text = gen.substitute_coord(gen.generate_fabrication_text(
                clients["A"], hk, sophisticated, rng, max_tokens=gen_max_tokens,
                cache_salt=f"{salt_prefix}:t1-{i}", target_words=target_words),
                pos[0], pos[1])
            duration = rng.choice([-5, 99999]) if not sophisticated else None  # implausible
            age_days = _event_age_days(rng)  # original draw, reused for the epoch
            atk_epoch = _attack_epoch(age_days, anchor, event_epoch, window_start, now)
            job = _FabJob(i=i, sophisticated=sophisticated, adv=adv, pos=pos, hk=hk,
                          duration=duration,
                          quality=_weighted(random.Random(f"{seed}|t1q-{i}"), qt),
                          target_words=target_words, atk_epoch=atk_epoch,
                          issue_offset_h=rng.uniform(0, 24),
                          submission_pos=gen.noise_position(pos, rng.uniform(1.0, 8.0), rng),
                          split=_attack_split(rng, dev_frac), salt=f"{salt_prefix}:t1-{i}",
                          fb_rng=None)
            add(_build_fab_record(job, text, degraded))
            prog.tick()
        prog.done()

    # --- T2 replay (real expired text) ---
    replay_leftover = _add_replay(cfg, rng, now, dev_frac, n_events, bcfg, add,
                                  verbose, pop=pop, window_start=window_start,
                                  seed=seed, degraded=degraded)

    # --- T3 tamper (programmatic perturbation of genuine crowd) ---
    crowd_pool = [r for r in records if r.gt_category == Category.GENUINE_CROWD]
    n_tamper = int(round(bcfg["attack_multipliers"]["tamper"] * n_events))
    for i in range(min(n_tamper, len(crowd_pool))):
        src = crowd_pool[rng.randrange(len(crowd_pool))]
        add(_tamper(src, i, rng, dev_frac))

    # --- T4 impersonation (programmatic provenance forgery) ---
    n_imp = int(round(bcfg["attack_multipliers"]["impersonation"] * n_events))
    for i in range(min(n_imp, len(crowd_pool))):
        src = crowd_pool[rng.randrange(len(crowd_pool))]
        add(_impersonate(src, i, rng, dev_frac))

    # --- T5 Sybil campaigns ---
    _add_sybil(cfg, rng, now, dev_frac, event_list, clients, pop, bcfg, add,
               salt_prefix, event_epoch=event_epoch, seed=seed, degraded=degraded)

    # --- attack reserve ---
    # An enlarged attack pool for the fixed-genuine adversarial-pressure sweep
    # (E6c): the base overlay above keeps its native ~14% prevalence for every
    # other experiment, while the reserve holds `reserve_attack_multiplier` ×
    # the base attack counts as EXTRA records under a disjoint namespace
    # (t1r-/t2r-/t3r-/t4r-/t5r-). All reserve randomness comes from RNGs
    # derived from f"{seed}|reserve…", so the shared draw sequence — and every
    # base record — is byte-identical whether or not the reserve is built.
    mult = (reserve_multiplier if reserve_multiplier is not None
            else float(bcfg.get("reserve_attack_multiplier", 0) or 0))
    reserve: List[MSIRecord] = []
    if mult > 0:
        reserve = _build_attack_reserve(
            cfg, mult, bcfg, seed, now, dev_frac, event_list, event_epoch,
            window_start, pop, clients, crowd_pool, replay_leftover,
            salt_prefix, gen_max_tokens, concurrency, degraded, verbose)

    # Farmed adversarial identities registered their histories on `pop` after the
    # snapshot above was taken; merge them so HistoryScore sees the aged records
    # (genuine entries always win on a key collision — there are none by design).
    for pid, h in pop.histories.items():
        histories.setdefault(pid, h)

    reserve_counts: Dict[str, int] = {}
    for r in reserve:
        key = r.gt_category.value if r.gt_category else "?"
        reserve_counts[key] = reserve_counts.get(key, 0) + 1
    meta = {
        "n_events": n_events,
        "now": now.isoformat(),
        "degraded": degraded,
        "category_counts": counts,
        "total_records": len(records),
        "generators": {k: v.tag for k, v in clients.items()},
        "seed": seed,
        "attack_prevalence_pct": round(
            100.0 * sum(v for k, v in counts.items() if k.startswith("ATTACK")) / max(1, len(records)), 2),
        "reserve_multiplier": mult,
        "reserve_counts": reserve_counts,
        "reserve_records": len(reserve),
    }
    if verbose:
        print(f"[build] assembled {len(records)} records: {counts}")
        cache = clients["A"].cache
        if cache is not None:
            print(f"[build] LLM cache: {cache.hits} hit(s), {cache.misses} miss(es) "
                  f"-> {cache.path}")
    return Benchmark(records=records, events={
        eid: {"event_type": ev.event_type.value, "centroid": ev.centroid,
              "split": ev.split, "cancelled": ev.cancelled} for eid, ev in events.items()
    }, histories=histories, meta=meta, reserve=reserve)


# --------------------------------------------------------------------------
# helpers for attack overlays
# --------------------------------------------------------------------------
def _mask_lifecycle(text: str) -> str:
    """Mask in-text lifecycle references (Section 4.3 leakage rule)."""
    return re.sub(r"CANCEL\s+NAVAREA\s+IV\s+\d+\s*/\s*\d+", "[REF MASKED]", text or "", flags=re.I)


def _weighted(rng: random.Random, tiers: Dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for name, w in tiers.items():
        acc += w
        if r <= acc:
            return name
    return list(tiers)[-1]


def _expertise(contrib) -> str:
    return {"PROFESSIONAL": "officer", "MARINE_CERTIFIED": "officer",
            "CONSUMER_GPS": "recreational", "MOBILE_DEVICE": "recreational",
            "UNKNOWN": "fishing"}.get(contrib.device_class, "recreational")


def _crowd_structured(text: str, quality: str) -> Dict:
    """Structured attributes extracted from the report's OWN text (Section 4.3).

    ``hazard_type`` is recovered by the same keyword extractor applied to official
    text — so it is derived, not inherited from the seed event. The crowd report's
    existence assertion (``in_force``) and a quality-conditioned severity remain
    report-level statements, not event ground truth.
    """
    s = {"hazard_type": _hazard_key(text), "existence_status": "in_force"}
    if quality == "good":
        s["severity_level"] = "major"
    elif quality == "poor":
        s.pop("existence_status", None)  # sparse
    return s


def _attack_split(rng: random.Random, dev_frac: float) -> str:
    return "dev" if rng.random() < dev_frac else "test"


def _tamper(src: MSIRecord, i: int, rng: random.Random, dev_frac: float,
            id_prefix: str = "t3") -> MSIRecord:
    from copy import deepcopy
    r = deepcopy(src)
    r.record_id = f"{id_prefix}-{i}"
    mode = rng.choice(["shift", "type", "severity", "existence"])
    perturb = {}
    if mode == "shift" and r.centroid:
        shift_nm = rng.uniform(5, 20)
        r.centroid = gen.noise_position(r.centroid, shift_nm, rng)
        r.coordinates = [r.centroid]
        perturb = {"mode": "coord_shift", "shift_nm": round(shift_nm, 1)}
    elif mode == "type":
        new = rng.choice(["wreck", "rocket", "iceberg"])
        r.structured_data = dict(r.structured_data, hazard_type=new)
        perturb = {"mode": "type_swap", "to": new}
    elif mode == "existence":
        # deny a real hazard → affirm-vs-deny (irreconcilable) conflict with the
        # genuine reports of the event; exercises CRITICAL partitioning
        r.structured_data = dict(r.structured_data, existence_status="cancelled")
        perturb = {"mode": "existence_flip", "to": "cancelled"}
    else:
        r.structured_data = dict(r.structured_data, severity_level="minor")
        perturb = {"mode": "severity_downgrade"}
    r.gt_category = Category.ATTACK_TAMPER
    r.gt_event_id = None
    r.gt_quality_tier = None
    r.provenance = {"split": _attack_split(rng, dev_frac), "tampered_from": src.record_id,
                    "perturbation": perturb, "orig_event": src.gt_event_id}
    return r


def _impersonate(src: MSIRecord, i: int, rng: random.Random, dev_frac: float,
                 id_prefix: str = "t4") -> MSIRecord:
    from copy import deepcopy
    r = deepcopy(src)
    r.record_id = f"{id_prefix}-{i}"
    r.provider_type = ProviderType.OFFICIAL          # claims official provenance
    r.authority_norm = f"FAKE HYDROGRAPHIC OFFICE {rng.randint(1,99)}"  # not in real registry
    r.auth_level = "ANONYMOUS"                        # fails PKI credential check
    r.device_class = "UNKNOWN"
    r.gt_category = Category.ATTACK_IMPERSONATION
    r.gt_event_id = None
    r.provenance = {"split": _attack_split(rng, dev_frac), "impersonated_from": src.record_id,
                    "credential_valid": False}
    return r


def _replay_record(m, i, jitter_h, split, rng2, pop, farmed, w_start, span_s,
                   now, qt, degraded, id_prefix="t2", adv_tag="t2a") -> MSIRecord:
    """Build one T2 replay record (shared by the base build and the reserve).

    Replay (Section 4.3): obsolete 2020–2021 text resubmitted with a
    CURRENT (in-window) timestamp but its ORIGINAL (long-expired) validity
    period. Detection is via temporal/lifecycle consistency (the issue time
    falls after the validity window has ended), not issue-time age or identity
    metadata. ``rng2`` is the record's derived RNG; its draw ORDER (identity,
    issue time, submission offset, then the field-shape quality) is fixed, so
    T2 identity/timing metadata does not depend on later draws.
    """
    adv = (pop.adversary(rng2, i, farmed=farmed, tag=adv_tag) if pop is not None
           else None)
    r_issue = (w_start + timedelta(seconds=rng2.uniform(0, span_s))
               - timedelta(hours=jitter_h)) if span_s > 0 else now - timedelta(hours=jitter_h)
    submission_pos = gen.noise_position(m.centroid, rng2.uniform(1.0, 8.0), rng2)
    quality = _weighted(rng2, qt)  # parity draw, appended AFTER the original draws
    v_start = m.issue_time
    v_end = m.cancel_time or m.self_expiry_time or (m.issue_time + timedelta(hours=72))
    text = _mask_lifecycle(m.raw_text)
    return MSIRecord(
        record_id=f"{id_prefix}-{i}", event_type=m.event_type,
        provider_type=ProviderType.CROWDSOURCE,
        provider_id=adv.provider_id if adv else f"replay-{i}",
        raw_text=text,
        structured_data=_attack_structured(text, quality, degraded),
        geometry_kind=GeometryKind.POINT, coordinates=[m.centroid], centroid=m.centroid,
        issue_time=r_issue, valid_start=v_start, valid_end=v_end,
        device_class=adv.device_class if adv else "MOBILE_DEVICE",
        auth_level=adv.auth_level if adv else "ANONYMOUS",
        network_origin=adv.network_origin if adv else None,
        submission_position=submission_pos, observation_type="visual",
        gt_category=Category.ATTACK_REPLAY,
        provenance={"split": split, "replayed_from": m.uid,
                    "identity_tier": "farmed" if farmed else "throwaway"},
    )


def _add_replay(cfg, rng, now, dev_frac, n_events, bcfg, add, verbose,
                pop=None, window_start=None, seed=0, degraded=False):
    """T2: real cancelled warnings from the replay years, resubmitted in-window.

    Adaptive-adversary hardening: replays are submitted
    through alternating throwaway/farmed identities — not a hardcoded
    anonymous-mobile signature — from a realistic observer offset rather than
    the hazard position itself, and their submission times are spread across
    the operational window like every other report. The detection signal is the
    designed one: expired validity at submission time (cte + validity lapse),
    not identity or recency fingerprints. Original shared-rng draws (shuffle,
    per-record uniform, split) are preserved; new randomness is derived.

    Returns the UNUSED remainder of the replay pool so the attack reserve
    can draw additional distinct replays without reparsing.
    """
    n_replay = int(round(bcfg["attack_multipliers"]["replay"] * n_events))
    if verbose:
        print(f"[build] sourcing {n_replay} real replay-attack messages …")
    pool = parse_corpus(cfg, cfg.years("replay"), nav_area=cfg.nav_area,
                        functions=[MessageFunction.NEW_WARNING, MessageFunction.UPDATE])
    pool = [m for m in pool if m.centroid and m.status == "C" and m.issue_time]
    rng.shuffle(pool)
    w_start = window_start or now
    span_s = max(0.0, (now - w_start).total_seconds())
    qt = bcfg["quality_tiers"]
    for i, m in enumerate(pool[:n_replay]):
        jitter_h = rng.uniform(0, 12)  # original draw, kept as intra-day jitter
        split = _attack_split(rng, dev_frac)
        add(_replay_record(m, i, jitter_h, split, random.Random(f"{seed}|t2-{i}"),
                           pop, i % 2 == 1, w_start, span_s, now, qt, degraded))
    return pool[n_replay:]


def _add_sybil(cfg, rng, now, dev_frac, event_list, clients, pop, bcfg, add,
               salt_prefix="", event_epoch=None, seed=0, degraded=False,
               n_camp=None, id_prefix="t5", dev_tag="dev"):
    """T5: coordinated campaigns along a four-level sophistication axis.

    Identity metadata scales with sophistication: **low** =
    throwaway accounts sharing a device prefix and subnet (tests independence
    verification); **medium** = registered, thinly-farmed accounts still sharing
    a /24 and tight positions (tests the dispersion penalty); **high** = fully
    farmed identities with genuine-mix devices, aged histories, and distinct
    networks (tests the limits of both defences — and is caught by its scripted
    near-verbatim text via independence check (d)). **paraphrased** keeps the high
    tier's farmed identities and dispersion but authors DISTINCT prose per member —
    the residual limit of check (d), modelled so the per-tier table measures it instead of
    asserting it. Networks live in the same address space as genuine
    contributors — the *shared subnet* is the coordination signal, not a
    provenance prefix. Campaign timing is anchored to the real event the
    campaign parasitises. Original shared-rng draws preserved; base texts carry
    a genuine-anchored length target and every member's structured fields come
    from the platform extractor.

    ``n_camp``/``id_prefix``/``dev_tag`` let the attack reserve
    reuse this generator under a disjoint namespace ("t5r"/"rdev") and its own
    RNG, so reserve campaigns never collide with — or perturb — the base build.
    """
    # Scale campaign count with corpus size so Sybil reports stay a realistic
    # minority of the overlay (the configured value targets the full |E|).
    if n_camp is None:
        n_camp = min(bcfg["sybil_campaigns"], max(2, len(event_list) // 30))
    gen_max_tokens = int(cfg.get("llm", "gen_max_tokens", default=gen.GEN_MAX_TOKENS))
    qt = bcfg["quality_tiers"]
    levels = ["low", "medium", "high", "paraphrased"]
    # Per-level coordinated burst window (hours): low = minutes, medium = 2-hour
    # burst, high/paraphrased = 6-hour window (Section 4.3).
    # Only the WITHIN-campaign spread carries the coordination signal, which is
    # detected via independence and dispersion, not freshness.
    _CAMP_WINDOW_H = {"low": 0.1, "medium": 2.0, "high": 6.0, "paraphrased": 6.0}
    _TIER = {"low": "throwaway", "medium": "registered", "high": "farmed",
             "paraphrased": "paraphrased"}
    for c in range(n_camp):
        level = levels[c % 4]
        k = rng.randint(3, 8)
        # The campaign claims ONE fake hazard at `hazard`; members submit from
        # (faked) submission positions whose spread defines the sophistication.
        anchor = rng.choice(event_list)
        hazard = gen.noise_position(anchor.centroid, 3.0, rng)
        hk = rng.choice(["derelict", "wreck", "generic"])
        target_words = gen.sample_target_words(
            random.Random(f"{seed}|{id_prefix}len-{c}"))  # derived: no shared draw
        base_text = gen.generate_fabrication_text(clients["A"], hk, True, rng,
                                                  max_tokens=gen_max_tokens,
                                                  cache_salt=f"{salt_prefix}:{id_prefix}-{c}",
                                                  target_words=target_words)
        n1, n2 = rng.randint(0, 255), rng.randint(0, 255)   # original net draws
        shared_net = f"10.{n1}.{n2}.0/24"
        shared_dev_prefix = f"{dev_tag}{rng.randint(100,999)}"
        camp_age = _event_age_days(rng)  # original draw, reused for the epoch
        camp_epoch = _attack_epoch(camp_age, anchor, event_epoch, None, now) \
            if event_epoch else now - timedelta(days=camp_age)
        window_h = _CAMP_WINDOW_H[level]
        member_texts: List[str] = []  # accepted paraphrases (containment gate)
        for j in range(k):
            containment = None
            if level == "low":
                subpos = gen.noise_position(hazard, 0.1, rng)   # collocated, < δind
                dev_id, net = shared_dev_prefix, shared_net
                text = base_text
            elif level == "medium":
                subpos = gen.noise_position(hazard, 0.8, rng)   # tight, same /24
                dev_id, net = f"{shared_dev_prefix}-{j}", shared_net
                text = base_text + f" ({['confirm','also seen','still there'][j % 3]})"
            elif level == "high":  # dispersed, distinct networks
                subpos = gen.noise_position(hazard, 6.0, rng)
                dev_id = f"{dev_tag}{rng.randint(100,999)}-{j}"
                n3, n4 = rng.randint(0, 255), rng.randint(0, 255)  # original draws
                net = f"10.{n3}.{n4}.0/24"
                text = base_text + f" — {['per AIS','visual','radar'][j % 3]}"
            else:  # paraphrased: high-tier identity + distinct prose per member
                subpos = gen.noise_position(hazard, 6.0, rng)
                dev_id = f"{dev_tag}{rng.randint(100,999)}-{j}"
                n3, n4 = rng.randint(0, 255), rng.randint(0, 255)
                net = f"10.{n3}.{n4}.0/24"
                text = gen.generate_paraphrase(
                    clients["A"], base_text, hk,
                    random.Random(f"{seed}|{id_prefix}p|{c}|{j}"),
                    siblings=member_texts, max_tokens=gen_max_tokens,
                    cache_salt=f"{salt_prefix}:{id_prefix}-{c}-p{j}")
                containment = round(max(
                    [gen.text_duplication(text, base_text)]
                    + [gen.text_duplication(text, s) for s in member_texts]), 3)
                member_texts.append(text)
            raw = gen.substitute_coord(text, hazard[0], hazard[1])
            issue_t = camp_epoch + timedelta(hours=rng.uniform(0, window_h))
            legacy_device = rng.choice(["MOBILE_DEVICE", "UNKNOWN"])  # original draw
            if level == "low":
                device, auth = legacy_device, "ANONYMOUS"
            elif level == "medium":  # registered, thinly-farmed (λ=5) accounts
                ident = pop.farm_identity(dev_id, lam=5,
                                          auth_mix=[("REGISTERED", 1.0)],
                                          key=f"{id_prefix}|{c}|{j}")
                device, auth = ident.device_class, ident.auth_level
            else:                    # high/paraphrased: fully farmed identities
                ident = pop.farm_identity(dev_id, key=f"{id_prefix}|{c}|{j}")
                device, auth = ident.device_class, ident.auth_level
            quality = _weighted(random.Random(f"{seed}|{id_prefix}q|{c}|{j}"), qt)
            prov = {"split": _attack_split(rng, dev_frac), "campaign": c,
                    "sophistication": level, "identity_tier": _TIER[level]}
            if containment is not None:
                # realized word-3-gram containment vs base + accepted siblings —
                # whether the tier ACTUALLY defeats check (d) is measured, not
                # assumed (θ_ndup = 0.75 is the firing threshold).
                prov["containment_vs_campaign"] = containment
            add(MSIRecord(
                record_id=f"{id_prefix}-{c}-{j}", event_type=EventType.NAV_WARNING,
                provider_type=ProviderType.CROWDSOURCE, provider_id=dev_id,
                raw_text=raw, structured_data=_attack_structured(text, quality, degraded),
                geometry_kind=GeometryKind.POINT, coordinates=[hazard], centroid=hazard,
                issue_time=issue_t,
                device_class=device, auth_level=auth,
                network_origin=net, submission_position=subpos, observation_type="visual",
                gt_category=Category.ATTACK_SYBIL,
                provenance=prov,
            ))


def _build_attack_reserve(cfg, mult, bcfg, seed, now, dev_frac, event_list,
                          event_epoch, window_start, pop, clients, crowd_pool,
                          replay_leftover, salt_prefix, gen_max_tokens,
                          concurrency, degraded, verbose) -> List[MSIRecord]:
    """The attack reserve: ``mult`` × the base attack counts, as EXTRA records.

    Same generators, same sophistication tiers, same anchoring as the base
    overlay — only the namespace (record ids, cache salts, adversary identity
    tags, Sybil device prefixes and farm keys) is disjoint, so reserve
    identities can never collide with or overwrite base identities, and the
    reserve draws exclusively from RNGs derived from ``f"{seed}|reserve"``,
    leaving the base build's shared draw sequence untouched. T2 draws from the
    UNUSED remainder of the real cancelled-warning pool (distinct real texts);
    if the pool cannot supply ``mult`` × the base count the shortfall is
    reported in the meta rather than padded with duplicates.
    """
    rng_r = random.Random(f"{seed}|reserve")
    reserve: List[MSIRecord] = []

    def addr(rec: MSIRecord):
        rec.provenance["reserve"] = True
        reserve.append(rec)

    n_events = len(event_list)
    parallel = concurrency > 1 and not all(c.is_mock for c in clients.values())

    # T1 fabrications (LLM text; two-phase parallel like the base build)
    n_fab_r = int(round(mult * bcfg["attack_multipliers"]["fabrication"] * n_events))
    fab_jobs = [_draw_fab_job(i, rng_r, pop, event_list, now, dev_frac, salt_prefix,
                              seed, event_epoch, window_start, bcfg["quality_tiers"],
                              id_prefix="t1r", adv_tag="advr")
                for i in range(n_fab_r)]
    if verbose and fab_jobs:
        print(f"[build]   reserve: generating {len(fab_jobs)} fabrication texts …",
              flush=True)
    fab_texts = parallel_map(lambda j: _fab_text(j, clients, gen_max_tokens),
                             fab_jobs, concurrency if parallel else 1, verbose,
                             "reserve-t1")
    for job, text in zip(fab_jobs, fab_texts):
        addr(_build_fab_record(job, text, degraded))

    # T2 replays from the leftover real pool (no LLM; bounded by the archive)
    n_replay_r = int(round(mult * bcfg["attack_multipliers"]["replay"] * n_events))
    leftover = list(replay_leftover)
    rng_r.shuffle(leftover)
    w_start = window_start or now
    span_s = max(0.0, (now - w_start).total_seconds())
    qt = bcfg["quality_tiers"]
    for i, m in enumerate(leftover[:n_replay_r]):
        jitter_h = rng_r.uniform(0, 12)
        split = _attack_split(rng_r, dev_frac)
        addr(_replay_record(m, i, jitter_h, split, random.Random(f"{seed}|t2r-{i}"),
                            pop, i % 2 == 1, w_start, span_s, now, qt, degraded,
                            id_prefix="t2r", adv_tag="t2r"))

    # T3 tampers / T4 impersonations of base genuine crowd (programmatic)
    n_tamper_r = int(round(mult * bcfg["attack_multipliers"]["tamper"] * n_events))
    for i in range(min(n_tamper_r, len(crowd_pool))):
        src = crowd_pool[rng_r.randrange(len(crowd_pool))]
        addr(_tamper(src, i, rng_r, dev_frac, id_prefix="t3r"))
    n_imp_r = int(round(mult * bcfg["attack_multipliers"]["impersonation"] * n_events))
    for i in range(min(n_imp_r, len(crowd_pool))):
        src = crowd_pool[rng_r.randrange(len(crowd_pool))]
        addr(_impersonate(src, i, rng_r, dev_frac, id_prefix="t4r"))

    # T5 Sybil campaigns (LLM base texts + paraphrased-tier member texts)
    n_camp_base = min(bcfg["sybil_campaigns"], max(2, n_events // 30))
    _add_sybil(cfg, rng_r, now, dev_frac, event_list, clients, pop, bcfg, addr,
               salt_prefix, event_epoch=event_epoch, seed=seed, degraded=degraded,
               n_camp=int(round(mult * n_camp_base)), id_prefix="t5r", dev_tag="rdev")

    if verbose:
        cnt: Dict[str, int] = {}
        for r in reserve:
            cnt[r.gt_category.value] = cnt.get(r.gt_category.value, 0) + 1
        print(f"[build] attack reserve (x{mult:g}): {len(reserve)} records {cnt}")
    return reserve


def _datasheet(meta: Dict) -> str:
    return (
        "# MSI-Bench Datasheet\n\n"
        "Following Gebru et al., *Datasheets for Datasets* (CACM 2021).\n\n"
        "## Motivation\nA reproducible semi-synthetic benchmark for credibility "
        "assessment and fusion of multi-source Maritime Safety Information. No public "
        "corpus of crowdsourced MSI exists, and labelled real attacks are unobtainable; "
        "the crowdsourced and adversarial layers are therefore simulated by necessity "
        "(Section 4.1).\n\n"
        "## Composition\n"
        f"- Events (real backbone): {meta.get('n_events')}\n"
        f"- Total records: {meta.get('total_records')}\n"
        f"- Category counts: {meta.get('category_counts')}\n"
        f"- Attack prevalence: {meta.get('attack_prevalence_pct')}%\n"
        f"- Generators: {meta.get('generators')}\n"
        f"- Reference time: {meta.get('now')}\n"
        f"- Seed: {meta.get('seed')}\n\n"
        "## Provenance & channel separation\nThe real backbone is the public-domain "
        "NGA NAVAREA IV archive. Narrative text in synthetic records is LLM- or "
        "template-generated; ALL coordinates, timestamps, device/network identifiers, "
        "tampering perturbations, and forged credentials are produced by audited code. "
        "Every synthetic record stores its generator, split, and attack provenance.\n\n"
        "## Collection process\nGenerated by `msi_bench.generation.build_benchmark`; "
        "lifecycle references are masked from official text where chains define ground "
        "truth. Real replay text is drawn from cancelled 2020–2021 warnings.\n\n"
        "## Recommended uses\nFramework validation under an explicit threat model. "
        "NOT a substitute for operational field data. License: NGA broadcast warnings "
        "are U.S. government work (public domain); synthetic overlay released under the "
        "repository license.\n"
    )
