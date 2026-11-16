"""Property and sanity tests for the MSI-Bench framework core.

Runs under pytest *or* directly (``python tests/test_framework.py``). Covers:

* geometry/temporal parsing,
* MCAM properties (boundedness, monotonicity, neutral social default,
  cold-start history, dispersion penalty),
* STS-DBSCAN on a toy two-cluster set,
* D-S fusion BPA normalisation and combination behaviour.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msi_bench.clustering import STSDBSCAN
from msi_bench.credibility.domain import NavigabilityModel
from msi_bench.credibility.mcam import MCAM, MCAMParams, credibility_level
from msi_bench.experiments.freeze import freeze_dir
from msi_bench.fusion import build_bpa, dempster_combine, fuse_subcluster, mean_pairwise_conflict
from msi_bench.geo import extract_coordinates, haversine_nm, parse_geometry
from msi_bench.llm.cache import ResponseCache
from msi_bench.llm.client import LLMClient, MockClient
from msi_bench.schema import EventType, GeometryKind, MSIRecord, ProviderType
from msi_bench.temporal import parse_dtg

_NAV = NavigabilityModel(centroids=[(28.0, -80.0), (29.0, -79.0), (10.0, -60.0)])


def _rec(rid, *, cd_strong=True, centroid=(28.5, -80.0), structured=None,
         issue=datetime(2024, 5, 10, 0, 0), provider="p", net=None, subpos=None,
         text="derelict vessel adrift", ptype=ProviderType.CROWDSOURCE):
    return MSIRecord(
        record_id=rid,
        event_type=EventType.NAV_WARNING,
        provider_type=ptype,
        provider_id=provider,
        raw_text=text,
        structured_data=structured or {"hazard_type": "derelict", "position": 1},
        geometry_kind=GeometryKind.POINT,
        coordinates=[centroid],
        centroid=centroid,
        issue_time=issue,
        device_class="PROFESSIONAL" if cd_strong else "UNKNOWN",
        auth_level="PKI_FULL" if cd_strong else "ANONYMOUS",
        observation_type="visual",
        network_origin=net,
        submission_position=subpos or centroid,
    )


def test_geo_parsing():
    coords = extract_coordinates("28-47.00N 080-15.00W, 28-50.00N 080-21.00W")
    assert len(coords) == 2
    assert abs(coords[0][0] - 28.7833) < 1e-3 and abs(coords[0][1] + 80.25) < 1e-3
    # phone numbers / serials must not parse as coordinates
    assert extract_coordinates("PHONE: 305 415 6800, SERIAL 0/24") == []
    _, kind, _ = parse_geometry("IN AREAS BOUND BY: 28-47.00N 080-15.00W, 28-50N 080-21W, 28-40N 080-36W")
    assert kind == GeometryKind.POLYGON
    assert haversine_nm((28.0, -80.0), (29.0, -80.0)) > 55  # ~60 nm per degree lat


def test_temporal_parsing():
    dt, defect = parse_dtg("100049Z MAY 2024")
    assert defect is None and dt == datetime(2024, 5, 10, 0, 49)
    dt2, _ = parse_dtg("212231Z MAY 24")  # two-digit year
    assert dt2 == datetime(2024, 5, 21, 22, 31)
    _, d = parse_dtg("not a dtg")
    assert d == "no_dtg"


def test_mcam_boundedness():
    rng = np.random.default_rng(0)
    recs = [_rec(f"r{i}", cd_strong=bool(i % 2), centroid=(28 + 0.01 * i, -80))
            for i in range(20)]
    sim = rng.random((20, 20)); sim = (sim + sim.T) / 2; np.fill_diagonal(sim, 1.0)
    mcam = MCAM(MCAMParams(), navigability=_NAV, now=datetime(2024, 5, 11))
    scores = mcam.assess(recs, sim)
    for s in scores:
        for v in (s.cd, s.cc, s.cst, s.cso, s.cr):
            assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, (s.record_id, v)


def test_mcam_monotonicity():
    strong = _rec("strong", cd_strong=True,
                  structured={"hazard_type": "wreck", "position": 1, "severity_level": "major",
                              "existence_status": "confirmed", "restriction_type": "no_entry",
                              "valid_end": 1})
    weak = _rec("weak", cd_strong=False, centroid=(80.0, -10.0),  # far outside envelope
                structured={"hazard_type": "wreck"})
    sim = np.array([[1.0, 0.0], [0.0, 1.0]])  # isolated → equal neutral Cso
    mcam = MCAM(MCAMParams(), navigability=_NAV, now=datetime(2024, 5, 11))
    s = {x.record_id: x for x in mcam.assess([strong, weak], sim)}
    assert s["strong"].cd >= s["weak"].cd
    assert s["strong"].cc >= s["weak"].cc
    assert s["strong"].cr > s["weak"].cr


def test_social_neutral_default():
    r = _rec("solo")
    sim = np.array([[1.0]])
    mcam = MCAM(MCAMParams(), navigability=_NAV, now=datetime(2024, 5, 11))
    s = mcam.assess([r], sim)[0]
    assert abs(s.cso - 0.5) < 1e-6  # no neighbours → neutral (Eq. (6))


def test_history_cold_start():
    mcam = MCAM(MCAMParams(), now=datetime(2024, 5, 11))
    assert abs(mcam.history_score("never_seen") - 0.5) < 1e-9
    # all-correct, sufficient history → high
    hist = {"good": [(datetime(2024, 4, 1) + timedelta(days=i), "CORRECT") for i in range(15)]}
    mcam2 = MCAM(MCAMParams(), histories=hist, now=datetime(2024, 5, 11))
    assert mcam2.history_score("good") > 0.9


def test_dispersion_penalty():
    # Target plus three supportive neighbours, independent of the target
    # (distinct providers, ≈1.2 nm away → passes δind) but tightly clustered
    # among themselves (≈0.06 nm apart) → dispersion penalty < 1.
    base = (28.5, -80.0)
    target = _rec("t", provider="t", subpos=base)
    near = [_rec(f"n{i}", provider=f"n{i}",
                 centroid=base, subpos=(28.52 + 0.001 * i, -80.0)) for i in range(3)]
    recs = [target] + near
    sim = np.full((4, 4), 0.9); np.fill_diagonal(sim, 1.0)
    mcam = MCAM(MCAMParams(delta_disp_nm=5.0), navigability=_NAV, now=datetime(2024, 5, 11))
    s = mcam.assess(recs, sim)[0]
    assert s.n_support >= 2
    assert s.dispersion_nm < 5.0  # clustered corroboration
    assert s.cso < 1.0            # penalised


def test_credibility_levels():
    assert credibility_level(0.95) == "VERY HIGH"
    assert credibility_level(0.10) == "VERY LOW"


def test_clustering_two_groups():
    a = [_rec(f"a{i}", centroid=(28.5, -80.0)) for i in range(3)]
    b = [_rec(f"b{i}", centroid=(10.0, -60.0)) for i in range(3)]
    recs = a + b
    n = 6
    sim = np.full((n, n), 0.05)
    for i in range(3):
        for j in range(3):
            sim[i, j] = sim[i + 3, j + 3] = 0.9
    np.fill_diagonal(sim, 1.0)
    labels = STSDBSCAN(min_pts=2, eps=0.5).fit(recs, sim)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_fusion_bpa_and_combination():
    bpa = build_bpa(0.8, q=0.6)
    assert abs(sum(bpa.values()) - 1.0) < 1e-9
    m, k = dempster_combine(build_bpa(0.8), build_bpa(0.8))
    assert m["T"] > 0.8  # corroboration sharpens belief
    assert 0.0 <= mean_pairwise_conflict([build_bpa(0.9), build_bpa(0.1)]) <= 1.0
    fr = fuse_subcluster([_rec("x"), _rec("y")], [0.9, 0.85])
    assert 0.0 <= fr.belief <= fr.plausibility <= 1.0
    assert fr.centroid is not None


class _CountingClient(LLMClient):
    """Non-mock client that returns a unique string per call and counts calls."""

    def __init__(self):
        super().__init__(provider="fake", model="fake-1", temperature=0.8)
        self.calls = 0

    def _complete(self, system, user, max_tokens, temperature):
        self.calls += 1
        return f"resp{self.calls}:{user}"


def test_llm_response_cache():
    # Hit/miss, salt-based diversity, disk reload, and one transcript line per
    # unique generation.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "llm_cache.jsonl"
        c = _CountingClient(); c.cache = ResponseCache(path)
        a = c.complete("sys", "u", cache_salt="x")
        b = c.complete("sys", "u", cache_salt="x")        # same salt+prompt -> hit
        assert a == b and c.calls == 1                     # only one real API call
        other = c.complete("sys", "u", cache_salt="y")     # same prompt, new salt -> miss
        assert other != a and c.calls == 2                 # per-draw diversity preserved
        # A fresh cache+client reloads from disk and replays without any call.
        c2 = _CountingClient(); c2.cache = ResponseCache(path)
        assert c2.complete("sys", "u", cache_salt="x") == a and c2.calls == 0
        lines = [ln for ln in path.read_text("utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2                             # one line per unique generation
        rec = json.loads(lines[0])
        assert rec["response"] == a and rec["salt"] == "x" and rec["model"] == "fake-1"


def test_offline_cache_client():
    # Cross-machine replay: hit returns stored text without key/SDK/network; a
    # miss raises a clear error instead of calling out.
    from msi_bench.llm.client import OfflineCacheClient, OfflineCacheMiss
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "llm_cache.jsonl"
        gen = _CountingClient(); gen.cache = ResponseCache(path)   # "generated on the PC"
        text = gen.complete("sys", "u", cache_salt="s1:x")
        off = OfflineCacheClient(provider="fake", model="fake-1", temperature=0.8)
        off.cache = ResponseCache(path)                            # "replayed on the Mac"
        assert off.complete("sys", "u", cache_salt="s1:x") == text and gen.calls == 1
        try:
            off.complete("sys", "NOT-CACHED", cache_salt="s1:x")
            assert False, "expected OfflineCacheMiss"
        except OfflineCacheMiss:
            pass


def test_parallel_map_preserves_order():
    # Concurrency must not reorder results: output is positional, so the
    # assembled benchmark is identical regardless of worker count.
    from msi_bench.generation.benchmark import parallel_map
    items = list(range(64))
    expected = [x * x for x in items]
    assert parallel_map(lambda x: x * x, items, concurrency=1, verbose=False) == expected
    assert parallel_map(lambda x: x * x, items, concurrency=8, verbose=False) == expected


def test_mock_bypasses_cache():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "llm_cache.jsonl"
        m = MockClient(); m.cache = ResponseCache(path)
        assert m.complete("s", "u") == m.complete("s", "u")  # deterministic already
        assert not path.exists()                              # nothing cached/logged


def test_freeze_dir_manifest():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "benchmark"; src.mkdir()
        (src / "records.json").write_text('[{"record_id": "r1"}]', "utf-8")
        (src / "events.json").write_text("{}", "utf-8")
        (src / "histories.json").write_text("{}", "utf-8")
        (src / "meta.json").write_text('{"seed": 7, "n_events": 2, "total_records": 1}', "utf-8")
        (src / "DATASHEET.md").write_text("# DS", "utf-8")
        dst = Path(d) / "benchmark_release"
        freeze_dir(src, dst)
        man = json.loads((dst / "MANIFEST.json").read_text("utf-8"))
        assert man["seed"] == 7 and man["total_records"] == 1
        assert (dst / "records.json").read_text("utf-8") == '[{"record_id": "r1"}]'
        import hashlib
        h = hashlib.sha256((dst / "records.json").read_bytes()).hexdigest()
        assert man["files"]["records.json"]["sha256"] == h


def _trec(rid, issue, v_start, v_end, *, centroid=(28.5, -80.0), text="derelict vessel adrift"):
    """Point record with an explicit validity window (temporal-kernel tests)."""
    return MSIRecord(
        record_id=rid, event_type=EventType.NAV_WARNING,
        provider_type=ProviderType.OFFICIAL, provider_id="auth",
        raw_text=text, structured_data={},
        geometry_kind=GeometryKind.POINT, coordinates=[centroid], centroid=centroid,
        issue_time=issue, valid_start=v_start, valid_end=v_end,
        observation_type="shore", submission_position=centroid,
    )


def test_succession_kernel():
    from msi_bench.similarity import temporal_compatibility, temporal_similarity
    t0 = datetime(2023, 1, 1)
    # Lifecycle handoff: B's window starts exactly where A's ends, issues 30 d apart.
    a = _trec("a", t0, t0, t0 + timedelta(days=30))
    b = _trec("b", t0 + timedelta(days=30), t0 + timedelta(days=30), t0 + timedelta(days=55))
    assert temporal_similarity(a, b) < 0.2                      # overlap only: suppressed
    assert temporal_similarity(a, b, tau_succ_h=72.0) > 0.95    # succession: linked
    assert temporal_compatibility(a, b, tau_succ_h=72.0) > 0.95
    # Recurrence: same site, windows separated by 60 d of dead time.
    c = _trec("c", t0 + timedelta(days=90), t0 + timedelta(days=90), t0 + timedelta(days=95))
    assert temporal_compatibility(a, c, tau_succ_h=72.0) < 0.01
    # Same-event corroboration: near-identical windows → overlap carries the gate.
    d = _trec("d", t0 + timedelta(hours=6), t0, t0 + timedelta(days=30, hours=2))
    assert temporal_compatibility(a, d, tau_succ_h=72.0) > 0.9


def test_gated_composite_splits_recurrences_links_chains():
    from msi_bench.similarity import SimilarityModel, TextEncoder
    t0 = datetime(2023, 1, 1)
    chain_a = _trec("ca", t0, t0, t0 + timedelta(days=30))
    chain_b = _trec("cb", t0 + timedelta(days=30), t0 + timedelta(days=30),
                    t0 + timedelta(days=60))
    recur = _trec("rc", t0 + timedelta(days=120), t0 + timedelta(days=120),
                  t0 + timedelta(days=125))
    same = _trec("sm", t0 + timedelta(hours=3), t0, t0 + timedelta(days=30))
    enc = TextEncoder()
    enc.fit_tfidf(["derelict vessel adrift", "wreck reported dangerous to navigation",
                   "derelict vessel drifting northeast"])
    gate = SimilarityModel(encoder=enc, temporal_mode="gate", tau_succ_h=72.0)
    gate.prepare([chain_a, chain_b, recur, same])
    # Co-located, same-type, high text similarity everywhere: only the temporal
    # relation differs. The gate must separate recurrence from chain/same-event.
    assert gate.sim(chain_a, chain_b) > 0.8          # succession → linked
    assert gate.sim(chain_a, same) > 0.8             # co-present → linked
    assert gate.sim(chain_a, recur) < 0.1            # dead time → suppressed
    # The additive composite cannot: spatial+semantic alone keep recurrence high.
    add = SimilarityModel(encoder=enc, temporal_mode="additive")
    add.prepare([chain_a, recur])
    assert add.sim(chain_a, recur) > 0.6


def test_fast_matrix_matches_scalar_in_gate_mode():
    from msi_bench.similarity import SimilarityModel, TextEncoder
    t0 = datetime(2023, 1, 1)
    recs = [
        _trec("x1", t0, t0, t0 + timedelta(days=30)),
        _trec("x2", t0 + timedelta(days=30), t0 + timedelta(days=30),
              t0 + timedelta(days=60), text="wreck reported dangerous"),
        _trec("x3", t0 + timedelta(days=200), t0 + timedelta(days=200),
              t0 + timedelta(days=205), centroid=(28.6, -80.1)),
    ]
    enc = TextEncoder()
    enc.fit_tfidf([r.raw_text for r in recs] + ["hazard to navigation"])
    m = SimilarityModel(encoder=enc, temporal_mode="gate", tau_succ_h=72.0)
    S_fast = m.pairwise_matrix_fast(recs)
    m.prepare(recs)
    for i in range(3):
        for j in range(3):
            expect = 1.0 if i == j else m.sim(recs[i], recs[j])
            assert abs(S_fast[i, j] - expect) < 1e-9, (i, j, S_fast[i, j], expect)


def test_wildcard_attr_no_contradiction():
    # A contradiction needs two mutually exclusive LEAF categories; the
    # extractor wildcard ("generic"/"unknown") is the absence of a claim.
    from msi_bench.credibility.domain import attr_contradicts
    assert not attr_contradicts("hazard_type", "generic", "wreck")
    assert not attr_contradicts("hazard_type", "wreck", "unknown")
    assert attr_contradicts("hazard_type", "rocket", "wreck")
    assert attr_contradicts("existence_status", "in_force", "cancelled")
    assert not attr_contradicts("severity_level", "major", "minor")


def test_conflict_relatedness_floor():
    # Definition 2: a critical-attribute contradiction is a conflict
    # only between RELATED reports (composite Sim ≥ θ_rel). Same contradiction,
    # related pair → CONF_SEMANTIC; unrelated pair → nothing.
    from msi_bench.conflict import detect_conflicts
    from msi_bench.similarity import SimilarityModel, TextEncoder
    t0 = datetime(2024, 5, 10)
    a = _rec("a", structured={"hazard_type": "rocket", "existence_status": "in_force"},
             issue=t0)
    b_rel = _rec("b1", structured={"hazard_type": "wreck", "existence_status": "in_force"},
                 issue=t0)  # co-located, contemporaneous → related
    b_unrel = _rec("b2", centroid=(45.0, -40.0),
                   structured={"hazard_type": "wreck", "existence_status": "in_force"},
                   issue=t0 + timedelta(days=90))
    b_unrel.raw_text = "port entrance closed for scheduled survey operations"
    b_unrel.valid_start = b_unrel.issue_time
    b_unrel.valid_end = b_unrel.issue_time + timedelta(days=1)
    enc = TextEncoder()
    enc.fit_tfidf([a.raw_text, b_rel.raw_text, b_unrel.raw_text,
                   "derelict vessel adrift hazard",
                   "port entrance survey operations notice"])
    model = SimilarityModel(encoder=enc)
    model.prepare([a, b_rel, b_unrel])
    rel = detect_conflicts([a, b_rel], model, [0.8, 0.8])
    unrel = detect_conflicts([a, b_unrel], model, [0.8, 0.8])
    assert any(c.type == "CONF_SEMANTIC" for c in rel)
    assert not unrel


def test_r2_exempts_reference_linked_pairs():
    # Rule R2: disjoint validity windows are a temporal contradiction only when
    # the two reports are NOT linked by an update/cancellation reference. A
    # warning and the message that supersedes it are one lifecycle.
    from msi_bench.conflict import detect_conflicts
    from msi_bench.similarity import SimilarityModel, TextEncoder
    t0 = datetime(2024, 5, 10)

    def pair(linked):
        a = _rec("a", issue=t0, ptype=ProviderType.OFFICIAL)
        a.valid_start, a.valid_end = t0, t0 + timedelta(days=3)
        b = _rec("b", issue=t0 + timedelta(days=3), ptype=ProviderType.OFFICIAL)
        b.valid_start, b.valid_end = t0 + timedelta(days=3), t0 + timedelta(days=6)
        if linked:
            b.references = ["a"]
        return a, b

    enc = TextEncoder()
    enc.fit_tfidf(["derelict vessel adrift", "derelict vessel adrift", "derelict vessel adrift hazard",
                   "port entrance closed for scheduled survey operations",
                   "port entrance survey operations notice"])
    for linked in (False, True):
        a, b = pair(linked)
        model = SimilarityModel(encoder=enc)
        model.prepare([a, b])
        types = {c.type for c in detect_conflicts([a, b], model, [0.9, 0.9])}
        assert ("CONF_TEMPORAL" in types) == (not linked)

    # a three-message chain a <- b <- c: a and c are not linked directly, but
    # they are one lifecycle and must not be split by R2 either
    a, b = pair(True)
    c = _rec("c", issue=t0 + timedelta(days=6), ptype=ProviderType.OFFICIAL)
    c.valid_start, c.valid_end = t0 + timedelta(days=6), t0 + timedelta(days=9)
    for refs, expect_ac in ((["b"], False), ([], True)):
        c.references = refs
        model = SimilarityModel(encoder=enc)
        model.prepare([a, b, c])
        found = detect_conflicts([a, b, c], model, [0.9, 0.9, 0.9])
        temporal = {(x.i, x.j) for x in found if x.type == "CONF_TEMPORAL"}
        assert ((0, 2) in temporal) == expect_ac
        assert (0, 1) not in temporal


def test_references_serialisation_is_backward_compatible():
    from msi_bench.generation.benchmark import _record_from_dict, _record_to_dict
    r = _rec("x")
    d = _record_to_dict(r)
    assert "references" not in d                     # unchanged for old record sets
    assert _record_from_dict(d).references == []
    r.references = ["w-1"]
    assert _record_from_dict(_record_to_dict(r)).references == ["w-1"]


def test_text_duplication_measure_and_independence():
    # Near-verbatim (base text + token suffix) → containment ≈ 1 → the campaign
    # loses mutual support; paraphrases stay independent (Section 3.3 check (d)).
    from msi_bench.similarity import text_duplication
    base = "large derelict fishing vessel adrift and partially submerged near the shipping lane"
    assert text_duplication(base, base + " (confirm)") >= 0.9
    para = "half-sunken abandoned trawler drifting close to the traffic route, dangerous"
    assert text_duplication(base, para) < 0.3

    def campaign(theta):
        recs = []
        for i in range(3):
            r = _rec(f"c{i}", provider=f"c{i}",
                     subpos=(28.5 + 0.05 * i, -80.0 - 0.05 * i))  # passes δind
            r.raw_text = base + f" ({['confirm', 'also seen', 'still there'][i]})"
            recs.append(r)
        sim = np.full((3, 3), 0.9); np.fill_diagonal(sim, 1.0)
        mcam = MCAM(MCAMParams(theta_ndup=theta), navigability=_NAV,
                    now=datetime(2024, 5, 11))
        return mcam.assess(recs, sim)[0]
    with_check = campaign(0.75)
    without = campaign(1.01)
    assert without.n_support == 2       # near-dups corroborate when disabled
    assert with_check.n_support == 0    # one effective source under check (d)
    # paraphrased genuine corroboration survives the check
    texts = [base,
             "abandoned trawler partly under water drifting toward the traffic separation zone",
             "unmanned fishing boat awash, position uncertain, hazard to surface navigation"]
    recs = []
    for i, tx in enumerate(texts):
        r = _rec(f"g{i}", provider=f"g{i}", subpos=(28.5 + 0.05 * i, -80.0 + 0.05 * i))
        r.raw_text = tx
        recs.append(r)
    sim = np.full((3, 3), 0.9); np.fill_diagonal(sim, 1.0)
    mcam = MCAM(MCAMParams(theta_ndup=0.75), navigability=_NAV, now=datetime(2024, 5, 11))
    assert mcam.assess(recs, sim)[0].n_support == 2


def test_yager_rule_and_signed_bpa():
    from msi_bench.fusion import _iterative_dempster, _iterative_yager
    # signed BPA: a denial commits its credibility to {H_false}
    d = build_bpa(0.9, q=0.6, deny=True)
    assert d["F"] == 0.9 and abs(sum(d.values()) - 1.0) < 1e-9
    # Zadeh vignette: two highly credible contradicting reports
    zadeh = [build_bpa(0.99), build_bpa(0.99, deny=True)]
    y = _iterative_yager(zadeh)
    dm = _iterative_dempster(zadeh)
    assert y["Θ"] > 0.9                    # Yager: conflict → ignorance
    assert dm["T"] + dm["F"] > 0.9         # Dempster renormalises it away
    assert abs(sum(y.values()) - 1.0) < 1e-9
    # order invariance (associative TBM implementation)
    bpas = [build_bpa(0.8), build_bpa(0.3, deny=True), build_bpa(0.6)]
    y1 = _iterative_yager(bpas)
    y2 = _iterative_yager(list(reversed(bpas)))
    for k in ("T", "F", "Θ"):
        assert abs(y1[k] - y2[k]) < 1e-12


def test_fused_interpretation_and_hazard_vote():
    # weight-majority existence interpretation + wildcard-abstaining type vote
    affirm1 = _rec("a1", structured={"hazard_type": "wreck", "existence_status": "in_force"})
    affirm2 = _rec("a2", structured={"hazard_type": "generic", "existence_status": "in_force"})
    deny = _rec("d1", structured={"hazard_type": "wreck", "existence_status": "cancelled"})
    fr = fuse_subcluster([affirm1, affirm2, deny], [0.8, 0.7, 0.4])
    assert fr.interpretation == "affirm"
    assert fr.fused_hazard_type == "wreck"  # "generic" abstains
    fr2 = fuse_subcluster([deny], [0.8])
    assert fr2.interpretation == "deny"
    # uniform method must be credibility-blind end to end: unweighted centroid
    a = _rec("u1", centroid=(28.0, -80.0))
    b = _rec("u2", centroid=(29.0, -80.0))
    fu = fuse_subcluster([a, b], [1.0, 0.0], method="uniform")
    assert abs(fu.centroid[0] - 28.5) < 1e-9


def test_fusion_independence_discount():
    """Definition 4: near-verbatim members collapse to ONE effective source before
    BPA combination (the check-(d) Eq. (5) predicate — crowdsource-involving
    pair, ≥ 4 shingles each, containment ≥ θ_ndup); paraphrases, sensor
    channel pairs, and short fragments stay independent; content votes carry
    one voice per effective source; provenance still covers every member."""
    base = ("Large drifting container observed near the marked position, "
            "half submerged and barely visible, hazardous to small craft "
            "transiting the approach area")
    # scripted campaign: shared base text + per-member token suffix (the T5
    # generator's construction)
    camp = [_rec(f"s{i}", text=f"{base} extra token{i}") for i in range(4)]
    crs = [0.55, 0.62, 0.58, 0.57]
    fr_off = fuse_subcluster(camp, crs)                    # discount disabled
    fr_on = fuse_subcluster(camp, crs, theta_ndup=0.75)
    single = fuse_subcluster([camp[1]], [0.62], theta_ndup=0.75)
    assert fr_off.n_effective_sources == 4
    assert fr_off.belief > 0.8            # k-sharpening from one authored text
    assert fr_on.n_effective_sources == 1
    assert abs(fr_on.belief - single.belief) < 1e-12  # capped at best member
    assert len(fr_on.provenance) == 4                 # bookkept, not corroborating
    # genuinely distinct prose (independent observers or a paraphrasing
    # adversary) stays independent — corroboration still sharpens
    para = [_rec("p1", text=("Fishing gear and long lines reported drifting "
                             "north of the channel entrance, small buoys attached")),
            _rec("p2", text=("Observed several detached net floats moving "
                             "slowly with the current toward the fairway"))]
    fp = fuse_subcluster(para, [0.6, 0.6], theta_ndup=0.75)
    assert fp.n_effective_sources == 2
    assert fp.belief > 0.6
    # channel scoping: sensor feeds are templated by design — identical text
    # does not evidence common authorship on an authenticated channel
    sens = [_rec("sen1", text=base, ptype=ProviderType.SENSOR),
            _rec("sen2", text=base, ptype=ProviderType.SENSOR)]
    fs = fuse_subcluster(sens, [0.7, 0.7], theta_ndup=0.75)
    assert fs.n_effective_sources == 2
    # a crowdsourced verbatim relay of an official broadcast IS collapsed —
    # it confirms the broadcast's reach, not the hazard; the representative is
    # the higher-credibility member (the official original)
    off = _rec("off", text=base, ptype=ProviderType.OFFICIAL,
               structured={"hazard_type": "wreck"})
    relay = _rec("rel", text=f"{base} forwarded",
                 structured={"hazard_type": "wreck"})
    fo = fuse_subcluster([off, relay], [0.95, 0.6], theta_ndup=0.75)
    assert fo.n_effective_sources == 1
    assert abs(fo.belief - fuse_subcluster([off], [0.95]).belief) < 1e-12
    # short fragments (< 4 shingles) cannot evidence common authorship
    ft = fuse_subcluster([_rec("t1"), _rec("t2")], [0.6, 0.6], theta_ndup=0.75)
    assert ft.n_effective_sources == 2
    # content votes: one voice per effective source — three verbatim "wreck"
    # scripts no longer outvote one independent higher-credibility "rocket"
    wreckers = [_rec(f"w{i}", text=f"{base} tail{i}",
                     structured={"hazard_type": "wreck"}) for i in range(3)]
    rocket = _rec("r", text=("Spent rocket stage floating vertically near the "
                             "position, high visibility paint, drifting east"),
                  structured={"hazard_type": "rocket"})
    f_on = fuse_subcluster(wreckers + [rocket], [0.6, 0.6, 0.6, 0.9],
                           theta_ndup=0.75)
    f_off = fuse_subcluster(wreckers + [rocket], [0.6, 0.6, 0.6, 0.9])
    assert f_on.fused_hazard_type == "rocket"
    assert f_off.fused_hazard_type == "wreck"


class _ScriptedClient(LLMClient):
    """Non-mock client returning queued responses (records prompts)."""

    def __init__(self, responses):
        super().__init__(provider="scripted", model="s-1", temperature=0.0,
                         is_mock=False, cache=None)
        self._queue = list(responses)
        self.prompts = []

    def _complete(self, system, user, max_tokens, temperature):
        self.prompts.append((system, user))
        return self._queue.pop(0) if self._queue else self._queue_default()

    def _queue_default(self):
        return "Observed drifting hull with cargo nets near {{COORD}}, keeping clear."


def test_fabrication_length_target_prompt():
    """Length matching: the sophisticated prompt carries the word target;
    the naive prompt carries no length instruction."""
    import random as _r
    from msi_bench.generation import generators as g
    c = _ScriptedClient(["Drifting barge sighted near {{COORD}}, gave it a wide berth."] * 2)
    g.generate_fabrication_text(c, "wreck", True, _r.Random(0), target_words=41)
    assert c.prompts[-1][1] == ("Hazard type: wreck. Style: complete, plausible, "
                                "in a shipping lane, correct register. "
                                "Aim for roughly 41 words.")
    g.generate_fabrication_text(c, "wreck", False, _r.Random(0), target_words=None)
    assert c.prompts[-1][1] == "Hazard type: wreck. Style: terse and incomplete."
    tw = g.sample_target_words(_r.Random(7))
    assert tw in g.CROWD_WORDCOUNT_Q


def test_paraphrase_gate_and_fallback():
    """The paraphrased-Sybil generator retries near-verbatim candidates (the
    adaptive adversary proofs their own texts against check (d)), screens
    refusals, and guarantees the COORD token."""
    import random as _r
    from msi_bench.generation import generators as g
    from msi_bench.similarity import text_duplication
    base = ("Reporting a partially submerged container adrift near {{COORD}}, "
            "struck floating debris twice while passing, all vessels keep clear.")
    distinct = ("Heads up — our lookout spotted what looks like a sunken box "
                "floating low in the water by {{COORD}}, nearly hit it.")
    # attempt 0: refusal → skipped; attempt 1: verbatim copy → gated; attempt 2 ok
    c = _ScriptedClient(["I'm sorry, I can't help with that.", base, distinct])
    out = g.generate_paraphrase(c, base, "wreck", _r.Random(3))
    assert out == distinct and text_duplication(out, base) < 0.6
    # all candidates near-verbatim → keeps the least-contained, never crashes
    c2 = _ScriptedClient([base, base, base])
    out2 = g.generate_paraphrase(c2, base, "wreck", _r.Random(3))
    assert "{{COORD}}" in out2
    # mock path: deterministic, distinct across members, below the gate
    mock = MockClient()
    texts = [g.generate_paraphrase(mock, base, "wreck", _r.Random(f"m{j}"))
             for j in range(4)]
    assert len(set(texts)) == len(texts)
    assert all(text_duplication(t, base) < 0.6 for t in texts)


def test_attack_structured_parity():
    """Extractor parity: attack structured fields come from the same
    quality-conditioned platform extractor as genuine crowd; the naive
    duration form-field rides on top; the degraded bracket strips schema
    from every crowdsource-channel record, attacks included."""
    from msi_bench.generation.benchmark import _attack_structured, _crowd_structured
    text = "saw a submerged wreck right by 28-00.00N 080-00.00W, keep clear"
    for q in ("good", "moderate", "poor"):
        assert _attack_structured(text, q, False) == _crowd_structured(text, q)
    s = _attack_structured(text, "good", False, duration=-5)
    assert s["duration_hours"] == -5 and s["hazard_type"] == "wreck"
    assert _attack_structured(text, "good", True) == {}
    assert _attack_structured(text, "good", True, duration=-5) == {}


def test_view_at_prevalence_fixed_genuine():
    """The pressure sweep holds genuine fixed, scales attacks from
    base+reserve stratified by type, and flags pool exhaustion."""
    from msi_bench.experiments.e6_endtoend import _view_at_prevalence
    from msi_bench.generation.benchmark import Benchmark
    from msi_bench.schema import Category
    genuine = []
    for i in range(40):
        r = _rec(f"g{i}")
        r.gt_category = Category.GENUINE_CROWD
        genuine.append(r)
    base_atk, reserve = [], []
    for i in range(6):
        r = _rec(f"a{i}")
        r.gt_category = Category.ATTACK_FABRICATION if i % 2 else Category.ATTACK_SYBIL
        base_atk.append(r)
    for i in range(54):
        r = _rec(f"r{i}")
        r.gt_category = Category.ATTACK_FABRICATION if i % 2 else Category.ATTACK_SYBIL
        r.provenance["reserve"] = True
        reserve.append(r)
    bench = Benchmark(records=genuine + base_atk, events={}, histories={},
                      meta={}, reserve=reserve)
    for target in (0.0, 0.05, 0.3, 0.5):
        view, info = _view_at_prevalence(bench, target, seed=1)
        n_gen = sum(1 for r in view.records
                    if not r.gt_category.name.startswith("ATTACK"))
        assert n_gen == 40 and info["n_genuine"] == 40      # genuine NEVER thinned
        need = 0 if target == 0 else round(target * 40 / (1 - target))
        assert abs(info["n_attacks"] - min(need, 60)) <= 1  # stratified rounding
        assert not info["pool_exhausted"]
    view, info = _view_at_prevalence(bench, 0.9, seed=1)    # needs 360 > 60
    assert info["pool_exhausted"] and info["n_attacks"] == 60


def test_mock_build_composition():
    """Mock-mode end-to-end build: paraphrased tier present with realized
    containment logged; extractor parity on T1/T2/T5; the attack reserve is
    namespaced, flagged, and its generation leaves every base record — and
    every base provider history — byte-identical (draw-sequence isolation)."""
    from msi_bench.config import get_config
    from msi_bench.generation.benchmark import build_benchmark, _record_to_dict, _hazard_key
    from msi_bench.schema import Category
    cfg = get_config()
    cfg.raw["llm"] = {"enabled": False}          # deterministic mock clients
    b0 = build_benchmark(cfg, verbose=False, limit_events=120, reserve_multiplier=0)
    b1 = build_benchmark(cfg, verbose=False, limit_events=120, reserve_multiplier=1.0)
    # (1) reserve isolation: base records identical with/without the reserve
    assert ([_record_to_dict(r) for r in b0.records]
            == [_record_to_dict(r) for r in b1.records])
    for pid, h in b0.histories.items():          # base histories undisturbed
        assert b1.histories.get(pid) == h
    assert not b0.reserve and len(b1.reserve) > 0
    base_ids = {r.record_id for r in b1.records}
    assert all(r.record_id not in base_ids for r in b1.reserve)
    assert all(r.provenance.get("reserve") for r in b1.reserve)
    assert all(r.record_id.split("-")[0].endswith("r") for r in b1.reserve)
    # (2) the four-level Sybil axis, with containment measured on paraphrased
    tiers = {r.provenance["identity_tier"] for r in b1.records
             if r.gt_category == Category.ATTACK_SYBIL}
    assert tiers == {"throwaway", "registered", "farmed", "paraphrased"}
    para = [r for r in b1.records if r.provenance.get("identity_tier") == "paraphrased"]
    assert para and all("containment_vs_campaign" in r.provenance for r in para)
    camp = [r for r in para if r.provenance["campaign"] == para[0].provenance["campaign"]]
    assert len({r.raw_text for r in camp}) == len(camp)  # distinct prose per member
    # (3) extractor parity: attack hazard_type is extracted from the text itself
    for r in b1.records + b1.reserve:
        if r.gt_category in (Category.ATTACK_FABRICATION, Category.ATTACK_SYBIL,
                             Category.ATTACK_REPLAY):
            ht = r.structured_data.get("hazard_type")
            if ht is not None:
                assert ht == _hazard_key(r.raw_text)
            n = len(r.structured_data)
            assert 0 <= n <= 4                    # genuine-shaped 1–3 (+ naive form field)
    # (4) same-seed rebuild reproduces the reserve byte-for-byte
    b2 = build_benchmark(cfg, verbose=False, limit_events=120, reserve_multiplier=1.0)
    assert ([_record_to_dict(r) for r in b1.reserve]
            == [_record_to_dict(r) for r in b2.reserve])
    # (5) round-trips through save/load
    with tempfile.TemporaryDirectory() as td:
        b1.save(Path(td))
        b3 = __import__("msi_bench.generation.benchmark", fromlist=["Benchmark"]).Benchmark.load(Path(td))
        assert len(b3.reserve) == len(b1.reserve)
        assert b3.meta["reserve_records"] == len(b1.reserve)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\nAll {passed} tests passed.")


if __name__ == "__main__":
    _run_all()
