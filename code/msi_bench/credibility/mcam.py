"""The Multi-dimensional Credibility Assessment Model (MCAM, Section 3.3).

Computes device (Cd), content (Cc), spatiotemporal (Cst), and social (Cso)
credibility and integrates them with a two-pass strategy (Section 3.3) that
resolves the Cso↔Cr circular dependency. Includes the Sybil defences of
Section 3.3: reporter-independence verification and the dispersion penalty
(Eq. (7)).

The model satisfies the three properties of Proposition 1 (boundedness,
monotonicity, conditional temporal decay); see ``tests/test_framework.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geo import haversine_nm
from ..schema import MSIRecord, ProviderType, VerificationStatus
from ..similarity import type_similarity, word_ngrams
from . import domain

_EPS = 0.01  # ε in Eq. (6)


@dataclass
class MCAMParams:
    """All MCAM parameters, with the Section 3.3 defaults."""

    # dimension weights (αd, αc, αst, αso)
    w_device: float = 0.25
    w_content: float = 0.30
    w_spatiotemporal: float = 0.20
    w_social: float = 0.25
    # device sub-weights (ω1, ω2, ω3)
    w_type: float = 0.30
    w_cert: float = 0.35
    w_history: float = 0.35
    # content sub-weights (β1, β2, β3) and consistency weights (δ1, δ2, δ3)
    w_compl: float = 0.25
    w_consis: float = 0.40
    w_reason: float = 0.35
    d_csd: float = 0.40
    d_cte: float = 0.35
    d_cnr: float = 0.25
    # completeness weights (γ1, γ2) and reasonability weights (η1, η2, η3)
    g_mand: float = 0.70
    g_opt: float = 0.30
    e_pnw: float = 0.40
    e_pet: float = 0.35
    e_ppf: float = 0.25
    # scales / thresholds
    lambda_history_days: float = 90.0
    n_min_history: int = 10
    theta_rel: float = 0.50
    theta_compat: float = 0.50
    delta_ind_nm: float = 0.5
    delta_disp_nm: float = 2.0
    trajectory_penalty: float = 0.5
    # Near-verbatim text-duplication threshold for the reporter-independence
    # check (d) of Section 3.3: a pair of free-text (crowdsourced) reports
    # whose word-3-gram containment reaches θ_ndup is treated as ONE effective
    # source — verbatim relay or campaign-scripted text is evidence of common
    # authorship, not an independent observation. Grounded on the dev split:
    # genuine same-event paraphrase pairs p99 = 0.57 vs Sybil within-campaign
    # p1 = 0.92, so 0.75 sits mid-gap with wide margins (sensitivity in E5).
    # Values > 1 disable the check. Distinct from θ_dup (Definition 2), which
    # is a COMPOSITE-similarity redundancy notion presuming independent authors.
    theta_ndup: float = 0.75
    # Validity-lapse decay (Section 3.3). Information whose asserted validity
    # window has already closed loses temporal credibility in proportion to how
    # long ago it lapsed, on this timescale (days). This makes Dt robust to a
    # forged-fresh issue_time on expired content — the signature of a replay (T2)
    # — without penalising genuinely current reports. A secondary parameter (cf. the
    # trajectory penalty); its sensitivity is examined in Section 5.5.
    lapse_sigma_days: float = 30.0
    # Assessment reference clock (Section 3.3). In
    # "submission" mode each report's spatiotemporal credibility is evaluated as
    # of its own arrival (issue_time + assessment_lag_h) — the operationally
    # faithful choice: a live service scores a report when it is received, not
    # at a later global snapshot, so freshness decay measures the report's
    # currency at ingestion and reports about long-running events are not
    # punished for the archive's calendar span. The validity-lapse penalty
    # (expired content resubmitted "fresh" — T2) is unaffected: the lapse is
    # measured at submission time, exactly when a real system would see it.
    # "global" restores the single-clock behaviour. HistoryScore always uses the
    # global reference; Eq. (3)'s exponential weights cancel a common shift, so
    # it is invariant to that choice.
    assessment_mode: str = "submission"  # "submission" | "global"
    assessment_lag_h: float = 1.0
    # ablation / variants
    active_dims: Tuple[str, ...] = ("device", "content", "spatiotemporal", "social")
    social_mode: str = "polarity"  # "polarity" | "naive" | "binary"

    def dim_weight(self, dim: str) -> float:
        return {
            "device": self.w_device,
            "content": self.w_content,
            "spatiotemporal": self.w_spatiotemporal,
            "social": self.w_social,
        }[dim]


@dataclass
class DimensionScores:
    record_id: str
    cd: float = 0.0
    cc: float = 0.0
    cst: float = 0.0
    cso: float = 0.5
    cr: float = 0.0
    cr_pre: float = 0.0
    n_support: int = 0
    n_neutral: int = 0
    n_oppose: int = 0
    dispersion_nm: float = 0.0


class MCAM:
    """Evaluator over a set of records sharing one precomputed similarity matrix."""

    def __init__(
        self,
        params: Optional[MCAMParams] = None,
        navigability: Optional[domain.NavigabilityModel] = None,
        histories: Optional[Dict[str, List[Tuple[datetime, str]]]] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.p = params or MCAMParams()
        self.nav = navigability
        self.histories = histories or {}
        self.now = now

    # ------------------------------------------------------------------ Cd
    def history_score(self, provider_id: str) -> float:
        """Time-decayed reputation (Eq. (3)). Evaluation-time invariant."""
        records = self.histories.get(provider_id, [])
        if not records:
            return 0.5  # cold start with no history → neutral prior
        now = self.now or max(t for t, _ in records)
        lam_s = self.p.lambda_history_days * 86400.0
        num = den = 0.0
        for t, outcome in records:
            # weight ∝ exp(-(now - t)/λ); the common exp(-now/λ) cancels in the
            # ratio (Section 3.3 note), so this is evaluation-time invariant.
            w = math.exp(-max(0.0, (now - t).total_seconds()) / lam_s)
            v = {"CORRECT": 1.0, "INCORRECT": 0.0}.get(outcome, 0.5)
            num += w * v
            den += w
        p0 = num / den if den > 0 else 0.5
        n = len(records)
        if n < self.p.n_min_history:  # cold-start interpolation (Eq. (3))
            frac = n / self.p.n_min_history
            return p0 * frac + 0.5 * (1.0 - frac)
        return p0

    def device_credibility(self, rec: MSIRecord) -> float:
        type_score = domain.DEVICE_TYPE_SCORE.get(rec.device_class, 0.30)
        cert_score = domain.CERT_SCORE.get(rec.auth_level, 0.30)
        hist = self.history_score(rec.provider_id)
        cd = self.p.w_type * type_score + self.p.w_cert * cert_score + self.p.w_history * hist
        # Identity–device binding / credential verification (Section 3.2, T4):
        # a report CLAIMING official provenance without a valid PKI credential
        # fails verification and is heavily penalised — the primary defence
        # against impersonation.
        if rec.provider_type == ProviderType.OFFICIAL and rec.auth_level not in (
                "PKI_FULL", "PKI_BASIC"):
            cd *= 0.25
        return cd

    # ------------------------------------------------------------------ Cc
    def _completeness(self, rec: MSIRecord) -> float:
        req = domain.REQUIRED_FIELDS.get(rec.event_type, {"description"})
        opt = domain.OPTIONAL_FIELDS.get(rec.event_type, set())
        filled = set(k for k, v in rec.structured_data.items() if v not in (None, "", []))
        if rec.centroid is not None:
            filled.add("position")
        if rec.valid_end is not None:
            filled.add("valid_end")
        c_mand = len(req & filled) / len(req) if req else 1.0
        c_opt = len(opt & filled) / len(opt) if opt else 1.0
        return self.p.g_mand * c_mand + self.p.g_opt * c_opt

    def _consistency(self, rec: MSIRecord) -> float:
        csd = self.nav.navigability(rec.centroid) if self.nav else (
            1.0 if rec.centroid else 0.5)
        cte = self._temporal_consistency(rec)
        cnr = self._numeric_consistency(rec)
        return self.p.d_csd * csd + self.p.d_cte * cte + self.p.d_cnr * cnr

    def _temporal_consistency(self, rec: MSIRecord) -> float:
        vs = rec.valid_start or rec.issue_time
        ve = rec.valid_end
        if vs and ve and ve <= vs:
            return 0.0  # logical violation
        if rec.issue_time and vs and rec.issue_time > vs + timedelta(hours=1):
            return 0.0
        if vs and ve:
            dur_h = (ve - vs).total_seconds() / 3600.0
            if dur_h <= 0 or dur_h > domain.NUMERIC_RANGES["duration_hours"][1]:
                return 0.2  # implausibly long/short window
        return 1.0

    def _numeric_consistency(self, rec: MSIRecord) -> float:
        checks = []
        for key, val in rec.structured_data.items():
            if key in domain.NUMERIC_RANGES and isinstance(val, (int, float)):
                lo, hi = domain.NUMERIC_RANGES[key]
                checks.append(1.0 if lo <= val <= hi else 0.0)
        return float(np.mean(checks)) if checks else 1.0

    def _reasonability(self, rec: MSIRecord) -> float:
        pnw = self.nav.navigability(rec.centroid) if self.nav else (
            1.0 if rec.centroid else 0.5)
        pet = float(rec.structured_data.get("_type_plausibility", 1.0))
        ppf = self._numeric_consistency(rec)
        return self.p.e_pnw * pnw + self.p.e_pet * pet + self.p.e_ppf * ppf

    def content_credibility(self, rec: MSIRecord) -> float:
        return (
            self.p.w_compl * self._completeness(rec)
            + self.p.w_consis * self._consistency(rec)
            + self.p.w_reason * self._reasonability(rec)
        )

    # ----------------------------------------------------------------- Cst
    def _assessment_time(self, rec: MSIRecord) -> Optional[datetime]:
        """Per-record reference clock (submission-time scoring; Section 3.3)."""
        if self.p.assessment_mode == "submission" and rec.issue_time is not None:
            return rec.issue_time + timedelta(hours=self.p.assessment_lag_h)
        return self.now

    def spatiotemporal_credibility(self, rec: MSIRecord,
                                   now: Optional[datetime] = None) -> float:
        ref = now if now is not None else self.now
        # spatial proximity factor Ds (Eq. (4))
        if rec.submission_position is not None and rec.centroid is not None:
            d = haversine_nm(rec.submission_position, rec.centroid)
            sigma_s = domain.OBSERVATION_SIGMA_S.get(rec.observation_type, 5.0)
            ds = math.exp(-d / sigma_s)
        else:
            ds = 0.5  # observer position unavailable
        # temporal freshness factor Dt (Eq. (4))
        if rec.issue_time is not None and ref is not None:
            dt_h = max(0.0, (ref - rec.issue_time).total_seconds() / 3600.0)
            sigma_t = domain.EVENT_SIGMA_T_H.get(rec.event_type, 24.0)
            dt = math.exp(-dt_h / sigma_t)
        else:
            dt = 1.0
        cst = ds * dt
        # Validity-lapse penalty (Section 3.3): issue-time freshness alone is
        # forgeable — a replay (T2) resubmits expired content under a current
        # issue_time, scoring a spuriously high Dt. We therefore decay Cst by how
        # long ago the asserted validity window closed, on a lapse timescale that
        # leaves genuinely current reports (lapsed by hours/days) almost untouched
        # while collapsing long-expired replays (lapsed by months/years).
        if rec.valid_end is not None and ref is not None and ref > rec.valid_end:
            lapse_h = (ref - rec.valid_end).total_seconds() / 3600.0
            cst *= math.exp(-lapse_h / (self.p.lapse_sigma_days * 24.0))
        # auxiliary trajectory consistency check (Section 3.3)
        if rec.vessel_max_speed_kn and len(rec.trajectory) >= 2:
            for (t0, p0), (t1, p1) in zip(rec.trajectory, rec.trajectory[1:]):
                hours = abs((t1 - t0).total_seconds()) / 3600.0
                if hours > 0:
                    v = haversine_nm(p0, p1) / hours
                    if v / rec.vessel_max_speed_kn > 1.5:
                        cst *= self.p.trajectory_penalty
                        break
        return cst

    # ----------------------------------------------------------------- Cso
    def _independent(self, r: MSIRecord, j: MSIRecord,
                     ng_r: Optional[frozenset] = None,
                     ng_j: Optional[frozenset] = None) -> bool:
        """Reporter-independence check (Section 3.3, extensions (a)–(d))."""
        if j.provider_id == r.provider_id and r.provider_id != "unknown":
            return False
        if (r.network_origin and j.network_origin
                and r.network_origin == j.network_origin):
            if r.submission_position and j.submission_position:
                if haversine_nm(r.submission_position, j.submission_position) < self.p.delta_ind_nm:
                    return False
        if r.submission_position and j.submission_position:
            if haversine_nm(r.submission_position, j.submission_position) < self.p.delta_ind_nm:
                return False
        # (d) near-verbatim text duplication:
        # verbatim relay, copy-paste, or campaign-scripted variants of one base
        # text are ONE effective source regardless of the identities attached.
        # Applies when either report is free-text (crowdsourced); structured
        # sensor feeds are templated by design, and their independence is
        # attested by channel authentication, not prose diversity. Both texts
        # must carry ≥ 4 shingles (≈ six tokens): a shorter fragment is too
        # little prose to evidence common authorship, and containment against
        # a near-empty shingle set is degenerate.
        if (ng_r is not None and ng_j is not None
                and len(ng_r) >= 4 and len(ng_j) >= 4
                and self.p.theta_ndup <= 1.0
                and (r.provider_type == ProviderType.CROWDSOURCE
                     or j.provider_type == ProviderType.CROWDSOURCE)):
            if len(ng_r & ng_j) >= self.p.theta_ndup * min(len(ng_r), len(ng_j)):
                return False
        return True

    def _classify_neighbor(self, r: MSIRecord, j: MSIRecord) -> str:
        """Return 'support' | 'oppose' | 'neutral' (Section 3.3)."""
        if self.p.social_mode == "naive":
            return "support"
        acrit = domain.CRITICAL_ATTRS.get(r.event_type, set()) & domain.CRITICAL_ATTRS.get(
            j.event_type, set())
        shared = [a for a in acrit if a in r.structured_data and a in j.structured_data]
        if not shared:
            # No shared critical attribute → agreement is undeterminable. Treat
            # as neutral; the binary baseline (no neutral class) then simply
            # omits it from support/oppose rather than counting it as support.
            return "neutral"
        for a in shared:
            if domain.attr_contradicts(a, r.structured_data[a], j.structured_data[a]):
                return "oppose"
        return "support"

    def _social_credibility(
        self,
        records: Sequence[MSIRecord],
        sim: np.ndarray,
        cr_pre: np.ndarray,
        scores: List[DimensionScores],
    ) -> None:
        n = len(records)
        # word-3-gram shingles for the near-verbatim duplication check (d);
        # computed once per record, compared per qualifying pair.
        ngram = ([word_ngrams(r.raw_text) for r in records]
                 if self.p.theta_ndup <= 1.0 else [None] * n)
        for i in range(n):
            r = records[i]
            support = neutral = oppose = 0.0
            supp_positions: List = []
            n_s = n_n = n_o = 0
            for jx in range(n):
                if jx == i:
                    continue
                s = sim[i, jx]
                if s < self.p.theta_rel:
                    continue
                j = records[jx]
                if type_similarity(r.event_type, j.event_type) < self.p.theta_compat:
                    continue
                if not self._independent(r, j, ngram[i], ngram[jx]):
                    continue
                cls = self._classify_neighbor(r, j)
                contrib = cr_pre[jx] * s
                if cls == "support":
                    support += contrib
                    n_s += 1
                    if j.submission_position is not None:
                        supp_positions.append(j.submission_position)
                elif cls == "oppose":
                    oppose += contrib
                    n_o += 1
                else:
                    neutral += contrib
                    n_n += 1

            if self.p.social_mode == "binary":
                cso_raw = (support + 0.5 * _EPS) / (support + oppose + _EPS)
            else:
                cso_raw = (support + 0.5 * neutral + 0.5 * _EPS) / (
                    support + neutral + oppose + _EPS)

            # dispersion penalty over supportive neighbours (Eq. (7))
            disp = self._dispersion(supp_positions)
            penalty = 1.0 if len(supp_positions) < 2 else min(1.0, disp / self.p.delta_disp_nm)
            scores[i].cso = cso_raw * penalty
            scores[i].n_support, scores[i].n_neutral, scores[i].n_oppose = n_s, n_n, n_o
            scores[i].dispersion_nm = disp

    @staticmethod
    def _dispersion(positions: Sequence) -> float:
        k = len(positions)
        if k < 2:
            return 0.0
        total = 0.0
        for a in range(k):
            for b in range(a + 1, k):
                total += haversine_nm(positions[a], positions[b])
        return total * 2.0 / (k * (k - 1))

    # ------------------------------------------------------------- assess
    def assess(
        self, records: Sequence[MSIRecord], sim: np.ndarray
    ) -> List[DimensionScores]:
        """Two-pass credibility assessment over a record set (Section 3.3)."""
        if self.now is None:
            issue_times = [r.issue_time for r in records if r.issue_time is not None]
            self.now = max(issue_times) if issue_times else datetime(2024, 1, 1)

        n = len(records)
        scores = [DimensionScores(record_id=r.record_id) for r in records]

        active = set(self.p.active_dims)
        # --- Pass 1: independent dimensions ---
        cr_pre = np.zeros(n)
        for i, r in enumerate(records):
            cd = self.device_credibility(r) if "device" in active else 0.0
            cc = self.content_credibility(r) if "content" in active else 0.0
            cst = (self.spatiotemporal_credibility(r, now=self._assessment_time(r))
                   if "spatiotemporal" in active else 0.0)
            scores[i].cd, scores[i].cc, scores[i].cst = cd, cc, cst
            wsum = sum(self.p.dim_weight(d) for d in ("device", "content", "spatiotemporal")
                       if d in active)
            pre = (self.p.w_device * cd + self.p.w_content * cc
                   + self.p.w_spatiotemporal * cst)
            scores[i].cr_pre = pre / wsum if wsum > 0 else 0.5
            cr_pre[i] = scores[i].cr_pre

        # --- Pass 2: social credibility, then integrate ---
        if "social" in active:
            self._social_credibility(records, sim, cr_pre, scores)
        else:
            for s in scores:
                s.cso = 0.5

        total_w = sum(self.p.dim_weight(d) for d in active)
        for i, s in enumerate(scores):
            num = 0.0
            if "device" in active:
                num += self.p.w_device * s.cd
            if "content" in active:
                num += self.p.w_content * s.cc
            if "spatiotemporal" in active:
                num += self.p.w_spatiotemporal * s.cst
            if "social" in active:
                num += self.p.w_social * s.cso
            s.cr = num / total_w if total_w > 0 else 0.0
            records[i].credibility_score = s.cr
            # NOTE: verification_status is a workflow state (Section 3.2) and is
            # deliberately NOT set from the score here. The MCAM
            # outputs a continuous score and a credibility LEVEL (credibility_level);
            # status is assigned by the verification workflow, and the fusion stage
            # may set CROSS_VERIFIED on its OUTPUT when belief > 0.75 (Section 3.6).
        return scores


def credibility_level(cr: float) -> str:
    """Map a score to the five-level classification."""
    if cr >= 0.90:
        return "VERY HIGH"
    if cr >= 0.75:
        return "HIGH"
    if cr >= 0.50:
        return "MEDIUM"
    if cr >= 0.25:
        return "LOW"
    return "VERY LOW"
