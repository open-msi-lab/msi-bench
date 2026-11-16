"""Statistical protocol utilities (Section 4.5).

Bootstrap confidence intervals over events, AUC with a bootstrap CI, a paired
bootstrap test for AUC differences, and a fast DeLong test for two correlated
ROC AUCs. These back the headline metrics and significance claims in E1–E7.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score

    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC (1 = positive/credible). Returns NaN if a class is absent."""
    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    if _HAS_SK:
        return float(roc_auc_score(y, np.asarray(scores)))
    return _auc_manual(y, np.asarray(scores))


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels)
    if len(np.unique(y)) < 2 or not _HAS_SK:
        return float("nan")
    return float(average_precision_score(y, np.asarray(scores)))


def _auc_manual(y: np.ndarray, s: np.ndarray) -> float:
    """Mann–Whitney U estimate of AUC (handles ties)."""
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    point = float(statistic(v))
    if len(v) < 2:
        return point, point, point
    boot = np.empty(n_resamples)
    n = len(v)
    for b in range(n_resamples):
        boot[b] = statistic(v[rng.integers(0, n, n)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for ROC-AUC (resampling reports/events with replacement)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    point = roc_auc(y, s)
    n = len(y)
    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(roc_auc(y[idx], s[idx]))
    if not boot:
        return point, point, point
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_auc_ci_grouped(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Event-level bootstrap CI for ROC-AUC (Section 4.5).

    Resampling is over GROUPS (events) with replacement, not individual
    records, so correlated reports of the same event are kept together. Each
    attack with no event id forms its own singleton group.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    point = roc_auc(y, s)
    boot = []
    G = len(uniq)
    for _ in range(n_resamples):
        chosen = rng.choice(uniq, size=G, replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(roc_auc(y[idx], s[idx]))
    if not boot:
        return point, point, point
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_auc_diff(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Paired bootstrap for AUC(a) − AUC(b). Returns (diff, two-sided p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = roc_auc(y, a) - roc_auc(y, b)
    n = len(y)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc(y[idx], a[idx]) - roc_auc(y[idx], b[idx]))
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return diff, float("nan")
    # two-sided p: proportion of resamples on the opposite side of 0
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diff), float(min(1.0, p))


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> Tuple[list, list]:
    """Holm–Bonferroni step-down correction (Section 4.5).

    Given m raw p-values, returns ``(adjusted, reject)`` aligned to the input
    order. Adjusted p-values are the monotone step-down values; ``reject[i]`` is
    True iff hypothesis i is rejected at family-wise level ``alpha``. Robust to
    NaN p-values (treated as 1.0, never rejected).
    """
    p = [1.0 if (x is None or x != x) else float(x) for x in pvalues]  # NaN→1
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    reject = [False] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)  # enforce monotonicity along the step-down
        adjusted[i] = running
        reject[i] = running <= alpha
    return adjusted, reject


def delong_test(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Tuple[float, float, float]:
    """Fast DeLong test for two correlated ROC AUCs.

    Returns (auc_a, auc_b, p_value). Implementation of Sun & Xu (2014).
    """
    y = np.asarray(labels)
    pos = y == 1
    X = np.asarray(scores_a, dtype=float)[pos]
    Yn = np.asarray(scores_a, dtype=float)[~pos]
    X2 = np.asarray(scores_b, dtype=float)[pos]
    Y2 = np.asarray(scores_b, dtype=float)[~pos]
    m, n = len(X), len(Yn)
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    def _midrank(x):
        order = np.argsort(x)
        ranked = x[order]
        N = len(x)
        r = np.empty(N)
        i = 0
        while i < N:
            j = i
            while j < N and ranked[j] == ranked[i]:
                j += 1
            r[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(N)
        out[order] = r
        return out

    def _structural(pos_s, neg_s):
        tx = _midrank(pos_s)
        ty = _midrank(neg_s)
        txy = _midrank(np.concatenate([pos_s, neg_s]))
        auc = (txy[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01 = (txy[:m] - tx) / n
        v10 = 1.0 - (txy[m:] - ty) / m
        return auc, v01, v10

    auc_a, v01a, v10a = _structural(X, Yn)
    auc_b, v01b, v10b = _structural(X2, Y2)
    v01 = np.vstack([v01a, v01b])
    v10 = np.vstack([v10a, v10b])
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float(auc_a), float(auc_b), 1.0
    from math import erfc, sqrt

    z = (auc_a - auc_b) / sqrt(var)
    p = erfc(abs(z) / sqrt(2.0))
    return float(auc_a), float(auc_b), float(p)


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, Optional[float]]:
    """Spearman rank correlation (ρ, p). Uses scipy when available; otherwise
    Pearson over average ranks with ``p=None`` (the E1 quality-tier ranking
    only needs the coefficient's sign and magnitude)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 3:
        return float("nan"), None
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(xa, ya)
        return float(rho), float(p)
    except Exception:  # pragma: no cover
        def _ranks(v):
            order = np.argsort(v, kind="mergesort")
            ranks = np.empty(len(v), dtype=float)
            ranks[order] = np.arange(1, len(v) + 1, dtype=float)
            for val in np.unique(v):
                m = v == val
                if m.sum() > 1:
                    ranks[m] = ranks[m].mean()
            return ranks
        rx, ry = _ranks(xa), _ranks(ya)
        rx -= rx.mean(); ry -= ry.mean()
        denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return (float((rx * ry).sum() / denom) if denom else float("nan")), None
