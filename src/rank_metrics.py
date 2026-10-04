"""Within-vacancy ranking metrics with bootstrap intervals.

A ranking is judged inside each vacancy: did the hired candidate(s) land at the top of that
vacancy's candidate list? Equal scores are treated as randomly ordered, and the expected metric
is computed exactly (no sampling), so a score with many ties is neither rewarded nor punished
by row order and the numbers do not move between runs.
"""

from collections.abc import Sequence
from math import comb

import numpy as np
import pandas as pd


def _first_hire_rank_distribution(scores: np.ndarray, y: np.ndarray) -> list[tuple[int, float]]:
    """(rank, probability) of the best-ranked hire when ties are ordered at random.

    Candidates are grouped into blocks of equal score. The first hire is in the first block that
    contains one. Inside a block of t candidates with h hires the first hire sits at position j
    with probability C(t - j, h - 1) / C(t, h).
    """
    before = 0
    for value in np.unique(scores)[::-1]:
        in_block = scores == value
        size, hires = int(in_block.sum()), int(y[in_block].sum())
        if hires:
            denominator = comb(size, hires)
            return [
                (before + j, comb(size - j, hires - 1) / denominator)
                for j in range(1, size - hires + 2)
            ]
        before += size
    raise ValueError("no hire in the vacancy")


def per_vacancy_metrics(
    frame: pd.DataFrame,
    scores: Sequence[float] | np.ndarray,
    ks: tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    """Expected hit@k (any hire in the top k) and reciprocal rank of the first hire, per vacancy."""
    values = np.asarray(scores, dtype=float)
    if len(values) != len(frame):
        raise ValueError("one score per row is required")
    if np.isnan(values).any():
        raise ValueError("scores must not contain NaN")
    y_all = frame["y"].to_numpy()
    rows = {}
    for vacancy, idx in frame.groupby("vacancy_id").indices.items():
        if y_all[idx].sum() == 0:
            raise ValueError(f"vacancy {vacancy!r} has no hire, so it cannot be ranked")
        ranks = _first_hire_rank_distribution(values[idx], y_all[idx])
        row = {f"hit@{k}": sum(p for r, p in ranks if r <= k) for k in ks}
        row["mrr"] = sum(p / r for r, p in ranks)
        rows[vacancy] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def random_metrics(frame: pd.DataFrame, ks: tuple[int, ...] = (1, 3)) -> pd.DataFrame:
    """What a random order scores on the same vacancies (exact expectation)."""
    return per_vacancy_metrics(frame, np.zeros(len(frame)), ks=ks)


def gain_with_ci(
    method: pd.Series, baseline: pd.Series, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean per-vacancy difference (method - baseline) and a 95% bootstrap interval over
    vacancies. Both series are aligned by vacancy first and the vacancies are resampled
    together, so the comparison stays paired."""
    if set(method.index) != set(baseline.index):
        raise ValueError("method and baseline must cover the same vacancies")
    diff = (method - baseline.reindex(method.index)).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
