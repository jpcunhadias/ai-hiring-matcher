"""Within-vacancy ranking metrics with bootstrap intervals.

A ranking is judged inside each vacancy: did the hired candidate(s) land at the top of that
vacancy's candidate list? Ties are broken at random and averaged over several draws, so a score
with many equal values is neither rewarded nor punished by row order.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

TIE_DRAWS = 20
RANDOM_DRAWS = 200


def per_vacancy_metrics(
    frame: pd.DataFrame,
    scores: Sequence[float] | np.ndarray,
    ks: tuple[int, ...] = (1, 3),
    draws: int = TIE_DRAWS,
    seed: int = 0,
) -> pd.DataFrame:
    """hit@k (any hire in the top k) and reciprocal rank of the first hire, per vacancy."""
    values = np.asarray(scores, dtype=float)
    if len(values) != len(frame):
        raise ValueError("one score per row is required")
    rng = np.random.default_rng(seed)
    rows = {}
    y_all = frame["y"].to_numpy()
    for vacancy, idx in frame.groupby("vacancy_id").indices.items():
        s, y = values[idx], y_all[idx]
        if y.sum() == 0:
            raise ValueError(f"vacancy {vacancy!r} has no hire, so it cannot be ranked")
        acc = {f"hit@{k}": 0.0 for k in ks} | {"mrr": 0.0}
        for _ in range(draws):
            order = np.lexsort((rng.random(len(s)), -s))  # best first, ties shuffled
            first = int(np.argmax(y[order] == 1)) + 1
            for k in ks:
                acc[f"hit@{k}"] += float(first <= k)
            acc["mrr"] += 1.0 / first
        rows[vacancy] = {k: v / draws for k, v in acc.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def random_metrics(frame: pd.DataFrame, ks: tuple[int, ...] = (1, 3)) -> pd.DataFrame:
    """What a random order scores on the same vacancies."""
    return per_vacancy_metrics(frame, np.zeros(len(frame)), ks=ks, draws=RANDOM_DRAWS, seed=1)


def gain_with_ci(
    method: pd.Series, baseline: pd.Series, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean per-vacancy difference (method - baseline) and a 95% bootstrap interval over
    vacancies. The vacancies are resampled together, so the comparison stays paired."""
    diff = (method - baseline.reindex(method.index)).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
