import itertools

import numpy as np
import pandas as pd
import pytest

from src.rank_metrics import gain_with_ci, per_vacancy_metrics, random_metrics


def frame(y_per_vacancy: list[list[int]]) -> pd.DataFrame:
    rows = [
        {"vacancy_id": f"v{i}", "candidate_id": f"c{i}_{j}", "y": y}
        for i, ys in enumerate(y_per_vacancy)
        for j, y in enumerate(ys)
    ]
    return pd.DataFrame(rows)


def test_a_perfect_ranking_scores_one_and_a_reversed_one_scores_the_last_place():
    data = frame([[0, 0, 1], [1, 0, 0]])
    perfect = per_vacancy_metrics(data, data["y"].astype(float))
    reversed_ = per_vacancy_metrics(data, -data["y"].astype(float))

    assert perfect["hit@1"].tolist() == [1.0, 1.0] and perfect["mrr"].tolist() == [1.0, 1.0]
    assert reversed_["hit@1"].tolist() == [0.0, 0.0]
    assert reversed_["mrr"].tolist() == [pytest.approx(1 / 3), pytest.approx(1 / 3)]


def test_tied_scores_have_an_exact_expectation_independent_of_row_order():
    data = frame([[1, 0, 0, 0]])

    metrics = per_vacancy_metrics(data, np.zeros(4), ks=(1, 3))

    assert metrics.loc["v0", "hit@1"] == pytest.approx(0.25)
    assert metrics.loc["v0", "hit@3"] == pytest.approx(0.75)
    assert metrics.loc["v0", "mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    reordered = frame([[0, 0, 0, 1]])
    pd.testing.assert_frame_equal(metrics, per_vacancy_metrics(reordered, np.zeros(4), ks=(1, 3)))


def test_a_hire_tied_for_first_with_one_other_scores_one_half_and_is_deterministic():
    data = frame([[1, 0, 0]])
    scores = [0.9, 0.9, 0.1]

    first = per_vacancy_metrics(data, scores)

    assert first.loc["v0", "hit@1"] == 0.5
    pd.testing.assert_frame_equal(first, per_vacancy_metrics(data, scores))


def test_the_exact_expectation_matches_brute_force_over_every_tie_order():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(2, 7))
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, size=int(rng.integers(1, min(n, 3) + 1)), replace=False)] = 1
        scores = rng.integers(0, 3, size=n).astype(float)  # many ties
        data = pd.DataFrame({"vacancy_id": "v", "candidate_id": range(n), "y": y})

        got = per_vacancy_metrics(data, scores, ks=(1, 2)).iloc[0]

        hits1 = hits2 = mrr = orders = 0
        for perm in itertools.permutations(range(n)):  # a tie order = a tie-breaking key
            key = np.empty(n)
            key[list(perm)] = np.arange(n)
            order = np.lexsort((key, -scores))
            first = int(np.argmax(y[order] == 1)) + 1
            hits1 += first <= 1
            hits2 += first <= 2
            mrr += 1 / first
            orders += 1
        assert got["hit@1"] == pytest.approx(hits1 / orders)
        assert got["hit@2"] == pytest.approx(hits2 / orders)
        assert got["mrr"] == pytest.approx(mrr / orders)


def test_any_hire_in_the_top_k_counts():
    data = frame([[1, 1, 0, 0, 0]])

    metrics = per_vacancy_metrics(data, [0.1, 0.8, 0.9, 0.2, 0.0], ks=(1, 2))

    assert metrics.loc["v0", "hit@1"] == 0.0 and metrics.loc["v0", "hit@2"] == 1.0
    assert metrics.loc["v0", "mrr"] == pytest.approx(0.5)


def test_random_metrics_are_exact():
    random = random_metrics(frame([[1, 0, 0, 0], [1, 1, 0, 0]]))

    assert random.loc["v0", "hit@1"] == pytest.approx(0.25)
    assert random.loc["v1", "hit@1"] == pytest.approx(0.5)
    assert random.loc["v1", "hit@3"] == 1.0  # two non-hires cannot fill the top 3


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="no hire"):
        per_vacancy_metrics(frame([[0, 0]]), [0.0, 1.0])
    with pytest.raises(ValueError, match="one score per row"):
        per_vacancy_metrics(frame([[1, 0]]), [0.0])
    with pytest.raises(ValueError, match="NaN"):
        per_vacancy_metrics(frame([[1, 0]]), [0.0, float("nan")])


def test_gain_ci_is_paired_even_when_the_baseline_varies_and_the_index_order_differs():
    baseline = pd.Series([0.0, 0.5, 1.0, 0.25, 0.75, 0.1], index=list("abcdef"))
    method = (baseline + 0.1).sample(frac=1.0, random_state=3)  # same vacancies, shuffled order

    mean, low, high = gain_with_ci(method, baseline, n_boot=300)

    # paired: every vacancy gains exactly 0.1, so the interval collapses; positional or
    # unpaired subtraction would give a wide interval around a different mean
    assert (mean, low, high) == pytest.approx((0.1, 0.1, 0.1))


def test_gain_ci_is_wide_when_the_vacancy_differences_vary():
    method = pd.Series([1.0, 1.0, 0.0, 1.0], index=list("abcd"))
    baseline = pd.Series([0.5, 0.5, 0.5, 0.5], index=list("dcba"))

    mean, low, high = gain_with_ci(method, baseline, n_boot=500)

    assert mean == pytest.approx(0.25) and low < mean < high


def test_gain_ci_rejects_different_vacancy_sets():
    with pytest.raises(ValueError, match="same vacancies"):
        gain_with_ci(pd.Series([1.0], index=["a"]), pd.Series([1.0], index=["b"]))
