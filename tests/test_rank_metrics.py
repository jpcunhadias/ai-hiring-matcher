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


def test_tied_scores_are_judged_by_expectation_not_by_row_order():
    data = frame([[1, 0, 0, 0]] * 40)  # the hire is always the first row

    metrics = per_vacancy_metrics(data, np.zeros(len(data)), draws=50)

    assert metrics["hit@1"].mean() == pytest.approx(0.25, abs=0.07)  # not 1.0


def test_any_hire_in_the_top_k_counts():
    data = frame([[1, 1, 0, 0, 0]])

    metrics = per_vacancy_metrics(data, [0.1, 0.8, 0.9, 0.2, 0.0], ks=(1, 2))

    assert metrics.loc["v0", "hit@1"] == 0.0 and metrics.loc["v0", "hit@2"] == 1.0
    assert metrics.loc["v0", "mrr"] == pytest.approx(0.5)


def test_random_metrics_match_the_analytic_expectation():
    data = frame([[1, 0, 0, 0]] * 30 + [[1, 1, 0, 0]] * 30)

    random = random_metrics(data)

    one_hire = [f"v{i}" for i in range(30)]
    two_hires = [f"v{i}" for i in range(30, 60)]
    assert random.loc[one_hire, "hit@1"].mean() == pytest.approx(0.25, abs=0.04)
    assert random.loc[two_hires, "hit@1"].mean() == pytest.approx(0.5, abs=0.04)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="no hire"):
        per_vacancy_metrics(frame([[0, 0]]), [0.0, 1.0])
    with pytest.raises(ValueError, match="one score per row"):
        per_vacancy_metrics(frame([[1, 0]]), [0.0])


def test_gain_ci_is_paired_and_degenerates_for_a_constant_difference():
    method = pd.Series([1.0, 1.0, 0.0, 1.0], index=list("abcd"))
    baseline = pd.Series([0.5, 0.5, 0.5, 0.5], index=list("dcba"))  # index order differs

    mean, low, high = gain_with_ci(method, baseline, n_boot=500)

    assert mean == pytest.approx(0.25)
    assert low <= mean <= high

    same = gain_with_ci(pd.Series([1.0, 1.0]), pd.Series([0.0, 0.0]), n_boot=50)
    assert same == (1.0, 1.0, 1.0)
