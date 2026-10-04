import numpy as np
import pandas as pd
import pytest

from src.masked_data import _shift_month, build_pairs, rolling_splits
from src.rank_metrics import per_vacancy_metrics
from src.ranker import (
    KINDS,
    Ranker,
    RollingResult,
    _method_scores,
    center_within_vacancy,
    evaluate_rolling,
    fold_gains,
    mean_coefficients,
    summarize,
)
from tests.test_features import COOK_CV, SAP_CV, hired, pair, small
from tests.test_masked_data import make_tables


def signal_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 4 == 0).astype(int)
    x = pd.DataFrame(
        {
            "signal": y + rng.normal(0, 0.4, n),
            "noise": rng.normal(0, 1, n),
            "sometimes_missing": np.where(rng.random(n) < 0.3, np.nan, rng.normal(0, 1, n)),
        }
    )
    return x, pd.Series(y)


@pytest.mark.parametrize("kind", KINDS)
def test_a_ranker_scores_hires_above_the_rest_when_a_feature_carries_signal(kind):
    x, y = signal_data()

    ranker = Ranker(kind).fit(x, y)
    score = ranker.score(x)

    assert score[y == 1].mean() > score[y == 0].mean() + 0.1
    assert Ranker(kind).fit(x, y).score(x).tolist() == pytest.approx(
        score.tolist()
    )  # deterministic


def test_the_logistic_ranker_exposes_standardized_weights_and_the_boosted_one_does_not():
    x, y = signal_data()

    weights = Ranker("logistic").fit(x, y).coefficients

    assert list(weights.index) == list(x.columns)
    assert weights["signal"] > abs(weights["noise"]) * 3
    with pytest.raises(ValueError, match="only"):
        Ranker("boosted").fit(x, y).coefficients  # noqa: B018


def test_ranker_input_errors_are_explicit():
    x, y = signal_data()

    with pytest.raises(ValueError, match="unknown kind"):
        Ranker("forest")
    with pytest.raises(ValueError, match="both hired"):
        Ranker().fit(x, np.zeros(len(x)))
    with pytest.raises(RuntimeError, match="fit"):
        Ranker().score(x)
    with pytest.raises(ValueError, match="missing feature columns"):
        Ranker().fit(x, y).score(x.drop(columns=["signal"]))


def test_scoring_uses_the_training_columns_whatever_the_order_or_extras():
    x, y = signal_data()
    ranker = Ranker().fit(x, y)

    shuffled = x[["noise", "sometimes_missing", "signal"]].assign(extra=1.0)

    assert ranker.score(shuffled).tolist() == pytest.approx(ranker.score(x).tolist())


@pytest.mark.parametrize("kind", KINDS)
def test_a_column_missing_for_every_training_row_is_ignored(kind):
    x, y = signal_data()
    x["never_seen"] = np.nan

    ranker = Ranker(kind).fit(x, y)

    assert "never_seen" not in ranker.columns
    assert ranker.score(x.assign(never_seen=1.0)).shape == (len(x),)
    with pytest.raises(ValueError, match="every feature is missing"):
        Ranker(kind).fit(x[["never_seen"]], y)


def test_centering_removes_vacancy_level_effects_and_keeps_differences():
    x = pd.DataFrame({"a": [1.0, 3.0, 10.0, 14.0], "b": [5.0, 5.0, np.nan, 7.0]})
    groups = pd.Series(["v1", "v1", "v2", "v2"])

    centered = center_within_vacancy(x, groups)

    assert centered["a"].tolist() == [-1.0, 1.0, -2.0, 2.0]
    assert centered["b"].tolist()[:2] == [0.0, 0.0] and np.isnan(centered["b"].iloc[2])
    shifted = x.assign(a=x["a"] + groups.map({"v1": 100.0, "v2": -50.0}))
    pd.testing.assert_frame_equal(center_within_vacancy(shifted, groups), centered)


def test_centered_models_ignore_a_vacancy_level_shift():
    pairs = synthetic_pairs(24)
    x = small().fit(pairs).transform(pairs, history=pairs)
    groups = pairs["vacancy_id"]
    shift = groups.map({v: float(i) for i, v in enumerate(groups.unique())})

    base, _ = _method_scores(x, pairs["y"], x, 0, groups, groups)
    moved, _ = _method_scores(
        x.assign(cv_len_log=x["cv_len_log"] + shift),
        pairs["y"],
        x.assign(cv_len_log=x["cv_len_log"] + shift),
        0,
        groups,
        groups,
    )

    for name in ("logistic, centered", "boosted trees, centered"):
        assert moved[name].tolist() == pytest.approx(base[name].tolist(), abs=1e-6), name


def test_month_shifting_crosses_year_boundaries():
    assert _shift_month("2020-11", 3) == "2021-02"
    assert _shift_month("2021-01", -1) == "2020-12"


def test_rolling_windows_are_disjoint_ordered_and_train_only_on_the_past():
    pairs = build_pairs(make_tables(12))

    splits = list(rolling_splits(pairs, "all_prospects", "2020-03", step_months=3))

    assert [s.cutoff for s in splits] == ["2020-03", "2020-06", "2020-09"]
    tested = [set(s.test["vacancy_id"]) for s in splits]
    assert set.union(*tested) == {f"v{i}" for i in range(3, 12)}  # each later vacancy, once
    assert sum(len(t) for t in tested) == len(set.union(*tested))
    for split in splits:
        assert split.train["requested_month"].max() <= split.cutoff
        assert split.test["requested_month"].min() > split.cutoff
    assert len({len(s.train) for s in splits}) == 3  # the training window grows


def test_rolling_windows_with_nothing_to_test_are_skipped_and_the_step_is_validated():
    pairs = build_pairs(make_tables(12))

    assert list(rolling_splits(pairs, "all_prospects", "2020-12", step_months=3)) == []
    with pytest.raises(ValueError, match="step_months"):
        list(rolling_splits(pairs, "all_prospects", "2020-03", step_months=0))


def synthetic_pairs(n_vacancies=36):
    """The hired candidate's CV matches the vacancy; the others' does not."""
    rows = []
    for i in range(n_vacancies):
        month = f"{2020 + i // 12}-{i % 12 + 1:02d}"
        common = {"requested_month": month, "candidacy_month": month, "updated_month": month}
        rows += [
            hired(vacancy_id=f"v{i}", candidate_id=f"h{i}", cv_text=SAP_CV, **common),
            pair(f"v{i}", f"r{i}", cv_text=COOK_CV, **common),
            pair(f"v{i}", f"p{i}", cv_text=COOK_CV, outcome="pending", resolved=False, **common),
        ]
    return pd.DataFrame(rows)


def run(pairs, setting="all_prospects"):
    return evaluate_rolling(pairs, setting, "2020-12", 6, make_builder=small)


def test_rolling_evaluation_finds_a_real_signal_and_tests_each_vacancy_once():
    result = run(synthetic_pairs())

    table = summarize(result).set_index("method")
    tested = pd.concat(result.per_vacancy["random"]).index

    assert len(result.folds) >= 2 and tested.is_unique
    assert {"random", "tfidf zero-shot", "logistic", "boosted trees", "logistic + lag",
            "boosted trees + lag", "lag_months alone (artifact)", "logistic, centered",
            "boosted trees, centered"} <= set(table.index)  # fmt: skip
    assert table.loc["random", "hit@1"] == pytest.approx(1 / 3)
    assert table.loc["logistic", "hit@1"] > 0.9 and table.loc["logistic", "low"] > 0.4
    assert table.loc["random", "gain@1"] == 0.0


def test_the_clean_model_never_receives_the_process_artifact():
    pairs = synthetic_pairs()

    builder = small().fit(pairs)
    x = builder.transform(pairs, history=pairs)
    _, weights = _method_scores(x, pairs["y"], x, seed=0)

    assert "lag_months" in x.columns and "lag_months" not in weights.index


def test_a_fold_does_not_see_labels_of_vacancies_requested_after_it():
    pairs = synthetic_pairs()
    altered = pairs.copy()
    later = altered["requested_month"] > "2021-09"  # beyond the first two test windows
    flipped = altered["candidate_id"].str.startswith("h") & later
    altered.loc[flipped, ["outcome", "y"]] = ["rejected", 0]
    altered.loc[altered["candidate_id"].str.startswith("r") & later, ["outcome", "y"]] = [
        "hired",
        1,
    ]

    before, after = run(pairs), run(altered)

    for name in ("logistic", "boosted trees", "tfidf zero-shot"):
        pd.testing.assert_frame_equal(before.per_vacancy[name][0], after.per_vacancy[name][0])


def test_resolved_only_trains_and_tests_on_resolved_candidacies():
    result = run(synthetic_pairs(), "resolved_only")

    table = summarize(result).set_index("method")

    assert table.loc["random", "hit@1"] == pytest.approx(1 / 2)  # hired + rejected only
    assert table.loc["logistic", "hit@1"] > 0.9


def test_summaries_handle_an_empty_result_and_order_weights_by_magnitude():
    assert summarize(RollingResult()).empty
    result = RollingResult(
        coefficients=[pd.Series({"a": 0.1, "b": -0.9}), pd.Series({"a": 0.3, "b": -0.7, "c": 0.5})]
    )

    ordered = mean_coefficients(result)

    assert list(ordered.index) == ["b", "c", "a"]
    assert ordered["b"] == pytest.approx(-0.8)


def test_metrics_of_a_score_equal_to_the_label_are_perfect():
    pairs = synthetic_pairs(3)

    metrics = per_vacancy_metrics(pairs, pairs["y"].astype(float))

    assert metrics["hit@1"].eq(1.0).all()


def test_fold_gains_and_the_paired_difference_to_zero_shot():
    result = run(synthetic_pairs())

    gains = fold_gains(result, "logistic")
    table = summarize(result).set_index("method")

    assert [g["cutoff"] for g in gains] == [f["cutoff"] for f in result.folds]
    assert all(g["gain"] > 0.4 for g in gains)  # a real signal is consistent in every fold
    assert fold_gains(result, "random") == [
        {"cutoff": f["cutoff"], "vacancies": f["test_vacancies"], "gain": 0.0} for f in result.folds
    ]
    assert table.loc["tfidf zero-shot", ["vs_tfidf", "vs_low", "vs_high"]].eq(0.0).all()
