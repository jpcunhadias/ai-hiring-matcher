import numpy as np
import pandas as pd

from src.feature_report import RESULT_COLUMNS, SINGLE, single_feature_table


def test_single_feature_table_ranks_a_useful_feature_above_noise_and_skips_constants():
    rows = [
        {"vacancy_id": f"v{i}", "candidate_id": f"c{i}_{j}", "y": int(j == 0)}
        for i in range(30)
        for j in range(4)
    ]
    test = pd.DataFrame(rows)
    feats = pd.DataFrame(
        {
            "tfidf_word": test["y"] + 0.01 * (test.index % 3),  # the hire always scores highest
            "cv_len_log": (test.index * 7919 % 13).astype(float),  # unrelated to the label
            "prior_hires": 0.0,  # constant inside every vacancy: cannot rank
        }
    )

    table = single_feature_table(test, feats)

    assert list(table["feature"]) == ["tfidf_word", "cv_len_log"]
    best = table.iloc[0]
    assert best["hit@1"] == 1.0 and best["gain"] > 0.6 and best["low"] > 0.5
    assert abs(table.iloc[1]["gain"]) < 0.3


def vacancies(n=20, size=3):
    rows = [
        {"vacancy_id": f"v{i}", "candidate_id": f"c{i}_{j}", "y": int(j == 0)}
        for i in range(n)
        for j in range(size)
    ]
    return pd.DataFrame(rows)


def test_a_missing_value_ranks_last_so_a_mostly_missing_feature_still_ranks():
    test = vacancies()
    values = [1.0 if row.y else np.nan for row in test.itertuples()]  # only the hire has a value

    table = single_feature_table(test, pd.DataFrame({"tfidf_word": values}))

    assert list(table["feature"]) == ["tfidf_word"]
    assert table.iloc[0]["hit@1"] == 1.0


def test_an_all_missing_or_all_constant_report_is_empty_not_an_error():
    test = vacancies()
    feats = pd.DataFrame({"tfidf_word": 1.0, "title_sim": np.nan}, index=test.index)

    table = single_feature_table(test, feats)

    assert table.empty and list(table.columns) == RESULT_COLUMNS


def test_the_report_covers_every_rankable_feature():
    assert {"spanish_gap", "client_prior_hire_rate"} <= set(SINGLE)
