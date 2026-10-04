import pandas as pd

from src.feature_report import single_feature_table


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
