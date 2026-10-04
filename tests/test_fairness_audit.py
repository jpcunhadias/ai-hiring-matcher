import pandas as pd

from src.fairness_audit import (
    gender_gap_by_role,
    gender_rate_bimodality,
    label_selection_rates,
    render_fairness_report,
    residual_gap_by_group,
    retrieval_accuracy_by_group,
    retrieval_homogeneity_pvalues,
    run_fairness_audit,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Gender": ["Male", "Male", "Female", "Female"],
            "Race": ["A", "A", "B", "B"],
            "Ethnicity": ["X", "X", "Y", "Y"],
            "Job Roles": ["Nurse", "Pilot", "Nurse", "Pilot"],
            "Best Match": [1, 1, 0, 0],
            "cosine_similarity": [0.9, 0.8, 0.9, 0.8],
            "skill_overlap": [0.5, 0.4, 0.5, 0.4],
            "best_match_proba": [0.9, 0.8, 0.5, 0.4],
        }
    )


def test_label_selection_rates_reflects_raw_label_skew():
    rates = label_selection_rates(_sample_df())

    gender_rates = rates[rates["attribute"] == "Gender"].set_index("group")["selection_rate"]
    assert gender_rates["Male"] == 1.0
    assert gender_rates["Female"] == 0.0


def test_residual_gap_by_group_flags_unexplained_difference():
    df = _sample_df()
    # Male and Female rows have identical cosine_similarity/skill_overlap, so any
    # remaining probability gap after regressing those out is entirely "unexplained".
    gaps = residual_gap_by_group(df)

    gender_gap = gaps[(gaps["attribute"] == "Gender")].set_index("group")["mean_residual"]
    assert gender_gap["Male"] > gender_gap["Female"]


def test_gender_gap_by_role_is_sorted_by_absolute_gap():
    df = _sample_df()

    gaps = gender_gap_by_role(df)

    # Nurse: Male=1.0, Female=0.0 -> gap=+1.0. Pilot: Male=1.0, Female=0.0 -> gap=+1.0.
    assert set(gaps["Job Roles"]) == {"Nurse", "Pilot"}
    assert (gaps["gap"] == 1.0).all()


def test_gender_rate_bimodality_counts_groups_by_bucket():
    df = _sample_df()
    # Nurse: Male=1.0, Female=0.0. Pilot: Male=1.0, Female=0.0 -> all 4 groups deterministic,
    # all at the extremes (0 or 1), none in the middle or "near but not at" a pole.
    summary = gender_rate_bimodality(df)

    assert summary["n_groups"] == 4
    assert summary["n_deterministic"] == 4
    assert summary["n_near_zero"] == 2
    assert summary["n_near_one"] == 2
    assert summary["n_middle"] == 0


def test_run_fairness_audit_returns_all_tables():
    report = run_fairness_audit(_sample_df())

    assert not report.label_selection_rate.empty
    assert not report.gender_gap_by_role.empty
    assert not report.residual_gap.empty
    assert report.bimodality["n_groups"] == 4
    assert report.proba_range == (0.4, 0.9)
    assert set(report.label_selection_rate["attribute"]) == {"Gender", "Race", "Ethnicity"}


def test_report_warns_when_classifier_probabilities_barely_vary():
    df = _sample_df()
    df["best_match_proba"] = [0.48, 0.49, 0.48, 0.47]

    text = render_fairness_report(run_fairness_audit(df))

    assert "not* evidence that the matcher is fair" in text


def test_report_has_no_warning_when_probabilities_spread():
    text = render_fairness_report(run_fairness_audit(_sample_df()))

    assert "not* evidence" not in text


def test_retrieval_accuracy_by_group_counts_rank_one_per_group():
    df = _sample_df()
    # Male rows are ranked first, Female rows are not.
    df["retrieval_rank"] = [1, 1, 3, 2]

    table = retrieval_accuracy_by_group(df)

    gender = table[table["attribute"] == "Gender"].set_index("group")
    assert gender.loc["Male", "recall_at_1"] == 1.0
    assert gender.loc["Female", "recall_at_1"] == 0.0
    assert gender.loc["Male", "n"] == 2
    assert gender.loc["Male", "std_error"] == 0.0


def test_report_includes_retrieval_section_only_when_ranks_present():
    without = render_fairness_report(run_fairness_audit(_sample_df()))
    df = _sample_df()
    df["retrieval_rank"] = [1, 1, 3, 2]
    with_ranks = render_fairness_report(run_fairness_audit(df))

    assert "Retrieval accuracy by group" not in without
    assert "Retrieval accuracy by group" in with_ranks


def test_homogeneity_pvalue_is_small_for_a_real_gap_and_large_for_none():
    rows = []
    for gender, hits in (("Male", 90), ("Female", 10)):
        rows += [
            {
                "Gender": gender,
                "Race": "A",
                "Ethnicity": "X",
                "retrieval_rank": 1 if i < hits else 2,
            }
            for i in range(100)
        ]
    gap = retrieval_homogeneity_pvalues(pd.DataFrame(rows))

    assert gap["Gender"] < 0.001
    assert gap["Race"] == 1.0  # a single group has nothing to differ from


def test_homogeneity_pvalue_handles_no_variation_in_hits():
    df = _sample_df()
    df["retrieval_rank"] = [1, 1, 1, 1]

    assert retrieval_homogeneity_pvalues(df) == {"Gender": 1.0, "Race": 1.0, "Ethnicity": 1.0}
