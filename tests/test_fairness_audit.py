import pandas as pd

from src.fairness_audit import (
    gender_gap_by_role,
    gender_rate_bimodality,
    label_selection_rates,
    residual_gap_by_group,
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
    assert set(report.label_selection_rate["attribute"]) == {"Gender", "Race", "Ethnicity"}
