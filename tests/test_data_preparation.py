import pandas as pd
import pytest

from src.data_preparation import (
    build_skill_vocabulary,
    extract_job_skills,
    parse_resume,
    skill_overlap,
    train_test_split_df,
)

SAMPLE_RESUME = (
    "Proficient in Injury Prevention, Motivation, Nutrition, Health Coaching, "
    "Strength Training, with mid-level experience in the field. Holds a "
    "Bachelors degree. Holds certifications such as Certified Personal Trainer "
    "(CPT) by NASM. Skilled in delivering results and adapting to dynamic "
    "environments."
)


def test_parse_resume_extracts_fields():
    parsed = parse_resume(SAMPLE_RESUME)

    assert parsed["skills"] == {
        "Injury Prevention",
        "Motivation",
        "Nutrition",
        "Health Coaching",
        "Strength Training",
    }
    assert parsed["level"] == "mid-level"
    assert parsed["degree"] == "Bachelors"
    assert parsed["certifications"] == "Certified Personal Trainer (CPT) by NASM"


def test_parse_resume_handles_unexpected_format():
    parsed = parse_resume("Not a templated resume at all.")

    assert parsed["skills"] == set()
    assert parsed["level"] is None


def test_build_skill_vocabulary_unions_across_resumes():
    other_resume = SAMPLE_RESUME.replace("Nutrition", "Budgeting")
    vocabulary = build_skill_vocabulary(pd.Series([SAMPLE_RESUME, other_resume]))

    assert "Nutrition" in vocabulary
    assert "Budgeting" in vocabulary


def test_extract_job_skills_uses_whole_word_matching():
    vocabulary = {"R", "GIS", "Logistics", "Inventory Management"}
    job_description = "You will manage logistics and inventory management for the team."

    found = extract_job_skills(job_description, vocabulary)

    # "GIS" and "R" must NOT match as substrings of "logistics" / other words.
    assert found == {"Logistics", "Inventory Management"}


def test_skill_overlap_jaccard():
    assert skill_overlap({"Python", "SQL"}, {"Python", "R"}) == pytest.approx(1 / 3)
    assert skill_overlap(set(), {"Python"}) == 0.0
    assert skill_overlap({"Python"}, set()) == 0.0


def test_train_test_split_df_is_stratified():
    df = pd.DataFrame({"Best Match": [0, 1] * 50, "x": range(100)})

    train_df, test_df = train_test_split_df(df, test_size=0.2)

    assert len(test_df) == 20
    assert train_df["Best Match"].mean() == pytest.approx(0.5)
    assert test_df["Best Match"].mean() == pytest.approx(0.5)
