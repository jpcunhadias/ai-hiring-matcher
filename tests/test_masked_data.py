import pandas as pd
import pytest

from src.ingest import TABLE_COLUMNS
from src.masked_data import (
    MaskedTables,
    audit_attributes,
    build_pairs,
    load_masked,
    select_setting,
    summarize,
    temporal_split,
)

QUERY = "analista de sistemas sap " * 4  # long enough to pass the vacancy-text filter
CV = "experiencia com sap e abap " * 12  # long enough to pass the CV filter


def _frame(name: str, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TABLE_COLUMNS[name]).fillna("")


def make_tables(n_vacancies: int = 10) -> MaskedTables:
    """Vacancy i is requested in month i; it has a hired, a rejected and a pending candidate."""
    vacancies, candidacies, applicants, sensitive = [], [], [], []
    for i in range(n_vacancies):
        month = f"2020-{i + 1:02d}"
        vacancies.append(
            {"vacancy_id": f"v{i}", "title": "analista", "activities": QUERY, "competencies": "",
             "requested_month": month, "seniority": "Pleno", "state": "SP"}
        )  # fmt: skip
        for role, outcome in (("hired", "hired"), ("rej", "rejected"), ("pend", "pending")):
            cid = f"c{i}{role}"
            applicants.append({"candidate_id": cid, "cv_text": CV, "has_cv": True, "area": "TI"})
            sensitive.append({"candidate_id": cid, "sex": "F", "pcd": "", "age_band": "25-34"})
            candidacies.append(
                {"vacancy_id": f"v{i}", "candidate_id": cid, "status": outcome, "outcome": outcome,
                 "candidacy_month": month, "updated_month": month, "modality": "",
                 "has_vacancy": True, "has_applicant": True}
            )  # fmt: skip
    return MaskedTables(
        _frame("vacancies", vacancies),
        _frame("applicants", applicants),
        _frame("sensitive", sensitive),
        _frame("candidacies", candidacies),
    )


def test_load_masked_reads_the_four_tables(tmp_path):
    tables = make_tables(3)
    for name in TABLE_COLUMNS:
        getattr(tables, name).to_parquet(tmp_path / f"{name}.parquet")

    loaded = load_masked(tmp_path)

    assert len(loaded.candidacies) == 9
    assert list(loaded.sensitive.columns) == TABLE_COLUMNS["sensitive"]


def test_load_masked_fails_loudly_on_a_missing_table_or_changed_schema(tmp_path):
    with pytest.raises(FileNotFoundError, match="make ingest"):
        load_masked(tmp_path)

    tables = make_tables(2)
    for name in TABLE_COLUMNS:
        getattr(tables, name).to_parquet(tmp_path / f"{name}.parquet")
    tables.sensitive.drop(columns=["sex"]).to_parquet(tmp_path / "sensitive.parquet")
    with pytest.raises(ValueError, match="unexpected columns"):
        load_masked(tmp_path)


def test_pairs_carry_the_label_and_never_the_sensitive_attributes():
    pairs = build_pairs(make_tables(2))

    assert set(pairs.loc[pairs["outcome"] == "hired", "y"]) == {1}
    assert set(pairs.loc[pairs["outcome"] != "hired", "y"]) == {0}
    assert pairs["resolved"].tolist() == [True, True, False] * 2
    assert not {"sex", "pcd", "age_band"} & set(pairs.columns)
    assert {"v_seniority", "a_area", "query", "cv_text"} <= set(pairs.columns)


def test_pairs_drop_short_texts_missing_records_and_duplicates():
    tables = make_tables(2)
    tables.applicants.loc[0, "cv_text"] = "curto"  # c0hired has no usable CV
    tables.vacancies.loc[1, "activities"] = ""  # v1 has no usable text
    dup = tables.candidacies.iloc[[1]]
    candidacies = pd.concat([tables.candidacies, dup, dup.assign(vacancy_id="ghost")])
    tables = MaskedTables(tables.vacancies, tables.applicants, tables.sensitive, candidacies)

    pairs = build_pairs(tables)

    assert sorted(pairs["candidate_id"]) == ["c0pend", "c0rej"]  # v1 gone, c0hired gone, ghost gone


def test_resolved_only_drops_pending_and_unrankable_vacancies():
    tables = make_tables(3)
    # v1 has a hire but nobody passed over; v2 has no hire at all
    keep = ~tables.candidacies["candidate_id"].isin(["c1rej", "c2hired"])
    candidacies = tables.candidacies[keep | (tables.candidacies["outcome"] == "pending")]
    pairs = build_pairs(
        MaskedTables(tables.vacancies, tables.applicants, tables.sensitive, candidacies)
    )

    frame = select_setting(pairs, "resolved_only")

    assert sorted(frame["vacancy_id"].unique()) == ["v0"]
    assert frame["resolved"].all()
    assert len(frame) == 2


def test_all_prospects_counts_pending_as_not_hired():
    pairs = build_pairs(make_tables(2))

    frame = select_setting(pairs, "all_prospects")

    assert len(frame) == 6
    assert frame.loc[frame["outcome"] == "pending", "y"].eq(0).all()


def test_a_vacancy_with_one_candidate_is_not_rankable():
    tables = make_tables(1)
    one = tables.candidacies[tables.candidacies["outcome"] == "hired"]
    pairs = build_pairs(MaskedTables(tables.vacancies, tables.applicants, tables.sensitive, one))

    assert select_setting(pairs, "all_prospects").empty


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="unknown setting"):
        select_setting(build_pairs(make_tables(2)), "everything")  # type: ignore[arg-type]


def test_temporal_split_trains_on_the_past_and_tests_on_the_future():
    frame = select_setting(build_pairs(make_tables(10)), "all_prospects")

    split = temporal_split(frame, test_fraction=0.2)

    assert split.cutoff == "2020-08"
    assert (
        split.train["requested_month"].max() <= split.cutoff < split.test["requested_month"].min()
    )
    assert set(split.train["vacancy_id"]).isdisjoint(split.test["vacancy_id"])
    assert split.test["vacancy_id"].nunique() == 2


def test_training_rows_whose_outcome_is_recorded_after_the_cutoff_are_dropped():
    tables = make_tables(10)
    late = tables.candidacies["candidate_id"] == "c0hired"
    tables.candidacies.loc[late, "updated_month"] = "2020-12"  # decided long after the cutoff
    frame = select_setting(build_pairs(tables), "all_prospects")

    split = temporal_split(frame, test_fraction=0.2)

    assert "c0hired" not in set(split.train["candidate_id"])
    assert "c0rej" in set(split.train["candidate_id"])  # the rest of that vacancy stays


def test_temporal_split_rejects_unusable_input():
    frame = select_setting(build_pairs(make_tables(10)), "all_prospects")

    with pytest.raises(ValueError, match="test_fraction"):
        temporal_split(frame, test_fraction=1.5)
    with pytest.raises(ValueError, match="requested_month"):
        temporal_split(frame.assign(requested_month=None))
    with pytest.raises(ValueError, match="two distinct months"):
        temporal_split(frame.assign(requested_month="2020-01"))


def test_summary_is_aggregates_only():
    summary = summarize(select_setting(build_pairs(make_tables(4)), "all_prospects"))

    assert summary == {
        "vacancies": 4,
        "candidacies": 12,
        "hires": 4,
        "hire_rate": 0.3333,
        "median_candidates_per_vacancy": 3.0,
    }
    assert summarize(pd.DataFrame({"vacancy_id": [], "y": []}))["hires"] == 0


def test_audit_attributes_are_available_separately():
    audit = audit_attributes(make_tables(2))

    assert {"sex", "pcd", "age_band"} <= set(audit.columns)
