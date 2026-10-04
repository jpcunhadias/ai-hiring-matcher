import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureBuilder,
    _area_parts,
    education_rank,
    language_rank,
    model_features,
    month_index,
)

SAP_QUERY = "consultor sap abap fiori s4hana modulo fi co integracao financeira " * 2
SAP_CV = "consultor sap com experiencia em abap fiori s4hana e modulo fi co financeira " * 4
COOK_CV = "cozinheira com experiencia em restaurante cardapio receitas e atendimento " * 5


def small(**kwargs) -> FeatureBuilder:
    """A builder whose vocabulary limits suit a handful of rows."""
    return FeatureBuilder(char_max_features=2000, min_df=1, max_df=1.0, common_share=1.0, **kwargs)


def pair(vacancy_id, candidate_id, **overrides) -> dict:
    row = {
        "vacancy_id": vacancy_id, "candidate_id": candidate_id, "status": "x",
        "outcome": "rejected",
        "candidacy_month": "2020-03", "updated_month": "2020-04", "v_title": "consultor sap",
        "query": SAP_QUERY, "requested_month": "2020-02", "v_client_id": "k1",
        "v_is_sap": "Sim", "v_contract_type": "PJ", "v_priority": "", "v_seniority": "Sênior",
        "v_education_level": "Ensino Superior Completo", "v_english_level": "Avançado",
        "v_spanish_level": "", "v_areas": "TI - SAP-", "v_state": "SP", "cv_text": COOK_CV,
        "a_education_level": "", "a_english_level": "", "a_spanish_level": "", "a_area": "",
        "a_seniority": "", "a_professional_title": "", "a_technical_skills": "",
        "a_certifications": "", "y": 0, "resolved": True,
    }  # fmt: skip
    return row | overrides


def hired(**overrides) -> dict:
    return pair(outcome="hired", y=1, **overrides)


def basic_frame() -> pd.DataFrame:
    rows = [
        hired(vacancy_id="v1", candidate_id="c1", cv_text=SAP_CV, a_area="TI - SAP",
              a_education_level="Pós Graduação Completo", a_english_level="Intermediário",
              a_professional_title="consultor sap sênior", a_technical_skills="abap fiori sap"),
        pair("v1", "c2", cv_text=COOK_CV, a_area="Administrativa"),
        pair("v1", "c3", cv_text=COOK_CV),
        hired(vacancy_id="v2", candidate_id="c4", cv_text=SAP_CV, requested_month="2020-08",
              candidacy_month="2020-09", updated_month="2020-10"),
        pair("v2", "c5", cv_text=COOK_CV, requested_month="2020-08", candidacy_month="2020-09",
             updated_month="2020-10"),
    ]  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def built():
    frame = basic_frame()
    builder = small().fit(frame)
    return frame, builder, builder.transform(frame, history=frame)


def test_helpers_parse_levels_months_and_areas():
    assert education_rank("Pós Graduação Completo") > education_rank("Ensino Superior Completo")
    assert education_rank("Ensino Superior Cursando") < education_rank("Ensino Superior Completo")
    assert np.isnan(education_rank("")) and np.isnan(language_rank("desconhecido"))
    assert language_rank("Fluente") > language_rank("Básico")
    assert month_index("2021-03") - month_index("2020-12") == 3
    assert np.isnan(month_index(None)) and np.isnan(month_index("21-3"))
    assert _area_parts("TI - Projetos-TI - SAP-") == ["ti - projetos", "ti - sap"]
    assert _area_parts("Administrativa-") == ["administrativa"]


def test_text_features_rank_the_relevant_cv_above_the_unrelated_one(built):
    frame, _, feats = built
    relevant, unrelated = (
        feats.loc[frame["candidate_id"] == "c1"],
        feats.loc[frame["candidate_id"] == "c2"],
    )

    for name in ("tfidf_word", "tfidf_char", "query_coverage", "skills_coverage", "title_sim"):
        assert relevant[name].iloc[0] > unrelated[name].iloc[0], name


def test_structured_features_compare_the_two_sides(built):
    frame, _, feats = built
    c1, c2, c3 = (feats[frame["candidate_id"] == c].iloc[0] for c in ("c1", "c2", "c3"))

    assert c1["education_gap"] == 1.0  # post-graduate against a bachelor's requirement
    assert c1["english_gap"] == -1.0  # intermediate against advanced
    assert np.isnan(c3["education_gap"]) and c3["a_education_filled"] == 0.0
    assert (c1["area_exact"], c1["area_group"], c1["sap_match"]) == (1.0, 1.0, 1.0)
    assert (c2["area_exact"], c2["area_group"], c2["sap_match"]) == (0.0, 0.0, 0.0)
    assert c1["seniority_in_cv"] == 1.0 and c2["seniority_in_cv"] == 0.0
    assert c1["a_skills_filled"] == 1.0 and c2["a_skills_filled"] == 0.0


def test_area_group_matches_the_broad_area_but_not_the_exact_one():
    frame = pd.DataFrame(
        [
            hired(
                vacancy_id="v1", candidate_id="c1", a_area="TI - Governança", v_areas="TI - SAP-"
            ),
            pair("v1", "c2"),
        ]
    )

    feats = small().fit(frame).transform(frame)

    assert (feats["area_exact"].iloc[0], feats["area_group"].iloc[0]) == (0.0, 1.0)


def test_within_vacancy_context_is_relative_to_the_same_vacancy(built):
    frame, _, feats = built
    v1 = feats[frame["vacancy_id"] == "v1"]

    assert (
        v1["n_candidates"].eq(3).all()
        and feats.loc[frame["vacancy_id"] == "v2", "n_candidates"].eq(2).all()
    )
    assert v1["tfidf_word_z"].sum() == pytest.approx(0.0, abs=1e-9)
    assert v1["tfidf_word_rank"].iloc[0] == 1.0 and v1["tfidf_word_rank"].iloc[1] < 1.0
    assert feats["cv_len_log_z"].notna().all()


def test_a_vacancy_whose_candidates_tie_gets_zero_z_scores():
    frame = pd.DataFrame([pair("v1", "c1"), hired(vacancy_id="v1", candidate_id="c2")])

    feats = small().fit(frame).transform(frame)

    assert feats["cv_len_log_z"].eq(0.0).all()


def test_history_only_looks_at_the_strict_past():
    frame = basic_frame()
    earlier = pair(
        "v0",
        "c1",
        candidacy_month="2020-01",
        updated_month="2020-01",
        outcome="hired",
        y=1,
        requested_month="2019-12",
    )
    same_month = pair(
        "vx", "c1", candidacy_month="2020-03", updated_month="2020-03", outcome="hired", y=1
    )
    later = pair(
        "vy", "c1", candidacy_month="2020-06", updated_month="2020-06", outcome="hired", y=1
    )
    history = pd.concat([frame, pd.DataFrame([earlier, same_month, later])], ignore_index=True)
    builder = small().fit(frame)

    feats = builder.transform(frame, history=history)
    row = feats[frame["candidate_id"] == "c1"].iloc[0]  # evaluated at 2020-03

    assert (
        row["prior_candidacies"] == 1.0
    )  # only the January one; same month and later do not count
    assert row["prior_hires"] == 1.0
    # removing the rows that come later changes nothing for the rows evaluated at 2020-03
    columns = ["prior_candidacies", "prior_hires", "client_prior_hire_rate"]
    without_future = builder.transform(frame, history=pd.concat([frame, pd.DataFrame([earlier])]))
    march = frame["candidacy_month"] == "2020-03"
    pd.testing.assert_frame_equal(without_future.loc[march, columns], feats.loc[march, columns])


def test_an_outcome_counts_only_once_it_had_been_recorded():
    frame = basic_frame()
    slow = pair(
        "v0", "c1", candidacy_month="2020-01", updated_month="2020-05", outcome="hired", y=1
    )
    history = pd.concat([frame, pd.DataFrame([slow])], ignore_index=True)
    builder = small().fit(frame)

    feats = builder.transform(frame, history=history)
    at_march = feats[frame["candidate_id"] == "c1"].iloc[0]

    assert at_march["prior_candidacies"] == 1.0  # the candidacy existed...
    assert at_march["prior_hires"] == 0.0  # ...but its outcome was recorded in May


def test_client_hire_rate_is_smoothed_towards_the_global_rate_and_bounded():
    frame = basic_frame()
    feats = small().fit(frame).transform(frame, history=frame)

    assert feats["client_prior_hire_rate"].between(0.0, 1.0).all()
    # nothing is known before the first candidacy: the rate is exactly the global one
    first = feats.loc[frame["candidacy_month"] == "2020-03", "client_prior_hire_rate"]
    assert first.eq(frame["y"].mean()).all()


def test_without_history_the_history_features_are_missing_but_lag_is_known():
    frame = basic_frame()

    feats = small().fit(frame).transform(frame)

    assert feats["prior_hires"].isna().all()
    assert feats["lag_months"].tolist() == [1, 1, 1, 1, 1]


def test_the_vocabulary_is_fit_on_the_training_frame_only():
    train, test = basic_frame().iloc[:3], basic_frame().iloc[3:].copy()
    test["cv_text"] = test["cv_text"] + " zzzunicornword"

    builder = small().fit(train)
    feats = builder.transform(test)

    assert "zzzunicornword" not in builder._word.vocabulary_  # type: ignore[union-attr]
    assert len(feats) == len(test) and feats["tfidf_word"].notna().all()


def test_an_injected_encoder_adds_the_embedding_feature():
    def encoder(texts: list[str]) -> np.ndarray:
        vectors = np.array(
            [[("sap" in t) + 0.1, ("cozinheira" in t) + 0.1] for t in texts], dtype=float
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    frame = basic_frame()
    feats = small(encoder=encoder).fit(frame).transform(frame)
    plain = small().fit(frame).transform(frame)

    assert feats["e5_cosine"].iloc[0] > feats["e5_cosine"].iloc[1]
    assert {"e5_cosine_z", "e5_cosine_rank"} <= set(feats.columns)
    assert "e5_cosine" not in plain.columns


def test_no_outcome_or_sensitive_column_can_leak_into_the_features(built):
    _, _, feats = built
    banned = {"y", "status", "outcome", "resolved", "updated_month", "sex", "pcd", "age_band",
              "vacancy_id", "candidate_id"}  # fmt: skip

    assert not banned & set(feats.columns)
    assert feats.index.equals(basic_frame().index)


def test_features_are_deterministic(built):
    frame, builder, feats = built

    pd.testing.assert_frame_equal(feats, builder.transform(frame, history=frame))


def test_process_artifacts_are_excluded_from_the_model_columns_unless_asked_for(built):
    _, _, feats = built

    assert "lag_months" in feats.columns
    assert "lag_months" not in model_features(feats)
    assert "lag_months" in model_features(feats, include_process=True)
    assert "tfidf_word" in model_features(feats)


def test_education_compound_names_pick_the_most_specific_level():
    assert education_rank("Ensino Médio Técnico Completo") == 2.5
    assert education_rank("Pós Doutorado Completo") == 6
    assert education_rank("Doutorado Cursando") == 5.5
    ladder = ["Ensino Fundamental Completo", "Ensino Médio Completo", "Ensino Técnico Completo",
              "Ensino Superior Completo", "Pós Graduação Completo", "Mestrado Completo",
              "Doutorado Completo"]  # fmt: skip
    ranks = [education_rank(v) for v in ladder]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)


@pytest.mark.parametrize("missing", [np.nan, None, pd.NA])
def test_missing_values_in_text_and_category_columns_do_not_crash(missing):
    frame = basic_frame()
    columns = ["a_education_level", "a_english_level", "a_area", "a_professional_title",
               "a_technical_skills", "v_areas", "v_seniority", "v_is_sap", "v_title"]  # fmt: skip
    for column in columns:
        frame[column] = frame[column].astype(object)
        frame.loc[frame.index[:3], column] = missing
    blank = frame.copy()
    blank = blank.fillna("").replace({pd.NA: ""})
    builder = small().fit(basic_frame())

    pd.testing.assert_frame_equal(builder.transform(frame), builder.transform(blank))


def test_the_character_and_query_vocabularies_are_fit_on_the_training_frame_only():
    train, test = basic_frame().iloc[:3], basic_frame().iloc[3:].copy()
    test["cv_text"] = test["cv_text"] + " qzxjvmarker"
    test["query"] = test["query"] + " wvkqzholdout"

    builder = small().fit(train)
    builder.transform(test)

    assert not {"qzx", "zxj", "xjv", "jvm"} & set(builder._char.vocabulary_)  # type: ignore[union-attr]
    assert "wvkqzholdout" not in builder._word.vocabulary_  # type: ignore[union-attr]
    assert "wvkqzholdout" not in builder._common


def _flipped(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        y=1 - frame["y"], outcome="pending", status="other", resolved=False, updated_month="2099-01"
    )


def test_a_feature_cannot_read_the_label_of_the_row_it_describes():
    frame = basic_frame()
    builder = small().fit(frame)

    plain = builder.transform(frame)  # no history: nothing may depend on any label

    pd.testing.assert_frame_equal(plain, builder.transform(_flipped(frame)))


def test_features_do_not_move_when_labels_of_the_latest_month_change():
    frame = basic_frame()  # v2's rows are the latest (2020-09)
    builder = small().fit(frame)
    latest = frame["candidacy_month"] == "2020-09"
    altered = pd.concat([frame[~latest], _flipped(frame[latest])]).sort_index()

    before = builder.transform(frame, history=frame)
    after = builder.transform(altered, history=altered)

    pd.testing.assert_frame_equal(before, after)  # nobody comes after them, so nothing can see it


def test_context_is_computed_over_the_pool_whatever_the_outcome_of_its_members():
    pool = basic_frame()
    evaluated = pool[pool["resolved"]].iloc[[0, 1]]  # only some candidates are scored
    builder = small().fit(pool)

    feats = builder.transform(evaluated, pool=pool)

    assert feats["n_candidates"].tolist() == [3, 3]  # the pool of v1, not the two scored rows
    changed = pool.copy()
    changed.loc[changed["candidate_id"] == "c3", ["outcome", "resolved", "y"]] = [
        "pending",
        False,
        0,
    ]
    pd.testing.assert_frame_equal(feats, builder.transform(evaluated, pool=changed))
    # the same rows scored without the pool see a smaller, outcome-filtered vacancy
    assert builder.transform(evaluated)["n_candidates"].tolist() == [2, 2]


def test_every_evaluated_row_must_be_part_of_the_pool():
    frame = basic_frame()
    builder = small().fit(frame)

    with pytest.raises(ValueError, match="part of the pool"):
        builder.transform(frame, pool=frame.iloc[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        builder.transform(frame, pool=pd.concat([frame, frame.iloc[[0]]]))
