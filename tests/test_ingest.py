import json
import os
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src import ingest
from src.masking import NoEntityMasker

SALT = b"t" * 32
CANARY = "canary"  # every sensitive value below contains this word


def _raw():
    vagas = {
        "100": {
            "informacoes_basicas": {
                "cliente": "CanaryClient Ltda",
                "analista_responsavel": "CanaryAnalyst",
                "requisitante": "CanaryRequester",
                "solicitante_cliente": "CanaryAsker",
                "superior_imediato": "CanaryBoss",
                "telefone": "(11) 91111-2222",
                "nome": "CanaryVacancyName",
                "nome_substituto": "CanarySubstitute",
                "objetivo_vaga": "CanaryObjective",
                "titulo_vaga": "Desenvolvedor Python",
                "vaga_sap": "Não",
                "tipo_contratacao": "CLT Full",
                "prioridade_vaga": "Alta",
                "data_requicisao": "05-03-2021",
            },
            "perfil_vaga": {
                "principais_atividades": "Falar com canary@leak.com ou (11) 99999-0000",
                "competencia_tecnicas_e_comportamentais": "Python e SQL",
                "bairro": "CanaryNeighborhood",
                "cidade": "CanaryCity",
                "local_trabalho": "CanaryPlace",
                "demais_observacoes": "CanaryNotes",
                "estado": "SP",
                "nivel profissional": "Sênior",
                "nivel_academico": "Ensino Superior Completo",
                "nivel_ingles": "Avançado",
                "nivel_espanhol": "Básico",
                "areas_atuacao": "TI",
            },
            "beneficios": {"valor_compra_1": "CanaryPrice", "valor_venda": "CanarySell"},
        }
    }
    prospects = {
        "100": {
            "titulo": "Desenvolvedor Python",
            "modalidade": "CLT",
            "prospects": [
                {
                    "codigo": "200",
                    "nome": "CanaryProspectName",
                    "situacao_candidado": "Contratado pela Decision",
                    "comentario": "CanaryComment",
                    "recrutador": "CanaryRecruiter",
                    "data_candidatura": "10-03-2021",
                    "ultima_atualizacao": "12-03-2021",
                },
                {
                    "codigo": "999",
                    "situacao_candidado": "Prospect",
                    "data_candidatura": "11-03-2021",
                },
            ],
        }
    }
    applicants = {
        "200": {
            "informacoes_pessoais": {
                "nome": "CanaryFirst CanaryLast",
                "email": "canary@leak.com",
                "cpf": "123.456.789-09",
                "telefone_celular": "(11) 98888-7777",
                "endereco": "Rua Canary 12",
                "data_nascimento": "1990-04-02",
                "sexo": "Feminino",
                "pcd": "Não",
                "estado_civil": "CanaryMarital",
                "url_linkedin": "linkedin.com/in/canary",
                "facebook": "fb.com/canary",
                "skype": "canary.skype",
                "fonte_indicacao": "CanarySource",
            },
            "infos_basicas": {
                "nome": "CanaryFirst CanaryLast",
                "email": "canary@leak.com",
                "local": "CanaryTown",
                "inserido_por": "CanaryRecruiter",
                "objetivo_profissional": "CanaryObjective",
            },
            "informacoes_profissionais": {
                "titulo_profissional": "Analista de Dados",
                "conhecimentos_tecnicos": "Python, SQL",
                "certificacoes": "AWS",
                "outras_certificacoes": "CanaryOther",
                "remuneracao": "CanarySalary",
                "area_atuacao": "TI",
                "nivel_profissional": "Pleno",
            },
            "formacao_e_idiomas": {
                "nivel_academico": "Ensino Superior Completo",
                "nivel_ingles": "Intermediário",
                "nivel_espanhol": "Básico",
                "instituicao_ensino_superior": "CanaryUniversity",
                "cursos": "CanaryCourse",
            },
            "cv_pt": (
                "CanaryFirst CanaryLast\nEmail canary@leak.com Tel (11) 98888-7777\n"
                "Experiência em Python 2015-2018"
            ),
            "cv_en": "",
        }
    }
    return vagas, prospects, applicants


def _write_zips(directory: Path):
    for name, data in zip(("vagas", "prospects", "applicants"), _raw(), strict=True):
        with zipfile.ZipFile(directory / f"{name}.zip", "w") as z:
            z.writestr(f"{name}.json", json.dumps(data))


# Raw identifiers planted in the fixture. The word "canary" alone is not enough: values like a
# phone number or a birth date don't contain it, so they are asserted individually.
RAW_IDENTIFIERS = [
    "canary",
    "98888",
    "7777",
    "99999-0000",
    "123.456.789",
    "1990-04-02",
    "Rua Canary",
    "leak.com",
    "91111-2222",
]


def _everything_written(out_dir: Path) -> str:
    """All output content as one string: every table cell, the report and any review sample."""
    parts = [(out_dir / "masking_report.json").read_text()]
    for path in out_dir.glob("*.parquet"):
        parts.append(pd.read_parquet(path).to_csv())
    for extra in ("review_sample.txt", ".salt"):
        if (out_dir / extra).exists():
            parts.append((out_dir / extra).read_text())
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def tiny_first_names(tmp_path_factory, monkeypatch):
    """The real IBGE list is gitignored; tests bring their own two-name list."""
    path = tmp_path_factory.mktemp("names") / "names.csv"
    path.write_text("Nome,1930,1940\nMARIA,100,200\nJOSE,50,60\n", encoding="utf-8")
    monkeypatch.setattr(ingest, "FIRST_NAMES_CSV", path)


@pytest.fixture
def raw_dir(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    _write_zips(directory)
    return directory


@pytest.fixture
def masked(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    report = ingest.run(raw_dir, out, NoEntityMasker(), review=1)
    return out, report


def test_no_sensitive_value_survives_anywhere_in_the_output(masked):
    out, _ = masked
    everything = _everything_written(out)

    for identifier in RAW_IDENTIFIERS:
        assert identifier.lower() not in everything.lower(), identifier


def test_output_schemas_are_exactly_the_allowlist(masked):
    out, _ = masked

    for name, columns in ingest.TABLE_COLUMNS.items():
        assert list(pd.read_parquet(out / f"{name}.parquet").columns) == columns


def test_ids_are_replaced_by_consistent_surrogates(masked):
    out, _ = masked
    applicants = pd.read_parquet(out / "applicants.parquet")
    candidacies = pd.read_parquet(out / "candidacies.parquet")
    vacancies = pd.read_parquet(out / "vacancies.parquet")

    assert "200" not in set(applicants["candidate_id"]) | set(candidacies["candidate_id"])
    assert "100" not in set(vacancies["vacancy_id"]) | set(candidacies["vacancy_id"])
    assert set(candidacies["candidate_id"]) >= set(applicants["candidate_id"])
    assert set(candidacies["vacancy_id"]) == set(vacancies["vacancy_id"])


def test_candidacies_keep_outcome_month_and_link_flags(masked):
    out, _ = masked
    c = pd.read_parquet(out / "candidacies.parquet").set_index("status")

    assert c.loc["Contratado pela Decision", "outcome"] == "hired"
    assert c.loc["Contratado pela Decision", "candidacy_month"] == "2021-03"
    assert bool(c.loc["Contratado pela Decision", "has_applicant"]) is True
    assert c.loc["Prospect", "outcome"] == "pending"
    assert bool(c.loc["Prospect", "has_applicant"]) is False  # candidate 999 has no record


def test_candidate_code_zero_keeps_its_identity_and_join():
    salt = b"s" * 32
    prospects = {"1": {"prospects": [{"codigo": 0, "situacao_candidado": "Prospect"}]}}

    row = ingest.build_candidacies(prospects, {"1"}, {"0"}, salt).iloc[0]

    assert row["candidate_id"] == ingest.surrogate_id("cand", "0", salt) != ""
    assert bool(row["has_applicant"]) is True


def test_sensitive_attributes_live_in_their_own_generalized_table(masked):
    out, report = masked
    sensitive = pd.read_parquet(out / "sensitive.parquet").iloc[0]
    applicants = pd.read_parquet(out / "applicants.parquet")

    assert sensitive["sex"] == "Feminino"
    # born 1990-04-02, snapshot 2021-03-12 -> aged 30 then (not 35, as a fixed 2025 would say)
    assert sensitive["age_band"] == "25-34"
    assert report["age_reference_date"] == "2021-03-12"  # the latest date in the data
    assert not {"sex", "pcd", "age_band", "birth"} & set(applicants.columns)


def test_cv_text_keeps_useful_content_and_loses_identifiers(masked):
    out, _ = masked
    cv = pd.read_parquet(out / "applicants.parquet").iloc[0]["cv_text"]

    assert "Python" in cv and "2015-2018" in cv
    assert "[NAME]" in cv and "[EMAIL]" in cv and "[PHONE]" in cv


def test_retained_category_fields_are_scrubbed_too(raw_dir, tmp_path, monkeypatch):
    # Regression: `area` was copied as-is, so an email or phone there reached the output.
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    vagas, prospects, applicants = _raw()
    applicants["200"]["informacoes_profissionais"]["area_atuacao"] = "TI - ligar 98765 4321"
    vagas["100"]["perfil_vaga"]["areas_atuacao"] = "TI, contato canary@leak.com"
    for name, data in zip(
        ("vagas", "prospects", "applicants"), (vagas, prospects, applicants), strict=True
    ):
        with zipfile.ZipFile(raw_dir / f"{name}.zip", "w") as z:
            z.writestr(f"{name}.json", json.dumps(data))
    out = tmp_path / "out"

    ingest.run(raw_dir, out, NoEntityMasker())

    assert "98765" not in _everything_written(out)
    assert "leak.com" not in _everything_written(out)


def test_name_dictionary_layer_masks_names_the_structured_record_does_not_know(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    names_csv = tmp_path / "names.csv"
    names_csv.write_text("Nome,ate2010\nROBERTA,1000\nCLAUDIO,900\n", encoding="utf-8")
    vagas, prospects, applicants = _raw()
    applicants["200"]["cv_pt"] += "\nreferencia: roberta canary-silva"  # an unrelated person
    applicants["200"]["cv_pt"] = applicants["200"]["cv_pt"].replace("canary-silva", "claudio")
    for name, data in zip(
        ("vagas", "prospects", "applicants"), (vagas, prospects, applicants), strict=True
    ):
        with zipfile.ZipFile(raw_dir / f"{name}.zip", "w") as z:
            z.writestr(f"{name}.json", json.dumps(data))
    out = tmp_path / "out"

    report = ingest.run(raw_dir, out, NoEntityMasker(), first_names_csv=names_csv)

    cv = pd.read_parquet(out / "applicants.parquet").iloc[0]["cv_text"]
    assert "roberta" not in cv and "claudio" not in cv
    assert report["name_dictionary"]["first_names"] == 2


def test_a_missing_first_name_list_stops_the_run_unless_explicitly_waived(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    missing = tmp_path / "nope.csv"

    with pytest.raises(ValueError, match="first-name list .* is required"):
        ingest.run(raw_dir, tmp_path / "out", NoEntityMasker(), first_names_csv=missing)
    assert not (tmp_path / "out").exists()  # nothing was written

    report = ingest.run(
        raw_dir, tmp_path / "out", NoEntityMasker(), first_names_csv=missing,
        require_name_dictionary=False,
    )  # fmt: skip
    assert report["name_dictionary"] is None


def test_report_is_aggregate_only_and_separates_self_check_from_the_independent_one(masked):
    _, report = masked

    assert report["rows"] == {"vacancies": 1, "applicants": 1, "sensitive": 1, "candidacies": 2}
    assert report["mode"] == "full"
    assert report["cv_texts_with_surviving_own_name"] == 0
    for hits in report["self_consistency_residual_hits"].values():
        assert sum(hits.values()) == 0
    for hits in report["independent_residual_estimate"].values():
        assert sum(hits.values()) == 0
    assert "proof" in report["residual_notes"]
    assert report["applicants_empty_rate_by_structured_field"]["seniority"] == 0.0


def test_independent_estimate_sees_a_leak_the_masker_misses(masked):
    # The two checks must be able to disagree; otherwise "0 residuals" proves nothing.
    out, _ = masked
    applicants = pd.read_parquet(out / "applicants.parquet")
    applicants["cv_text"] = ["a pessoa nasceu em 1990-04-02 aqui"]
    tables = {
        "vacancies": pd.read_parquet(out / "vacancies.parquet"),
        "applicants": applicants,
        "sensitive": pd.read_parquet(out / "sensitive.parquet"),
        "candidacies": pd.read_parquet(out / "candidacies.parquet"),
    }

    report = ingest.build_report(tables, ingest.TextMasker(), 0, {}, date(2021, 3, 12), "full")

    assert report["independent_residual_estimate"]["applicants.cv_text"]["iso_date"] == 1


# --- fail-closed, staging, salt and permissions -------------------------------------------


class _Broken(NoEntityMasker):
    def mask_many(self, texts):
        raise RuntimeError("model unavailable")


def test_ner_failure_writes_nothing_and_creates_no_salt(raw_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"

    with pytest.raises(RuntimeError):
        ingest.run(raw_dir, out, _Broken())

    assert not out.exists()  # not even the output directory, so no .salt either
    assert not list(tmp_path.glob(".out.staging-*"))  # and no half-finished staging left behind


def test_a_missing_archive_creates_no_salt(tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        ingest.run(empty, tmp_path / "out", NoEntityMasker())

    assert not (tmp_path / "out").exists()


def test_a_failed_run_leaves_the_previous_generation_untouched(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    with pytest.raises(RuntimeError):
        ingest.run(raw_dir, out, _Broken())

    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not list(tmp_path.glob(".out.staging-*"))


def test_a_failure_while_writing_a_table_publishes_nothing(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    original = pd.DataFrame.to_parquet
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # the second table fails after the first was written
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", flaky)

    with pytest.raises(OSError):
        ingest.run(raw_dir, out, NoEntityMasker())

    assert {p.name: p.read_bytes() for p in out.iterdir()} == before  # no mixed generation


def test_outputs_are_private(raw_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    old_umask = os.umask(0o022)  # a permissive umask must not leak into the outputs
    try:
        ingest.run(raw_dir, out, NoEntityMasker(), review=1)
    finally:
        os.umask(old_umask)

    assert oct(out.stat().st_mode & 0o777) == "0o700"
    for path in out.iterdir():
        assert oct(path.stat().st_mode & 0o777) == "0o600", path.name


def test_the_salt_is_created_on_success_reused_and_never_changes(raw_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"

    ingest.run(raw_dir, out, NoEntityMasker())
    first = (out / ".salt").read_text()
    ids_first = pd.read_parquet(out / "applicants.parquet")["candidate_id"].tolist()
    ingest.run(raw_dir, out, NoEntityMasker())

    assert (out / ".salt").read_text() == first
    assert pd.read_parquet(out / "applicants.parquet")["candidate_id"].tolist() == ids_first


@pytest.mark.parametrize("bad", ["", "abcd", "zz" * 20])
def test_a_weak_or_corrupt_salt_file_is_rejected(tmp_path, monkeypatch, bad):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    out.mkdir()
    (out / ".salt").write_text(bad)

    with pytest.raises(ValueError):
        ingest.resolve_salt(out)


def test_a_group_readable_salt_file_is_tightened(tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    out.mkdir()
    (out / ".salt").write_text("ab" * 32)
    (out / ".salt").chmod(0o644)

    ingest.resolve_salt(out)

    assert oct((out / ".salt").stat().st_mode & 0o777) == "0o600"


def test_a_too_short_env_salt_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("MASKING_SALT", "short")

    with pytest.raises(ValueError):
        ingest.resolve_salt(tmp_path)


def test_different_salts_give_unlinkable_ids(raw_dir, tmp_path, monkeypatch):
    ids = []
    for salt in ("a" * 32, "b" * 32):
        monkeypatch.setenv("MASKING_SALT", salt)
        out = tmp_path / f"out_{salt[0]}"
        ingest.run(raw_dir, out, NoEntityMasker())
        ids.append(pd.read_parquet(out / "applicants.parquet").iloc[0]["candidate_id"])

    assert ids[0] != ids[1]


# --- --review ----------------------------------------------------------------------------


def test_review_sample_is_only_written_on_request(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())

    ingest.run(raw_dir, tmp_path / "plain", NoEntityMasker(), review=0)
    ingest.run(raw_dir, tmp_path / "rev", NoEntityMasker(), review=1)

    assert not (tmp_path / "plain" / "review_sample.txt").exists()
    assert (tmp_path / "rev" / "review_sample.txt").exists()


def test_review_larger_than_the_cv_population_is_clamped_not_a_crash(
    raw_dir, tmp_path, monkeypatch
):
    # Regression: ten applicants, one with a CV, --review 2 raised AFTER writing everything.
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    vagas, prospects, applicants = _raw()
    base = applicants["200"]
    extra = {str(300 + i): {**base, "cv_pt": ""} for i in range(9)}
    for name, data in zip(
        ("vagas", "prospects", "applicants"),
        (vagas, prospects, {**applicants, **extra}),
        strict=True,
    ):
        with zipfile.ZipFile(raw_dir / f"{name}.zip", "w") as z:
            z.writestr(f"{name}.json", json.dumps(data))
    out = tmp_path / "out"

    ingest.run(raw_dir, out, NoEntityMasker(), review=2)

    assert len((out / "review_sample.txt").read_text().split("=====")) == 1  # exactly one CV


def test_negative_review_is_rejected_before_anything_is_written(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())

    with pytest.raises(ValueError):
        ingest.run(raw_dir, tmp_path / "out", NoEntityMasker(), review=-1)

    assert not (tmp_path / "out").exists()


# --- repair ------------------------------------------------------------------------------


def test_repair_reapplies_the_layers_to_existing_tables_without_a_full_rerun(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())
    # Simulate output produced by an older, weaker masker: leaks that today's layers catch.
    applicants = pd.read_parquet(out / "applicants.parquet")
    applicants["cv_text"] = ["nasceu em 1990-04-02, estado civil: casado, ligar 98765 4321"]
    applicants["professional_title"] = ["analista (canary@leak.com)"]
    applicants.to_parquet(out / "applicants.parquet", index=False)
    other_tables_before = {
        n: (out / f"{n}.parquet").read_bytes() for n in ("sensitive", "candidacies")
    }

    report = ingest.repair(raw_dir, out, NoEntityMasker())

    repaired = pd.read_parquet(out / "applicants.parquet").iloc[0]
    assert "1990" not in repaired["cv_text"] and "casado" not in repaired["cv_text"]
    assert "98765" not in repaired["cv_text"] and "leak.com" not in repaired["professional_title"]
    assert report["mode"] == "repair"
    assert report["independent_residual_estimate"]["applicants.cv_text"]["iso_date"] == 0
    # the sensitive table is rebuilt from the raw archive; the candidacies are carried over
    assert pd.read_parquet(out / "candidacies.parquet").equals(
        pd.read_parquet(__import__("io").BytesIO(other_tables_before["candidacies"]))
    )
    for identifier in RAW_IDENTIFIERS:
        assert identifier.lower() not in _everything_written(out).lower(), identifier


def test_repair_is_idempotent(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())
    ingest.repair(raw_dir, out, NoEntityMasker())
    once = pd.read_parquet(out / "applicants.parquet")

    ingest.repair(raw_dir, out, NoEntityMasker())

    assert pd.read_parquet(out / "applicants.parquet").equals(once)


def test_repair_needs_an_existing_generation(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())

    with pytest.raises(FileNotFoundError):
        ingest.repair(raw_dir, tmp_path / "missing", NoEntityMasker())


# --- the publication gate -----------------------------------------------------------------


def _clean_report():
    zero = {c: 0 for c in ("iso_date", "labeled_value", "email_like", "phone_like")}
    return {
        "self_consistency_residual_hits": {"applicants.cv_text": {"EMAIL": 0}},
        "independent_residual_estimate": {"applicants.cv_text": dict(zero)},
        "cv_texts_with_surviving_own_name": 0,
    }


def test_the_gate_passes_a_clean_report_and_ignores_the_soft_estimates():
    report = _clean_report()
    report["independent_residual_estimate"]["applicants.cv_text"].update(
        phone_like=98, url_or_domain=1163, age_like=198, long_digit_run=3471
    )

    ingest.privacy_gate(report)  # soft checks have false positives: reported, not enforced


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r["self_consistency_residual_hits"]["applicants.cv_text"].update(EMAIL=2),
         "self_consistency"),
        (lambda r: r.update(cv_texts_with_surviving_own_name=1), "own_name_survivors"),
        (lambda r: r["independent_residual_estimate"]["applicants.cv_text"].update(email_like=3),
         "applicants.cv_text.email_like"),
        (lambda r: r["independent_residual_estimate"]["applicants.cv_text"].update(iso_date=1),
         "applicants.cv_text.iso_date"),
        (lambda r: r["independent_residual_estimate"]["applicants.cv_text"].update(
            labeled_value=1), "applicants.cv_text.labeled_value"),
    ],
)  # fmt: skip
def test_the_gate_blocks_on_each_hard_check_and_its_message_holds_counts_only(mutate, expected):
    report = _clean_report()
    mutate(report)

    with pytest.raises(ingest.PrivacyGateError, match="blocked") as caught:
        ingest.privacy_gate(report)

    assert expected in str(caught.value)
    assert not any(c.isalpha() and c.isupper() for c in str(caught.value).split(": ", 1)[1])


def test_a_blocked_publication_writes_nothing_and_keeps_the_previous_generation(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())
    before = (out / "applicants.parquet").read_bytes()
    monkeypatch.setattr(ingest, "residual_pattern_counts", lambda texts: {"EMAIL": 1})

    with pytest.raises(ingest.PrivacyGateError):
        ingest.run(raw_dir, out, NoEntityMasker())

    assert (out / "applicants.parquet").read_bytes() == before
    assert not list(tmp_path.glob(".out.staging-*"))


def test_a_normal_run_passes_the_gate_and_records_it(masked):
    _, report = masked

    assert report["privacy_gate"] == "passed"
    assert report["categories_replaced"] == {}


# --- retained categories ------------------------------------------------------------------


def test_a_personal_string_in_a_retained_category_is_replaced_and_counted(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    applicants = json.loads(zipfile.ZipFile(raw_dir / "applicants.zip").read("applicants.json"))
    person = applicants["200"]
    person["formacao_e_idiomas"]["nivel_ingles"] = "jane.doe@example.com"
    person["informacoes_profissionais"]["nivel_profissional"] = "ligar (11) 98765-4321"
    person["informacoes_pessoais"]["sexo"] = "nasc 1990-04-02"
    with zipfile.ZipFile(raw_dir / "applicants.zip", "w") as z:
        z.writestr("applicants.json", json.dumps(applicants))

    report = ingest.run(raw_dir, tmp_path / "out", NoEntityMasker())

    out = tmp_path / "out"
    everything = _everything_written(out)
    for leaked in ("jane.doe", "98765", "1990-04-02"):
        assert leaked not in everything
    table = pd.read_parquet(out / "applicants.parquet")
    assert table["english_level"].tolist() == ["[REDACTED]"]
    assert table["education_level"].tolist() == ["Ensino Superior Completo"]  # untouched
    assert report["categories_replaced"] == {
        "applicants.english_level": 1, "applicants.seniority": 1, "sensitive.sex": 1,
    }  # fmt: skip


def test_category_columns_cover_every_retained_category():
    retained = {
        "vacancies": {"is_sap", "contract_type", "priority", "seniority", "education_level",
                      "english_level", "spanish_level", "state"},
        "applicants": {"education_level", "english_level", "spanish_level", "seniority"},
        "sensitive": {"sex", "pcd"},
        "candidacies": {"status", "modality"},
    }  # fmt: skip

    assert {t: set(c) for t, c in ingest.CATEGORY_COLUMNS.items()} == retained


# --- own-name spellings -------------------------------------------------------------------


def test_applicant_names_keep_each_spelling_apart_and_drop_duplicates():
    person = {"informacoes_pessoais": {"nome": "Li Wu"}, "infos_basicas": {"nome": "LI  WU"}}
    other = {"informacoes_pessoais": {"nome": "João Silva"}, "infos_basicas": {"nome": "Joao S."}}

    assert ingest.applicant_names(person) == ["Li Wu"]
    assert ingest.applicant_names(other) == ["João Silva", "Joao S."]
    assert ingest.applicant_names({}) == []


def test_a_short_name_present_in_both_fields_is_masked_and_not_reported_as_a_survivor(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    applicants = json.loads(zipfile.ZipFile(raw_dir / "applicants.zip").read("applicants.json"))
    person = applicants["200"]
    person["informacoes_pessoais"]["nome"] = person["infos_basicas"]["nome"] = "Li Wu"
    person["cv_pt"] = "li wu\nexperiencia em python"
    with zipfile.ZipFile(raw_dir / "applicants.zip", "w") as z:
        z.writestr("applicants.json", json.dumps(applicants))

    report = ingest.run(raw_dir, tmp_path / "out", NoEntityMasker())

    text = pd.read_parquet(tmp_path / "out" / "applicants.parquet")["cv_text"].iloc[0]
    assert "li wu" not in text.lower() and report["cv_texts_with_surviving_own_name"] == 0


# --- what the NER model may not be told to ignore ------------------------------------------


def test_protection_comes_from_role_words_and_generic_tokens_never_from_a_frequent_first_name():
    from src.masking import NameDictionary

    texts = [f"Maria Analista Python {i}" for i in range(40)]
    dictionary = NameDictionary.build(["maria"], [])

    protected = ingest.protected_vocabulary(texts, dictionary)

    assert "maria" not in protected  # frequent, but a person's name
    assert {"analista", "junior", "pleno"} <= protected  # role and seniority words
    assert "python" in protected  # generic and not a name
    assert "maria" in ingest.protected_vocabulary(texts, None)  # no dictionary, no exclusion


def test_a_first_name_that_is_frequent_in_titles_is_still_masked_by_the_ner_layer(
    raw_dir, tmp_path, monkeypatch
):
    from src.masking import HFEntityMasker

    def pipe(chunks, batch_size, stride):  # flags every "Maria" as a person
        found = []
        for chunk in chunks:
            at = chunk.find("Maria")
            hit = {"entity_group": "PER", "score": 0.99, "start": at, "end": at + 5}
            found.append([hit] if at >= 0 else [])
        return found

    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    vagas = json.loads(zipfile.ZipFile(raw_dir / "vagas.zip").read("vagas.json"))
    template = vagas["100"]
    vagas = {str(1000 + i): json.loads(json.dumps(template)) for i in range(35)}
    for i, vacancy in enumerate(vagas.values()):
        vacancy["informacoes_basicas"]["titulo_vaga"] = f"Maria Analista {i} Python"
    with zipfile.ZipFile(raw_dir / "vagas.zip", "w") as z:
        z.writestr("vagas.json", json.dumps(vagas))

    ingest.run(raw_dir, tmp_path / "out", HFEntityMasker(pipe=pipe))

    titles = pd.read_parquet(tmp_path / "out" / "vacancies.parquet")["title"]
    assert not titles.str.contains("Maria").any()
    assert titles.str.contains("Analista").all()  # the role word is protected


# --- salt and repair ----------------------------------------------------------------------


def test_repair_without_the_original_salt_is_refused_and_changes_nothing(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())  # creates a file salt
    (out / ".salt").unlink()  # ... which is then lost
    before = {p.name: p.read_bytes() for p in out.glob("*.parquet")}

    with pytest.raises(ValueError, match="original salt"):
        ingest.repair(raw_dir, out, NoEntityMasker())

    assert {p.name: p.read_bytes() for p in out.glob("*.parquet")} == before
    assert not (out / ".salt").exists()  # no invented salt either


def test_repair_with_a_salt_that_does_not_reproduce_the_published_ids_is_refused(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())  # an env salt leaves no .salt file behind
    monkeypatch.setenv("MASKING_SALT", "x" * 32)

    with pytest.raises(ValueError, match="does not reproduce"):
        ingest.repair(raw_dir, out, NoEntityMasker())


def test_an_environment_salt_that_differs_from_the_stored_one_is_refused(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())  # file salt
    monkeypatch.setenv("MASKING_SALT", "y" * 32)

    with pytest.raises(ValueError, match="differs from the salt"):
        ingest.run(raw_dir, out, NoEntityMasker())
    with pytest.raises(ValueError, match="differs from the salt"):
        ingest.repair(raw_dir, out, NoEntityMasker())


def test_repair_with_the_right_salt_still_works_after_the_checks(raw_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())

    report = ingest.repair(raw_dir, out, NoEntityMasker())

    assert report["mode"] == "repair" and report["privacy_gate"] == "passed"


# --- an interrupted publish ---------------------------------------------------------------


def test_a_crash_between_the_two_renames_is_recovered_and_the_salt_survives(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    out = tmp_path / "out"
    ingest.run(raw_dir, out, NoEntityMasker())
    ids = pd.read_parquet(out / "applicants.parquet")["candidate_id"].tolist()
    out.rename(tmp_path / "out.previous")  # the crash: no live directory, only the backup
    (tmp_path / ".out.staging-dead").mkdir()  # and a staging directory nobody finished

    ingest.run(raw_dir, out, NoEntityMasker())  # the next run must not invent a new salt

    assert pd.read_parquet(out / "applicants.parquet")["candidate_id"].tolist() == ids
    assert not (tmp_path / "out.previous").exists() and not list(tmp_path.glob(".out.staging-*"))


def test_each_generation_carries_a_marker_that_changes_on_every_publish(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    out = tmp_path / "out"

    ingest.run(raw_dir, out, NoEntityMasker())
    first = (out / ingest.GENERATION_FILE).read_text()
    ingest.run(raw_dir, out, NoEntityMasker())

    assert first and (out / ingest.GENERATION_FILE).read_text() != first
    assert (out / ingest.GENERATION_FILE).stat().st_mode & 0o077 == 0


# --- errors never carry personal text -----------------------------------------------------


class _Leaky:
    def mask_many(self, texts):
        raise RuntimeError("model failed on: " + texts[0])


def test_a_masking_failure_surfaces_without_the_text_or_the_original_message(
    raw_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())

    with pytest.raises(ingest.MaskingError) as caught:
        ingest.run(raw_dir, tmp_path / "out", _Leaky())

    message = str(caught.value)
    assert "Desenvolvedor" not in message and "model failed" not in message
    assert "RuntimeError" in message  # the type is safe and useful
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert not (tmp_path / "out").exists()


def test_the_command_line_withholds_unexpected_error_details(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("secret person text")

    monkeypatch.setattr(ingest, "run", explode)

    with pytest.raises(SystemExit) as caught:
        ingest.main(["--ner", "none"])
    assert "secret" not in str(caught.value) and "RuntimeError" in str(caught.value)

    monkeypatch.setenv("INGEST_DEBUG", "1")
    with pytest.raises(RuntimeError, match="secret person text"):
        ingest.main(["--ner", "none"])


def test_the_command_line_shows_our_own_static_messages(monkeypatch):
    monkeypatch.setattr(
        ingest, "run", lambda *a, **k: (_ for _ in ()).throw(ValueError("needs a salt"))
    )

    with pytest.raises(SystemExit, match="needs a salt"):
        ingest.main(["--ner", "none"])
