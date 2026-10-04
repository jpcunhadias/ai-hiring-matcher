import json
import zipfile
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


def _everything_written(out_dir: Path) -> str:
    """All output content as one string: every table cell plus the report."""
    parts = [(out_dir / "masking_report.json").read_text()]
    for path in out_dir.glob("*.parquet"):
        parts.append(pd.read_parquet(path).to_csv())
    return "\n".join(parts)


@pytest.fixture
def masked(tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    _write_zips(tmp_path)
    out = tmp_path / "out"
    report = ingest.run(tmp_path, out, NoEntityMasker(), limit=None, review=0)
    return out, report


def test_no_sensitive_value_survives_anywhere_in_the_output(masked):
    out, _ = masked

    assert CANARY not in _everything_written(out).lower()


def test_output_schemas_are_exactly_the_allowlist(masked):
    out, _ = masked
    expected = {
        "vacancies": ingest.VACANCY_COLUMNS,
        "applicants": ingest.APPLICANT_COLUMNS,
        "sensitive": ingest.SENSITIVE_COLUMNS,
        "candidacies": ingest.CANDIDACY_COLUMNS,
    }

    for name, columns in expected.items():
        assert list(pd.read_parquet(out / f"{name}.parquet").columns) == columns


def test_ids_are_replaced_by_consistent_surrogates(masked):
    out, _ = masked
    applicants = pd.read_parquet(out / "applicants.parquet")
    candidacies = pd.read_parquet(out / "candidacies.parquet")
    vacancies = pd.read_parquet(out / "vacancies.parquet")

    assert "200" not in set(applicants["candidate_id"]) | set(candidacies["candidate_id"])
    assert "100" not in set(vacancies["vacancy_id"]) | set(candidacies["vacancy_id"])
    # the join keys still line up after pseudonymization
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


def test_sensitive_attributes_live_in_their_own_generalized_table(masked):
    out, _ = masked
    sensitive = pd.read_parquet(out / "sensitive.parquet").iloc[0]
    applicants = pd.read_parquet(out / "applicants.parquet")

    assert sensitive["sex"] == "Feminino"
    assert sensitive["age_band"] == "35-44"  # not a birth date
    assert not {"sex", "pcd", "age_band", "birth"} & set(applicants.columns)


def test_cv_text_keeps_useful_content_and_loses_identifiers(masked):
    out, _ = masked
    cv = pd.read_parquet(out / "applicants.parquet").iloc[0]["cv_text"]

    assert "Python" in cv and "2015-2018" in cv
    assert "[NAME]" in cv and "[EMAIL]" in cv and "[PHONE]" in cv


def test_report_is_aggregate_only_and_shows_zero_residuals(masked):
    _, report = masked

    assert report["rows"] == {"vacancies": 1, "applicants": 1, "sensitive": 1, "candidacies": 2}
    assert report["cv_texts_with_surviving_own_name"] == 0
    for hits in report["residual_pattern_hits"].values():
        assert sum(hits.values()) == 0
    assert report["replacements"]["patterns_and_own_name"]["OWN_NAME"] >= 2
    assert report["applicants_empty_rate_by_structured_field"]["seniority"] == 0.0


def test_ner_failure_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    _write_zips(tmp_path)

    class Broken(NoEntityMasker):
        def mask_many(self, texts):
            raise RuntimeError("model unavailable")

    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        ingest.run(tmp_path, out, Broken(), limit=None, review=0)

    assert not list(out.glob("*.parquet")) if out.exists() else True
    assert not (out / "masking_report.json").exists()


def test_salt_is_created_once_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("MASKING_SALT", raising=False)
    path = tmp_path / "sub" / ".salt"

    first = ingest.load_salt(path)
    second = ingest.load_salt(path)

    assert first == second and len(first) == 32
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_different_salts_give_unlinkable_ids(tmp_path, monkeypatch):
    _write_zips(tmp_path)
    ids = []
    for salt in ("a" * 32, "b" * 32):
        monkeypatch.setenv("MASKING_SALT", salt)
        out = tmp_path / f"out_{salt[0]}"
        ingest.run(tmp_path, out, NoEntityMasker(), limit=None, review=0)
        ids.append(pd.read_parquet(out / "applicants.parquet").iloc[0]["candidate_id"])

    assert ids[0] != ids[1]


def test_review_sample_is_only_written_on_request(tmp_path, monkeypatch):
    monkeypatch.setenv("MASKING_SALT", SALT.decode())
    _write_zips(tmp_path)

    ingest.run(tmp_path, tmp_path / "plain", NoEntityMasker(), limit=None, review=0)
    ingest.run(tmp_path, tmp_path / "rev", NoEntityMasker(), limit=None, review=1)

    assert not (tmp_path / "plain" / "review_sample.txt").exists()
    assert (tmp_path / "rev" / "review_sample.txt").exists()
