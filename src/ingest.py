"""Private ingest of the real ATS data: raw archives -> masked, analysis-ready tables.

The raw data (vacancies, prospects, applicants) holds personal data and is not ours to
publish, so this stage is the only code that touches it. It reads the zips straight into
memory, builds each output table from an explicit allowlist of fields (anything not named
here is dropped), masks all free text, and writes parquet files plus an aggregate-only
report under data/masked/ (gitignored). It never logs record content.

    uv run python -m src.ingest              # full run with NER (slow, run once)
    uv run python -m src.ingest --ner none   # fast run, regex + own-name masking only

Fail-closed: any error aborts before a single file is written.
"""

import argparse
import hashlib
import json
import os
import secrets
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.masking import (
    EntityMasker,
    HFEntityMasker,
    NoEntityMasker,
    age_band,
    mask_own_name,
    mask_texts,
    residual_pattern_counts,
    surrogate_id,
    to_month,
)
from src.utils import logger

RAW_DIR = Path(os.getenv("RAW_ARCHIVE_DIR", "datathon data"))
OUT_DIR = Path("data/masked")
SALT_PATH = OUT_DIR / ".salt"
POLICY_VERSION = 1

HIRED_STATUSES = {"Contratado pela Decision", "Contratado como Hunting", "Proposta Aceita"}
REJECTED_STATUSES = {
    "Não Aprovado pelo Cliente",
    "Não Aprovado pelo RH",
    "Não Aprovado pelo Requisitante",
    "Desistiu",
    "Sem interesse nesta vaga",
    "Recusado",
    "Desistiu da Contratação",
}

VACANCY_COLUMNS = [
    "vacancy_id", "client_id", "title", "activities", "competencies", "is_sap", "contract_type",
    "priority", "seniority", "education_level", "english_level", "spanish_level", "areas",
    "state", "requested_month",
]  # fmt: skip
APPLICANT_COLUMNS = [
    "candidate_id", "education_level", "english_level", "spanish_level", "area", "seniority",
    "professional_title", "technical_skills", "certifications", "cv_text", "has_cv",
]  # fmt: skip
SENSITIVE_COLUMNS = ["candidate_id", "sex", "pcd", "age_band"]
CANDIDACY_COLUMNS = [
    "vacancy_id", "candidate_id", "status", "outcome", "candidacy_month", "updated_month",
    "modality", "has_vacancy", "has_applicant",
]  # fmt: skip
QUASI_IDENTIFIERS = ["education_level", "seniority", "area", "age_band", "sex", "pcd"]


def outcome(status: str | None) -> str:
    if status in HIRED_STATUSES:
        return "hired"
    if status in REJECTED_STATUSES:
        return "rejected"
    return "pending"  # still in the funnel: the outcome is not known yet


def dig(record: Any, *path: str) -> str:
    """Safely read a nested string field; anything missing or not a string is ''."""
    for key in path:
        record = record.get(key) if isinstance(record, dict) else None
    return record.strip() if isinstance(record, str) else ""


class TextMasker:
    """Light = patterns + own name (short structured text); deep = light + NER (long text)."""

    def __init__(self, ner: EntityMasker | None = None) -> None:
        self.ner: EntityMasker = ner or NoEntityMasker()
        self.stats: Counter = Counter()

    def light(self, texts: list[str], names: list[str] | None = None) -> list[str]:
        return mask_texts(texts, names, None, self.stats)

    def deep(self, texts: list[str], names: list[str] | None = None) -> list[str]:
        return mask_texts(texts, names, self.ner, self.stats)


# --- table builders (pure: take the parsed JSON dicts) --------------------------------------


def build_vacancies(vagas: dict, masker: TextMasker, salt: bytes) -> pd.DataFrame:
    ids = list(vagas)
    ib = [vagas[i].get("informacoes_basicas") or {} for i in ids]
    pv = [vagas[i].get("perfil_vaga") or {} for i in ids]
    return pd.DataFrame(
        {
            "vacancy_id": [surrogate_id("vac", i, salt) for i in ids],
            "client_id": [surrogate_id("client", dig(b, "cliente"), salt) for b in ib],
            "title": masker.light([dig(b, "titulo_vaga") for b in ib]),
            "activities": masker.deep([dig(p, "principais_atividades") for p in pv]),
            "competencies": masker.deep(
                [dig(p, "competencia_tecnicas_e_comportamentais") for p in pv]
            ),
            "is_sap": [dig(b, "vaga_sap") for b in ib],
            "contract_type": [dig(b, "tipo_contratacao") for b in ib],
            "priority": [dig(b, "prioridade_vaga") for b in ib],
            "seniority": [dig(p, "nivel profissional") for p in pv],
            "education_level": [dig(p, "nivel_academico") for p in pv],
            "english_level": [dig(p, "nivel_ingles") for p in pv],
            "spanish_level": [dig(p, "nivel_espanhol") for p in pv],
            "areas": [dig(p, "areas_atuacao") for p in pv],
            "state": [dig(p, "estado") for p in pv],
            "requested_month": [to_month(dig(b, "data_requicisao")) for b in ib],
        }
    )[VACANCY_COLUMNS]


def build_applicants(
    applicants: dict, masker: TextMasker, salt: bytes
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Returns (applicants, sensitive, residual_own_name_hits)."""
    ids = list(applicants)
    pers = [applicants[i].get("informacoes_pessoais") or {} for i in ids]
    prof = [applicants[i].get("informacoes_profissionais") or {} for i in ids]
    edu = [applicants[i].get("formacao_e_idiomas") or {} for i in ids]
    basic = [applicants[i].get("infos_basicas") or {} for i in ids]
    # Every spelling of the person's own name, from both structured name fields.
    names = [f"{dig(p, 'nome')} {dig(b, 'nome')}".strip() for p, b in zip(pers, basic, strict=True)]
    cvs_raw = [dig(applicants[i], "cv_pt") for i in ids]

    cvs = masker.deep(cvs_raw, names)
    # Residual check: if masking the output again still changes it, a name token survived.
    residual = sum(mask_own_name(t, n) != t for t, n in zip(cvs, names, strict=True))

    candidate_ids = [surrogate_id("cand", i, salt) for i in ids]
    table = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "education_level": [dig(e, "nivel_academico") for e in edu],
            "english_level": [dig(e, "nivel_ingles") for e in edu],
            "spanish_level": [dig(e, "nivel_espanhol") for e in edu],
            "area": [dig(p, "area_atuacao") for p in prof],
            "seniority": [dig(p, "nivel_profissional") for p in prof],
            "professional_title": masker.light(
                [dig(p, "titulo_profissional") for p in prof], names
            ),
            "technical_skills": masker.light(
                [dig(p, "conhecimentos_tecnicos") for p in prof], names
            ),
            "certifications": masker.light([dig(p, "certificacoes") for p in prof], names),
            "cv_text": cvs,
            "has_cv": [bool(t) for t in cvs_raw],
        }
    )[APPLICANT_COLUMNS]
    sensitive = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "sex": [dig(p, "sexo") for p in pers],
            "pcd": [dig(p, "pcd") for p in pers],
            "age_band": [age_band(dig(p, "data_nascimento")) for p in pers],
        }
    )[SENSITIVE_COLUMNS]
    return table, sensitive, residual


def build_candidacies(
    prospects: dict, vacancy_ids: set[str], applicant_ids: set[str], salt: bytes
) -> pd.DataFrame:
    rows = []
    for vid, entry in prospects.items():
        for p in (entry or {}).get("prospects") or []:
            cid = str(p.get("codigo") or "")
            status = dig(p, "situacao_candidado")
            rows.append(
                {
                    "vacancy_id": surrogate_id("vac", vid, salt),
                    "candidate_id": surrogate_id("cand", cid, salt),
                    "status": status,
                    "outcome": outcome(status),
                    "candidacy_month": to_month(dig(p, "data_candidatura")),
                    "updated_month": to_month(dig(p, "ultima_atualizacao")),
                    "modality": dig(entry, "modalidade"),
                    "has_vacancy": vid in vacancy_ids,
                    "has_applicant": cid in applicant_ids,
                }
            )
    return pd.DataFrame(rows, columns=CANDIDACY_COLUMNS)


# --- report (aggregates only) ---------------------------------------------------------------


def quasi_identifier_uniqueness(applicants: pd.DataFrame, sensitive: pd.DataFrame) -> dict:
    merged = applicants.merge(sensitive, on="candidate_id")[QUASI_IDENTIFIERS].fillna("")
    sizes = merged.groupby(QUASI_IDENTIFIERS, dropna=False).size()
    per_row = merged.merge(sizes.rename("k").reset_index(), on=QUASI_IDENTIFIERS)["k"]
    return {
        "quasi_identifiers": QUASI_IDENTIFIERS,
        "share_unique": round(float((per_row == 1).mean()), 4),
        "share_in_groups_smaller_than_5": round(float((per_row < 5).mean()), 4),
    }


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - a checksum for provenance, not security
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    tables: dict[str, pd.DataFrame], masker: TextMasker, residual_names: int, sources: dict
) -> dict:
    texts = {
        "applicants.cv_text": tables["applicants"]["cv_text"],
        "applicants.professional_title": tables["applicants"]["professional_title"],
        "applicants.technical_skills": tables["applicants"]["technical_skills"],
        "applicants.certifications": tables["applicants"]["certifications"],
        "vacancies.title": tables["vacancies"]["title"],
        "vacancies.activities": tables["vacancies"]["activities"],
        "vacancies.competencies": tables["vacancies"]["competencies"],
    }
    structured = ["education_level", "english_level", "spanish_level", "area", "seniority"]
    return {
        "policy_version": POLICY_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources_md5": sources,
        "rows": {name: int(len(df)) for name, df in tables.items()},
        "applicants_with_cv_text": int(tables["applicants"]["has_cv"].sum()),
        "applicants_empty_rate_by_structured_field": {
            c: round(float((tables["applicants"][c] == "").mean()), 4) for c in structured
        },
        "replacements": {
            "patterns_and_own_name": dict(masker.stats),
            "named_entities": dict(masker.ner.counts),
        },
        "residual_pattern_hits": {k: residual_pattern_counts(v.tolist()) for k, v in texts.items()},
        "cv_texts_with_surviving_own_name": residual_names,
        "ner_covers": sorted(getattr(masker.ner, "entity_types", [])),
        "not_masked": "organizations/employers and unlabeled person names missed by the NER",
        "k_anonymity": quasi_identifier_uniqueness(tables["applicants"], tables["sensitive"]),
        "candidacy_links": {
            "with_applicant_record": int(tables["candidacies"]["has_applicant"].sum()),
            "with_vacancy_record": int(tables["candidacies"]["has_vacancy"].sum()),
        },
    }


# --- IO and CLI -----------------------------------------------------------------------------


def load_salt(path: Path = SALT_PATH) -> bytes:
    """MASKING_SALT wins; otherwise a random salt is created once and kept outside git."""
    if os.getenv("MASKING_SALT"):
        return os.environ["MASKING_SALT"].encode()
    if path.exists():
        return bytes.fromhex(path.read_text().strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(32)
    path.write_text(salt.hex())
    path.chmod(0o600)
    logger.info("Created a new masking salt at %s (keep it private)", path)
    return salt


def read_archive(raw_dir: Path, name: str) -> dict:
    with zipfile.ZipFile(raw_dir / f"{name}.zip") as z, z.open(f"{name}.json") as f:
        return json.load(f)


def run(raw_dir: Path, out_dir: Path, ner: EntityMasker, limit: int | None, review: int) -> dict:
    salt = load_salt(out_dir / ".salt")
    vagas, prospects, applicants = (
        read_archive(raw_dir, n) for n in ("vagas", "prospects", "applicants")
    )
    logger.info(
        "Loaded %d vacancies, %d prospect lists, %d applicants",
        len(vagas),
        len(prospects),
        len(applicants),
    )
    if limit:
        vagas = dict(list(vagas.items())[:limit])
        applicants = dict(list(applicants.items())[:limit])

    masker = TextMasker(ner)
    vacancies = build_vacancies(vagas, masker, salt)
    logger.info("Vacancies masked: %d", len(vacancies))
    applicants_t, sensitive, residual = build_applicants(applicants, masker, salt)
    logger.info("Applicants masked: %d", len(applicants_t))
    candidacies = build_candidacies(prospects, set(vagas), set(applicants), salt)
    tables = {
        "vacancies": vacancies,
        "applicants": applicants_t,
        "sensitive": sensitive,
        "candidacies": candidacies,
    }
    sources = {n: md5_of(raw_dir / f"{n}.zip") for n in ("vagas", "prospects", "applicants")}
    report = build_report(tables, masker, residual, sources)

    # Nothing is written until everything above succeeded.
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "masking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if review:
        sample = applicants_t[applicants_t["has_cv"]].sample(
            min(review, len(applicants_t)), random_state=0
        )
        (out_dir / "review_sample.txt").write_text(
            "\n\n=====\n\n".join(sample["cv_text"]), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ner", choices=["hf", "none"], default="hf")
    parser.add_argument("--mask-orgs", action="store_true", help="also mask organizations")
    parser.add_argument("--limit", type=int, help="only the first N vacancies/applicants (testing)")
    parser.add_argument(
        "--review", type=int, default=0, help="write N masked CVs for manual review"
    )
    args = parser.parse_args(argv)

    ner: EntityMasker
    if args.ner == "hf":
        ner = HFEntityMasker(
            entity_types=("PER", "LOC", "ORG") if args.mask_orgs else ("PER", "LOC")
        )
    else:
        ner = NoEntityMasker()
    report = run(RAW_DIR, OUT_DIR, ner, args.limit, args.review)
    logger.info("Done. Aggregate report written to %s/masking_report.json", OUT_DIR)
    logger.info("Rows: %s", report["rows"])


if __name__ == "__main__":
    main()
