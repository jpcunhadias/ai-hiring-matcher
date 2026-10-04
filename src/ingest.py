"""Private ingest of the real ATS data: raw archives -> masked, analysis-ready tables.

The raw data (vacancies, prospects, applicants) holds personal data and is not ours to
publish, so this stage is the only code that touches it. It reads the zips straight into
memory, builds each output table from an explicit allowlist of fields (anything not named
here is dropped), masks all free text, and publishes parquet files plus an aggregate-only
report under data/masked/ (gitignored). It never logs record content.

    uv run python -m src.ingest                  # full run
    uv run python -m src.ingest --repair         # re-apply the current layers to the existing
                                                 # masked tables (minutes, no full re-run)
    uv run python -m src.ingest --ner none       # skip the NER layer entirely

All output is written to a private staging directory and swapped into place only when every
file is complete: an error leaves the previous generation untouched and creates no salt file.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.masking import (
    EntityMasker,
    HFEntityMasker,
    NameDictionary,
    NoEntityMasker,
    age_band,
    fold,
    independent_residual_counts,
    load_ibge_first_names,
    mask_own_name,
    mask_texts,
    name_tokens,
    parse_date,
    residual_pattern_counts,
    surrogate_id,
    to_month,
)
from src.utils import logger

RAW_DIR = Path(os.getenv("RAW_ARCHIVE_DIR", "datathon data"))
OUT_DIR = Path("data/masked")
FIRST_NAMES_CSV = Path(os.getenv("FIRST_NAMES_CSV", "data/reference/nomes-censos-ibge.csv"))
POLICY_VERSION = 2
PROTECTED_MIN_DF = 30  # tokens in at least this many short texts are generic, not names
MIN_SALT_BYTES = 16

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
TABLE_COLUMNS = {
    "vacancies": VACANCY_COLUMNS,
    "applicants": APPLICANT_COLUMNS,
    "sensitive": SENSITIVE_COLUMNS,
    "candidacies": CANDIDACY_COLUMNS,
}
QUASI_IDENTIFIERS = ["education_level", "seniority", "area", "age_band", "sex", "pcd"]
STAFF_NAME_FIELDS = (
    "analista_responsavel", "requisitante", "solicitante_cliente", "superior_imediato", "nome",
    "nome_substituto",
)  # fmt: skip


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


def frequent_tokens(texts: list[str], min_df: int) -> set[str]:
    """Folded tokens that appear in at least `min_df` of the texts: generic terms such as
    technologies, which an NER model would otherwise mistake for people."""
    counts: Counter = Counter()
    for text in set(texts):
        counts.update({fold(w) for w in text.split()})
    return {t for t, n in counts.items() if n >= min_df}


class TextMasker:
    """`plain`: patterns + own name + name chains (lowercase-safe, no model).
    `mixed`: plain + NER, for short fields that keep their capitalization."""

    def __init__(self, ner: EntityMasker | None = None, dictionary: NameDictionary | None = None):
        self.ner: EntityMasker = ner or NoEntityMasker()
        self.dictionary = dictionary
        self.stats: Counter = Counter()

    def plain(self, texts: list[str], names: list[str] | None = None) -> list[str]:
        return mask_texts(texts, names, None, self.stats, self.dictionary)

    def mixed(self, texts: list[str], names: list[str] | None = None) -> list[str]:
        if isinstance(self.ner, HFEntityMasker):
            self.ner.protected = {fold(t) for t in frequent_tokens(texts, PROTECTED_MIN_DF)}
        return mask_texts(texts, names, self.ner, self.stats, self.dictionary)


# --- table builders (pure: take the parsed JSON dicts) --------------------------------------


def build_vacancies(vagas: dict, masker: TextMasker, salt: bytes) -> pd.DataFrame:
    ids = list(vagas)
    ib = [vagas[i].get("informacoes_basicas") or {} for i in ids]
    pv = [vagas[i].get("perfil_vaga") or {} for i in ids]
    return pd.DataFrame(
        {
            "vacancy_id": [surrogate_id("vac", i, salt) for i in ids],
            "client_id": [surrogate_id("client", dig(b, "cliente"), salt) for b in ib],
            "title": masker.mixed([dig(b, "titulo_vaga") for b in ib]),
            "activities": masker.mixed([dig(p, "principais_atividades") for p in pv]),
            "competencies": masker.mixed(
                [dig(p, "competencia_tecnicas_e_comportamentais") for p in pv]
            ),
            "is_sap": [dig(b, "vaga_sap") for b in ib],
            "contract_type": [dig(b, "tipo_contratacao") for b in ib],
            "priority": [dig(b, "prioridade_vaga") for b in ib],
            "seniority": [dig(p, "nivel profissional") for p in pv],
            "education_level": [dig(p, "nivel_academico") for p in pv],
            "english_level": [dig(p, "nivel_ingles") for p in pv],
            "spanish_level": [dig(p, "nivel_espanhol") for p in pv],
            "areas": masker.plain([dig(p, "areas_atuacao") for p in pv]),
            "state": [dig(p, "estado") for p in pv],
            "requested_month": [to_month(dig(b, "data_requicisao")) for b in ib],
        }
    )[VACANCY_COLUMNS]


def applicant_names(record: dict) -> str:
    """Every spelling of the person's own name, from both structured name fields."""
    first = dig(record, "informacoes_pessoais", "nome")
    second = dig(record, "infos_basicas", "nome")
    return f"{first} {second}".strip()


def build_sensitive(applicants: dict, salt: bytes, snapshot: date) -> pd.DataFrame:
    ids = list(applicants)
    pers = [applicants[i].get("informacoes_pessoais") or {} for i in ids]
    return pd.DataFrame(
        {
            "candidate_id": [surrogate_id("cand", i, salt) for i in ids],
            "sex": [dig(p, "sexo") for p in pers],
            "pcd": [dig(p, "pcd") for p in pers],
            "age_band": [age_band(dig(p, "data_nascimento"), snapshot) for p in pers],
        }
    )[SENSITIVE_COLUMNS]


def build_applicants(
    applicants: dict, masker: TextMasker, salt: bytes, snapshot: date
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Returns (applicants, sensitive, residual_own_name_hits)."""
    ids = list(applicants)
    prof = [applicants[i].get("informacoes_profissionais") or {} for i in ids]
    edu = [applicants[i].get("formacao_e_idiomas") or {} for i in ids]
    names = [applicant_names(applicants[i]) for i in ids]
    cvs_raw = [dig(applicants[i], "cv_pt") for i in ids]

    cvs = masker.plain(cvs_raw, names)  # lowercase text: no NER (see docs/masking.md)
    # Residual check: if masking the output again still changes it, a name token survived.
    residual = sum(mask_own_name(t, n) != t for t, n in zip(cvs, names, strict=True))

    table = pd.DataFrame(
        {
            "candidate_id": [surrogate_id("cand", i, salt) for i in ids],
            "education_level": [dig(e, "nivel_academico") for e in edu],
            "english_level": [dig(e, "nivel_ingles") for e in edu],
            "spanish_level": [dig(e, "nivel_espanhol") for e in edu],
            "area": masker.plain([dig(p, "area_atuacao") for p in prof], names),
            "seniority": [dig(p, "nivel_profissional") for p in prof],
            "professional_title": masker.mixed(
                [dig(p, "titulo_profissional") for p in prof], names
            ),
            "technical_skills": masker.plain(
                [dig(p, "conhecimentos_tecnicos") for p in prof], names
            ),
            "certifications": masker.plain([dig(p, "certificacoes") for p in prof], names),
            "cv_text": cvs,
            "has_cv": [bool(t) for t in cvs_raw],
        }
    )[APPLICANT_COLUMNS]
    return table, build_sensitive(applicants, salt, snapshot), residual


def build_candidacies(
    prospects: dict, vacancy_ids: set[str], applicant_ids: set[str], salt: bytes
) -> pd.DataFrame:
    rows = []
    for vid, entry in prospects.items():
        for p in (entry or {}).get("prospects") or []:
            code = p.get("codigo")
            cid = "" if code is None else str(code)  # 0 is a real code, not "missing"
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


def snapshot_date(prospects: dict) -> date:
    """The data snapshot: the latest date in the prospect records (ages are as of then)."""
    latest = date.min
    for entry in prospects.values():
        for p in (entry or {}).get("prospects") or []:
            for field in ("data_candidatura", "ultima_atualizacao"):
                parsed = parse_date(dig(p, field))
                if parsed and parsed.date() > latest:
                    latest = parsed.date()
    return latest if latest != date.min else datetime.now(UTC).date()


# --- name dictionary ------------------------------------------------------------------------


def build_name_dictionary(
    applicants: dict, vagas: dict, first_names_csv: Path, top_k: int
) -> NameDictionary | None:
    """Common first names (IBGE census) + every name token seen in the system of record.
    Without the IBGE file the layer is disabled (and the report says so)."""
    if not first_names_csv.exists():
        logger.warning(
            "First-name list not found at %s: name-chain layer disabled", first_names_csv
        )
        return None
    pool: set[str] = set()
    for a in applicants.values():
        pool.update(name_tokens(applicant_names(a)))
    for v in vagas.values():
        for field in STAFF_NAME_FIELDS:
            pool.update(name_tokens(dig(v, "informacoes_basicas", field)))
    return NameDictionary.build(load_ibge_first_names(first_names_csv, top_k), pool)


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
    tables: dict[str, pd.DataFrame],
    masker: TextMasker,
    residual_names: int,
    sources: dict,
    snapshot: date,
    mode: str,
) -> dict:
    texts = {
        "applicants.cv_text": tables["applicants"]["cv_text"],
        "applicants.professional_title": tables["applicants"]["professional_title"],
        "applicants.technical_skills": tables["applicants"]["technical_skills"],
        "applicants.certifications": tables["applicants"]["certifications"],
        "applicants.area": tables["applicants"]["area"],
        "vacancies.title": tables["vacancies"]["title"],
        "vacancies.activities": tables["vacancies"]["activities"],
        "vacancies.competencies": tables["vacancies"]["competencies"],
        "vacancies.areas": tables["vacancies"]["areas"],
    }
    structured = ["education_level", "english_level", "spanish_level", "area", "seniority"]
    dictionary = masker.dictionary
    return {
        "policy_version": POLICY_VERSION,
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "age_reference_date": snapshot.isoformat(),
        "sources_md5": sources,
        "rows": {name: int(len(df)) for name, df in tables.items()},
        "applicants_with_cv_text": int(tables["applicants"]["has_cv"].sum()),
        "applicants_empty_rate_by_structured_field": {
            c: round(float((tables["applicants"][c] == "").mean()), 4) for c in structured
        },
        "replacements_this_pass": {
            "patterns_names_chains": dict(masker.stats),
            "named_entities": dict(masker.ner.counts),
        },
        "name_dictionary": (
            {
                "first_names": len(dictionary.first_names),
                "name_tokens": len(dictionary.name_tokens),
                "benchmark_note": (
                    "~90% token recall on injected lowercase names; see docs/masking.md"
                ),
            }
            if dictionary
            else None
        ),
        "ner_covers_mixed_case_fields_only": sorted(getattr(masker.ner, "entity_types", [])),
        "self_consistency_residual_hits": {
            k: residual_pattern_counts(v.tolist()) for k, v in texts.items()
        },
        "independent_residual_estimate": {
            k: independent_residual_counts(v.tolist()) for k, v in texts.items()
        },
        "residual_notes": (
            "self_consistency_* re-runs the masker's own patterns and only proves it agrees with "
            "itself. independent_* uses separately written, looser checks and estimates what may "
            "have slipped through; neither is proof of anonymity."
        ),
        "cv_texts_with_surviving_own_name": residual_names,
        "not_masked": "employers/organizations, unlabeled person names the dictionary and NER miss",
        "k_anonymity": quasi_identifier_uniqueness(tables["applicants"], tables["sensitive"]),
        "candidacy_links": {
            "with_applicant_record": int(tables["candidacies"]["has_applicant"].sum()),
            "with_vacancy_record": int(tables["candidacies"]["has_vacancy"].sum()),
        },
    }


# --- salt, staging and atomic publish -------------------------------------------------------


def _write_new_private_file(path: Path, content: str) -> None:
    """Create with mode 0600 from the start (no window where it is world-readable)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)


def resolve_salt(out_dir: Path) -> tuple[bytes, bool]:
    """(salt, is_new). A new salt only exists in memory until the run has succeeded."""
    if os.getenv("MASKING_SALT"):
        salt = os.environ["MASKING_SALT"].encode()
        if len(salt) < MIN_SALT_BYTES:
            raise ValueError(f"MASKING_SALT must be at least {MIN_SALT_BYTES} bytes")
        return salt, False
    path = out_dir / ".salt"
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{path} is not a regular file")
        salt = bytes.fromhex(path.read_text().strip())
        if len(salt) < MIN_SALT_BYTES:
            raise ValueError(f"{path} holds a weak salt (under {MIN_SALT_BYTES} bytes)")
        if path.stat().st_mode & 0o077:
            path.chmod(0o600)
            logger.warning("Tightened permissions on %s to 0600", path)
        return salt, False
    return secrets.token_bytes(32), True


def publish(out_dir: Path, write: Callable[[Path], None], new_salt: bytes | None) -> None:
    """Write a complete generation into a private staging dir, then swap it into place.

    On any error the staging dir is removed, the previous generation is untouched, and no salt
    file is created. Directories are 0700 and files 0600.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    staging.chmod(0o700)
    backup = out_dir.with_name(f"{out_dir.name}.previous")
    try:
        write(staging)
        if new_salt is not None:
            _write_new_private_file(staging / ".salt", new_salt.hex())
        elif (out_dir / ".salt").exists():
            shutil.copy2(out_dir / ".salt", staging / ".salt")
        for path in staging.iterdir():
            path.chmod(0o600)
        if backup.exists():
            shutil.rmtree(backup)
        if out_dir.exists():
            out_dir.rename(backup)
        try:
            staging.rename(out_dir)
        except BaseException:
            if backup.exists() and not out_dir.exists():
                backup.rename(out_dir)  # put the previous generation back
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# --- run and repair ------------------------------------------------------------------------


def read_archive(raw_dir: Path, name: str) -> dict:
    with zipfile.ZipFile(raw_dir / f"{name}.zip") as z, z.open(f"{name}.json") as f:
        return json.load(f)


def _sample_cvs(applicants: pd.DataFrame, review: int) -> str:
    eligible = applicants[applicants["has_cv"] & (applicants["cv_text"] != "")]
    n = min(review, len(eligible))
    if n == 0:
        return ""
    return "\n\n=====\n\n".join(eligible.sample(n, random_state=0)["cv_text"])


def _writer(
    tables: dict[str, pd.DataFrame], report: dict, review_text: str
) -> Callable[[Path], None]:
    def write(staging: Path) -> None:
        for name, df in tables.items():
            df.to_parquet(staging / f"{name}.parquet", index=False)
        (staging / "masking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if review_text:
            (staging / "review_sample.txt").write_text(review_text, encoding="utf-8")

    return write


def run(
    raw_dir: Path,
    out_dir: Path,
    ner: EntityMasker,
    limit: int | None = None,
    review: int = 0,
    first_names_csv: Path = FIRST_NAMES_CSV,
    top_k: int = 3000,
) -> dict:
    if review < 0:
        raise ValueError("--review must be zero or positive")
    salt, is_new = resolve_salt(out_dir)
    vagas, prospects, applicants = (
        read_archive(raw_dir, n) for n in ("vagas", "prospects", "applicants")
    )
    logger.info(
        "Loaded %d vacancies, %d prospect lists, %d applicants",
        len(vagas),
        len(prospects),
        len(applicants),
    )
    snapshot = snapshot_date(prospects)
    dictionary = build_name_dictionary(applicants, vagas, first_names_csv, top_k)
    if limit:
        vagas = dict(list(vagas.items())[:limit])
        applicants = dict(list(applicants.items())[:limit])

    masker = TextMasker(ner, dictionary)
    vacancies = build_vacancies(vagas, masker, salt)
    logger.info("Vacancies masked: %d", len(vacancies))
    applicants_t, sensitive, residual = build_applicants(applicants, masker, salt, snapshot)
    logger.info("Applicants masked: %d", len(applicants_t))
    candidacies = build_candidacies(prospects, set(vagas), set(applicants), salt)
    tables = {
        "vacancies": vacancies,
        "applicants": applicants_t,
        "sensitive": sensitive,
        "candidacies": candidacies,
    }
    sources = {n: md5_of(raw_dir / f"{n}.zip") for n in ("vagas", "prospects", "applicants")}
    report = build_report(tables, masker, residual, sources, snapshot, mode="full")
    publish(
        out_dir,
        _writer(tables, report, _sample_cvs(applicants_t, review)),
        salt if is_new else None,
    )
    return report


def repair(
    raw_dir: Path,
    out_dir: Path,
    ner: EntityMasker,
    review: int = 0,
    first_names_csv: Path = FIRST_NAMES_CSV,
    top_k: int = 3000,
) -> dict:
    """Re-apply the current layers to the EXISTING masked tables, without a full re-run.

    Every layer is idempotent (placeholders are left alone), so improvements to the patterns,
    the name dictionary or the own-name rule can be applied to already-masked text in minutes.
    The raw archives are read only for structured names, the dictionary and birth dates; the
    expensive NER pass over the CV text is not repeated (it does not apply to lowercase text).
    """
    if review < 0:
        raise ValueError("--review must be zero or positive")
    salt, _ = resolve_salt(out_dir)
    existing = {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in TABLE_COLUMNS}
    vagas, prospects, applicants = (
        read_archive(raw_dir, n) for n in ("vagas", "prospects", "applicants")
    )
    snapshot = snapshot_date(prospects)
    masker = TextMasker(ner, build_name_dictionary(applicants, vagas, first_names_csv, top_k))

    names_by_id = {surrogate_id("cand", i, salt): applicant_names(a) for i, a in applicants.items()}
    app = existing["applicants"].copy()
    names = [names_by_id.get(c, "") for c in app["candidate_id"]]
    app["cv_text"] = masker.plain(app["cv_text"].tolist(), names)
    app["area"] = masker.plain(app["area"].tolist(), names)
    app["technical_skills"] = masker.plain(app["technical_skills"].tolist(), names)
    app["certifications"] = masker.plain(app["certifications"].tolist(), names)
    app["professional_title"] = masker.mixed(app["professional_title"].tolist(), names)
    residual = sum(mask_own_name(t, n) != t for t, n in zip(app["cv_text"], names, strict=True))
    logger.info("Applicants repaired: %d", len(app))

    vac = existing["vacancies"].copy()
    vac["title"] = masker.mixed(vac["title"].tolist())
    vac["activities"] = masker.plain(vac["activities"].tolist())  # already NER-masked (mixed case)
    vac["competencies"] = masker.plain(vac["competencies"].tolist())
    vac["areas"] = masker.plain(vac["areas"].tolist())
    logger.info("Vacancies repaired: %d", len(vac))

    sensitive = build_sensitive(
        applicants, salt, snapshot
    )  # age bands re-derived from a real snapshot
    tables = {
        "vacancies": vac[VACANCY_COLUMNS],
        "applicants": app[APPLICANT_COLUMNS],
        "sensitive": sensitive,
        "candidacies": existing["candidacies"][CANDIDACY_COLUMNS],
    }
    sources = {n: md5_of(raw_dir / f"{n}.zip") for n in ("vagas", "prospects", "applicants")}
    report = build_report(tables, masker, residual, sources, snapshot, mode="repair")
    publish(out_dir, _writer(tables, report, _sample_cvs(app, review)), None)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ner", choices=["hf", "none"], default="hf", help="mixed-case fields only"
    )
    parser.add_argument("--mask-orgs", action="store_true", help="also mask organizations")
    parser.add_argument("--repair", action="store_true", help="re-apply layers to existing tables")
    parser.add_argument("--limit", type=int, help="only the first N vacancies/applicants (testing)")
    parser.add_argument(
        "--review", type=int, default=0, help="write N masked CVs for manual review"
    )
    parser.add_argument(
        "--first-names", type=Path, default=FIRST_NAMES_CSV, help="IBGE first-name CSV"
    )
    parser.add_argument(
        "--top-k", type=int, default=3000, help="how many common first names to use"
    )
    args = parser.parse_args(argv)

    ner: EntityMasker
    if args.ner == "hf":
        ner = HFEntityMasker(entity_types=("PER", "ORG") if args.mask_orgs else ("PER",))
    else:
        ner = NoEntityMasker()
    if args.repair:
        report = repair(RAW_DIR, OUT_DIR, ner, args.review, args.first_names, args.top_k)
    else:
        report = run(RAW_DIR, OUT_DIR, ner, args.limit, args.review, args.first_names, args.top_k)
    logger.info(
        "Done (%s). Aggregate report written to %s/masking_report.json", report["mode"], OUT_DIR
    )
    logger.info("Rows: %s", report["rows"])


if __name__ == "__main__":
    main()
