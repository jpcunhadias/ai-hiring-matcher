from typing import cast

import pandas as pd

from src.data_preparation import extract_job_skills, parse_resume, skill_overlap
from src.embeddings import embed_texts
from src.matcher import JobCatalog, rank_jobs
from src.utils import MODELS_DIR, load_model, logger


def load_artifacts() -> tuple[object, set[str], JobCatalog]:
    logger.info("Carregando modelo, vocabulário de skills e catálogo de vagas...")
    model = load_model(MODELS_DIR / "matcher_classifier.joblib")
    vocabulary = cast(set, load_model(MODELS_DIR / "skill_vocabulary.joblib"))
    catalog = cast(JobCatalog, load_model(MODELS_DIR / "job_catalog.joblib"))
    return model, vocabulary, catalog


def match_resume(
    resume_text: str,
    model: object,
    vocabulary: set[str],
    catalog: JobCatalog,
    top_n: int = 5,
) -> pd.DataFrame:
    """Ranks the catalog's jobs for a resume and scores each with the Best Match classifier."""
    resume_embedding = embed_texts([resume_text])[0]
    ranked = rank_jobs(resume_embedding, catalog, top_n=top_n)

    resume_skills = parse_resume(resume_text)["skills"]
    role_to_description = dict(zip(catalog.roles, catalog.descriptions))
    ranked["skill_overlap"] = [
        skill_overlap(resume_skills, extract_job_skills(role_to_description[role], vocabulary))
        for role in ranked["job_role"]
    ]

    features = ranked[["similarity", "skill_overlap"]].rename(
        columns={"similarity": "cosine_similarity"}
    )
    ranked["best_match_proba"] = model.predict_proba(features)[:, 1]  # type: ignore[attr-defined]

    return ranked
