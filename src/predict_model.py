from typing import cast

import pandas as pd

from src.data_preparation import extract_job_skills, parse_resume, skill_overlap
from src.embeddings import embed_texts
from src.matcher import JobCatalog, rank_jobs
from src.utils import MODELS_DIR, load_model, logger


def load_artifacts() -> tuple[set[str], JobCatalog]:
    logger.info("Loading skill vocabulary and job catalog...")
    vocabulary = cast(set, load_model(MODELS_DIR / "skill_vocabulary.joblib"))
    catalog = cast(JobCatalog, load_model(MODELS_DIR / "job_catalog.joblib"))
    return vocabulary, catalog


def match_resume(
    resume_text: str,
    vocabulary: set[str],
    catalog: JobCatalog,
    top_n: int = 5,
) -> pd.DataFrame:
    """Ranks the catalog's jobs for a resume by embedding similarity, with the skill overlap
    for each. Deliberately returns no "match probability": the Best Match classifier is an
    audit probe (see src/train_model.py), not something to score people with."""
    resume_embedding = embed_texts([resume_text])[0]
    ranked = rank_jobs(resume_embedding, catalog, top_n=top_n)

    resume_skills = parse_resume(resume_text)["skills"]
    role_to_description = dict(zip(catalog.roles, catalog.descriptions))
    ranked["skill_overlap"] = [
        skill_overlap(resume_skills, extract_job_skills(role_to_description[role], vocabulary))
        for role in ranked["job_role"]
    ]

    return ranked
