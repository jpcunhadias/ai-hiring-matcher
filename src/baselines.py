"""Retrieval baselines, to show whether the embedding matcher earns its recall@1.

Every method produces a (resumes x jobs) score matrix and is judged by the same
function, so the comparison is apples-to-apples. Ties are scored by expectation (a
tied block is treated as randomly ordered) instead of by array position, which would
quietly favor or punish methods that produce many equal scores.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_preparation import (
    build_skill_vocabulary,
    extract_job_skills,
    load_raw_data,
    parse_resume,
    skill_overlap,
)
from src.embeddings import embed_texts
from src.matcher import JobCatalog, build_job_catalog_from_df
from src.utils import REPORTS_DIR, logger


def metrics_from_scores(
    scores: np.ndarray, true_idx: np.ndarray, k_values: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    """Expected recall@k and MRR of ranking each row's true job by `scores`.

    The true job sits in a block of `ties` equal scores that starts after `greater`
    strictly better ones; with random order inside the block its rank is uniform over
    greater+1 .. greater+ties.
    """
    n = len(true_idx)
    true_score = scores[np.arange(n), true_idx][:, None]
    greater = (scores > true_score).sum(axis=1)
    ties = (scores == true_score).sum(axis=1)  # includes the true job itself

    out = {f"recall_at_{k}": float(np.mean(np.clip((k - greater) / ties, 0, 1))) for k in k_values}
    out["mrr"] = float(
        np.mean([np.mean(1.0 / (g + np.arange(1, t + 1))) for g, t in zip(greater, ties)])
    )
    return out


def skill_overlap_scores(
    resumes: list[str], catalog: JobCatalog, vocabulary: set[str]
) -> np.ndarray:
    job_skills = [extract_job_skills(desc, vocabulary) for desc in catalog.descriptions]
    scores = np.zeros((len(resumes), len(job_skills)))
    for i, resume in enumerate(resumes):
        resume_skills = parse_resume(resume)["skills"]
        for j, skills in enumerate(job_skills):
            scores[i, j] = skill_overlap(resume_skills, skills)
    return scores


def tfidf_scores(resumes: list[str], catalog: JobCatalog) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectorizer.fit(catalog.descriptions + resumes)  # unsupervised: no labels involved
    resume_matrix = vectorizer.transform(resumes)
    job_matrix = vectorizer.transform(catalog.descriptions)
    return (resume_matrix @ job_matrix.T).toarray()


def compare_methods(df: pd.DataFrame) -> pd.DataFrame:
    catalog = build_job_catalog_from_df(df)
    resumes = df["Resume"].tolist()
    role_to_idx = {role: i for i, role in enumerate(catalog.roles)}
    true_idx = np.array([role_to_idx[role] for role in df["Job Roles"]])
    vocabulary = build_skill_vocabulary(df["Resume"])

    logger.info("Scoring %d resumes with each method...", len(resumes))
    methods = {
        "Random (no information)": np.zeros((len(resumes), len(catalog.roles))),
        "Skill overlap only": skill_overlap_scores(resumes, catalog, vocabulary),
        "TF-IDF cosine": tfidf_scores(resumes, catalog),
        "Embeddings (this project)": embed_texts(resumes) @ catalog.embeddings.T,
    }
    rows = [{"method": name, **metrics_from_scores(s, true_idx)} for name, s in methods.items()]
    return pd.DataFrame(rows)


def main() -> None:
    table = compare_methods(load_raw_data())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    text = (
        "# Retrieval baselines\n\n"
        "Recall and MRR of ranking each resume's own job role among the 51 known jobs. "
        "Ties are scored by expectation.\n\n"
        + table.to_markdown(index=False, floatfmt=".3f")
        + "\n"
    )
    (REPORTS_DIR / "baselines.md").write_text(text, encoding="utf-8")
    logger.info("Baselines:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
