from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.embeddings import embed_texts


@dataclass
class JobCatalog:
    """The closed set of jobs a resume can be matched against.

    Only 51 unique job descriptions exist in the dataset, so this is a
    closed-set retrieval problem: ranking a resume among *known* jobs, not
    generalizing to unseen postings.
    """

    roles: list[str]
    descriptions: list[str]
    embeddings: np.ndarray  # (n_jobs, dim), L2-normalized


def build_job_catalog(job_roles: list[str], job_descriptions: list[str]) -> JobCatalog:
    embeddings = embed_texts(job_descriptions)
    return JobCatalog(roles=job_roles, descriptions=job_descriptions, embeddings=embeddings)


def build_job_catalog_from_df(df: pd.DataFrame) -> JobCatalog:
    catalog_df = df.drop_duplicates(subset="Job Roles")[["Job Roles", "Job Description"]]
    return build_job_catalog(
        catalog_df["Job Roles"].tolist(), catalog_df["Job Description"].tolist()
    )


def rank_jobs(resume_embedding: np.ndarray, catalog: JobCatalog, top_n: int = 5) -> pd.DataFrame:
    """Ranks every job in the catalog by cosine similarity to one resume embedding."""
    scores = catalog.embeddings @ resume_embedding
    order = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame(
        {
            "job_role": [catalog.roles[i] for i in order],
            "similarity": scores[order],
        }
    )


def retrieval_ranks(
    resume_embeddings: np.ndarray, true_roles: list[str], catalog: JobCatalog
) -> np.ndarray:
    """1-based rank of each resume's true Job Roles among the catalog's jobs."""
    role_to_idx = {role: i for i, role in enumerate(catalog.roles)}
    scores_matrix = resume_embeddings @ catalog.embeddings.T  # (n_resumes, n_jobs)
    orders = np.argsort(-scores_matrix, axis=1)

    ranks = np.empty(len(true_roles), dtype=int)
    for i, role in enumerate(true_roles):
        true_idx = role_to_idx[role]
        ranks[i] = int(np.where(orders[i] == true_idx)[0][0]) + 1
    return ranks


def evaluate_retrieval(
    resume_embeddings: np.ndarray,
    true_roles: list[str],
    catalog: JobCatalog,
    k_values: tuple[int, ...] = (1, 5),
) -> dict:
    """Recall@k and MRR for ranking each resume's true Job Roles among the 51 known jobs."""
    ranks = retrieval_ranks(resume_embeddings, true_roles, catalog)

    # Named recall_at_k (not recall@k): mlflow metric names reject "@".
    metrics = {f"recall_at_{k}": float(np.mean(ranks <= k)) for k in k_values}
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    return metrics
