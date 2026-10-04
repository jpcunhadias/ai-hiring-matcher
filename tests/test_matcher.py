import numpy as np
import pandas as pd

from src.matcher import (
    JobCatalog,
    build_job_catalog_from_df,
    evaluate_retrieval,
    rank_jobs,
    retrieval_ranks,
)


def _unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=float)
    return arr / np.linalg.norm(arr)


def test_rank_jobs_orders_by_similarity():
    catalog = JobCatalog(
        roles=["Nurse", "Pilot", "Chef"],
        descriptions=["...", "...", "..."],
        embeddings=np.stack([_unit([1, 0]), _unit([0, 1]), _unit([0.6, 0.4])]),
    )
    resume_embedding = _unit([1, 0])

    ranked = rank_jobs(resume_embedding, catalog, top_n=3)

    assert ranked["job_role"].tolist() == ["Nurse", "Chef", "Pilot"]
    assert ranked["similarity"].is_monotonic_decreasing


def test_evaluate_retrieval_recall_and_mrr():
    catalog = JobCatalog(
        roles=["Nurse", "Pilot", "Chef"],
        descriptions=["...", "...", "..."],
        embeddings=np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]),
    )
    # Resume 1 is exactly its true role (Nurse) -> rank 1.
    # Resume 2's true role (Pilot) scores lower than Chef for it -> rank 2.
    resume_embeddings = np.stack([_unit([1, 0, 0]), _unit([0, 0.6, 0.8])])
    true_roles = ["Nurse", "Pilot"]

    metrics = evaluate_retrieval(resume_embeddings, true_roles, catalog, k_values=(1, 2))

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
    assert metrics["mrr"] == (1 / 1 + 1 / 2) / 2


def test_build_job_catalog_from_df_dedupes_job_roles(monkeypatch):
    def fake_embed_texts(texts):
        return np.stack([_unit([len(t), 1]) for t in texts])

    monkeypatch.setattr("src.matcher.embed_texts", fake_embed_texts)

    df = pd.DataFrame(
        {
            "Job Roles": ["Nurse", "Nurse", "Pilot"],
            "Job Description": ["care for patients", "care for patients", "fly planes"],
        }
    )

    catalog = build_job_catalog_from_df(df)

    assert catalog.roles == ["Nurse", "Pilot"]
    assert catalog.embeddings.shape == (2, 2)


def test_retrieval_ranks_are_one_based_positions_of_the_true_role():
    catalog = JobCatalog(
        roles=["Nurse", "Pilot", "Chef"],
        descriptions=["...", "...", "..."],
        embeddings=np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]),
    )
    resume_embeddings = np.stack([_unit([1, 0, 0]), _unit([0, 0.6, 0.8])])

    ranks = retrieval_ranks(resume_embeddings, ["Nurse", "Pilot"], catalog)

    assert ranks.tolist() == [1, 2]
