import numpy as np

from src.embeddings import embed_texts


def test_embed_texts_returns_normalized_vectors():
    embeddings = embed_texts(["Python developer", "Financial analyst"])

    assert embeddings.shape[0] == 2
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_texts_similar_texts_score_higher_than_unrelated():
    resume = "Proficient in Python, SQL, Machine Learning."
    similar_job = "We are looking for a Python and Machine Learning engineer."
    unrelated_job = "We need a chef skilled in pastry and knife work."

    resume_emb, similar_emb, unrelated_emb = embed_texts([resume, similar_job, unrelated_job])

    assert np.dot(resume_emb, similar_emb) > np.dot(resume_emb, unrelated_emb)
