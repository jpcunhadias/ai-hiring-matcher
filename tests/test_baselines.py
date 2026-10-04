import numpy as np
import pytest

from src.baselines import metrics_from_scores, tfidf_scores
from src.matcher import JobCatalog


def test_a_clear_winner_scores_perfectly():
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])

    metrics = metrics_from_scores(scores, np.array([0, 1]), k_values=(1,))

    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_all_equal_scores_behave_like_random_guessing():
    scores = np.zeros((3, 4))

    metrics = metrics_from_scores(scores, np.array([0, 1, 2]), k_values=(1, 5))

    assert metrics["recall_at_1"] == pytest.approx(1 / 4)
    assert metrics["recall_at_5"] == 1.0  # k larger than the catalog
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_a_tie_at_the_top_is_split_not_awarded_by_position():
    # The true job (index 1) is tied with index 0 for the best score: half credit.
    scores = np.array([[0.7, 0.7, 0.1]])

    metrics = metrics_from_scores(scores, np.array([1]), k_values=(1,))

    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2) / 2)


def test_tfidf_prefers_the_job_that_shares_words_with_the_resume():
    catalog = JobCatalog(
        roles=["Nurse", "Pilot"],
        descriptions=["patient care and medication", "flight navigation and aircraft"],
        embeddings=np.zeros((2, 2)),  # unused by TF-IDF
    )

    scores = tfidf_scores(["experienced in patient care", "aircraft navigation expert"], catalog)

    assert scores.argmax(axis=1).tolist() == [0, 1]
