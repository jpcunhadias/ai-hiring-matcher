from fastapi.testclient import TestClient

from src.api import app
from src.utils import REQUEST_LOG_PATH

client = TestClient(app)

SAMPLE_RESUME = (
    "Proficient in Firewalls, Cyber Threats, Penetration Testing, Vulnerability "
    "Analysis, Ethical Hacking, with senior-level experience in the field. Holds "
    "a Masters degree. Holds certifications such as Certified Ethical Hacker "
    "(CEH). Skilled in delivering results and adapting to dynamic environments."
)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_match_endpoint_returns_ranked_jobs():
    response = client.post("/match", json={"resume": SAMPLE_RESUME, "top_n": 3})

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 3

    top = matches[0]
    assert set(top.keys()) >= {"job_role", "similarity", "skill_overlap", "best_match_proba"}
    assert 0.0 <= top["best_match_proba"] <= 1.0
    # Similarity should be sorted descending.
    assert matches[0]["similarity"] >= matches[1]["similarity"] >= matches[2]["similarity"]


def test_match_endpoint_logs_the_request():
    REQUEST_LOG_PATH.unlink(missing_ok=True)

    client.post("/match", json={"resume": SAMPLE_RESUME, "top_n": 1})

    assert REQUEST_LOG_PATH.exists()
    logged_lines = REQUEST_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(logged_lines) == 1
