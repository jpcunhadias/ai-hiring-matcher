# AI Hiring Matcher

[![CI](https://github.com/jpcunhadias/ai-hiring-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/jpcunhadias/ai-hiring-matcher/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![uv](https://img.shields.io/badge/managed%20with-uv-de5fe9)

A semantic resume-to-job matcher with an automated fairness audit and
drift monitoring, built on a fully local MLOps stack (`uv`, DVC, MLflow,
Evidently) — no cloud account required to run it.

Given a resume, it ranks a catalog of job descriptions by embedding
similarity, scores each match with a classifier, and — while validating
that classifier's training label — surfaces a severe, quantifiable gender
bias baked into the dataset itself. That finding, and how the model is
deliberately designed not to reproduce it, is the core of this project.

This started as a Datathon submission: a plain classifier over 7
categorical fields, with no text signal, hosted on AWS S3 because the
course was an AWS partner. This is the rebuild — a real retrieval-based
matcher instead of a blind categorical classifier, and an honest audit of
the data it's trained on.

## Contents

- [Features](#features)
- [The dataset](#the-dataset)
- [Key finding: the label is gender-biased](#key-finding-the-label-is-gender-biased)
- [How the matcher works](#how-the-matcher-works)
- [Drift monitoring](#drift-monitoring)
- [Stack](#stack)
- [Quick start](#quick-start)
- [API](#api)
- [Project structure](#project-structure)
- [Testing & quality](#testing--quality)
- [Limitations](#limitations)

## Features

- **Semantic retrieval matcher** — resumes ranked against a 51-job catalog
  by `sentence-transformers` embedding similarity, not keyword matching.
- **Fairness audit** — checks the training label itself for demographic
  skew before trusting it, independent of any model.
- **Drift monitoring** — compares real logged requests against a training
  reference built the same way, with a threshold alert and a minimum
  sample-size guard.
- **Fully local MLOps stack** — `uv` for dependencies, DVC for data
  versioning, MLflow for experiment tracking, all with local defaults and
  zero required cloud credentials.
- **FastAPI service + Docker**, with a Streamlit drift dashboard.

## The dataset

[`job_applicant_dataset.csv`](data/external/job_applicant_dataset.csv)
(10,000 rows, [via Kaggle](https://www.kaggle.com/datasets/surendra365/recruitement-dataset))
provides free-text `Resume` and `Job Description` pairs, a binary
`Best Match` label, and demographic columns (`Age`, `Gender`, `Race`,
`Ethnicity`). Only **51 unique job descriptions** exist — so this is a
*closed-set retrieval* problem (ranking among 51 known jobs), not
generalization to unseen postings.

The dataset's own Kaggle card describes it as material for "HR Analytics &
Hiring Bias Studies" and for analyzing hiring bias by age, gender, or
ethnicity — a bias finding here is the dataset's stated purpose, not a
surprise. What *is* notable: the same card defines `Best Match` as
reflecting "qualifications and experience." The next section shows that
definition doesn't hold up.

## Key finding: the label is gender-biased

Before trusting `Best Match` as a training target, the fairness audit
([src/fairness_audit.py](src/fairness_audit.py)) checks its selection rate
by demographic group, directly on the raw data, with no model involved:

- **Aggregate:** `Best Match = 1` for **61.3%** of male rows vs. **35.4%**
  of female rows.
- **Per job role**, the effect is far more extreme and **not uniform** —
  some roles favor men, others favor women, with gaps up to **89
  percentage points**:

| Job role | Female rate | Male rate | Gap |
|---|---:|---:|---:|
| Journalist | 6.2% | 95.6% | +89.4 pp |
| Financial Analyst | 10.0% | 95.7% | +85.7 pp |
| Content Writer | 91.7% | 6.6% | −85.1 pp |
| Psychologist | 92.5% | 8.3% | −84.2 pp |

(full table generated in `reports/fairness_report.md` on every training run)

Across all 102 `(Job Role, Gender)` groups, none is perfectly deterministic
(rate = 0% or 100%), but the rates are sharply **bimodal**: 53 groups sit at
≤20%, 49 at ≥80%, and **none** fall in between — the signature of
`Best Match` having been sampled as `Bernoulli(p)` per group, with `p`
fixed near 0.1 or 0.9. Age, race, ethnicity, experience level, and
certifications show no comparable signal; this is specific to gender, and
specific to role. Correlation between `Best Match` and the actual
similarity features (`cosine_similarity`, `skill_overlap`) is close to
zero — the bimodal group pattern dominates the label, not qualification.

**Direct design consequence:** the `Best Match` classifier below **never
receives Gender/Race/Ethnicity as a feature** — even though it's by far the
strongest signal for the label. Training on a protected attribute to
predict "match" would reproduce exactly the bias this audit exists to
catch. The cost of that choice shows up in the next section.

## How the matcher works

[src/embeddings.py](src/embeddings.py) embeds resumes and job descriptions
with `sentence-transformers` (`all-MiniLM-L6-v2`, local, no API key).
[src/matcher.py](src/matcher.py) ranks the 51 catalog jobs by cosine
similarity — this is the actual matching logic.

Evaluated as retrieval (does a resume recover its own `Job Roles` among the
51 known jobs?):

| Metric | Value |
|---|---:|
| Recall@1 | 63.1% |
| Recall@5 | 88.2% |
| MRR | 0.745 |

This works well because each resume was generated with a skill vocabulary
that reflects its target role — text similarity recovers that role most of
the time.

### The Best Match classifier: honest about its limits

[src/train_model.py](src/train_model.py) trains a logistic regression on
`[cosine_similarity, skill_overlap]` → `Best Match`. Correlation between
those features and the label is close to zero (~−0.02 and ~0.00) — because,
per the finding above, `Best Match` is dominated by gender, not genuine
semantic fit. Result: F1 ≈ 0.08 on the positive class, barely above chance.

This isn't a bug to fix by tuning hyperparameters — it's the expected
result of deliberately withholding the attribute that most predicts the
label. `skill_overlap` itself is computed via regex extraction over the
resumes' template format ([src/data_preparation.py](src/data_preparation.py)),
with whole-word matching, verified against all 10,000 rows.

## Drift monitoring

[src/drift_monitor.py](src/drift_monitor.py) compares real requests logged
to `data/logs/requests.jsonl` (every `/match` call records its own
features) against a training-time reference built the same way a live
request is scored — top-1 catalog match, not the dataset's arbitrary
resume/job pairing. Evidently runs a per-column statistical test; below
`DRIFT_MIN_WINDOW_SIZE` (default 100) logged requests, the check refuses to
run, since drift tests are unreliable on small samples.

```bash
make drift-check   # run once
make monitor       # Streamlit dashboard
```

Meant to run on a schedule (cron/systemd timer), not just manually.

## Stack

Everything runs on `uv` — no manual `pip`/`venv`, no `requirements.txt`.
The dataset ([`data/external/`](data/external)) is tracked with
[DVC](https://dvc.org): git stores only a small pointer file with its
checksum, which is how you verify you have the right CSV. No shared remote
is configured, so the project works fully offline with zero credentials.

- **MLflow** tracks experiments to a local sqlite database
  (`sqlite:///mlflow.db`) by default. Point it at any remote tracking
  server via `MLFLOW_TRACKING_URI` in `.env`.
- **DVC** has no default remote. To push/pull the data yourself, add any
  DVC-supported one (local path, S3, MinIO, GCS, Azure...) with
  `dvc remote add --local -d <name> <url>` — `--local` keeps it out of the
  committed config.

See [`.env.example`](.env.example) for all configurable environment
variables.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (it installs the pinned Python 3.12
on its own). The first run downloads the `all-MiniLM-L6-v2` embedding model
(~80 MB) from the Hugging Face Hub.

### Get the data

The dataset isn't stored in the repo. Download it from
[Kaggle](https://www.kaggle.com/datasets/surendra365/recruitement-dataset)
(free account required) and save the CSV as
`data/external/job_applicant_dataset.csv`. Then confirm it's the exact file
the project was built on by comparing its MD5 with the `md5:` line in
`data/external/job_applicant_dataset.csv.dvc` (the two must match):

```bash
uv run python -c "import hashlib, pathlib; print(hashlib.md5(pathlib.Path('data/external/job_applicant_dataset.csv').read_bytes()).hexdigest())"
grep md5 data/external/job_applicant_dataset.csv.dvc
```

(`dvc status` isn't a reliable check on a fresh clone — it reports "not in
cache" whether or not the file is correct.)

### Run it

```bash
uv sync                        # install everything (runtime + dev)
make train                     # embeddings, classifier, catalog,
                                # fairness audit, drift reference
make serve                     # API at http://localhost:8000
make test                      # pytest
make lint                      # ruff + mypy
```

`make train` writes the artifacts the API loads (`models/`). Until it has
run, the API tests are skipped rather than failed.

### Docker (no credentials needed)

```bash
docker compose up
```

`models/` and `data/` are mounted as volumes — train locally (`make train`)
before bringing the container up for the first time.

## API

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"resume": "Proficient in Python, SQL, Machine Learning, with senior-level experience in the field. Holds a Masters degree. Skilled in delivering results and adapting to dynamic environments.", "top_n": 3}'
```

```json
{
  "matches": [
    {"job_role": "Software Engineer", "similarity": 0.564, "skill_overlap": 0.0, "best_match_proba": 0.480},
    {"job_role": "AI Specialist", "similarity": 0.530, "skill_overlap": 0.25, "best_match_proba": 0.484},
    {"job_role": "Machine Learning Engineer", "similarity": 0.484, "skill_overlap": 0.125, "best_match_proba": 0.487}
  ]
}
```

(real output, generated from the model trained in this repository)

## Project structure

```
.
├── data/
│   ├── external/            # Kaggle dataset, versioned with DVC
│   ├── processed/           # drift reference (generated by `make train`)
│   └── logs/                 # real requests logged at runtime
├── models/                   # classifier, skill vocabulary, catalog (generated)
├── reports/                  # fairness audit (generated by `make train`)
├── src/
│   ├── data_preparation.py   # resume parsing, skill extraction, splitting
│   ├── embeddings.py         # sentence-transformers wrapper
│   ├── matcher.py            # job catalog, ranking, retrieval metrics
│   ├── train_model.py        # full training pipeline (MLflow + fairness + drift ref)
│   ├── predict_model.py      # inference: match_resume()
│   ├── fairness_audit.py     # demographic bias audit
│   ├── drift_monitor.py      # batch-live drift comparison with alerting
│   ├── api.py                 # FastAPI (/match)
│   ├── monitor_app.py         # Streamlit drift dashboard
│   └── utils.py                # logging, local I/O, request logging
└── tests/
```

## Testing & quality

```bash
make test     # pytest
make lint     # ruff check + mypy
uv run pre-commit run --all-files
```

## Limitations

- **Closed-set retrieval only** — the matcher ranks among the 51 jobs seen
  during training; it doesn't generalize to unseen job postings.
- **The `Best Match` classifier is weak by design** — see
  [above](#the-best-match-classifier-honest-about-its-limits). Improving
  its F1 would require feeding it the protected attributes that actually
  drive the label, which defeats the point of the fairness audit.
- **Drift alerts need real traffic volume** — fewer than
  `DRIFT_MIN_WINDOW_SIZE` logged requests, and the check won't run at all.

## License

[MIT](LICENSE)
