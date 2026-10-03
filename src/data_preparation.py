import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import logger

RAW_DATA_PATH = Path("data/external/job_applicant_dataset.csv")

# Resumes in this dataset are generated from a fixed template:
# "Proficient in X, Y, Z, with mid-level experience in the field. Holds a
# Bachelors degree. Holds certifications such as W. Skilled in delivering
# results..." — verified against all 10,000 rows before relying on it here.
_RESUME_PATTERN = re.compile(
    r"^Proficient in (?P<skills>.+?), with (?P<level>[\w\- ]+?) experience in the field\. "
    r"Holds an? (?P<degree>.+?) degree\."
    r"(?: Holds certifications such as (?P<certs>.+?)\.)? Skilled in"
)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Baixe-o do Kaggle e salve nesse caminho "
            "(ver a seção 'Get the data' do README)."
        )
    logger.info("Carregando dataset bruto de: %s", path)
    df = pd.read_csv(path)
    df["Best Match"] = df["Best Match"].astype(int)
    return df


def parse_resume(resume: str) -> dict:
    """Extracts skills/degree/certifications/experience level from a templated resume."""
    match = _RESUME_PATTERN.match(resume)
    if not match:
        return {"skills": set(), "level": None, "degree": None, "certifications": None}

    skills = {s.strip() for s in match.group("skills").split(",")}
    return {
        "skills": skills,
        "level": match.group("level"),
        "degree": match.group("degree"),
        "certifications": match.group("certs"),
    }


def build_skill_vocabulary(resumes: pd.Series) -> set[str]:
    """Controlled vocabulary of skill phrases, extracted from every resume in the dataset."""
    vocabulary: set[str] = set()
    for resume in resumes:
        vocabulary.update(parse_resume(resume)["skills"])
    return vocabulary


def extract_job_skills(job_description: str, vocabulary: set[str]) -> set[str]:
    """Which known skill phrases appear (whole-word, case-insensitive) in a job description.

    Whole-word matching matters here: a naive substring check matches "R" or "GIS"
    inside unrelated words (e.g. "GIS" inside "logistics").
    """
    text = job_description.lower()
    return {
        skill
        for skill in vocabulary
        if re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", text)
    }


def skill_overlap(resume_skills: set[str], job_skills: set[str]) -> float:
    """Jaccard overlap between a resume's skills and a job's required skills."""
    if not resume_skills or not job_skills:
        return 0.0
    union = resume_skills | job_skills
    return len(resume_skills & job_skills) / len(union)


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on Best Match.

    A plain random row-level split is safe here: although 547 applicant names repeat
    ~18x each, 9,996 of the 10,000 Resume texts are distinct, so this doesn't leak
    duplicate resumes across the split (verified against the raw CSV).
    """
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["Best Match"]
    )
