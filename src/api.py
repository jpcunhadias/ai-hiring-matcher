from fastapi import FastAPI
from pydantic import BaseModel

from src.predict_model import load_artifacts, match_resume
from src.utils import log_request, logger

app = FastAPI(
    title="AI Hiring Matcher",
    description="Ranks a resume against a catalog of job descriptions by semantic "
    "similarity and skill overlap.",
    version="2.0.0",
)

_vocabulary, _catalog = load_artifacts()


class ResumeRequest(BaseModel):
    resume: str
    top_n: int = 5


@app.get("/")
def read_root():
    logger.info("Root route accessed.")
    return {"message": "AI Hiring Matcher API is up!"}


@app.post("/match")
def match(request: ResumeRequest):
    logger.info("Received /match request (top_n=%d)", request.top_n)
    try:
        ranked = match_resume(request.resume, _vocabulary, _catalog, top_n=request.top_n)
        top = ranked.iloc[0]

        log_request(
            {
                "resume_length": len(request.resume),
                "cosine_similarity": float(top["similarity"]),
                "skill_overlap": float(top["skill_overlap"]),
            }
        )

        logger.info("Match complete. Best job: %s", top["job_role"])
        return {"matches": ranked.to_dict(orient="records")}
    except Exception as e:
        logger.error("Error during matching: %s", str(e))
        raise
