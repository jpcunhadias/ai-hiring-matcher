import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.predict_model import load_model, predict_proba
from src.utils import logger

app = FastAPI(
    title="Prioriza Candidatos",
    description="API para prever probabilidade de contratação de candidatos",
    version="1.0.0",
)

model = load_model()


class Candidate(BaseModel):
    nivel_academico: str
    ingles: str
    espanhol: str
    area_atuacao: str
    nivel_profissional: str
    sap: str
    cliente: str


@app.get("/")
def read_root():
    logger.info("Rota raiz acessada.")
    return {"message": "API de priorização de candidatos está no ar!"}


@app.post("/predict")
def predict(data: Candidate):
    logger.info("Recebida requisição para /predict: %s", data.model_dump())
    try:
        df = pd.DataFrame([data.model_dump()])
        result_df = predict_proba(model, df, top_n=1)
        prob = result_df.iloc[0]["prob_contratacao"]
        logger.info("Predição realizada com sucesso: %.5f", prob)
        return {"prob_contratacao": round(float(prob), 8)}
    except Exception as e:
        logger.error("Erro durante a predição: %s", str(e))
        raise
