import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.predict_model import load_model, predict_proba

app = FastAPI(
    title="Prioriza Candidatos",
    description="API para prever probabilidade de contratação de candidatos",
    version="1.0.0",
)

# Carrega modelo na inicialização da API
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
    return {"message": "API de priorização de candidatos está no ar!"}


@app.post("/predict")
def predict(data: Candidate):
    # Converte o input para DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Faz a predição
    result_df = predict_proba(model, df, top_n=1)

    # Retorna a probabilidade do primeiro (e único) candidato
    prob = result_df.iloc[0]["prob_contratacao"]
    return {"prob_contratacao": round(float(prob), 8)}
