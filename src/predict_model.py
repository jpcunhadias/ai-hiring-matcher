import pandas as pd

from src.utils import logger, load_model_from_s3


def load_model(path: str = "models/modelo_priorizacao_xgb.pkl"):
    """
    Carrega o modelo treinado a partir do S3.
    """
    logger.info("Carregando modelo a partir de: %s", path)
    model = load_model_from_s3(path)
    logger.info("Modelo carregado com sucesso.")
    return model


def predict_proba(model, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Retorna as probabilidades de contratação para os candidatos.
    """
    logger.info("Realizando predições de probabilidade...")
    proba = model.predict_proba(df)[:, 1]  # Probabilidade da classe positiva
    df_result = df.copy()
    df_result["prob_contratacao"] = proba

    logger.info("Ordenando candidatos pela probabilidade...")
    return df_result.sort_values(by="prob_contratacao", ascending=False).head(top_n)
