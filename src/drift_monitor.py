import pandas as pd
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils import load_df_from_s3, setup_logging

logger = setup_logging("drift")
load_dotenv()

# Caminho no S3
DF_MODEL_KEY = "data/processed/df_model.csv"


def simulate_production_data(df: pd.DataFrame, fraction: float = 0.1) -> pd.DataFrame:
    """Simula dados de produção com uma amostra da base de treino."""
    return df.sample(frac=fraction, random_state=42)


def run_drift_report():
    logger.info("Carregando dados de treinamento do S3...")
    df_train = load_df_from_s3(DF_MODEL_KEY)

    logger.info("Simulando dados de produção...")
    df_prod = simulate_production_data(df_train)

    logger.info("Executando análise de drift com Evidently...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_train, current_data=df_prod)

    output_path = "drift_report.html"
    report.save_html(output_path)
    logger.info(f"Relatório de drift salvo em: {output_path}")


if __name__ == "__main__":
    run_drift_report()
