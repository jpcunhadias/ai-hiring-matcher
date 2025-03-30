import json
import os

import boto3
import pandas as pd
from dotenv import load_dotenv

from src.utils import save_df_to_s3, logger

load_dotenv()

BUCKET = os.getenv("S3_BUCKET_NAME")
RAW_PATH = os.getenv("RAW_PATH")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)


def read_json_from_s3(key: str) -> dict:
    response = s3.get_object(Bucket=BUCKET, Key=key)
    content = response["Body"].read()
    return json.loads(content)


def load_json_files():
    logger.info("Iniciando leitura dos arquivos JSON do S3...")
    jobs_dict = read_json_from_s3(f"{RAW_PATH}jobs.json")
    logger.info("Carregado: jobs.json")

    prospects_dict = read_json_from_s3(f"{RAW_PATH}prospects.json")
    logger.info("Carregado: prospects.json")

    applicants_dict = read_json_from_s3(f"{RAW_PATH}applicants.json")
    logger.info("Carregado: applicants.json")

    return jobs_dict, prospects_dict, applicants_dict


def flatten_prospects(prospects_dict: dict) -> pd.DataFrame:
    rows = []

    for vaga_id, dados_vaga in prospects_dict.items():
        titulo = dados_vaga.get("titulo")
        modalidade = dados_vaga.get("modalidade")

        for prospect in dados_vaga.get("prospects", []):
            rows.append(
                {
                    "vaga_id": vaga_id,
                    "candidato_id": prospect.get("codigo"),
                    "situacao": prospect.get("situacao_candidado"),
                    "comentario": prospect.get("comentario"),
                    "nome": prospect.get("nome"),
                    "recrutador": prospect.get("recrutador"),
                    "data_candidatura": prospect.get("data_candidatura"),
                    "ultima_atualizacao": prospect.get("ultima_atualizacao"),
                    "titulo_vaga": titulo,
                    "modalidade_vaga": modalidade,
                }
            )

    return pd.DataFrame(rows)

def safe_key(k):
    try:
        return int(float(k))
    except (ValueError, TypeError):
        return str(k).strip()

def merge_all(
        jobs_dict: dict, applicants_dict: dict, prospects_df: pd.DataFrame
) -> pd.DataFrame:
    merged_rows = []

    applicants_dict_clean = {safe_key(k): v for k, v in applicants_dict.items()}
    jobs_dict_clean = {safe_key(k): v for k, v in jobs_dict.items()}

    logger.info("Iniciando merge de candidatos, vagas e prospecções...")

    for _, row in prospects_df.iterrows():
        vaga_key = safe_key(row["vaga_id"])
        candidato_key = safe_key(row["candidato_id"])

        vaga_data = jobs_dict_clean.get(vaga_key, {})
        candidato_data = applicants_dict_clean.get(candidato_key, {})

        informacoes_basicas = vaga_data.get("informacoes_basicas", {})
        perfil_vaga = vaga_data.get("perfil_vaga", {})

        merged_row = {
            "vaga_id": vaga_key,
            "candidato_id": candidato_key,
            "situacao": row["situacao"],
            "comentario": row["comentario"],
            "nome": row["nome"],
            "titulo_vaga": row.get("titulo_vaga"),
            "modalidade_vaga": row.get("modalidade_vaga"),
            "cliente": informacoes_basicas.get("cliente"),
            "sap": informacoes_basicas.get("vaga_sap"),
            "nivel_profissional": perfil_vaga.get("nivel profissional"),
            "idioma": perfil_vaga.get("nivel_ingles"),
            "competencias_tecnicas": perfil_vaga.get(
                "competencia_tecnicas_e_comportamentais"
            ),
            "nivel_academico": candidato_data.get("formacao_e_idiomas", {}).get(
                "nivel_academico"
            ),
            "ingles": candidato_data.get("formacao_e_idiomas", {}).get("nivel_ingles"),
            "espanhol": candidato_data.get("formacao_e_idiomas", {}).get(
                "nivel_espanhol"
            ),
            "area_atuacao": candidato_data.get("informacoes_profissionais", {}).get(
                "area_atuacao"
            ),
            "conhecimentos": candidato_data.get("informacoes_profissionais", {}).get(
                "conhecimentos_tecnicos"
            ),
            "cv": candidato_data.get("cv_pt"),
        }

        merged_rows.append(merged_row)

    return pd.DataFrame(merged_rows)


def load_and_merge_data(save: bool = True) -> pd.DataFrame:
    jobs_dict, prospects_dict, applicants_dict = load_json_files()
    prospects_df = flatten_prospects(prospects_dict)
    merged_df = merge_all(jobs_dict, applicants_dict, prospects_df)

    if save:
        save_df_to_s3(merged_df, "data/processed/merged_data.csv")

    return merged_df


if __name__ == "__main__":
    load_and_merge_data()
