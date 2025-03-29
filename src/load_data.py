import json
import pandas as pd


def load_json_files(jobs_path: str, prospects_path: str, applicants_path: str):
    """
    Carrega os arquivos JSON e retorna os dados como dicionários.
    """
    with open(jobs_path, 'r', encoding='utf-8') as f:
        jobs_dict = json.load(f)

    with open(prospects_path, 'r', encoding='utf-8') as f:
        prospects_dict = json.load(f)

    with open(applicants_path, 'r', encoding='utf-8') as f:
        applicants_dict = json.load(f)

    return jobs_dict, prospects_dict, applicants_dict


def flatten_prospects(prospects_dict: dict) -> pd.DataFrame:
    """
    Converte o dicionário de prospecções em um DataFrame plano com colunas:
    vaga_id, candidato_id, situacao, nome, comentario, etc.
    """
    rows = []

    for vaga_id, dados_vaga in prospects_dict.items():
        titulo = dados_vaga.get("titulo")
        modalidade = dados_vaga.get("modalidade")

        for prospect in dados_vaga.get("prospects", []):
            rows.append({
                "vaga_id": vaga_id,
                "candidato_id": prospect.get("codigo"),
                "situacao": prospect.get("situacao_candidado"),
                "comentario": prospect.get("comentario"),
                "nome": prospect.get("nome"),
                "recrutador": prospect.get("recrutador"),
                "data_candidatura": prospect.get("data_candidatura"),
                "ultima_atualizacao": prospect.get("ultima_atualizacao"),
                "titulo_vaga": titulo,
                "modalidade_vaga": modalidade
            })

    return pd.DataFrame(rows)


def merge_all(jobs_dict: dict, applicants_dict: dict, prospects_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une os dados de candidatos, vagas e prospecções em um único DataFrame consolidado.
    """
    merged_rows = []
    for _, row in prospects_df.iterrows():
        vaga_id = row["vaga_id"]
        candidato_id = row["candidato_id"]

        vaga_data = jobs_dict.get(str(vaga_id), {})
        candidato_data = applicants_dict.get(str(candidato_id), {})

        merged_row = {
            "vaga_id": vaga_id,
            "candidato_id": candidato_id,
            "situacao": row["situacao"],
            "comentario": row["comentario"],
            "nome": row["nome"],
            # Inclui campos úteis diretamente do dicionário da vaga
            "cliente": vaga_data.get("cliente"),
            "sap": vaga_data.get("eh_vaga_sap"),
            "nivel_profissional": vaga_data.get("nivel_profissional"),
            "idioma": vaga_data.get("idioma"),
            "competencias_tecnicas": vaga_data.get("competencias_tecnicas"),
            # E do candidato
            "nivel_academico": candidato_data.get("nivel_academico"),
            "ingles": candidato_data.get("nivel_ingles"),
            "espanhol": candidato_data.get("nivel_espanhol"),
            "area_atuacao": candidato_data.get("area_atuacao"),
            "conhecimentos": candidato_data.get("conhecimentos_tecnicos"),
            "cv": candidato_data.get("cv")
        }

        merged_rows.append(merged_row)

    return pd.DataFrame(merged_rows)


if __name__ == "__main__":
    # Caminhos esperados dos arquivos
    jobs_path = "../data/raw/jobs.json"
    prospects_path = "../data/raw/prospects.json"
    applicants_path = "../data/raw/applicants.json"

    # Executa pipeline de carregamento
    jobs, prospects, applicants = load_json_files(jobs_path, prospects_path, applicants_path)
    prospects_df = flatten_prospects(prospects)
    merged_df = merge_all(jobs, applicants, prospects_df)

    # Salva consolidado
    merged_df.to_csv("../data/processed/merged_data.csv", index=False)
    print("Arquivo salvo em ../data/processed/merged_data.csv")
