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
    Une os dados de candidatos, vagas e prospecções em um único DataFrame consolidado,
    com diagnóstico e acesso correto a campos aninhados.
    """

    merged_rows = []

    def safe_key(k):
        try:
            return int(float(k))
        except:
            return str(k).strip()

    applicants_dict_clean = {safe_key(k): v for k, v in applicants_dict.items()}
    jobs_dict_clean = {safe_key(k): v for k, v in jobs_dict.items()}

    for _, row in prospects_df.iterrows():
        vaga_key = safe_key(row["vaga_id"])
        candidato_key = safe_key(row["candidato_id"])

        vaga_data = jobs_dict_clean.get(vaga_key, {})
        candidato_data = applicants_dict_clean.get(candidato_key, {})

        # Acesso seguro aos campos da vaga
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
            # Campos da vaga (corrigidos)
            "cliente": informacoes_basicas.get("cliente"),
            "sap": informacoes_basicas.get("vaga_sap"),
            "nivel_profissional": perfil_vaga.get("nivel profissional"),
            "idioma": perfil_vaga.get("nivel_ingles"),
            "competencias_tecnicas": perfil_vaga.get("competencia_tecnicas_e_comportamentais"),
            # Campos do candidato
            "nivel_academico": candidato_data.get("formacao_e_idiomas", {}).get("nivel_academico"),
            "ingles": candidato_data.get("formacao_e_idiomas", {}).get("nivel_ingles"),
            "espanhol": candidato_data.get("formacao_e_idiomas", {}).get("nivel_espanhol"),
            "area_atuacao": candidato_data.get("informacoes_profissionais", {}).get("area_atuacao"),
            "conhecimentos": candidato_data.get("informacoes_profissionais", {}).get("conhecimentos_tecnicos"),
            "cv": candidato_data.get("cv_pt")
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
