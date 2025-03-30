# AI Hiring Matcher

Este projeto foi desenvolvido para o Datathon de Machine Learning Engineering com o objetivo de priorizar candidatos com maior probabilidade de contratacao. A solução envolve um modelo preditivo de classificação, uma API em FastAPI, e empacotamento via Docker, além de testes automatizados.

---

## Objetivo

Desenvolver uma solução de Machine Learning Engineering que:

- Analise dados históricos de processos seletivos
- Classifique candidatos de acordo com sua chance de contratação
- Ofereça uma API que possa ser consultada por outros sistemas
- Seja facilmente testável, reproduzível e containerizada

---

## Tecnologias utilizadas

- Python 3.9
- Pandas, Scikit-Learn, XGBoost
- FastAPI
- Pytest
- Docker
- Joblib

---

## Estrutura do projeto

```
.
├── Dockerfile
├── README.md
├── data/
│   └── df_model.csv              # Base com variável target
├── models/
│   └── modelo_priorizacao_xgb.pkl
├── notebooks/
│   ├── eda.ipynb
│   ├── modeling_baseline.ipynb
│   └── modelo_priorizacao_final.ipynb
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── api.py                    # FastAPI
│   ├── data_preparation.py       # Pipeline de pré-processamento
│   ├── load_data.py              # (opcional)
│   ├── predict_model.py          # Predição com modelo treinado
│   └── train_model.py            # Treinamento do modelo
└── tests/
    ├── test_api.py
    └── test_preprocessing.py
```

---

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/ai-hiring-matcher.git
cd ai-hiring-matcher
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
PYTHONPATH=. python src/train_model.py
```

Isso irá gerar o arquivo `modelo_priorizacao_xgb.pkl` na pasta `models/`.

---

## Testes

Execute os testes automatizados com:

```bash
PYTHONPATH=. pytest tests/
```

---

## Executar via Docker

### 1. Build da imagem

```bash
docker build -t ai-hiring-matcher .
```

### 2. Executar o container

```bash
docker run -p 8000:8000 ai-hiring-matcher
```

A API estará disponível em:  
`http://localhost:8000`

---

## Uso da API

### Rota `/predict`

- **Método:** `POST`
- **Endpoint:** `/predict`
- **Descrição:** Recebe os dados de um candidato e retorna a probabilidade de contratação.

#### Exemplo de entrada (JSON)

```json
{
  "nivel_academico": "Ensino Superior Completo",
  "ingles": "Avançado",
  "espanhol": "Intermediário",
  "area_atuacao": "Ti - Projetos",
  "nivel_profissional": "Sênior",
  "sap": "Não",
  "cliente": "Gonzalez And Sons"
}
```

#### Exemplo de resposta

```json
{
  "prob_contratacao": 0.71837991
}
```

---

## Modelo e métricas

O modelo utilizado foi um XGBoost com pré-processamento baseado em codificação categórica (OneHotEncoder).  

As principais métricas na base de validação foram:

- **Accuracy:** ~0.75
- **Precision (classe positiva):** ~0.50
- **Recall (classe positiva):** ~0.54
- **F1-score (classe positiva):** ~0.52

O desempenho pode ser melhorado com mais dados ou novas features.

---

## Conclusão

A solução entregue atende todos os critérios propostos pelo desafio:

- Pipeline de modelagem funcional e modularizado
- API documentada e testada
- Container Docker funcional
- Testes automatizados
- Documentação clara e reprodutível

---