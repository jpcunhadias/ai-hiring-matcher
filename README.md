# AI Hiring Matcher

Este projeto foi desenvolvido para o Datathon de Machine Learning Engineering com o objetivo de priorizar candidatos com maior probabilidade de contratação. A solução envolve um modelo preditivo de classificação, uma API em FastAPI, empacotamento via Docker, integração com AWS S3, além de testes automatizados e logging estruturado.

---

## Objetivo

Desenvolver uma solução de Machine Learning Engineering que:

- Analise dados históricos de processos seletivos
- Classifique candidatos de acordo com sua chance de contratação
- Ofereça uma API que possa ser consultada por outros sistemas
- Seja facilmente testável, reprodutível e containerizada
- Armazene os dados e o modelo em ambiente remoto (S3)

---

## Tecnologias utilizadas

- Python 3.9
- Pandas, Scikit-Learn, XGBoost
- FastAPI
- Pytest
- Docker
- Joblib
- AWS S3
- Logging com módulo `logging`

---

## Estrutura do projeto

```
.
├── Dockerfile
├── README.md
├── notebooks/
│   ├── eda.ipynb
│   ├── modeling_baseline.ipynb
│   └── modelo_priorizacao_final.ipynb
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── api.py                    # FastAPI
│   ├── data_preparation.py       # Pipeline de pré-processamento
│   ├── load_data.py              # Carregamento de dados do S3
│   ├── predict_model.py          # Predição com modelo treinado
│   ├── train_model.py            # Treinamento completo com persistência no S3
│   └── utils.py                  # Utilitários para salvar/carregar S3 e logging
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
└── Makefile                      # Comandos automatizados
```

> **Observação:** os arquivos de dados e modelos são salvos no S3 e não estão presentes localmente.

---

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/ai-hiring-matcher.git
cd ai-hiring-matcher
```

### 2. Configurar variáveis de ambiente

Crie um arquivo `.env` com as seguintes variáveis:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-2
S3_BUCKET_NAME=ml-recruitment-data-joaopd
RAW_PATH=data/raw/
PROCESSED_PATH=data/processed/
MODELS_PATH=models/
```

### 3. Criar ambiente virtual e instalar dependências

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Executar pipeline de dados e treino do modelo

```bash
PYTHONPATH=. python src/train_model.py
```

Isso irá:
- Carregar os dados brutos do S3
- Criar a variável target
- Treinar o modelo
- Salvar o modelo e dados processados de volta no S3

---

## Testes

Execute os testes automatizados com:

```bash
make test
```

---

## Linting e formatação

```bash
make lint
```

---

## Executar via Docker

### 1. Build da imagem

```bash
docker build -t ai-hiring-matcher .
```

### 2. Executar o container

```bash
docker run -p 8000:8000 --env-file .env ai-hiring-matcher
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

O modelo utilizado foi um XGBoost com codificação categórica via `OneHotEncoder`.

As principais métricas de avaliação:

- **Accuracy:** ~0.75
- **Precision (classe positiva):** ~0.50
- **Recall (classe positiva):** ~0.54
- **F1-score (classe positiva):** ~0.52

---

## Conclusão

A solução entregue atende aos critérios do desafio:

- Pipeline modularizado com integração ao S3
- Modelo preditivo funcional
- API REST com FastAPI e documentação via Swagger
- Testes automatizados com Pytest
- Logging estruturado com `logging`
- Empacotamento com Docker
- Automação com Makefile

---