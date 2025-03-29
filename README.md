# AI Hiring Matcher 🤖💼

Projeto desenvolvido para o Datathon da Pós Tech, com o objetivo de aplicar Machine Learning no processo de recrutamento e seleção da empresa Decision.

## 📌 Objetivo

Criar um modelo preditivo que, a partir de dados de candidatos(as) e vagas anteriores, estime a probabilidade de sucesso no processo seletivo. A solução será exposta via API e empacotada com Docker.

## 📂 Estrutura do Projeto

ml_recruitment_model/  
├── data/              # Dados brutos e processados  
├── notebooks/         # EDA e análises exploratórias  
├── src/               # Scripts de carregamento, modelagem e API  
├── models/            # Modelos salvos  
├── tests/             # Testes automatizados  
├── Dockerfile         # Configuração do ambiente com Docker  
├── requirements.txt   # Dependências do projeto  
└── README.md          # Documentação inicial

## 🚀 Entregáveis

- Modelo preditivo treinado e serializado (.pkl ou .joblib)  
- API com FastAPI no endpoint `/predict`  
- Empacotamento com Docker  
- Deploy funcional (local ou nuvem)  
- Código organizado em repositório público  
- Vídeo explicativo (até 5 minutos)

## 📋 Status

🛠️ Em desenvolvimento — estruturação inicial do projeto e exploração de dados

## 🧪 Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

Ou use um ambiente virtual com Poetry, Conda ou Virtualenv.
