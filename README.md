# AI Hiring Matcher

Este projeto começou como a entrega de um Datathon de Machine Learning
Engineering: um classificador XGBoost sobre 7 campos categóricos, sem nenhum
sinal de texto, hospedado no AWS S3. Funcionava, mas nunca "casava" nada — o
nome prometia um *matcher* e entregava um classificador binário cego à vaga.

Esta é a reconstrução (v2): um matcher semântico de verdade — que ranqueia o
currículo contra um catálogo de vagas por similaridade de embeddings —, uma
auditoria de fairness que **encontrou um viés real e severo embutido no
próprio dataset**, monitoramento de drift que compara tráfego real (não uma
reamostragem de si mesmo), e todo o ambiente reconstruído em `uv`, rodando
localmente sem nenhuma credencial externa.

---

## O dataset

[`job_applicant_dataset.csv`](data/external/job_applicant_dataset.csv) (10.000
linhas, via Kaggle) traz `Resume` e `Job Description` em texto livre, um
rótulo binário `Best Match`, e colunas demográficas sintéticas (`Age`,
`Gender`, `Race`, `Ethnicity`). Só existem **51 descrições de vaga únicas** —
ou seja, este é um problema de *retrieval em conjunto fechado* (ranquear
entre 51 vagas conhecidas), não generalização para vagas nunca vistas.

## O achado principal: o rótulo do dataset é enviesado por gênero

Antes de confiar em `Best Match` como alvo de treino, a auditoria de fairness
([src/fairness_audit.py](src/fairness_audit.py)) mede a taxa de match por
grupo demográfico **direto no dado bruto, sem nenhum modelo envolvido**. O
resultado:

- Agregado: `Best Match = 1` em **61,3%** das linhas de homens vs. **35,4%**
  das linhas de mulheres.
- Por vaga, o efeito é muito mais extremo e **não é uniforme** — em algumas
  vagas favorece homens, em outras mulheres, com gaps de até **89 pontos
  percentuais**:

| Vaga | Taxa (mulheres) | Taxa (homens) | Gap |
|---|---:|---:|---:|
| Journalist | 6,2% | 95,6% | +89,4 pp |
| Financial Analyst | 10,0% | 95,7% | +85,7 pp |
| Content Writer | 91,7% | 6,6% | −85,1 pp |
| Psychologist | 92,5% | 8,3% | −84,2 pp |

(tabela completa gerada em `reports/fairness_report.md` a cada treino)

Nem idade, raça, etnia, nível de experiência ou certificações mostram sinal
comparável — o efeito é especificamente de gênero, e específico por vaga.
Olhando para todos os 102 grupos `(Job Role, Gender)`, nenhum é
perfeitamente determinístico (0% ou 100%), mas as taxas são fortemente
**bimodais**: 53 grupos ficam em ≤20%, 49 em ≥80%, e **nenhum** cai entre
esses dois polos (calculado em [src/fairness_audit.py](src/fairness_audit.py)
via `gender_rate_bimodality`, reproduzido a cada `make train`). Isso é a
assinatura de `Best Match` tendo sido amostrado como `Bernoulli(p)` por
grupo, com `p` fixado perto de 0,1 ou 0,9 — não ruído incidental, e não uma
regra determinística fixa.

A [ficha do dataset no Kaggle](https://www.kaggle.com/datasets/surendra365/recruitement-dataset)
confirma o contexto: ele é descrito explicitamente como material para
"HR Analytics & Hiring Bias Studies" e para "analisar tendências e vieses
na contratação com base em idade, gênero ou etnia" — então a existência de
viés aqui não é surpresa, é o propósito declarado do dataset. O que *é*
notável: a mesma ficha define `Best Match` como "indicando o quão bem o
candidato corresponde à vaga **com base em qualificações e experiência**"
— e isso não bate com o que os dados mostram. A correlação entre
`Best Match` e as features reais de similaridade (`cosine_similarity`,
`skill_overlap`) é praticamente zero; o padrão bimodal por grupo é quem
domina o rótulo, não qualificação. Não dá para saber, só pelos dados, se
essa divergência entre a documentação e o gerador foi intencional — mas o
achado prático vale de qualquer forma: **audite o rótulo antes de treinar
em cima dele**, mesmo quando a fonte descreve o que ele deveria significar.

**Decisão de design direta consequência disso:** o classificador de
`Best Match` (abaixo) **nunca recebe Gender/Race/Ethnicity como feature** —
mesmo sendo, de longe, o sinal mais forte do rótulo. Usar um atributo
protegido para prever "match" seria reproduzir exatamente o viés que esta
auditoria existe para pegar. O preço dessa escolha aparece na seção seguinte.

## O matcher

[src/embeddings.py](src/embeddings.py) usa `sentence-transformers`
(`all-MiniLM-L6-v2`, local, sem chave de API) para embutir currículos e
descrições de vaga. [src/matcher.py](src/matcher.py) ranqueia as 51 vagas do
catálogo por similaridade de cosseno — isso é o "matcher" de fato.

Avaliado como retrieval (o currículo recupera sua própria `Job Roles` entre
as 51 vagas conhecidas):

| Métrica | Valor |
|---|---:|
| Recall@1 | 63,1% |
| Recall@5 | 88,2% |
| MRR | 0,745 |

Isso funciona bem porque cada currículo foi gerado com um vocabulário de
skills que reflete a vaga alvo — similaridade textual recupera essa vaga na
maior parte das vezes.

### O classificador de Best Match: honesto sobre seus limites

[src/train_model.py](src/train_model.py) treina uma regressão logística sobre
`[cosine_similarity, skill_overlap]` → `Best Match`. A correlação entre essas
features e o rótulo é essencialmente zero (~ -0,02 e ~0,00) — porque, como o
achado acima mostra, `Best Match` é dominado por Gender, não por similaridade
semântica real. Resultado: F1 ~0,08 na classe positiva, pouco acima do acaso.

Isso não é um bug para "consertar" ajustando hiperparâmetros — é o resultado
esperado de deliberadamente não alimentar o modelo com o atributo que mais
prediz o rótulo. O `skill_overlap` é calculado por extração baseada em regex
sobre o formato template dos currículos (`src/data_preparation.py`), com
correspondência de palavra inteira — verificado contra as 10.000 linhas antes
de virar código.

## Monitoramento de drift (batch-live, não só sob demanda)

[src/drift_monitor.py](src/drift_monitor.py) compara requisições reais
logadas em `data/logs/requests.jsonl` (cada chamada a `/match` grava suas
próprias features) contra uma referência de treino — construída da **mesma
forma** que uma requisição real (top-1 do catálogo, não o pareamento
arbitrário do dataset; comparar essas duas distribuições diferentes já gerou
um alerta falso de 75-100% de drift antes desse ajuste). Evidently roda o
teste estatístico por coluna; abaixo de `DRIFT_MIN_WINDOW_SIZE` (padrão 100)
requisições, o check se recusa a rodar — testes estatísticos em amostras
pequenas são ruidosos por natureza (uma janela de 30 requisições já gerou
100% de "drift" só por variância amostral).

```bash
make drift-check   # roda uma vez
make monitor       # dashboard Streamlit
```

Pensado para rodar em um agendamento (cron/systemd timer no servidor), não
apenas manualmente.

---

## Stack e por que não tem AWS

Tudo roda com `uv` — sem `pip`/`venv` manual, sem `requirements.txt`. Dados
(`data/external/`) são versionados com [DVC](https://dvc.org), com um remote
local (`~/.local/share/dvc-storage/`) por padrão — funciona 100% offline, sem
credencial nenhuma.

O AWS S3 da v1 só existia porque a pós-graduação era parceira da AWS. O
ambiente real da equipe é um servidor Ubuntu compartilhado (MLflow + MySQL +
Postgres + MinIO, documentado em `~/Desktop/Projects/server-onboarding`,
acessado via Tailscale) — mas esse servidor está sendo fisicamente
transportado no momento desta reconstrução, então tudo aqui roda local por
padrão:

- **MLflow**: `sqlite:///mlflow.db` local por padrão. Para apontar para o
  MLflow do servidor quando ele voltar: `MLFLOW_TRACKING_URI=http://<ip-tailscale>:5000`
  no `.env`.
- **DVC**: remote local por padrão. Para apontar para o MinIO do servidor:
  `dvc remote add -d server-storage s3://<bucket> --endpointurl http://<ip-tailscale>:9000`
  (ver `server-onboarding` para credenciais).

## Como rodar

```bash
uv sync                        # instala tudo (runtime + dev)
make train                     # treina: embeddings, classificador, catálogo,
                                # auditoria de fairness, referência de drift
make serve                     # sobe a API em http://localhost:8000
make test                      # pytest
make lint                      # ruff + mypy
```

### Docker (sem credencial nenhuma)

```bash
docker compose up
```

`models/` e `data/` são montados como volumes — treine localmente
(`make train`) antes de subir o container pela primeira vez.

### API

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"resume": "Proficient in Python, SQL, Machine Learning, with senior-level experience in the field. Holds a Masters degree. Skilled in delivering results and adapting to dynamic environments.", "top_n": 3}'
```

```json
{
  "matches": [
    {"job_role": "Software Engineer", "similarity": 0.564, "skill_overlap": 0.0, "best_match_proba": 0.480},
    {"job_role": "AI Specialist", "similarity": 0.530, "skill_overlap": 0.25, "best_match_proba": 0.484},
    {"job_role": "Machine Learning Engineer", "similarity": 0.484, "skill_overlap": 0.125, "best_match_proba": 0.487}
  ]
}
```

(saída real, gerada a partir do modelo treinado neste repositório)

---

## Estrutura

```
.
├── data/
│   ├── external/           # dataset Kaggle, versionado com DVC
│   ├── processed/          # referência de drift (gerado por make train)
│   └── logs/                # requisições reais logadas (gerado em runtime)
├── models/                  # classificador, vocabulário de skills, catálogo (gerado)
├── reports/                  # auditoria de fairness (gerado por make train)
├── src/
│   ├── data_preparation.py   # parsing de currículo, extração de skills, split
│   ├── embeddings.py         # wrapper sentence-transformers
│   ├── matcher.py            # catálogo de vagas, ranking, métricas de retrieval
│   ├── train_model.py        # pipeline de treino completo (MLflow + fairness + drift ref)
│   ├── predict_model.py      # inferência: match_resume()
│   ├── fairness_audit.py     # auditoria de viés demográfico
│   ├── drift_monitor.py      # comparação de drift batch-live com alerta
│   ├── api.py                 # FastAPI (/match)
│   ├── monitor_app.py         # dashboard Streamlit de drift
│   └── utils.py                # logging, I/O local, log de requisições
└── tests/
```

## Testes e qualidade

```bash
make test     # pytest (26 testes)
make lint     # ruff check + mypy
uv run pre-commit run --all-files
```
