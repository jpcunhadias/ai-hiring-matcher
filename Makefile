.PHONY: sync train test lint format serve drift-check monitor

# Instala/sincroniza dependências (runtime + dev) via uv
sync:
	uv sync

# Treina o modelo, gera embeddings/catálogo de vagas e a referência de drift
train:
	uv run python -m src.train_model

# Roda os testes com pytest
test:
	uv run pytest

# Verifica estilo com ruff e tipos com mypy
lint:
	uv run ruff check .
	uv run mypy src

# Corrige automaticamente com ruff
format:
	uv run ruff check . --fix
	uv run ruff format .

# Sobe a API localmente com reload
serve:
	uv run uvicorn src.api:app --reload

# Compara requisições reais (data/logs/requests.jsonl) contra a referência de treino.
# Pensado para rodar em um agendamento (cron/systemd timer), não só sob demanda.
drift-check:
	uv run python -m src.drift_monitor

# Sobe o dashboard de monitoramento de drift
monitor:
	uv run streamlit run src/monitor_app.py
