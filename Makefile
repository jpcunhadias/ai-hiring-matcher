.PHONY: sync train baselines figures test lint format serve drift-check monitor

# Install/sync dependencies (runtime + dev) via uv
sync:
	uv sync

# Train the model; build the job catalog, fairness report and drift reference
train:
	uv run python -m src.train_model

# Compare the matcher against simple baselines (random, skill overlap, TF-IDF)
baselines:
	uv run python -m src.baselines

# Regenerate the README fairness figures (docs/images/); needs the dataset
figures:
	uv run python -m src.plots

# Run the tests
test:
	uv run pytest

# Lint with ruff and type-check with mypy
lint:
	uv run ruff check .
	uv run mypy src

# Auto-fix and format with ruff
format:
	uv run ruff check . --fix
	uv run ruff format .

# Serve the API locally with auto-reload
serve:
	uv run uvicorn src.api:app --reload

# Compare real requests (data/logs/requests.jsonl) against the training reference.
# Meant to run on a schedule (cron/systemd timer), not only on demand.
drift-check:
	uv run python -m src.drift_monitor

# Launch the drift monitoring dashboard
monitor:
	uv run streamlit run src/monitor_app.py
