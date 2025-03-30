.PHONY: test lint format

# Roda os testes com pytest
test:
	PYTHONPATH=. pytest --disable-warnings

# Verifica estilo com ruff
lint:
	ruff check .

# Corrige automaticamente com ruff
format:
	ruff check . --fix
