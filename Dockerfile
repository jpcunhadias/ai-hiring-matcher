# Base image
FROM python:3.9-slim

# Diretório de trabalho no container
WORKDIR /app

# Copia apenas os arquivos necessários
COPY requirements.txt .

# Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copia o restante do código da aplicação
COPY . .

# Define variáveis de ambiente (opcional: pode ser usado um docker-compose.yml também)
ENV PYTHONPATH=/app

# Porta da API
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
