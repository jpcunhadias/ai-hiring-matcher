# Imagem base enxuta com Python 3.9
FROM python:3.9-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia todos os arquivos para o container
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expõe a porta da aplicação
EXPOSE 8000

# Comando para rodar a API com Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
