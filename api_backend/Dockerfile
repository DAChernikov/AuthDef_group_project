FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models_storage && \
    curl -o /models_storage/base_model.pkl https://storage.yandexcloud.net/auth-def-2024/models/base_model.pkl && \
    curl -o /models_storage/base_model_w2v.pkl https://storage.yandexcloud.net/auth-def-2024/models/base_model_w2v.pkl

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
