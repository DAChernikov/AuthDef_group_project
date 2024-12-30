FROM python:3.9-slim

WORKDIR /app

COPY pretrained_models /models_storage
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
