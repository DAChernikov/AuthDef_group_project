FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY streamlit_app .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.maxUploadSize=1024"]
