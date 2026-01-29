# Use official Python image as base
FROM python:3.11-slim

WORKDIR /app

COPY backend/ ./backend/

COPY KCDTA/models/ ./KCDTA/models/
COPY KCDTA/model_cnn_kiba.model ./KCDTA/model_cnn_kiba.model

RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 8000

CMD ["python", "backend/main.py"]
