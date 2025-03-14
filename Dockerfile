# Dockerfile
FROM python:3.10

WORKDIR /app

COPY backend /backend
WORKDIR /backend

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
