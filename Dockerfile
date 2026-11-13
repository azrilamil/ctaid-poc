FROM python:3.12-slim

WORKDIR /app

COPY requirements-run.txt .
RUN pip install --no-cache-dir -r requirements-run.txt

COPY app ./app
COPY models ./models

EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
