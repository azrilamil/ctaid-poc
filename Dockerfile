FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-run.txt .
RUN pip install --no-cache-dir -r requirements-run.txt

COPY app ./app
COPY models ./models

EXPOSE 80
HEALTHCHECK --interval=5s --timeout=5s --start-period=10s --retries=10 \
  CMD curl -fsS http://127.0.0.1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
