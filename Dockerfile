# CTAID POC container image: Streamlit behind a small nginx front.
# nginx listens on 80, answers /health itself (so the Coolify health check stays
# cheap) and proxies everything else to Streamlit on 8080 with websocket upgrade.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=127.0.0.1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nginx \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

COPY requirements-run.txt .
RUN pip install -r requirements-run.txt

COPY .streamlit ./.streamlit
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY streamlit_app.py ./
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Warm the demo artifacts at build time so the first prediction is instant.
RUN python scripts/train_demo_model.py --rows 1200

EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1/health || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]
