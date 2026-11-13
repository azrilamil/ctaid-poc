#!/bin/sh
# Container entrypoint: start Streamlit, then nginx in the foreground.
set -eu

cd /app

# Keep the demo workspace writable without volumes.
mkdir -p var models data

# Streamlit in the background.
streamlit run streamlit_app.py \
  --server.port "${STREAMLIT_SERVER_PORT:-8080}" \
  --server.address "${STREAMLIT_SERVER_ADDRESS:-127.0.0.1}" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  >> /tmp/streamlit.log 2>&1 &

STREAMLIT_PID=$!

# Wait for the app to answer before nginx starts proxying.
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${STREAMLIT_SERVER_PORT:-8080}/_stcore/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "streamlit starting (pid ${STREAMLIT_PID})"

exec nginx -g "daemon off;"
