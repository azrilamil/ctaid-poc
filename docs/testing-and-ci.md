# Testing and CI

## Local gates

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python scripts/train_demo_model.py
pytest -q
python scripts/scan_content.py
```

`pytest` covers generator shape and ranges, training, prediction, upload
validation, backend round trips, storage fallbacks, the hero asset, privacy
scanning and Streamlit page rendering. `scripts/scan_content.py` exits non zero
on any content leak.

## Test layout

| File | Covers |
| --- | --- |
| `tests/test_generator.py` | column order, row counts, schema ranges, determinism, record grouping, class signal, committed CSV shape |
| `tests/test_training.py` | all four models fit, metrics in range, artifact and sidecar written, prediction probabilities sum to one, upload rejections |
| `tests/test_backend.py` | SQLite round trip and reset, Convex URL fallback, local store put/get/list/clear, path traversal guard, S3 fallback, hero SVG validity |
| `tests/test_privacy.py` | work tree is clean, scanner detects injected leaks, dashes, `Patient ID`, credentials |
| `tests/test_app_smoke.py` | Home renders, all four pages render, training click produces tables, prediction page offers input methods |

## What the smoke tests guard

`tests/test_app_smoke.py` drives the real Streamlit script through
`streamlit.testing.v1.AppTest`:

1. Home renders with the demo notice in the markup.
2. Every sidebar page renders with no exception and no error element.
3. Clicking Train runs real training and produces at least two dataframes
   (metrics and classification report).
4. The Prediction page always offers a generated sample input.

A page that raises would fail these tests, so a broken import or a bad chart
argument cannot reach main unnoticed.

## Content scan

```bash
python scripts/scan_content.py --out /tmp/ctaid-poc-scan.txt
```

Checks the public work tree only, skipping `.git`, `.venv`, `__pycache__`,
`node_modules` and any local runtime state:

- institution, project and original file name markers
- emails, phone numbers, API keys, private key blocks
- `Patient ID` headers in committed CSVs
- em dashes and en dashes
- Malay leftovers in prose

The forbidden markers live base64 encoded inside `app/privacy.py`, so a plain
text search over the repository does not itself contain the strings.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull
request:

1. Python 3.12 with pip caching.
2. Install `requirements.txt`.
3. Regenerate the synthetic dataset and retrain the demo artifacts.
4. Run `pytest -q`.
5. Run the content scan and fail the job on any finding.

No secrets are required. Nothing in CI touches a cloud account.

## Docker gate

```bash
docker build -t ctaid-poc .
docker run --rm -p 8080:80 ctaid-poc
curl -fsS http://127.0.0.1:8080/health
```

The image installs runtime requirements only, warms the model artifacts at build
time and starts nginx in front of Streamlit. `/health` is answered by nginx
itself, so the container health check does not depend on Streamlit being ready.

## Gate log

Local runs write to `/tmp/ctaid-poc-gates.txt`:

```bash
pytest -q > /tmp/ctaid-poc-gates.txt 2>&1
python scripts/scan_content.py >> /tmp/ctaid-poc-gates.txt 2>&1
```
