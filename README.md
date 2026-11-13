# CTAID POC

A public portfolio proof of concept for a computed tomography atherosclerosis detection research app. Built from scratch as a clean demonstration of the engineering (Python, FastAPI, a small scikit-learn model, tests, and Docker). It is **not** a copy of any private research system and it does not contain any real clinical records.

**Live demo:** [https://ctaid-poc.azril.io](https://ctaid-poc.azril.io)

> **Demo with synthetic data, not a medical tool.** This app uses invented local data only. It is not a medical device, it does not give a diagnosis, and it is not affiliated with any clinic, university, or other institution. It must not be used for clinical decisions.

## What it shows

- A persistent demo banner so visitors never mistake this for a real medical product
- English-only interface
- A small logistic regression model trained on 2,500 fully synthetic rows produced by `scripts/train_model.py`
- A form that loads invented sample cases or accepts edited numeric inputs
- A FastAPI JSON endpoint at `/api/predict`
- Automated tests, lint-free Python, a production Docker image, and GitHub Actions CI

The feature names (age, blood pressure, cholesterol, a made-up calcium score, a made-up vessel narrowing percentage, smoking, and diabetes) were written for this demonstration. They are not taken from a private research file.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
pytest -q
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## Docker

```bash
python scripts/train_model.py
docker build -t ctaid-poc .
docker run --rm -p 8080:80 ctaid-poc
```

Then open [http://localhost:8080](http://localhost:8080).

## Stack

- Python 3.12 and FastAPI
- scikit-learn (logistic regression with standard scaling)
- Pytest
- Nginx is not required: Uvicorn serves the app on port 80 inside the container

## What this is not

This repository is a portfolio demonstration. It does not contain clinical images, private research tables, real personal records, or deployment secrets.

Built by [Mohd Azril Amil](https://github.com/azrilamil).
