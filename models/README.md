# Model artifacts

This directory holds demo model artifacts produced by:

```bash
python scripts/train_demo_model.py
```

Each run writes:

- `<model>-<timestamp>.joblib` plus a matching `.json` sidecar of metrics
- `<model>-latest.joblib` and `<model>-latest.json`, the pair the Prediction page loads

Models here are fitted only on `data/synthetic_features.csv`, a table generated
by `scripts/generate_synthetic_data.py`. They are not clinical models and must
never be used for care. Binary artifacts are git ignored on purpose: the training
script rebuilds them in a few seconds and the container does it at build time.
