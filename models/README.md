# Model artifacts

This directory holds demo model artifacts produced by:

```bash
python scripts/train_demo_model.py
```

Each run writes:

- `<model>-<timestamp>.joblib` plus a matching `.json` sidecar of metrics
- `<model>-latest.joblib` and `<model>-latest.json`, a convenience copy of the run

The Prediction page loads the best saved version of a model: every
`<model>-*.joblib` with a sidecar is ranked by held out ROC AUC and then
accuracy, and the winner is used. Random Forest is the default selection because
it is the strongest saved model (accuracy 0.8419, macro ROC AUC 0.9595).

Models here are fitted only on `data/synthetic_features.csv`, a table generated
by `scripts/generate_synthetic_data.py`. They are not clinical models and must
never be used for care. Binary artifacts are git ignored on purpose: the training
script rebuilds them in a few seconds and the container does it at build time
(`RUN python scripts/train_demo_model.py --rows 1200` in the Dockerfile), so a
visitor gets a working default model without pressing Train.
