# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog, and the project uses semantic versioning.

## [2.0.1] 2026-11-13

Visitor facing polish after a screenshot review. No change to the data, the
models or the training path.

### Changed

- The target column is labelled `Pattern` in every visitor facing surface: the
  Data Exploration sort dropdown, the table header, the downloaded CSV and the
  Home page copy. The CSV keeps the internal column name, mapped through
  `DISPLAY_LABELS` and `TARGET_DISPLAY` in `app/schema.py`.
- Prediction opens on the best saved Random Forest (accuracy 0.8419, macro ROC
  AUC 0.9595) instead of the first option, and saved artifact labels quote their
  held out scores. `app/training.py` gained `best_artifact`, which ranks saved
  versions by ROC AUC then accuracy.
- Top weighted features on a prediction are real non zero values: tree
  `feature_importances_`, or absolute mean coefficients for a linear model, via
  `predict.feature_weights`. Calibrated wrappers are unwrapped first.
- Model Training storage cards read `Demo storage` with the hint `Local to this
  demo, resets on a schedule` when Convex and Garage are not configured, and name
  the product only when it actually is. `app/labels.py` holds the mapping.
- Default trained models are Random Forest then Logistic Regression, so the demo
  script produces the default artifact first.

### Added

- `tests/test_ui_polish.py`: Pattern alias, Random Forest default selection,
  non zero importances for tree and linear models, storage badges and a guard
  that no page module contains the internal column name as copy.

## [2.0.0] 2026-11-13

Complete rebuild. The previous 1.x app was a thin FastAPI single form; this
release replaces it with a Streamlit product that mirrors the original research
interface, page for page, on freshly generated synthetic data.

### Added

- Streamlit app with a dark, wide theme and sidebar navigation: Home, Data
  Exploration, Model Training, Prediction.
- `scripts/generate_synthetic_data.py`: 2100 row, 68 column synthetic dataset with
  a fixed seed, a record and vessel grouping, and per class signal.
- `data/synthetic_features.csv`: committed synthetic dataset (reproducible).
- Four trainable classifiers in `app/training.py` with real metrics, artifacts and
  JSON sidecars: Logistic Regression, Random Forest, calibrated SVM, Gradient
  Boosting.
- Model Training page with metric comparison, classification report, confusion
  matrix, one versus rest ROC curves, feature importance and a version log.
- Prediction page with three input methods (form, generated sample, CSV upload)
  and class probability output.
- Upload validation: extension, 2 MB cap, content type, binary detection, 500 row
  cap, column alignment.
- Optional Convex integration (`convex/` plus `app/backend.py`) with a SQLite
  fallback for dataset metadata, model versions and prediction history.
- Optional Garage S3 integration (`app/storage.py`) with a local filesystem
  fallback for uploads and artifacts.
- `scripts/reseed-demo.py` to regenerate data, clear uploads and reset history.
- `app/privacy.py` and `scripts/scan_content.py`: content and privacy scan with
  base64 encoded markers, dash detection and identifier header detection.
- `app/artwork.py`: original SVG hero illustration, MIT licensed.
- Test suite: generator, training, prediction, upload rejection, backend, storage,
  privacy and Streamlit page smoke tests.
- Documentation: architecture with mermaid, features, data generation, data and
  privacy, model card, testing and CI, design decisions, assets.
- Docker image serving Streamlit behind nginx on port 80 with a local `/health`.

### Changed

- `Dockerfile` now builds a Streamlit image instead of running Uvicorn directly.
- `README.md` rewritten around the new product, the live demo and the gates.
- CI workflow runs the generator, training, tests and the content scan.

### Removed

- The FastAPI `/api/predict` endpoint, the hand written feature list and the
  `app/static` front end from 1.x.
- The previous `demo_model.joblib` artifact and `scripts/train_model.py`.

## [1.0.0] 2026-11-12

- Initial public proof of concept: FastAPI, a small logistic regression model and
  a single page static front end.
