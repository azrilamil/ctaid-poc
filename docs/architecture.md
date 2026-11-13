# Architecture

CTAID POC is a Streamlit application with an optional cloud backplane. It is a
single process, single page router, with no user accounts. Every dependency
below has a local fallback, so the app runs completely offline.

```mermaid
flowchart TD
    subgraph Browser
        U[Visitor]
    end

    subgraph Container["Docker container (Coolify)"]
        N[nginx :80<br/>/health answered locally]
        S["Streamlit :8080<br/>streamlit_app.py"]
        R[Page router]
        H[Home]
        D[Data Exploration]
        T[Model Training]
        P[Prediction]
    end

    subgraph LocalState["Local state (always present)"]
        CSV[("data/synthetic_features.csv<br/>2100 synthetic rows")]
        MODELS[("models/*.joblib<br/>fitted pipelines")]
        SQLITE[("var/ctaid-demo.sqlite3<br/>metadata, versions, predictions")]
        OBJ[("var/objects/<br/>uploads and artifacts")]
    end

    subgraph Cloud["Optional cloud wiring"]
        CONVEX[("Convex<br/>dataset meta, model versions,<br/>prediction history")]
        GARAGE[("Garage S3<br/>bucket ctaid-poc-demo<br/>uploads and artifacts")]
    end

    U --> N --> S --> R
    R --> H
    R --> D
    R --> T
    R --> P

    D --> CSV
    T --> CSV
    T --> MODELS
    P --> MODELS
    P --> CSV

    T -->|CONVEX_URL set| CONVEX
    T -->|unset| SQLITE
    P -->|CONVEX_URL set| CONVEX
    P -->|unset| SQLITE

    P -->|S3_ENDPOINT set| GARAGE
    P -->|unset| OBJ
    T -->|S3_ENDPOINT set| GARAGE
    T -->|unset| OBJ
```

## Request path

1. nginx answers `/health` and `/healthz` without touching Python, which keeps
   the Coolify health check fast and independent of Streamlit start up.
2. Everything else is proxied to Streamlit with the websocket upgrade headers
   on `/_stcore/stream`.
3. `streamlit_app.py` applies the theme, registers dataset metadata once per
   process, renders the sidebar and dispatches to one page module.
4. Each page module reads the synthetic CSV through `app/state.py`, which caches
   the frame on path plus modification time.

## Modules

| Module | Responsibility |
| --- | --- |
| `streamlit_app.py` | Entry point, theme, sidebar router |
| `app/theme.py` | Palette, CSS, cards, notices, page headers |
| `app/schema.py` | Column names, dtypes and sampling ranges |
| `app/data.py` | Load, summarise, correlate, sample the CSV |
| `app/training.py` | Model registry, pipelines, metrics, artifacts |
| `app/predict.py` | Upload validation, feature alignment, scoring |
| `app/charts.py` | Plotly figures on the project palette |
| `app/backend.py` | Convex HTTP client with a SQLite fallback |
| `app/storage.py` | Garage S3 client with a local filesystem fallback |
| `app/artwork.py` | Generates the original hero SVG |
| `app/privacy.py` | Privacy and content scan helpers |
| `app/bootstrap.py` | Registers dataset metadata at start up |

## Degradation rules

- `CONVEX_URL` unset or unreachable: SQLite in `var/` serves the same calls, and
  each row is tagged with the backend that answered.
- `S3_ENDPOINT` unset or unreachable: uploads and artifacts go to `var/objects`.
- Dataset missing: `app/data.py` regenerates it from the fixed seed.
- Model artifact missing: the Prediction page asks the visitor to train first
  rather than inventing a score.

## Reset flow

```mermaid
sequenceDiagram
    participant Cron
    participant Script as scripts/reseed-demo.py
    participant Gen as generate_synthetic_data.py
    participant Store as Garage S3 or var/objects
    participant DB as Convex or SQLite

    Cron->>Script: daily at 03:15
    Script->>Gen: build_frame(seed 20261113)
    Gen-->>Script: 2100 rows
    Script->>DB: resetDemo() / DELETE history
    Script->>Store: clear uploads and artifacts
    Script->>DB: saveDatasetMeta(digest)
    Script-->>Cron: exit 0
```
