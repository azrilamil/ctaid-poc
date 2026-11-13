# Design decisions

Short notes on the choices a reviewer is most likely to question.

## Streamlit, not a new framework

The original research app was Streamlit. Rebuilding on the same stack keeps the
information architecture, the wide layout and the sidebar navigation honest, and
it shows that a Streamlit app can look deliberate rather than default. The cost
is Streamlit's rerun model, which the code handles by caching the data frame and
keeping fitted pipelines in session state.

## Dark theme by hand, not by config alone

`.streamlit/config.toml` sets the dark base and the palette, but the real look
comes from `app/theme.py`: one CSS block with cards, a stat component, the notice
band, hero framing, buttons, tabs, metrics, uploader and expanders. Charts use a
matching Plotly palette in `app/charts.py`. Nothing in the UI relies on Streamlit
default surfaces.

## Synthetic data generated, not sampled

The brief shipped a schema with names and ranges and no rows. Rather than
approximating a table, the project owns a generator:

- one latent class per record, shared by its three vessels
- log normal draws for positive texture features, normal draws for signed ones
- a small per class effect so models can learn something
- fixed seed, so the committed CSV is reproducible

Every test asserts ranges, dtypes and determinism, so the generator cannot drift
silently.

## Neutral class labels

`Pattern A` to `Pattern D` instead of clinical vocabulary. The demo never claims
a class means plaque of a certain type. Helper descriptions exist, but they
describe the synthetic shape and are labelled as invented.

## Real training in the UI

The Train button fits scikit-learn pipelines on the loaded frame. Metrics come
from the held out split that run produced. Artifacts are written with joblib and
a JSON sidecar of metrics so a reviewer can inspect what was trained, when, and
on what.

The row cap exists so a demo click completes in a couple of seconds on a small
host; it never fabricates results, it just uses a subsample.

## Optional cloud, mandatory local fallback

Convex and Garage S3 are wired for real but are not required:

- `CONVEX_URL` set: dataset metadata, model versions and prediction history go
  through the Convex HTTP API from Python, with per call fallback to SQLite.
- `S3_ENDPOINT` set: uploads and artifacts go to the bucket, with per call
  fallback to `var/objects`.

Each stored row and each object records which backend served it, so a reviewer
can see whether the cloud path actually ran.

## SQLite rather than a fake in-memory store

A filesystem or in-memory history would have hidden schema problems. SQLite gives
real tables, a real reset and a real query path, with the same three collections
Convex holds. The Convex client talks to the same shapes, so switching backends
does not change page code.

## Upload limits, enforced in code and in config

Extension, size (2 MB), content type, binary detection, row cap (500) and column
alignment are checked in `app/predict.py`, and Streamlit's own upload limit is set
to 2 MB in `.streamlit/config.toml`. Tests cover each rejection path plus the
happy path.

## nginx in front of Streamlit

The previous image ran Uvicorn on port 80. Streamlit needs a websocket route and
a cheap health surface, so nginx answers `/health` and `/healthz` directly and
proxies everything else, including the `/_stcore/stream` upgrade. The Coolify app
layout (port 80, container health check) is unchanged.

## Privacy scanning as code, not as a checklist

`app/privacy.py` is a small scanner over the public tree, and
`tests/test_privacy.py` runs it in CI. The forbidden markers are base64 encoded
so the repository does not carry the strings it is meant to reject. The scanner
also enforces `Record ID` as the CSV identifier column and rejects forbidden
dashes.

## Hero asset generated in code

`app/artwork.py` builds an SVG of lungs, heart, a vessel trace and CT rings. It
carries a `<desc>` stating that it is generated and not a scan of a person. The
license is the repository MIT license, recorded in `docs/assets.md`. No stock
photo, no third party asset, nothing to attribute.

## What was deliberately left out

- No user accounts, no auth, no multi-tenant state. This is a demo.
- No image model: the feature table is the whole input, as in the original tool.
- No hyperparameter search in the UI. Four fixed pipelines keep the demo fast and
  the comparison fair.
- No CV in the UI. Held out metrics are enough for a demo, and the model card
  states the limitation.
