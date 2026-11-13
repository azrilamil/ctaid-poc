# Operations

Short notes for anyone running or inspecting the public CTAID demo.

## Live demo

The site is at [https://ctaid-poc.azril.io](https://ctaid-poc.azril.io). Source is this repository. The live demo talks to a dedicated Convex deployment at [https://convex-demo.azril.io](https://convex-demo.azril.io) and a dedicated Garage S3 bucket, used only for these public portfolio apps and never for any private deployment.

## Demo data reset

A Convex cron runs every day at 03:20 UTC (11:20 Malaysia Time). It clears prediction history and model-version rows. A fuller reset, including generated CSV and local files, is `python scripts/reseed-demo.py`. Visitors only ever see synthetic data.

## Backends

- Dataset metadata, model-version rows and the prediction log are stored in Convex when `CONVEX_URL` is set. The Model Training page then shows a Convex badge.
- Trained artifacts and uploads are stored in Garage S3 when the S3 environment is set. The page then shows a Garage S3 badge, with a caption that names both backends.
- If Convex or object storage is unavailable, the app falls back to SQLite and the local filesystem, and the badges read "Demo storage".

## What is not stored

There are no user accounts, no partner names, no clinical questionnaire items and no real participant or patient data. The bundled training CSV is synthetic. Uploads are not shared between visitors.

## Deploy

The live site is deployed from the `main` branch. Do a normal merge and a normal Coolify deploy. Do not force-push. Scan for secrets before every push.
