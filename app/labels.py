"""Visitor facing labels for backend and storage names.

The demo ships with local fallbacks (SQLite for metadata, files on disk for
artifacts). Those raw identifiers are internal details, so every screen shows a
finished looking badge instead. When Convex or Garage S3 is actually
configured, the real product name is fine to show.
"""

from __future__ import annotations

DEMO_STORAGE_LABEL = "Demo storage"
DEMO_STORAGE_HINT = "Local to this demo, resets on a schedule"

CONVEX_LABEL = "Convex"
CONVEX_HINT = "Managed metadata service"
GARAGE_LABEL = "Garage S3"
GARAGE_HINT = "Object storage for artifacts"

CONVEX_BACKENDS = {"convex"}
GARAGE_BACKENDS = {"garage-s3"}


def metadata_badge(backend_name: str) -> tuple[str, str]:
    """Label and hint for the metadata backend shown on screen."""
    if backend_name in CONVEX_BACKENDS:
        return CONVEX_LABEL, CONVEX_HINT
    return DEMO_STORAGE_LABEL, DEMO_STORAGE_HINT


def artifact_badge(store_name: str) -> tuple[str, str]:
    """Label and hint for the artifact store shown on screen."""
    if store_name in GARAGE_BACKENDS:
        return GARAGE_LABEL, GARAGE_HINT
    return DEMO_STORAGE_LABEL, DEMO_STORAGE_HINT


__all__ = [
    "DEMO_STORAGE_HINT",
    "DEMO_STORAGE_LABEL",
    "artifact_badge",
    "metadata_badge",
]
