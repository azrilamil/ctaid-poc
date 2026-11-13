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



def storage_caption(metadata_label: str, storage_label: str) -> str | None:
    """Visitor facing caption under the storage cards, or None if none is needed."""
    if storage_label == GARAGE_LABEL and metadata_label == CONVEX_LABEL:
        return (
            "Metadata is stored in Convex and clears on the scheduled reseed. "
            "Trained artifacts and uploads are stored in Garage S3."
        )
    if storage_label == GARAGE_LABEL and metadata_label == DEMO_STORAGE_LABEL:
        return (
            "Metadata is kept on this deployment and clears on the scheduled reseed. "
            "Trained artifacts and uploads are stored in Garage S3."
        )
    if storage_label == GARAGE_LABEL:
        return "Trained artifacts and uploads are stored in Garage S3."
    if DEMO_STORAGE_LABEL in (metadata_label, storage_label):
        return (
            "Demo storage is local to this deployment and is cleared by the scheduled reseed, "
            "so nothing here is a permanent record."
        )
    return None


__all__ = [
    "DEMO_STORAGE_HINT",
    "DEMO_STORAGE_LABEL",
    "GARAGE_LABEL",
    "artifact_badge",
    "metadata_badge",
    "storage_caption",
]
