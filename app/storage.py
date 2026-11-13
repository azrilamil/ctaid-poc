"""Object storage for the demo: local files first, Garage S3 when configured.

Garage speaks the S3 API, so boto3 works unchanged. Credentials come from the
environment only. When they are absent everything still works against
``var/objects`` on the local filesystem, which keeps the demo runnable offline.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import ObjectStoreConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StoredObject:
    key: str
    size: int
    location: str
    backend: str
    etag: str = ""


class LocalObjectStore:
    backend = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key or key.strip("/") == "":
            raise ValueError("Object key must not be empty")
        parts = Path(key).parts
        if any(part in {"..", "."} for part in parts):
            raise ValueError("Refusing to use a relative path segment in an object key")
        target = (self.root / key.lstrip("/")).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError("Refusing to write outside the object store")
        return target

    def put_bytes(self, key: str, payload: bytes, content_type: str = "application/octet-stream") -> StoredObject:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return StoredObject(
            key=key,
            size=len(payload),
            location=str(target.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in target.parents else str(target),
            backend=self.backend,
            etag=hashlib.sha256(payload).hexdigest()[:16],
        )

    def put_file(self, key: str, source: Path, content_type: str = "application/octet-stream") -> StoredObject:
        return self.put_bytes(key, Path(source).read_bytes(), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()

    def clear(self) -> int:
        removed = 0
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        return removed


class S3ObjectStore:
    backend = "garage-s3"

    def __init__(self, config: ObjectStoreConfig, fallback_root: Path) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        self.config = config
        self.fallback = LocalObjectStore(fallback_root)
        self.client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            region_name=config.region,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 2}),
        )

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.config.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.config.bucket)

    def put_bytes(self, key: str, payload: bytes, content_type: str = "application/octet-stream") -> StoredObject:
        try:
            self._ensure_bucket()
            self.client.put_object(Bucket=self.config.bucket, Key=key, Body=payload, ContentType=content_type)
            response = self.client.head_object(Bucket=self.config.bucket, Key=key)
            return StoredObject(
                key=key,
                size=len(payload),
                location=f"s3://{self.config.bucket}/{key}",
                backend=self.backend,
                etag=str(response.get("ETag", "")).strip('"')[:16],
            )
        except Exception:
            # A broken bucket must never take the demo down.
            stored = self.fallback.put_bytes(key, payload, content_type)
            stored.backend = "local-fallback"
            return stored

    def put_file(self, key: str, source: Path, content_type: str = "application/octet-stream") -> StoredObject:
        return self.put_bytes(key, Path(source).read_bytes(), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.config.bucket, Key=key)
            return response["Body"].read()
        except Exception:
            return self.fallback.get_bytes(key)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.config.bucket, Key=key)
            return True
        except Exception:
            return self.fallback.exists(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        try:
            response = self.client.list_objects_v2(Bucket=self.config.bucket, Prefix=prefix)
            return sorted(item["Key"] for item in response.get("Contents", []))
        except Exception:
            return self.fallback.list_keys(prefix)

    def delete(self, key: str) -> None:
        # A missing cloud object is not an error for the demo reset path.
        with contextlib.suppress(Exception):
            self.client.delete_object(Bucket=self.config.bucket, Key=key)
        self.fallback.delete(key)

    def clear(self) -> int:
        removed = 0
        for key in self.list_keys():
            self.delete(key)
            removed += 1
        removed += self.fallback.clear()
        return removed


def build_store(config: ObjectStoreConfig, local_root: Path):
    """Return the S3 store when configured, otherwise the local one."""
    local_root = Path(local_root)
    if config.enabled:
        try:
            return S3ObjectStore(config, local_root)
        except Exception:
            return LocalObjectStore(local_root)
    return LocalObjectStore(local_root)
