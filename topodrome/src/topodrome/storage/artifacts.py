"""Attempt-scoped staging with a verified manifest as the commit point."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from topodrome.storage.base import ObjectDigest, StorageAdapter
from topodrome.storage.blobs import blob_path, publish_artifact_blob
from topodrome.storage.paths import LogicalPathError, normalize_logical_path, validate_component
from topodrome.storage.progress import TransferObserver, check_transfer_cancelled, observe_transfers


class ArtifactCommitError(RuntimeError):
    """An attempt cannot publish a complete, verified artifact set."""


_MAX_COMMIT_MANIFEST_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CommittedArtifact:
    logical_path: str
    storage_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "logical_path": self.logical_path,
            "storage_path": self.storage_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CommitManifest:
    job_id: str
    attempt_id: str
    artifacts: tuple[CommittedArtifact, ...]
    committed_at: datetime
    schema_version: int = 1

    def to_bytes(self) -> bytes:
        payload = {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "committed_at": self.committed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
        content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(content) > _MAX_COMMIT_MANIFEST_BYTES:
            raise ArtifactCommitError("committed artifact manifest exceeds the metadata limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> CommitManifest:
        if len(content) > _MAX_COMMIT_MANIFEST_BYTES:
            raise ArtifactCommitError("committed artifact manifest exceeds the metadata limit")
        try:
            raw = json.loads(content)
            if not isinstance(raw, Mapping) or raw.get("schema_version") not in (1, 2):
                raise ValueError("unsupported manifest schema")
            job_id = validate_component(str(raw["job_id"]), label="job_id")
            attempt_id = validate_component(str(raw["attempt_id"]), label="attempt_id")
            committed_at = datetime.fromisoformat(str(raw["committed_at"]).replace("Z", "+00:00"))
            if committed_at.tzinfo is None:
                raise ValueError("committed_at must be timezone-aware")
            raw_artifacts = raw["artifacts"]
            if not isinstance(raw_artifacts, list):
                raise ValueError("artifacts must be an array")
            artifacts: list[CommittedArtifact] = []
            seen: set[str] = set()
            for item in raw_artifacts:
                if not isinstance(item, Mapping):
                    raise ValueError("artifact entries must be objects")
                logical_path = normalize_logical_path(str(item["logical_path"]))
                storage_path = normalize_logical_path(str(item["storage_path"]))
                expected_storage = f"jobs/{job_id}/attempts/{attempt_id}/committed/{logical_path}"
                if raw["schema_version"] == 2:
                    expected_storage = blob_path(str(item["sha256"]))
                if storage_path != expected_storage:
                    raise ValueError("artifact storage path does not belong to this attempt")
                if logical_path in seen:
                    raise ValueError("manifest contains duplicate artifact paths")
                seen.add(logical_path)
                digest = ObjectDigest(
                    size_bytes=int(item["size_bytes"]),
                    sha256=str(item["sha256"]),
                )
                artifacts.append(
                    CommittedArtifact(
                        logical_path=logical_path,
                        storage_path=storage_path,
                        size_bytes=digest.size_bytes,
                        sha256=digest.sha256,
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactCommitError("invalid committed artifact manifest") from exc
        return cls(
            job_id=job_id,
            attempt_id=attempt_id,
            artifacts=tuple(artifacts),
            committed_at=committed_at,
            schema_version=raw["schema_version"],
        )

    def artifact(self, logical_path: str) -> CommittedArtifact:
        normalized = normalize_logical_path(logical_path)
        for artifact in self.artifacts:
            if artifact.logical_path == normalized:
                return artifact
        raise KeyError(normalized)


class AttemptArtifactStore:
    """Stages one attempt and publishes only after every required file is durable.

    Attempt outputs are immutable and attempt-specific. The final atomic write
    of ``jobs/<job>/committed/manifest.json`` is the visibility boundary for
    downstream consumers; a failed attempt never publishes that manifest.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        *,
        job_id: str,
        attempt_id: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.storage = storage
        self.job_id = validate_component(job_id, label="job_id")
        self.attempt_id = validate_component(attempt_id, label="attempt_id")
        self._clock = clock

    @property
    def staging_prefix(self) -> str:
        return f"jobs/{self.job_id}/attempts/{self.attempt_id}/staging"

    @property
    def committed_prefix(self) -> str:
        return f"jobs/{self.job_id}/attempts/{self.attempt_id}/committed"

    @property
    def attempt_manifest_path(self) -> str:
        return f"jobs/{self.job_id}/attempts/{self.attempt_id}/manifest.json"

    @property
    def published_manifest_path(self) -> str:
        return f"jobs/{self.job_id}/committed/manifest.json"

    def stage_bytes(self, logical_path: str, content: bytes) -> Path:
        normalized = normalize_logical_path(logical_path)
        target = self.storage.resolve_writable(f"{self.staging_prefix}/{normalized}")
        self._atomic_local_write(target, content)
        return target

    def stage_file(self, source: Path | str, logical_path: str) -> Path:
        normalized = normalize_logical_path(logical_path)
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target = self.storage.resolve_writable(f"{self.staging_prefix}/{normalized}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
        try:
            with source_path.open("rb") as reader, temporary.open("xb") as writer:
                while chunk := reader.read(4 * 1024 * 1024):
                    check_transfer_cancelled()
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def commit(
        self,
        required_paths: Iterable[str],
        *,
        progress: TransferObserver | None = None,
    ) -> CommitManifest:
        normalized_paths = tuple(normalize_logical_path(path) for path in required_paths)
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ArtifactCommitError("required artifact paths must be unique")
        use_blobs = getattr(self.storage, "content_addressed", False)
        if use_blobs:
            previous = self.storage.resolve(self.attempt_manifest_path)
            if previous.is_file():
                saved = CommitManifest.from_bytes(previous.read_bytes())
                if (saved.job_id != self.job_id or saved.attempt_id != self.attempt_id
                        or saved.schema_version != 2
                        or {a.logical_path for a in saved.artifacts} != set(normalized_paths)):
                    raise ArtifactCommitError("immutable attempt manifest conflicts")
                for artifact in saved.artifacts:
                    self.storage.flush_and_verify(artifact.storage_path,
                                                  expected_sha256=artifact.sha256)
                content = saved.to_bytes()
                self._publish_manifest(self.published_manifest_path, content,
                                       hashlib.sha256(content).hexdigest())
                return saved

        sizes: dict[str, int] = {}
        for logical_path in normalized_paths:
            source = self.storage.resolve(f"{self.staging_prefix}/{logical_path}")
            if not source.is_file():
                source = self.storage.resolve(f"{self.committed_prefix}/{logical_path}")
            if not source.is_file():
                raise ArtifactCommitError(f"required artifact is missing: {logical_path}")
            sizes[logical_path] = source.stat().st_size
        total = sum(sizes.values())
        completed = 0

        def report(stage: str, current: int, _size: int) -> None:
            if progress is not None:
                # Telemetry never changes the success/failure of a commit.
                with suppress(Exception):
                    progress(stage, min(total, completed + current), total)

        committed: list[CommittedArtifact] = []
        for logical_path in sorted(normalized_paths):
            check_transfer_cancelled()
            staging_path = self.storage.resolve(f"{self.staging_prefix}/{logical_path}")
            committed_storage_path = f"{self.committed_prefix}/{logical_path}"
            committed_path = self.storage.resolve(committed_storage_path)
            if staging_path.is_file():
                committed_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging_path, committed_path)
            elif not committed_path.is_file():
                raise ArtifactCommitError(f"required artifact is missing: {logical_path}")

            report("verify", 0, sizes[logical_path])
            with observe_transfers(report):
                if use_blobs:
                    verified = publish_artifact_blob(
                        self.storage, committed_path,
                        job_id=self.job_id, attempt_id=self.attempt_id,
                    )
                    committed_storage_path = verified.logical_path
                else:
                    verified = self.storage.flush_and_verify(committed_storage_path)
            completed += verified.digest.size_bytes
            report("verify", 0, 0)
            committed.append(
                CommittedArtifact(
                    logical_path=logical_path,
                    storage_path=committed_storage_path,
                    size_bytes=verified.digest.size_bytes,
                    sha256=verified.digest.sha256,
                )
            )

        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("artifact clock must return a timezone-aware datetime")
        manifest = CommitManifest(
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            artifacts=tuple(committed),
            committed_at=now,
            schema_version=2 if use_blobs else 1,
        )
        content = manifest.to_bytes()
        digest = hashlib.sha256(content).hexdigest()
        report("manifest", 0, 0)
        self._publish_manifest(self.attempt_manifest_path, content, digest)
        self._publish_manifest(self.published_manifest_path, content, digest)
        if use_blobs:
            # Only after both manifests are durable may private output aliases
            # disappear. A retry before this point can still rebuild publication.
            for logical_path in normalized_paths:
                self.storage.resolve(f"{self.committed_prefix}/{logical_path}").unlink(
                    missing_ok=True
                )
        return manifest

    def _publish_manifest(self, logical_path: str, content: bytes, sha256: str) -> None:
        check_transfer_cancelled()
        target = self.storage.resolve_writable(logical_path)
        self._atomic_local_write(target, content)
        self.storage.flush_and_verify(logical_path, expected_sha256=sha256)

    @staticmethod
    def _atomic_local_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def load_published_manifest(storage: StorageAdapter, job_id: str) -> CommitManifest:
    safe_job_id = validate_component(job_id, label="job_id")
    logical_path = f"jobs/{safe_job_id}/committed/manifest.json"
    try:
        path = storage.resolve(logical_path)
        storage_root = getattr(storage, "root", None)
        lexical_path = (
            Path(storage_root).expanduser().resolve().joinpath(*logical_path.split("/"))
            if storage_root is not None
            else path
        )
        metadata = lexical_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactCommitError("committed artifact manifest must be a regular file")
        if metadata.st_size > _MAX_COMMIT_MANIFEST_BYTES:
            raise ArtifactCommitError("committed artifact manifest exceeds the metadata limit")
        content = path.read_bytes()
    except FileNotFoundError as exc:
        raise ArtifactCommitError("job has no committed artifact manifest") from exc
    except LogicalPathError as exc:
        raise ArtifactCommitError("committed artifact manifest must be a regular file") from exc
    manifest = CommitManifest.from_bytes(content)
    if manifest.job_id != safe_job_id:
        raise ArtifactCommitError("published manifest belongs to a different job")
    return manifest


def resolve_committed_artifact(
    storage: StorageAdapter,
    *,
    job_id: str,
    logical_path: str,
) -> Path:
    manifest = load_published_manifest(storage, job_id)
    try:
        artifact = manifest.artifact(logical_path)
    except KeyError as exc:
        raise ArtifactCommitError("artifact is not present in the committed manifest") from exc
    return storage.resolve(artifact.storage_path)


__all__ = [
    "ArtifactCommitError",
    "AttemptArtifactStore",
    "CommitManifest",
    "CommittedArtifact",
    "load_published_manifest",
    "resolve_committed_artifact",
]
