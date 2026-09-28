from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from importlib import resources
from itertools import pairwise
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pydantic import Field, field_validator

from topodrome.domain.specs import StrictModel
from topodrome.source_layout import (
    SOURCE_BUNDLE_FORMAT_VERSION,
    SOURCE_BUNDLE_MANIFEST_NAME,
    SOURCE_BUNDLE_METADATA_DIRECTORY,
)
from topodrome.source_limits import (
    HARD_MAX_SOURCE_ENCODED_FILE_CHARS,
    HARD_MAX_SOURCE_FILE_BYTES,
    HARD_MAX_SOURCE_FILES,
    HARD_MAX_SOURCE_PATH_CHARS,
    HARD_MAX_SOURCE_TOTAL_BYTES,
    LEGACY_SOURCE_LIMITS,
    SourceLimits,
)

_SHA256 = frozenset("0123456789abcdef")


class SourceBundle(StrictModel):
    files: dict[str, str] = Field(min_length=1, max_length=HARD_MAX_SOURCE_FILES)

    @field_validator("files")
    @classmethod
    def safe_paths(cls, value: dict[str, str]) -> dict[str, str]:
        total = 0
        logical_parts: list[tuple[str, ...]] = []
        for raw, encoded in value.items():
            path = PurePosixPath(raw)
            if (
                path.is_absolute()
                or ".." in path.parts
                or raw in {"", "."}
                or raw != path.as_posix()
                or len(raw) > HARD_MAX_SOURCE_PATH_CHARS
            ):
                raise ValueError("source paths must be relative")
            if path.parts[0] == SOURCE_BUNDLE_METADATA_DIRECTORY:
                raise ValueError("source path uses Topodrome's reserved metadata namespace")
            logical_parts.append(path.parts)
            if len(encoded) > HARD_MAX_SOURCE_ENCODED_FILE_CHARS:
                raise ValueError("source file exceeds the per-file decoded-size limit")
            try:
                size = len(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError):
                raise ValueError(f"source file {raw} is not valid base64") from None
            if size > HARD_MAX_SOURCE_FILE_BYTES:
                raise ValueError("source file exceeds the per-file decoded-size limit")
            total += size
            if total > HARD_MAX_SOURCE_TOTAL_BYTES:
                raise ValueError("source bundle exceeds the total decoded-size limit")
        logical_parts.sort()
        for shorter, longer in pairwise(logical_parts):
            if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
                raise ValueError("source paths cannot overlap as a file and directory")
        return value


class SourceResult(StrictModel):
    source_ref: str
    file_count: int
    size_bytes: int


class SourceAdmissionError(ValueError):
    """A syntactically valid bundle exceeds the active runtime policy."""


class SourceStore:
    """Content-addressed source store on the authoritative global filesystem."""

    def __init__(
        self,
        global_root: Path,
        *,
        limits_provider: Callable[[], SourceLimits] | None = None,
    ) -> None:
        self.root = global_root / "projects" / ".topodrome-sources"
        self._limits_provider = limits_provider or (lambda: LEGACY_SOURCE_LIMITS)

    def put(self, bundle: SourceBundle) -> SourceResult:
        # Capture exactly one immutable policy generation for the entire
        # upload. A concurrent reload may affect the next upload, never half of
        # this one.
        limits = self._limits_provider()
        if len(bundle.files) > limits.max_files:
            raise SourceAdmissionError(
                f"source bundle exceeds the active {limits.max_files}-file limit"
            )
        decoded: dict[str, bytes] = {}
        digest = hashlib.sha256()
        total = 0
        for logical_path, encoded in sorted(bundle.files.items()):
            if len(logical_path) > limits.max_path_chars:
                raise SourceAdmissionError(
                    "source path exceeds the active path-length limit"
                )
            if len(encoded) > limits.max_encoded_file_chars:
                raise SourceAdmissionError(
                    "source file exceeds the active per-file decoded-size limit"
                )
            try:
                body = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise ValueError(f"source file {logical_path} is not valid base64") from exc
            decoded[logical_path] = body
            if len(body) > limits.max_file_bytes:
                raise SourceAdmissionError(
                    "source file exceeds the active per-file decoded-size limit"
                )
            file_digest = hashlib.sha256(body).digest()
            digest.update(logical_path.encode())
            digest.update(b"\0")
            digest.update(file_digest)
            total += len(body)
            if total > limits.max_total_bytes:
                raise SourceAdmissionError(
                    "source bundle exceeds the active total decoded-size limit"
                )
        source_ref = f"sha256:{digest.hexdigest()}"
        target = self.root / digest.hexdigest()
        if target.exists():
            self._verify_existing(target, source_ref, decoded)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            staging = self.root / f".{digest.hexdigest()}.tmp-{os.getpid()}-{uuid4().hex}"
            staging.mkdir(mode=0o700)
            try:
                for logical_path, body in decoded.items():
                    output = (staging / logical_path).resolve()
                    if not output.is_relative_to(staging.resolve()):
                        raise ValueError("source path escaped source root")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("xb") as handle:
                        handle.write(body)
                        handle.flush()
                        os.fsync(handle.fileno())
                manifest = {
                    "format_version": SOURCE_BUNDLE_FORMAT_VERSION,
                    "source_ref": source_ref,
                    "files": {
                        name: hashlib.sha256(body).hexdigest()
                        for name, body in decoded.items()
                    },
                }
                metadata = staging / SOURCE_BUNDLE_METADATA_DIRECTORY
                metadata.mkdir(mode=0o700)
                manifest_path = metadata / SOURCE_BUNDLE_MANIFEST_NAME
                with manifest_path.open("x", encoding="utf-8") as handle:
                    json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_tree_directories(staging)
                try:
                    staging.rename(target)
                except OSError as exc:
                    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
                    # Another scheduler published this exact content while we
                    # were writing. Its complete manifest must still verify.
                    self._verify_existing(target, source_ref, decoded)
                _fsync_directory(self.root)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        return SourceResult(source_ref=source_ref, file_count=len(decoded), size_bytes=total)

    def ensure_builtin_kernel(self) -> SourceResult:
        """Publish the packaged kernel marker through the normal SHA-256 path."""

        body = resources.files("topodrome.jupyter").joinpath("builtin_kernel.py").read_bytes()
        return self.put(
            SourceBundle(
                files={
                    "__topodrome__/kernel.py": base64.b64encode(body).decode("ascii"),
                }
            )
        )

    def _verify_existing(
        self,
        target: Path,
        source_ref: str,
        decoded: dict[str, bytes],
    ) -> None:
        try:
            manifest_path, current_layout = _source_manifest_path(target)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            raise RuntimeError("existing content-addressed source is incomplete") from None
        _verify_manifest_format(manifest, current_layout=current_layout)
        expected = {name: hashlib.sha256(body).hexdigest() for name, body in decoded.items()}
        if manifest.get("source_ref") != source_ref or files != expected:
            raise RuntimeError("existing content-addressed source has invalid identity")
        for name, expected_digest in expected.items():
            candidate = (target / name).resolve()
            if not candidate.is_relative_to(target.resolve()) or not candidate.is_file():
                raise RuntimeError("existing content-addressed source is incomplete")
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_digest:
                raise RuntimeError("existing content-addressed source failed verification")

    def resolve(self, source_ref: str) -> Path:
        algorithm, separator, digest = source_ref.partition(":")
        if (
            algorithm != "sha256"
            or not separator
            or len(digest) != 64
            or any(character not in _SHA256 for character in digest)
        ):
            raise ValueError("invalid source reference")
        target = self.root / digest
        manifest_path, current_layout = _source_manifest_path(target)
        if not manifest_path.is_file():
            raise FileNotFoundError(source_ref)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            raise RuntimeError("content-addressed source is incomplete") from None
        _verify_manifest_format(manifest, current_layout=current_layout)
        if (
            manifest.get("source_ref") != source_ref
            or not isinstance(files, dict)
            or not files
            or len(files) > HARD_MAX_SOURCE_FILES
        ):
            raise RuntimeError("content-addressed source has invalid identity")
        identity = hashlib.sha256()
        total = 0
        for name, expected_digest in sorted(files.items()):
            if (
                not isinstance(name, str)
                or not isinstance(expected_digest, str)
                or len(name) > HARD_MAX_SOURCE_PATH_CHARS
                or len(expected_digest) != 64
                or any(character not in _SHA256 for character in expected_digest)
            ):
                raise RuntimeError("content-addressed source has invalid manifest")
            logical = PurePosixPath(name)
            candidate = (target / name).resolve()
            if (
                logical.is_absolute()
                or ".." in logical.parts
                or name in {"", "."}
                or name != logical.as_posix()
                or (
                    current_layout
                    and logical.parts[0] == SOURCE_BUNDLE_METADATA_DIRECTORY
                )
                or not candidate.is_relative_to(target.resolve())
                or not candidate.is_file()
            ):
                raise RuntimeError("content-addressed source is incomplete")
            size = candidate.stat().st_size
            total += size
            if (
                size > HARD_MAX_SOURCE_FILE_BYTES
                or total > HARD_MAX_SOURCE_TOTAL_BYTES
            ):
                raise RuntimeError("content-addressed source exceeds source limits")
            actual_digest = _file_sha256(candidate)
            if actual_digest != expected_digest:
                raise RuntimeError("content-addressed source failed verification")
            identity.update(name.encode())
            identity.update(b"\0")
            identity.update(bytes.fromhex(expected_digest))
        if identity.hexdigest() != digest:
            raise RuntimeError("content-addressed source has invalid identity")
        return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest_path(target: Path) -> tuple[Path, bool]:
    current = (
        target / SOURCE_BUNDLE_METADATA_DIRECTORY / SOURCE_BUNDLE_MANIFEST_NAME
    )
    if current.is_file():
        return current, True
    # Format v1 stored Topodrome's private manifest at the source root. Keep it
    # readable so already-persisted source references survive an upgrade.
    return target / SOURCE_BUNDLE_MANIFEST_NAME, False


def _verify_manifest_format(manifest: object, *, current_layout: bool) -> None:
    if not isinstance(manifest, dict):
        raise RuntimeError("content-addressed source has invalid manifest")
    version = manifest.get("format_version")
    if current_layout:
        valid = version == SOURCE_BUNDLE_FORMAT_VERSION
    else:
        valid = version is None
    if not valid:
        raise RuntimeError("content-addressed source has invalid manifest format")
