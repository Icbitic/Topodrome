"""Verification of a site-local Transformers checkpoint before advertising it.

The scheduler must never infer model identity from a filesystem path.  A site
does, however, need to prove that a configured path is a complete checkout of
the exact Hub revision it advertises.  Hugging Face ``local_dir`` downloads
leave a revision-scoped tree manifest which gives us that binding without
putting the private path into a heartbeat.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ModelPayloadVerificationError(RuntimeError):
    """The configured model directory is incomplete or has the wrong identity."""


@dataclass(frozen=True, slots=True)
class VerifiedModelPayload:
    revision: str
    weight_files: tuple[str, ...]
    weight_bytes: int
    identity_digest: str


_EXACT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
)
_MAX_EXTERNAL_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_FILES = 100_000
_MAX_LOGICAL_PATH_BYTES = 4096
_MANIFEST_ENTRY_KEYS = {
    "size",
    "sha256",
    "lfs_sha256",
    "lfs_size",
    "blob_id",
}


def record_huggingface_local_dir_manifest(
    path: Path | str,
    *,
    revision: str,
) -> Path:
    """Record exact Hub metadata emitted by ``snapshot_download(local_dir=...)``.

    Hugging Face writes a small metadata sidecar for every materialized file.
    Its first two lines bind that file to the resolved commit and its Git/LFS
    object identity.  Converting those sidecars into one atomic tree manifest
    lets later offline site startups verify the configured revision without a
    Hub request or access token.
    """

    if not _EXACT_REVISION.fullmatch(revision):
        raise ModelPayloadVerificationError(
            "Hugging Face acquisition requires an exact commit revision"
        )
    root = Path(path).expanduser().resolve()
    metadata_root = root / ".cache" / "huggingface" / "download"
    if not root.is_dir() or not metadata_root.is_dir():
        raise ModelPayloadVerificationError("Hugging Face local-dir metadata is unavailable")

    files: dict[str, dict[str, Any]] = {}
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.is_relative_to(root / ".cache"):
            continue
        logical = candidate.relative_to(root).as_posix()
        metadata = metadata_root / f"{logical}.metadata"
        try:
            lines = metadata.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            raise ModelPayloadVerificationError(
                "a downloaded model file has no trustworthy Hub metadata"
            ) from None
        if len(lines) < 2 or lines[0] != revision:
            raise ModelPayloadVerificationError(
                "a downloaded model file belongs to a different Hub revision"
            )
        object_id = lines[1].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id):
            raise ModelPayloadVerificationError("Hugging Face object metadata is invalid")
        size = candidate.stat().st_size
        entry: dict[str, Any] = {"size": size}
        if len(object_id) == 64:
            entry.update({"lfs_sha256": object_id, "lfs_size": size})
        else:
            entry["blob_id"] = object_id
        files[logical] = entry
    if not files:
        raise ModelPayloadVerificationError("downloaded model tree is empty")

    destination = root / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"format_version": 1, "revision": revision, "files": files},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def verify_transformers_payload(
    path: Path | str,
    *,
    revision: str,
    verify_lfs_sha256: bool = False,
    revision_manifest: Path | str | None = None,
) -> VerifiedModelPayload:
    """Verify completeness and, for commit revisions, Hub revision identity.

    Byte sizes alone cannot detect same-size corruption or tampering. Production
    site registration and acquisition enable full hashing; callers may disable
    it only for a separately attested payload or a lightweight diagnostic. An
    external revision manifest is an operator-managed trust anchor and must be
    protected independently from the model tree it attests.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ModelPayloadVerificationError("model directory is unavailable")
    _read_json_object(root / "config.json", "model config")
    tokenizer = next(
        (name for name in _TOKENIZER_FILES if _regular_nonempty(root, name)),
        None,
    )
    if tokenizer is None:
        raise ModelPayloadVerificationError("model tokenizer payload is missing")

    index_name = next((name for name in _INDEX_NAMES if (root / name).is_file()), None)
    required_files = {"config.json", tokenizer}
    if index_name is not None:
        index = _read_json_object(root / index_name, "model weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelPayloadVerificationError("model weight index has no weight map")
        shards = {_safe_relative(str(value)) for value in weight_map.values()}
        required_files.add(index_name)
    else:
        shards = {
            file.name
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for file in root.glob(pattern)
            if file.is_file() and file.stat().st_size > 0
        }
    if not shards:
        raise ModelPayloadVerificationError("model weight payload is missing")
    required_files.update(shards)

    sizes: dict[str, int] = {}
    for name in sorted(required_files):
        candidate = _resolve_member(root, name)
        if not candidate.is_file():
            raise ModelPayloadVerificationError("model payload is missing a required file")
        size = candidate.stat().st_size
        if size <= 0:
            raise ModelPayloadVerificationError("model payload contains an empty required file")
        sizes[name] = size

    tree_files: dict[str, Any] | None = None
    external_manifest = revision_manifest is not None
    if _EXACT_REVISION.fullmatch(revision):
        tree = _read_revision_manifest(
            root,
            revision=revision,
            external_path=revision_manifest,
        )
        tree_files = tree["files"]
        for name, observed_size in sizes.items():
            entry = tree_files.get(name)
            if not isinstance(entry, dict) or int(entry.get("size", -1)) != observed_size:
                raise ModelPayloadVerificationError(
                    "model file does not match the configured Hub revision"
                )

    elif revision_manifest is not None:
        raise ModelPayloadVerificationError(
            "an external revision manifest requires an exact Hub commit revision"
        )

    if verify_lfs_sha256:
        if tree_files is None:
            raise ModelPayloadVerificationError(
                "full model hashing requires an exact Hub commit revision"
            )
        names_to_hash = sorted(required_files if external_manifest else shards)
        for name in names_to_hash:
            entry = tree_files[name]
            if not isinstance(entry, dict):
                raise ModelPayloadVerificationError("revision manifest entry is invalid")
            _verify_manifest_content_id(
                _resolve_member(root, name),
                entry,
                require_any=external_manifest,
                require_lfs=name in shards and not external_manifest,
            )

    identity_files: dict[str, dict[str, Any]] = {}
    for name in sorted(required_files):
        identity_entry: dict[str, Any] = {"size": sizes[name]}
        if tree_files is not None:
            tree_entry = tree_files.get(name)
            expected_hash = (
                tree_entry.get("lfs_sha256") if isinstance(tree_entry, dict) else None
            )
            if isinstance(expected_hash, str):
                identity_entry["sha256"] = expected_hash
            elif isinstance(tree_entry, dict) and isinstance(
                tree_entry.get("sha256"), str
            ):
                identity_entry["sha256"] = tree_entry["sha256"]
            elif isinstance(tree_entry, dict) and isinstance(
                tree_entry.get("blob_id"), str
            ):
                identity_entry["blob_id"] = tree_entry["blob_id"]
        identity_files[name] = identity_entry
    identity = {
        "revision": revision,
        "files": identity_files,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return VerifiedModelPayload(
        revision=revision,
        weight_files=tuple(sorted(shards)),
        weight_bytes=sum(sizes[name] for name in shards),
        identity_digest=digest,
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ModelPayloadVerificationError(f"{label} is unavailable or invalid") from None
    if not isinstance(value, dict):
        raise ModelPayloadVerificationError(f"{label} must be a JSON object")
    return value


def _read_revision_manifest(
    root: Path,
    *,
    revision: str,
    external_path: Path | str | None,
) -> dict[str, Any]:
    if external_path is None:
        value = _read_json_object(
            root / ".cache" / "huggingface" / "trees" / f"{revision}.json",
            "Hugging Face revision manifest",
        )
    else:
        configured = Path(external_path).expanduser()
        if not configured.is_absolute():
            raise ModelPayloadVerificationError(
                "external revision manifest path must be absolute"
            )
        try:
            descriptor = os.open(
                configured,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError:
            raise ModelPayloadVerificationError(
                "external revision manifest is unavailable or unsafe"
            ) from None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid not in {0, os.geteuid()}
                or metadata.st_mode & 0o022
                or metadata.st_size <= 0
                or metadata.st_size > _MAX_EXTERNAL_MANIFEST_BYTES
            ):
                raise ModelPayloadVerificationError(
                    "external revision manifest is unavailable or unsafe"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                content = handle.read(_MAX_EXTERNAL_MANIFEST_BYTES + 1)
            if len(content) != metadata.st_size:
                raise ModelPayloadVerificationError(
                    "external revision manifest changed while it was being read"
                )
            try:
                decoded = json.loads(content)
            except (UnicodeError, json.JSONDecodeError):
                raise ModelPayloadVerificationError(
                    "Hugging Face revision manifest is unavailable or invalid"
                ) from None
            if not isinstance(decoded, dict):
                raise ModelPayloadVerificationError(
                    "Hugging Face revision manifest must be a JSON object"
                )
            value = decoded
        finally:
            os.close(descriptor)

    files = value.get("files")
    if (
        value.get("format_version") != 1
        or not isinstance(files, dict)
        or not files
        or len(files) > _MAX_MANIFEST_FILES
    ):
        raise ModelPayloadVerificationError("Hugging Face revision manifest is invalid")
    for raw_name, entry in files.items():
        if not isinstance(raw_name, str) or not isinstance(entry, dict):
            raise ModelPayloadVerificationError("revision manifest entry is invalid")
        _safe_relative(raw_name)
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ModelPayloadVerificationError("revision manifest entry is invalid")
    return value


def _verify_manifest_content_id(
    path: Path,
    entry: dict[str, Any],
    *,
    require_any: bool,
    require_lfs: bool,
) -> None:
    lfs_sha256 = entry.get("lfs_sha256")
    sha256 = entry.get("sha256")
    blob_id = entry.get("blob_id")
    if isinstance(lfs_sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", lfs_sha256):
        if _sha256(path) != lfs_sha256.casefold():
            raise ModelPayloadVerificationError("model shard failed its LFS checksum")
        return
    if require_lfs:
        raise ModelPayloadVerificationError("model shard has no authoritative LFS hash")
    if isinstance(sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        if _sha256(path) != sha256.casefold():
            raise ModelPayloadVerificationError("model file failed its SHA-256 checksum")
        return
    if isinstance(blob_id, str) and re.fullmatch(r"[0-9a-fA-F]{40}", blob_id):
        if _git_blob_sha1(path) != blob_id.casefold():
            raise ModelPayloadVerificationError("model file failed its Git blob checksum")
        return
    if require_any:
        raise ModelPayloadVerificationError(
            "external revision manifest entry has no authoritative content hash"
        )


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_nonempty(root: Path, name: str) -> bool:
    candidate = _resolve_member(root, name)
    return candidate.is_file() and candidate.stat().st_size > 0


def _safe_relative(name: str) -> str:
    logical = PurePosixPath(name)
    if logical.is_absolute() or ".." in logical.parts or name in {"", "."}:
        raise ModelPayloadVerificationError("model index contains an unsafe shard path")
    return logical.as_posix()


def _resolve_member(root: Path, name: str) -> Path:
    logical = _safe_relative(name)
    candidate = (root / Path(*PurePosixPath(logical).parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ModelPayloadVerificationError("model payload escaped its configured directory")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ModelPayloadVerificationError",
    "VerifiedModelPayload",
    "record_huggingface_local_dir_manifest",
    "verify_transformers_payload",
]
