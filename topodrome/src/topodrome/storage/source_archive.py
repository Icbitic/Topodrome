"""Bounded standard ZIP envelopes for immutable source bundles.

Only stored regular files are accepted. No generic archive extraction, implicit
paths, compression, or new content identity is introduced: the source manifest
and its existing SHA-256 identity remain authoritative.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

from topodrome.source_layout import (
    SOURCE_BUNDLE_FORMAT_VERSION,
    SOURCE_BUNDLE_MANIFEST_NAME,
    SOURCE_BUNDLE_METADATA_DIRECTORY,
)
from topodrome.source_limits import HARD_MAX_SOURCE_FILES, HARD_MAX_SOURCE_PATH_CHARS
from topodrome.storage.base import StorageVerificationError
from topodrome.storage.paths import normalize_logical_path
from topodrome.storage.progress import check_transfer_cancelled

_SOURCE_PATH = re.compile(r"^projects/\.topodrome-sources/([a-f0-9]{64})$")
_MANIFEST = f"{SOURCE_BUNDLE_METADATA_DIRECTORY}/{SOURCE_BUNDLE_MANIFEST_NAME}"


def source_bundle_path(path: str) -> bool:
    return _SOURCE_PATH.fullmatch(path) is not None


def _manifest(content: bytes, name: str, path: str) -> dict[str, str]:
    try:
        manifest = json.loads(content)
        files = manifest["files"]
        if (
            not isinstance(files, dict)
            or not 0 < len(files) <= HARD_MAX_SOURCE_FILES
            or manifest.get("source_ref") != "sha256:" + path.rsplit("/", 1)[-1]
            or manifest.get("format_version")
            != (SOURCE_BUNDLE_FORMAT_VERSION if name == _MANIFEST else None)
        ):
            raise ValueError
        names = []
        for filename, digest in files.items():
            if (
                not isinstance(filename, str)
                or len(filename) > HARD_MAX_SOURCE_PATH_CHARS
                or normalize_logical_path(filename) != filename
                or filename == name
                or filename.split("/", 1)[0] == SOURCE_BUNDLE_METADATA_DIRECTORY
                or not isinstance(digest, str)
                or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            ):
                raise ValueError
            names.append(tuple(filename.split("/")))
        for shorter, longer in pairwise(sorted(names)):
            if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
                raise ValueError
        identity = hashlib.sha256()
        for filename, digest in sorted(files.items()):
            identity.update(filename.encode())
            identity.update(b"\0")
            identity.update(bytes.fromhex(digest))
        if identity.hexdigest() != path.rsplit("/", 1)[-1]:
            raise ValueError
        return files
    except (ValueError, TypeError, KeyError, AttributeError):
        raise StorageVerificationError("source bundle manifest has invalid identity") from None


def pack_source_bundle(path: str, maximum: int, read: Callable[[str, int], bytes]) -> bytes:
    """Read only manifest-listed files, bounding all reads and wire overhead."""
    if not source_bundle_path(path):
        raise ValueError("not an immutable source directory")
    name = _MANIFEST
    try:
        content = read(f"{path}/{name}", maximum)
    except FileNotFoundError:
        name = SOURCE_BUNDLE_MANIFEST_NAME
        content = read(f"{path}/{name}", maximum)
    files = _manifest(content, name, path)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, content)
        total = len(content)
        overhead = 22 + sum(76 + 2 * len(n.encode()) for n in [name, *files])
        for filename, expected in sorted(files.items()):
            # Account conservatively for headers, central directory and UTF-8
            # filenames before opening even one potentially large member.
            remaining = maximum - total - overhead
            if remaining < 0:
                raise OverflowError("source bundle uses bulk transport")
            body = read(f"{path}/{filename}", remaining)
            if hashlib.sha256(body).hexdigest() != expected:
                raise StorageVerificationError("source bundle member failed verification")
            total += len(body)
            archive.writestr(filename, body)
    result = output.getvalue()
    if len(result) > maximum:
        raise OverflowError("source bundle uses bulk transport")
    return result


def unpack_source_bundle(path: str, content: bytes, destination: Path, maximum: int) -> None:
    """Verify the whole envelope before writing into a private staging tree."""
    if not source_bundle_path(path) or len(content) > maximum:
        raise StorageVerificationError("invalid source archive bound")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if not 1 < len(entries) <= HARD_MAX_SOURCE_FILES + 1 or len(names) != len(set(names)):
                raise ValueError
            total = 0
            for entry in entries:
                mode = entry.external_attr >> 16
                if (
                    entry.is_dir()
                    or entry.compress_type != zipfile.ZIP_STORED
                    or entry.file_size != entry.compress_size
                    or entry.flag_bits & 1
                    or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
                    or entry.file_size < 0
                ):
                    raise ValueError
                total += entry.file_size
                if total > maximum:
                    raise ValueError
            name = _MANIFEST if _MANIFEST in names else SOURCE_BUNDLE_MANIFEST_NAME
            manifest = archive.read(name)
            files = _manifest(manifest, name, path)
            if set(names) != {name, *files}:
                raise ValueError
            bodies = {name: manifest}
            for filename, expected in files.items():
                check_transfer_cancelled()
                body = archive.read(filename)
                if hashlib.sha256(body).hexdigest() != expected:
                    raise ValueError
                bodies[filename] = body
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, NotImplementedError):
        raise StorageVerificationError("source archive failed verification") from None
    # The caller owns this newly-created private directory and publishes it
    # with one atomic rename only after this function completes and fsyncs.
    for filename, body in bodies.items():
        check_transfer_cancelled()
        target = destination.joinpath(*filename.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    directories = [p for p in destination.rglob("*") if p.is_dir()]
    for directory in [*sorted(directories, key=lambda p: len(p.parts), reverse=True), destination]:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
