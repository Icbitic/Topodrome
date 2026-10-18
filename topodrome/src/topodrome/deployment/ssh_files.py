"""Atomic, bounded SSH-stdin uploads for administrative installation only.

Experiment data always uses the enrolled HTTP transport. This helper needs only
SSH and remote Python, before the worker's dependencies have been installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import shlex
import stat
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from asyncssh import SSHClientConnection

_CHUNK = 256 * 1024
_RECEIVER = r"""
import hashlib, os, signal, stat, sys, uuid

root, name, size, digest, mode = sys.argv[1:]
size, mode = int(size), int(mode, 8)
directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
temporary = None
try:
    for part in root.strip('/').split('/'):
        try:
            os.mkdir(part, 0o700, dir_fd=directory)
        except FileExistsError:
            pass
        child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory)
        os.close(directory)
        directory = child
    metadata = os.fstat(directory)
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise PermissionError('installation directory must be private and owned')
    try:
        existing = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and (not stat.S_ISREG(existing.st_mode)
                                 or existing.st_uid != os.geteuid()):
        raise PermissionError('unsafe installation target')
    temporary = '.ssh-upload-' + uuid.uuid4().hex
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW, 0o600, dir_fd=directory)
    def interrupted(*_):
        raise InterruptedError('installation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    actual, received = hashlib.sha256(), 0
    with os.fdopen(descriptor, 'wb') as output:
        while True:
            chunk = sys.stdin.buffer.read(256 * 1024)
            if not chunk:
                break
            received += len(chunk)
            if received > size:
                raise ValueError('installation length exceeded')
            output.write(chunk)
            actual.update(chunk)
        if received != size or actual.hexdigest() != digest:
            raise ValueError('installation integrity check failed')
        output.flush()
        os.fchmod(output.fileno(), mode)
        os.fsync(output.fileno())
    os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
    temporary = None
    os.fsync(directory)
    print('TOPODROME_SSH_FILE_OK ' + str(size) + ' ' + digest)
finally:
    if temporary is not None:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
    os.close(directory)
"""


def receiver_command(root: str, name: str, size: int, digest: str, mode: int) -> str:
    path = PurePosixPath(root)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or ".." in path.parts
        or "\0" in root
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        or name in {".", ".."}
    ):
        raise ValueError("installation target must be an exact managed directory and filename")
    if size < 0 or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid installation content identity")
    if mode < 0 or mode & ~0o777:
        raise ValueError("remote file mode must contain only permission bits")
    return shlex.join(
        ["python3", "-c", _RECEIVER, str(path), name, str(size), digest, oct(mode)[2:]]
    )


class _Input:
    """AsyncSSH's async file interface, with disk I/O off the event loop."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.lock = asyncio.Lock()

    async def read(self, size: int) -> bytes:
        async with self.lock:
            reading = asyncio.create_task(asyncio.to_thread(self.stream.read, min(size, _CHUNK)))
            try:
                return await asyncio.shield(reading)
            except asyncio.CancelledError:
                await reading
                raise

    async def close(self) -> None:
        async with self.lock:
            await asyncio.to_thread(self.stream.close)


def _fingerprint(stream: io.BufferedReader) -> tuple[int, str]:
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("installation source must be a regular file")
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("installation source changed during hashing")
    stream.seek(0)
    return before.st_size, digest


async def _upload(
    connection: SSHClientConnection,
    stream: BinaryIO,
    root: str,
    name: str,
    size: int,
    digest: str,
    mode: int,
    timeout_seconds: float,
) -> str:
    command = receiver_command(root, name, size, digest, mode)
    source = _Input(stream)
    try:
        result = await connection.run(
            command,
            stdin=source,
            encoding=None,
            bufsize=_CHUNK,
            check=True,
            timeout=timeout_seconds,
        )
        if result.stdout != f"TOPODROME_SSH_FILE_OK {size} {digest}\n".encode():
            raise RuntimeError("installation acknowledgement missing or invalid")
        return str(PurePosixPath(root) / name)
    finally:
        await source.close()


async def upload_file(
    connection: SSHClientConnection,
    source: Path,
    root: str,
    name: str,
    *,
    mode: int = 0o600,
    timeout_seconds: float = 1200,
) -> str:
    """Hash and stream a pinned regular file; publish only after remote fsync."""
    stream = os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb")
    hashing = asyncio.create_task(asyncio.to_thread(_fingerprint, stream))
    try:
        try:
            size, digest = await asyncio.shield(hashing)
        except asyncio.CancelledError:
            await hashing
            raise
        return await _upload(connection, stream, root, name, size, digest, mode, timeout_seconds)
    finally:
        stream.close()


async def upload_bytes(
    connection: SSHClientConnection,
    content: bytes,
    root: str,
    name: str,
    *,
    mode: int = 0o600,
    timeout_seconds: float = 1200,
) -> str:
    """Stream an already-materialized administrative payload in bounded blocks."""
    with io.BytesIO(content) as stream:
        return await _upload(
            connection,
            stream,
            root,
            name,
            len(content),
            hashlib.sha256(content).hexdigest(),
            mode,
            timeout_seconds,
        )
