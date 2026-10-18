"""One verified, resumable transfer engine for job and asset references."""

import asyncio
import fcntl
import hashlib
import os
import shutil
import stat
import threading
import time
from contextlib import suppress
from functools import partial
from pathlib import Path

import httpx

from topodrome.domain.downloads import DownloadManifest

from .asset_download import CHUNK_BYTES, AssetDownload, download_io

RENEW_SECONDS = 60
RENEW_FAILURE_SECONDS = 780  # Stop before the 15-minute server reference expires.


async def resource_io(operation, cleanup, cancel=None):
    """Cancellation must close resources produced by an already running thread."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if cancel:
            cancel.set()
        while not task.done():
            with suppress(BaseException):
                await asyncio.shield(task)
        try:
            result = task.result()
        except BaseException:  # noqa: S110 - Preserve cancellation after an aborted operation.
            pass
        else:
            await download_io(partial(cleanup, result))
        raise


class SyncLease:
    def __init__(self, http, identity, raise_response):
        self.http, self.identity, self.raise_response = http, identity, raise_response
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.keepalive, daemon=True)

    def keepalive(self):
        last = time.monotonic()
        while not self.stopped.wait(RENEW_SECONDS):
            try:
                response = self.http.post(f"/api/v1/downloads/{self.identity}/renew", timeout=30)
                self.raise_response(response)
                last = time.monotonic()
            except httpx.TransportError as exc:
                if time.monotonic() - last > RENEW_FAILURE_SECONDS:
                    self.error = exc
                    return
            except Exception as exc:
                self.error = exc
                return

    def check(self):
        if self.error is not None:
            raise self.error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()


async def keepalive(http, identity, raise_response):
    last = time.monotonic()
    while True:
        await asyncio.sleep(RENEW_SECONDS)
        try:
            response = await http.post(f"/api/v1/downloads/{identity}/renew", timeout=30)
            raise_response(response)
            last = time.monotonic()
        except httpx.TransportError:
            if time.monotonic() - last > RENEW_FAILURE_SECONDS:
                raise


class ResumableDownload(AssetDownload):
    def __init__(self, manifest, destination, origin):
        self.validate_manifest(manifest)
        self.asset = manifest
        self.destination = Path(destination).expanduser().absolute()
        if os.path.lexists(self.destination):
            raise FileExistsError(f"destination already exists: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(str(self.destination).encode()).hexdigest()[:24]
        self.stage = self.destination.parent / f".topodrome-download-{key}"
        self.stage.mkdir(mode=0o700, exist_ok=True)
        metadata = self.stage.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise ValueError("unsafe download staging directory")
        self.lock = os.open(self.stage / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = (
                hashlib.sha256(
                    (
                        origin + "\n" + manifest.ref + "\n" + manifest.kind + "\n" + manifest.digest
                    ).encode()
                )
                .hexdigest()
                .encode()
            )
            marker = self.stage / "identity"
            if marker.exists() or marker.is_symlink():
                fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as handle:
                    if handle.read(129) != identity:
                        raise ValueError("partial download belongs to different content or server")
            else:
                if set(p.name for p in self.stage.iterdir()) != {"lock"}:
                    raise ValueError("download staging identity is missing; inspect its contents")
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(identity)
                    handle.flush()
                    os.fsync(handle.fileno())
            self.payload = self.stage / "payload"
            if manifest.kind == "directory":
                self.payload.mkdir(mode=0o700, exist_ok=True)
                if not stat.S_ISDIR(self.payload.lstat().st_mode):
                    raise ValueError("unsafe download payload")
            self.committed = False
        except BaseException:
            os.close(self.lock)
            raise

    def __exit__(self, *_):
        try:
            if self.committed:
                shutil.rmtree(self.stage)
        finally:
            os.close(self.lock)

    def open_partial(self, item, check=lambda: None):
        check()
        path = self.payload / item.relative_path if self.asset.kind == "directory" else self.payload
        parent = path.parent
        while parent != self.stage:
            if parent.is_symlink():
                raise ValueError("symlink in download staging")
            parent = parent.parent
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1:
            os.close(fd)
            raise ValueError("unsafe partial download file")
        output = os.fdopen(fd, "r+b")
        digest = hashlib.sha256()
        count = 0
        try:
            if meta.st_size <= item.size_bytes:
                while block := output.read(CHUNK_BYTES):
                    check()
                    digest.update(block)
                    count += len(block)
            if meta.st_size > item.size_bytes or (
                count == item.size_bytes and digest.hexdigest() != item.sha256
            ):
                output.seek(0)
                output.truncate(0)
                count, digest = 0, hashlib.sha256()
            output.seek(count)
            return output, count, digest
        except BaseException:
            output.close()
            raise

    def verify(self, output, item, count, digest):
        try:
            self.verify_file(output, item, count, digest.hexdigest())
        except ValueError:
            output.seek(0)
            output.truncate(0)
            raise

    def commit(self):
        if self.asset.kind == "directory":
            expected = {item.relative_path for item in self.asset.files}
            found = set()
            for directory, dirs, files in os.walk(self.payload, followlinks=False):
                for name in dirs + files:
                    path = Path(directory) / name
                    if path.is_symlink():
                        raise ValueError("symlink in download staging")
                found.update(
                    str((Path(directory) / name).relative_to(self.payload)) for name in files
                )
            if found != expected:
                raise ValueError("download staging contains unrecognized or missing files")
        result = super().commit()
        self.committed = True
        return result


def check_response(response, item, offset, raise_response):
    raise_response(response)
    if response.headers.get("ETag") != f'"{item.sha256}"':
        raise ValueError("download content identity changed")
    if offset:
        expected = f"bytes {offset}-{item.size_bytes - 1}/{item.size_bytes}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("server did not honor the download resume range")
    elif response.status_code != 200:
        raise ValueError("unexpected download response")
    if int(response.headers.get("Content-Length", "-1")) != item.size_bytes - offset:
        raise ValueError("download response length does not match manifest")


def headers(item, count):
    result = {"Accept-Encoding": "identity", "If-Match": f'"{item.sha256}"'}
    if count:
        result["Range"] = f"bytes={count}-"
    return result


def download_sync(http, reference, destination, raise_response):
    response = http.post("/api/v1/downloads", json={"reference": reference})
    raise_response(response)
    manifest = DownloadManifest.model_validate(response.json())
    try:
        with (
            SyncLease(http, manifest.download_id, raise_response) as lease,
            ResumableDownload(manifest, destination, str(http.base_url)) as download,
        ):
            for index, item in enumerate(manifest.files):
                for retry in range(3):
                    try:
                        output, count, digest = download.open_partial(item, lease.check)
                        with output:
                            response = http.post(f"/api/v1/downloads/{manifest.download_id}/renew")
                            raise_response(response)
                            if count < item.size_bytes:
                                with http.stream(
                                    "GET",
                                    f"/api/v1/downloads/{manifest.download_id}/files/{index}",
                                    headers=headers(item, count),
                                    timeout=600,
                                ) as response:
                                    if not response.is_success:
                                        response.read()
                                    check_response(response, item, count, raise_response)
                                    for block in response.iter_bytes(chunk_size=CHUNK_BYTES):
                                        lease.check()
                                        count += len(block)
                                        if count > item.size_bytes:
                                            raise ValueError("download exceeds manifest size")
                                        output.write(block)
                                        digest.update(block)
                            download.verify(output, item, count, digest)
                        break
                    except httpx.TransportError:
                        if retry == 2:
                            raise
                        time.sleep(retry + 1)
            lease.check()
            return download.commit()
    finally:
        with suppress(Exception):
            http.delete(f"/api/v1/downloads/{manifest.download_id}")


async def download_async(http, reference, destination, raise_response):
    response = await http.post("/api/v1/downloads", json={"reference": reference})
    raise_response(response)
    manifest = DownloadManifest.model_validate(response.json())
    keeper = asyncio.create_task(keepalive(http, manifest.download_id, raise_response))
    canceled = threading.Event()

    def check():
        if canceled.is_set():
            raise RuntimeError("download canceled")
        if keeper.done():
            keeper.result()

    try:
        download = await resource_io(
            partial(ResumableDownload, manifest, destination, str(http.base_url)),
            lambda result: result.__exit__(None, None, None),
            canceled,
        )
        try:
            for index, item in enumerate(manifest.files):
                for retry in range(3):
                    try:
                        output, count, digest = await resource_io(
                            partial(download.open_partial, item, check),
                            lambda result: result[0].close(),
                            canceled,
                        )
                        with output:
                            response = await http.post(
                                f"/api/v1/downloads/{manifest.download_id}/renew"
                            )
                            raise_response(response)
                            if count < item.size_bytes:
                                async with http.stream(
                                    "GET",
                                    f"/api/v1/downloads/{manifest.download_id}/files/{index}",
                                    headers=headers(item, count),
                                    timeout=600,
                                ) as response:
                                    if not response.is_success:
                                        await response.aread()
                                    check_response(response, item, count, raise_response)
                                    async for block in response.aiter_bytes(chunk_size=CHUNK_BYTES):
                                        check()
                                        count += len(block)
                                        if count > item.size_bytes:
                                            raise ValueError("download exceeds manifest size")
                                        await download_io(partial(output.write, block))
                                        digest.update(block)
                            await download_io(partial(download.verify, output, item, count, digest))
                        break
                    except httpx.TransportError:
                        if retry == 2:
                            raise
                        await asyncio.sleep(retry + 1)
            check()
            return await download_io(download.commit)
        finally:
            await download_io(partial(download.__exit__, None, None, None))
    finally:
        keeper.cancel()
        await asyncio.gather(keeper, return_exceptions=True)
        with suppress(Exception):
            await http.delete(f"/api/v1/downloads/{manifest.download_id}")
