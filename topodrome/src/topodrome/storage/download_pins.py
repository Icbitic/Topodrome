"""Short-lived download references, serialized by the scheduler storage lock."""

import os
import re
import stat
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from topodrome.domain.downloads import DownloadFile, DownloadManifest
from topodrome.domain.errors import AdmissionRejectedError, NotFoundError
from topodrome.domain.specs import AssetMember, StrictModel, utc_now

TTL_SECONDS = 900
MAX_LIFETIME_SECONDS = 86400
MAX_METADATA_BYTES = 16 * 1024**2
_ID = re.compile(r"^download_[a-f0-9]{32}$")


class DownloadPin(StrictModel):
    owner: str
    created_at: float
    expires_at: float
    manifest: DownloadManifest
    members: tuple[AssetMember, ...] = Field(min_length=1, max_length=10000)


class DownloadPins:
    def __init__(self, root, *, clock=None):
        self.root = Path(root) / ".topodrome-downloads"
        self.clock = clock or utc_now

    def _path(self, identity):
        if not _ID.fullmatch(identity):
            raise NotFoundError("download does not exist or has expired")
        if self.root.is_symlink():
            raise ValueError("unsafe download directory")
        return self.root / f"{identity}.json"

    def read(self, identity):
        path = self._path(identity)
        meta = path.lstat()
        if not stat.S_ISREG(meta.st_mode) or meta.st_size > MAX_METADATA_BYTES:
            raise ValueError("invalid download reference")
        return DownloadPin.model_validate_json(path.read_bytes())

    def records(self):
        if self.root.is_symlink():
            raise ValueError("unsafe download directory")
        if not self.root.exists():
            return
        for path in self.root.glob("download_*.json"):
            yield path, self.read(path.stem)

    def write(self, pin):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self._path(pin.manifest.download_id)
        body = pin.model_dump_json().encode()
        if len(body) > MAX_METADATA_BYTES:
            raise AdmissionRejectedError("download metadata is too large")
        temporary = self.root / f".{uuid4().hex}.tmp"
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, resolved, owner):
        now = self.clock()
        active = []
        for path, pin in self.records():
            if pin.expires_at <= now.timestamp():
                path.unlink()
            else:
                active.append(pin)
        if len(active) >= 128 or sum(p.owner == owner for p in active) >= 16:
            raise AdmissionRejectedError("too many active downloads; finish one or wait for expiry")
        expires = now + timedelta(seconds=TTL_SECONDS)
        manifest = DownloadManifest(
            download_id="download_" + uuid4().hex,
            ref=resolved.source_ref,
            kind=resolved.kind,
            digest=resolved.digest,
            total_size_bytes=resolved.total_bytes,
            files=tuple(
                DownloadFile(
                    relative_path=m.relative_path, size_bytes=m.size_bytes, sha256=m.sha256
                )
                for m in resolved.members
            ),
            expires_at=expires,
        )
        pin = DownloadPin(
            owner=owner,
            created_at=now.timestamp(),
            expires_at=expires.timestamp(),
            manifest=manifest,
            members=tuple(
                AssetMember(
                    logical_path=m.logical_path,
                    relative_path=m.relative_path,
                    size_bytes=m.size_bytes,
                    sha256=m.sha256,
                )
                for m in resolved.members
            ),
        )
        self.write(pin)
        return manifest

    def get(self, identity, owner, *, renew=False):
        try:
            pin = self.read(identity)
        except FileNotFoundError:
            raise NotFoundError("download does not exist or has expired") from None
        now = self.clock()
        if pin.owner != owner or pin.expires_at <= now.timestamp():
            raise NotFoundError("download does not exist or has expired")
        if renew:
            expires = min(now.timestamp() + TTL_SECONDS, pin.created_at + MAX_LIFETIME_SECONDS)
            if expires <= now.timestamp():
                raise NotFoundError("download lifetime ended; start a new download")
            # Many small member requests must not fsync the entire descriptor
            # twice per file. Minute keepalives still renew well before expiry.
            if expires - pin.expires_at >= 30:
                pin.expires_at = expires
                pin.manifest.expires_at = datetime.fromtimestamp(expires, tz=now.tzinfo)
                self.write(pin)
        return pin

    def delete(self, identity, owner):
        try:
            pin = self.read(identity)
        except FileNotFoundError:
            return
        if pin.owner != owner:
            raise NotFoundError("download does not exist or has expired")
        self._path(identity).unlink(missing_ok=True)


def download_protected_paths(root, check=lambda: None, *, now=None):
    """Current live transfers stay pinned in future eligibility forecasts too."""
    now = utc_now().timestamp() if now is None else now
    protected = set()
    for _, pin in DownloadPins(root).records():
        check()
        if pin.expires_at > now:
            protected.update(member.logical_path for member in pin.members)
    return protected
