"""Immutable metadata catalog for reusable job outputs and direct uploads.

The catalog stores references to immutable payloads (shared blobs in blobs-v1,
attempt/upload paths in legacy storage). Publication never copies payloads.
Editable tags are separate metadata; deletion tombstones are audit records,
not storage pins.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote, urlsplit
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from topodrome.domain.errors import (
    AdmissionRejectedError,
    ConflictError,
    NotFoundError,
    TopodromeError,
)
from topodrome.domain.specs import StrictModel, utc_now
from topodrome.storage.artifacts import CommitManifest, CommittedArtifact
from topodrome.storage.assets import asset_manifest_digest
from topodrome.storage.blobs import blob_path, blob_transaction
from topodrome.storage.paths import normalize_logical_path, validate_component

AssetVisibility = Literal["private", "shared"]
AssetKind = Literal["file", "directory"]

_ASSET_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_MAX_CATALOG_MANIFEST_BYTES = 16 * 1024 * 1024
_TAG = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,63}$")


def normalize_tags(tags: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted({tag.strip().lower() for tag in tags}))
    if len(values) > 32 or any(not _TAG.fullmatch(tag) for tag in values):
        raise ValueError("use at most 32 tags, each 1-64 ASCII letters/digits or _ . : / -")
    return values


class AssetTagUpdate(StrictModel):
    add: tuple[str, ...] = Field(default=(), max_length=32)
    remove: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("add", "remove")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_tags(value)

    @model_validator(mode="after")
    def disjoint(self) -> AssetTagUpdate:
        if set(self.add) & set(self.remove):
            raise ValueError("cannot add and remove the same tag")
        return self


class AssetDeleteResult(StrictModel):
    ref: str
    state: Literal["deleted"] = "deleted"
    deleted_at: datetime
    already_deleted: bool
    payloads_deleted: Literal[False] = False


class AssetCatalogError(TopodromeError):
    """A catalog manifest is corrupt or cannot be published safely."""

    code = "asset_catalog_error"


def _asset_component(value: str, *, label: str) -> str:
    if not _ASSET_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return value


class AssetUploadRequest(StrictModel):
    """Name and visibility for a direct, immutable asset upload."""

    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    visibility: AssetVisibility = "private"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _asset_component(value, label="asset name")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _asset_component(value, label="asset version")


class AssetPublishRequest(AssetUploadRequest):
    """Create an immutable catalog name for one committed file or directory."""

    job_id: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=1200)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return validate_component(value, label="job_id")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        # A trailing slash is accepted as an ergonomic directory marker.  The
        # catalog records the normalized prefix and decides file-versus-prefix
        # from the committed manifest, never from the mutable filesystem.
        return normalize_logical_path(value.rstrip("/"))


class AssetFileSummary(StrictModel):
    relative_path: str = Field(min_length=1, max_length=1200)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_logical_path(value)


class AssetSummary(StrictModel):
    schema_version: Literal[1] = 1
    ref: str
    name: str
    version: str
    visibility: AssetVisibility
    owner_username: str
    producer_job_id: str | None = None
    producer_attempt_id: str | None = None
    upload_id: str | None = None
    source_path: str
    kind: AssetKind
    files: tuple[AssetFileSummary, ...]
    total_size_bytes: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_at: datetime
    tags: tuple[str, ...] = ()


class _AssetMember(StrictModel):
    """Private manifest member, including its exact immutable storage path."""

    logical_path: str = Field(min_length=1, max_length=2000)
    relative_path: str = Field(min_length=1, max_length=1200)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("logical_path", "relative_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return normalize_logical_path(value)


class _AssetManifest(StrictModel):
    schema_version: Literal[1, 2] = 1
    name: str
    version: str
    visibility: AssetVisibility
    owner_username: str = Field(min_length=1, max_length=64)
    producer_job_id: str | None = None
    producer_attempt_id: str | None = None
    upload_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    source_path: str
    kind: AssetKind
    members: tuple[_AssetMember, ...]
    total_size_bytes: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_at: datetime

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _asset_component(value, label="asset name")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _asset_component(value, label="asset version")

    @field_validator("producer_job_id", "producer_attempt_id")
    @classmethod
    def validate_ids(cls, value: str | None) -> str | None:
        return None if value is None else validate_component(value)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        return normalize_logical_path(value)

    @field_validator("owner_username")
    @classmethod
    def validate_owner_username(cls, value: str) -> str:
        return _validate_owner(value)

    @model_validator(mode="after")
    def validate_content_identity(self) -> _AssetManifest:
        if not self.members:
            raise ValueError("asset must contain at least one file")
        if self.kind == "file" and len(self.members) != 1:
            raise ValueError("file assets must contain exactly one member")
        relative_paths = tuple(member.relative_path for member in self.members)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("asset contains duplicate relative paths")
        if tuple(sorted(relative_paths)) != relative_paths:
            raise ValueError("asset members must be sorted by relative path")
        if self.upload_id is not None:
            if self.producer_job_id is not None or self.producer_attempt_id is not None:
                raise ValueError("uploaded asset cannot name a producer job")
            prefix = f"artifacts/uploads/{self.upload_id}/files/"
            if self.schema_version == 1 and any(
                m.logical_path != prefix + m.relative_path for m in self.members
            ):
                raise ValueError("uploaded asset member is outside its upload")
            if self.kind == "file" and self.source_path != self.members[0].relative_path:
                raise ValueError("uploaded file does not match its source path")
        elif self.producer_job_id is None or self.producer_attempt_id is None:
            raise ValueError("job asset requires a producer job and attempt")
        expected_prefix = (
            f"jobs/{self.producer_job_id}/attempts/{self.producer_attempt_id}/committed/"
        )
        if (
            self.schema_version == 1
            and self.upload_id is None
            and any(not member.logical_path.startswith(expected_prefix) for member in self.members)
        ):
            raise ValueError("asset member does not belong to the producer attempt")
        for member in self.members:
            if self.schema_version == 2:
                if member.logical_path != blob_path(member.sha256):
                    raise ValueError("asset member does not match its content identity")
                continue
            if self.upload_id is not None:
                continue
            artifact_path = member.logical_path.removeprefix(expected_prefix)
            if self.kind == "file":
                expected_relative = Path(self.source_path).name
                if artifact_path != self.source_path or member.relative_path != expected_relative:
                    raise ValueError("file asset member does not match its source path")
            else:
                source_prefix = f"{self.source_path}/"
                if not artifact_path.startswith(source_prefix):
                    raise ValueError("directory asset member does not belong to its source prefix")
                if member.relative_path != artifact_path.removeprefix(source_prefix):
                    raise ValueError("directory asset member has an invalid relative path")
        if sum(member.size_bytes for member in self.members) != self.total_size_bytes:
            raise ValueError("asset total size does not match its members")
        if _members_digest(self.members) != self.digest:
            raise ValueError("asset digest does not match its members")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        return self


class _AssetDeletion(StrictModel):
    schema_version: Literal[1] = 1
    manifest: _AssetManifest
    deleted_at: datetime
    deleted_by: str


class _AssetTags(StrictModel):
    tags: tuple[str, ...] = Field(default=(), max_length=32)
    updated_at: datetime
    updated_by: str

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_tags(value)


@dataclass(frozen=True, slots=True)
class ResolvedAssetMember:
    """One member ready to be copied into a canonical job-input pin."""

    logical_path: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedAssetInput:
    """Scheduler-facing immutable coordinates for one job input tree."""

    source_ref: str
    kind: AssetKind
    digest: str
    total_bytes: int
    members: tuple[ResolvedAssetMember, ...]


@dataclass(frozen=True, slots=True)
class AssetPublishResult:
    asset: AssetSummary
    created: bool


@dataclass(frozen=True, slots=True)
class _Selection:
    path: str
    kind: AssetKind
    members: tuple[_AssetMember, ...]
    digest: str
    total_bytes: int


class AssetCatalog:
    """Filesystem-backed, immutable catalog of verified asset manifests."""

    def __init__(
        self,
        global_root: Path | str,
        *,
        clock: Callable[[], datetime] = utc_now,
        content_addressed: bool = False,
    ) -> None:
        self.global_root = Path(global_root).expanduser().resolve()
        self._root = self.global_root / "artifacts" / "catalog" / "v1"
        self._clock = clock
        self.content_addressed = content_addressed

    def publish(
        self,
        request: AssetPublishRequest,
        *,
        owner_username: str,
        source_manifest: CommitManifest,
    ) -> AssetPublishResult:
        """Publish a name without copying bytes from the committed attempt."""

        owner = _validate_owner(owner_username)
        if source_manifest.job_id != request.job_id:
            raise ValueError("source manifest belongs to a different job")
        selection = _select_artifacts(source_manifest, request.path)
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("asset catalog clock must return a timezone-aware datetime")
        manifest = _AssetManifest(
            schema_version=source_manifest.schema_version,
            name=request.name,
            version=request.version,
            visibility=request.visibility,
            owner_username=owner,
            producer_job_id=source_manifest.job_id,
            producer_attempt_id=source_manifest.attempt_id,
            source_path=selection.path,
            kind=selection.kind,
            members=selection.members,
            total_size_bytes=selection.total_bytes,
            digest=selection.digest,
            published_at=now.astimezone(UTC),
        )
        return self._publish_manifest(manifest)

    def publish_uploaded(
        self,
        request: AssetUploadRequest,
        *,
        owner_username: str,
        upload_id: str,
        kind: AssetKind,
        files: tuple[AssetFileSummary, ...],
        _blob_lock_held: bool = False,
    ) -> AssetPublishResult:
        """Publish only after the upload service has verified and fsynced files."""
        members = tuple(
            _AssetMember(
                logical_path=(
                    blob_path(item.sha256)
                    if self.content_addressed
                    else f"artifacts/uploads/{upload_id}/files/{item.relative_path}"
                ),
                **item.model_dump(),
            )
            for item in files
        )
        return self._publish_manifest(
            _AssetManifest(
                schema_version=2 if self.content_addressed else 1,
                **request.model_dump(),
                owner_username=_validate_owner(owner_username),
                upload_id=upload_id,
                source_path=files[0].relative_path if kind == "file" else "files",
                kind=kind,
                members=members,
                total_size_bytes=sum(item.size_bytes for item in files),
                digest=_members_digest(members),
                published_at=self._clock(),
            ),
            _blob_lock_held=_blob_lock_held,
        )

    def _publish_manifest(
        self,
        manifest: _AssetManifest,
        *,
        _blob_lock_held: bool = False,
    ) -> AssetPublishResult:
        if not _blob_lock_held:
            with blob_transaction(self.global_root):
                return self._publish_manifest_unlocked(manifest)
        return self._publish_manifest_unlocked(manifest)

    def _publish_manifest_unlocked(self, manifest: _AssetManifest) -> AssetPublishResult:
        target = self._manifest_path(
            name=manifest.name,
            version=manifest.version,
            visibility=manifest.visibility,
            owner_username=manifest.owner_username,
        )
        if _is_deleted(target):
            raise ConflictError("asset version was deleted; publish under a new version")
        content = _manifest_bytes(manifest)
        created = _link_immutable(target, content)
        if not created:
            existing, existing_content = _read_manifest(target)
            _require_manifest_coordinates(
                existing,
                name=manifest.name,
                version=manifest.version,
                visibility=manifest.visibility,
                private_owner=manifest.owner_username,
            )
            # Name+version is content-addressed semantically.  Re-publishing
            # identical files is idempotent even if another job produced them.
            if existing.digest != manifest.digest or existing.kind != manifest.kind:
                raise ConflictError(
                    f"asset {manifest.name}@{manifest.version} already names different content"
                )
            manifest, content = existing, existing_content
        return AssetPublishResult(
            asset=_summary(manifest, content, tags=_read_tags(target)),
            created=created,
        )

    def delete(
        self,
        name: str,
        version: str,
        *,
        requester_username: str,
        visibility: AssetVisibility = "private",
        is_admin: bool = False,
    ) -> AssetDeleteResult:
        """Durably remove exactly one reference, never its payload or job pins."""
        target = self._manifest_path(
            name=name,
            version=version,
            visibility=visibility,
            owner_username=requester_username,
        )
        with blob_transaction(self.global_root):
            tombstone = target.with_name("deleted.json")
            already_deleted = _is_deleted(target)
            if already_deleted:
                deletion = _read_metadata(tombstone, _AssetDeletion)
                manifest = deletion.manifest
            else:
                manifest, _ = _read_manifest_or_not_found(target)
            _require_manifest_coordinates(
                manifest,
                name=name,
                version=version,
                visibility=visibility,
                private_owner=requester_username,
            )
            _require_mutation_owner(manifest, requester_username, is_admin)
            if not already_deleted:
                deletion = _AssetDeletion(
                    manifest=manifest,
                    deleted_by=requester_username,
                    deleted_at=self._clock(),
                )
                # The tombstone is durable before removing the active manifest.
                # A crash between these operations hides the ref but keeps its
                # payload protected until a repeated delete finishes the unlink.
                _link_immutable(tombstone, deletion.model_dump_json().encode() + b"\n")
            if target.exists() or target.is_symlink():
                current, _ = _read_manifest(target)
                if current != manifest:
                    raise AssetCatalogError("asset changed during deletion")
                target.unlink()
            _fsync_directory(target.parent)
            return AssetDeleteResult(
                ref=_asset_ref(visibility, name, version),
                deleted_at=deletion.deleted_at,
                already_deleted=already_deleted,
            )

    def update_tags(
        self,
        name: str,
        version: str,
        update: AssetTagUpdate,
        *,
        requester_username: str,
        visibility: AssetVisibility = "private",
        is_admin: bool = False,
    ) -> AssetSummary:
        """Apply atomic set additions/removals independently of immutable content."""
        target = self._manifest_path(
            name=name,
            version=version,
            visibility=visibility,
            owner_username=requester_username,
        )
        with blob_transaction(self.global_root):
            if _is_deleted(target):
                raise NotFoundError(f"asset {name}@{version} does not exist")
            manifest, content = _read_manifest_or_not_found(target)
            _require_manifest_coordinates(
                manifest,
                name=name,
                version=version,
                visibility=visibility,
                private_owner=requester_username,
            )
            _require_mutation_owner(manifest, requester_username, is_admin)
            before = _read_tags(target)
            try:
                tags = normalize_tags((set(before) | set(update.add)) - set(update.remove))
            except ValueError as exc:
                raise AdmissionRejectedError(str(exc)) from None
            if tags != before:
                metadata = _AssetTags(
                    tags=tags, updated_at=self._clock(), updated_by=requester_username
                )
                _replace_metadata(
                    target.with_name("tags.json"), metadata.model_dump_json().encode() + b"\n"
                )
            return _summary(manifest, content, tags=tags)

    def get(
        self,
        name: str,
        version: str,
        *,
        requester_username: str,
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        """Get a shared asset or the requester's own private asset."""

        owner = _validate_owner(requester_username)
        visibility = _validate_visibility(visibility)
        try:
            safe_name = _asset_component(name, label="asset name")
            safe_version = _asset_component(version, label="asset version")
        except ValueError:
            raise NotFoundError(f"asset {name}@{version} does not exist") from None
        target = self._manifest_path(
            name=safe_name,
            version=safe_version,
            visibility=visibility,
            owner_username=owner,
        )
        if _is_deleted(target):
            raise NotFoundError(f"asset {name}@{version} does not exist")
        manifest, content = _read_manifest_or_not_found(target)
        _require_manifest_coordinates(
            manifest,
            name=name,
            version=version,
            visibility=visibility,
            private_owner=owner,
        )
        return _summary(manifest, content, tags=_read_tags(target))

    def list_visible(
        self,
        *,
        requester_username: str,
        visibility: AssetVisibility | None = None,
        limit: int = 100,
        tags: Iterable[str] = (),
    ) -> list[AssetSummary]:
        """List shared assets and the requester's private assets."""

        owner = _validate_owner(requester_username)
        if visibility is not None:
            visibility = _validate_visibility(visibility)
        if not 1 <= limit <= 1000:
            raise ValueError("asset list limit must be between 1 and 1000")
        roots: list[tuple[Path, AssetVisibility]] = []
        if visibility in {None, "shared"}:
            roots.append((self._root / "shared", "shared"))
        if visibility in {None, "private"}:
            roots.append((self._root / "private" / _owner_key(owner), "private"))
        assets: list[AssetSummary] = []
        required_tags = set(normalize_tags(tags))
        for root, expected_visibility in roots:
            if not root.is_dir():
                continue
            for target in root.glob("*/*/manifest.json"):
                try:
                    asset = self.get(
                        target.parent.parent.name,
                        target.parent.name,
                        requester_username=owner,
                        visibility=expected_visibility,
                    )
                except NotFoundError:
                    continue  # A concurrent deletion may remove an enumerated ref.
                if required_tags.issubset(asset.tags):
                    assets.append(asset)
        assets.sort(key=lambda value: (value.published_at, value.ref), reverse=True)
        return assets[:limit]

    def resolve(
        self,
        source_ref: str,
        *,
        requester_username: str,
    ) -> ResolvedAssetInput:
        """Resolve ``asset://`` into exact attempt paths for a persisted job pin."""

        visibility, name, version = _parse_asset_ref(source_ref)
        self.get(
            name,
            version,
            requester_username=requester_username,
            visibility=visibility,
        )
        target = self._manifest_path(
            name=name,
            version=version,
            visibility=visibility,
            owner_username=requester_username,
        )
        manifest, _ = _read_manifest(target)
        return _resolved(_asset_ref(visibility, name, version), manifest)

    def _manifest_path(
        self,
        *,
        name: str,
        version: str,
        visibility: AssetVisibility,
        owner_username: str,
    ) -> Path:
        try:
            _asset_component(name, label="asset name")
            _asset_component(version, label="asset version")
        except ValueError:
            raise NotFoundError(f"asset {name}@{version} does not exist") from None
        _validate_visibility(visibility)
        _validate_owner(owner_username)
        base = self._root / visibility
        if visibility == "private":
            base /= _owner_key(owner_username)
        target = base / name / version / "manifest.json"
        for parent in target.parents:
            if parent == self.global_root:
                break
            if parent.is_symlink():
                raise AssetCatalogError("symlink in asset catalog")
        return target


def resolve_job_asset(
    source_ref: str,
    source_manifest: CommitManifest,
) -> ResolvedAssetInput:
    """Pin ``job://job-id/path`` using the same tree semantics as named assets."""

    job_id, requested_path = _parse_job_ref(source_ref)
    if source_manifest.job_id != job_id:
        raise NotFoundError("job artifact reference does not match its committed manifest")
    selection = _select_artifacts(source_manifest, requested_path)
    canonical_ref = _job_ref(job_id, selection.path, kind=selection.kind)
    return ResolvedAssetInput(
        source_ref=canonical_ref,
        kind=selection.kind,
        digest=selection.digest,
        total_bytes=selection.total_bytes,
        members=tuple(
            ResolvedAssetMember(
                logical_path=member.logical_path,
                relative_path=member.relative_path,
                size_bytes=member.size_bytes,
                sha256=member.sha256,
            )
            for member in selection.members
        ),
    )


def job_id_from_ref(source_ref: str) -> str:
    """Return the validated producer ID from a direct ``job://`` reference."""

    job_id, _ = _parse_job_ref(source_ref)
    return job_id


def _select_artifacts(manifest: CommitManifest, requested_path: str) -> _Selection:
    path = normalize_logical_path(requested_path.rstrip("/"))
    exact = tuple(item for item in manifest.artifacts if item.logical_path == path)
    kind: AssetKind = "file" if exact else "directory"
    selected: Iterable[CommittedArtifact]
    if exact:
        selected = exact
    else:
        prefix = f"{path}/"
        selected = (item for item in manifest.artifacts if item.logical_path.startswith(prefix))

    members: list[_AssetMember] = []
    for artifact in selected:
        relative_path = (
            artifact.logical_path.removeprefix(f"{path}/")
            if kind == "directory"
            else Path(artifact.logical_path).name
        )
        members.append(
            _AssetMember(
                logical_path=artifact.storage_path,
                relative_path=relative_path,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
            )
        )
    members.sort(key=lambda item: item.relative_path)
    if not members:
        raise NotFoundError(f"committed artifact path does not exist: {path}")
    member_tuple = tuple(members)
    return _Selection(
        path=path,
        kind=kind,
        members=member_tuple,
        digest=_members_digest(member_tuple),
        total_bytes=sum(member.size_bytes for member in member_tuple),
    )


def _members_digest(members: Iterable[_AssetMember]) -> str:
    return asset_manifest_digest(member.model_dump(mode="json") for member in members)


def _resolved(source_ref: str, manifest: _AssetManifest) -> ResolvedAssetInput:
    return ResolvedAssetInput(
        source_ref=source_ref,
        kind=manifest.kind,
        digest=manifest.digest,
        total_bytes=manifest.total_size_bytes,
        members=tuple(
            ResolvedAssetMember(
                logical_path=member.logical_path,
                relative_path=member.relative_path,
                size_bytes=member.size_bytes,
                sha256=member.sha256,
            )
            for member in manifest.members
        ),
    )


def _summary(manifest: _AssetManifest, content: bytes, *, tags=()) -> AssetSummary:
    return AssetSummary(
        ref=_asset_ref(manifest.visibility, manifest.name, manifest.version),
        name=manifest.name,
        version=manifest.version,
        visibility=manifest.visibility,
        owner_username=manifest.owner_username,
        producer_job_id=manifest.producer_job_id,
        producer_attempt_id=manifest.producer_attempt_id,
        upload_id=manifest.upload_id,
        source_path=manifest.source_path,
        kind=manifest.kind,
        files=tuple(
            AssetFileSummary(
                relative_path=member.relative_path,
                size_bytes=member.size_bytes,
                sha256=member.sha256,
            )
            for member in manifest.members
        ),
        total_size_bytes=manifest.total_size_bytes,
        digest=manifest.digest,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        published_at=manifest.published_at,
        tags=tags,
    )


def _require_mutation_owner(manifest, requester, is_admin):
    if manifest.owner_username != requester and not is_admin:
        raise NotFoundError("asset does not exist or is not owned by this user")


def _is_deleted(target):
    marker = target.with_name("deleted.json")
    return marker.exists() or marker.is_symlink()


def _read_metadata(path, model):
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_CATALOG_MANIFEST_BYTES + 4096:
            raise ValueError("invalid metadata file")
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise AssetCatalogError("invalid asset catalog metadata") from exc


def _read_tags(target):
    path = target.with_name("tags.json")
    return _read_metadata(path, _AssetTags).tags if path.exists() or path.is_symlink() else ()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_metadata(path, content):
    temporary = path.parent / f".metadata-{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_bytes(manifest: _AssetManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(content) > _MAX_CATALOG_MANIFEST_BYTES:
        raise AssetCatalogError("asset catalog manifest exceeds the metadata safety limit")
    return content


def _read_manifest(path: Path) -> tuple[_AssetManifest, bytes]:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("asset manifest must be a regular file")
        if metadata.st_size > _MAX_CATALOG_MANIFEST_BYTES:
            raise ValueError("asset manifest exceeds the metadata safety limit")
        content = path.read_bytes()
        raw = json.loads(content)
        manifest = _AssetManifest.model_validate(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AssetCatalogError("invalid asset catalog manifest") from exc
    return manifest, content


def _require_manifest_coordinates(
    manifest: _AssetManifest,
    *,
    name: str,
    version: str,
    visibility: AssetVisibility,
    private_owner: str | None,
) -> None:
    if (
        manifest.name != name
        or manifest.version != version
        or manifest.visibility != visibility
        or (
            visibility == "private"
            and (private_owner is None or manifest.owner_username != private_owner)
        )
    ):
        # Treat a misplaced private manifest as absent rather than disclosing
        # another owner through a corrupted or manually moved catalog file.
        if visibility == "private":
            raise NotFoundError(f"asset {name}@{version} does not exist")
        raise AssetCatalogError("asset catalog manifest coordinates do not match its path")


def _read_manifest_or_not_found(path: Path) -> tuple[_AssetManifest, bytes]:
    if not path.is_file():
        raise NotFoundError(f"asset {path.parent.parent.name}@{path.parent.name} does not exist")
    try:
        return _read_manifest(path)
    except AssetCatalogError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise NotFoundError("asset no longer exists") from None
        raise


def _link_immutable(path: Path, content: bytes) -> bool:
    """Atomically create ``path`` without ever replacing an existing version."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".manifest-{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        os.chmod(path, 0o444)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _asset_ref(visibility: AssetVisibility, name: str, version: str) -> str:
    return f"asset://{visibility}/{quote(name, safe='')}@{quote(version, safe='')}"


def _job_ref(job_id: str, path: str, *, kind: AssetKind) -> str:
    suffix = "/" if kind == "directory" else ""
    return f"job://{job_id}/{quote(path, safe='/._-')}" + suffix


def _parse_asset_ref(source_ref: str) -> tuple[AssetVisibility, str, str]:
    parsed = urlsplit(source_ref)
    if parsed.scheme != "asset" or parsed.query or parsed.fragment:
        raise ValueError("asset reference must be asset://private|shared/name@version")
    if parsed.netloc not in {"private", "shared"}:
        raise ValueError("asset reference visibility must be private or shared")
    value = unquote(parsed.path.removeprefix("/"))
    name, separator, version = value.rpartition("@")
    if not separator or not name or not version or "/" in value:
        raise ValueError("asset reference must include one name and version")
    return (
        parsed.netloc,  # type: ignore[return-value]
        _asset_component(name, label="asset name"),
        _asset_component(version, label="asset version"),
    )


def _parse_job_ref(source_ref: str) -> tuple[str, str]:
    parsed = urlsplit(source_ref)
    if parsed.scheme != "job" or parsed.query or parsed.fragment:
        raise ValueError("job artifact reference must be job://job-id/path")
    if not parsed.netloc or any(value in parsed.netloc for value in "@:[%"):
        raise ValueError("job artifact reference has an invalid job id")
    job_id = validate_component(parsed.netloc, label="job_id")
    path = normalize_logical_path(unquote(parsed.path.removeprefix("/")).rstrip("/"))
    return job_id, path


def _validate_owner(owner_username: str) -> str:
    if (
        not isinstance(owner_username, str)
        or not owner_username
        or owner_username != owner_username.strip()
        or len(owner_username) > 64
        or any(ord(character) < 32 or ord(character) == 127 for character in owner_username)
    ):
        raise ValueError("owner username is invalid")
    return owner_username


def _validate_visibility(value: str) -> AssetVisibility:
    if value not in {"private", "shared"}:
        raise ValueError("asset visibility must be private or shared")
    return value  # type: ignore[return-value]


def _owner_key(owner_username: str) -> str:
    return hashlib.sha256(owner_username.encode("utf-8")).hexdigest()


__all__ = [
    "AssetCatalog",
    "AssetCatalogError",
    "AssetDeleteResult",
    "AssetFileSummary",
    "AssetKind",
    "AssetPublishRequest",
    "AssetPublishResult",
    "AssetSummary",
    "AssetTagUpdate",
    "AssetUploadRequest",
    "AssetVisibility",
    "ResolvedAssetInput",
    "ResolvedAssetMember",
    "job_id_from_ref",
    "normalize_tags",
    "resolve_job_asset",
]
