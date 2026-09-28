"""Build a secret-free, crash-safe AutoDL site bootstrap release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import SplitResult, urlsplit

import yaml

_FORMAT_VERSION = 2
_PERSISTENT_ROOT = PurePosixPath("/root/autodl-fs")
_EPHEMERAL_ROOT = PurePosixPath("/root/autodl-tmp")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:token|secrets?|passwords?|credentials?|authorizations?|"
    r"api[_-]?keys?|access[_-]?keys?|client[_-]?secrets?|private[_-]?keys?|"
    r"verification[_-]?keys?|signing[_-]?keys?)"
    r"(?:$|[_-])",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.|"
    r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_CREDENTIAL_URL = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@|"
    r"[?&](?:access_token|api[_-]?key|auth|password|secret|token)=[^\s&#]+)"
)
_TEXT_BUNDLE_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)
_RCLONE_REMOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_RCLONE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_HF_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SSH_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,252}$")
_SSH_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_RUNTIME_SEED_TARGET = re.compile(
    r"^(?P<machine>x86_64|aarch64)-(?P<libc>manylinux|musllinux)_"
    r"(?P<major>[0-9]+)_(?P<minor>[0-9]+)$"
)
_LIBC_VERSION = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:\.[0-9]+)?$")
_SEED_METADATA_FILES = frozenset({"METADATA", "WHEEL"})
_MAX_SEED_METADATA_FILE_BYTES = 16 * 1024 * 1024
_MAX_SEED_METADATA_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AutoDLBootstrapArtifacts:
    bootstrap_script: Path
    release_archive: Path
    checksum_file: Path
    current_pointer: Path
    digest: str
    runtime_seed_remote: str | None = None


@dataclass(frozen=True, slots=True)
class _RuntimeSeed:
    path: Path
    sha256: str
    size_bytes: int
    member_count: int
    uncompressed_bytes: int
    platform_key: str
    remote: str
    requires_online_repair: bool


@dataclass(frozen=True, slots=True)
class _RuntimePlatform:
    key: str
    uv_target: str


@dataclass(frozen=True, slots=True)
class _RuntimeSeedInspection:
    member_count: int
    uncompressed_bytes: int
    distribution_metadata: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True, slots=True)
class _BundleMember:
    name: str
    mode: int
    source: Path | None = None
    content: bytes | None = None

    @property
    def size(self) -> int:
        if self.source is not None:
            return self.source.stat().st_size
        return len(self.content or b"")

    def open(self) -> BinaryIO:
        if self.source is not None:
            return self.source.open("rb")
        return io.BytesIO(self.content or b"")


def build_autodl_bootstrap(
    *,
    repository_root: str | Path,
    site_config: str | Path,
    model_config: str | Path,
    output_dir: str | Path,
    uv_binary: str | Path | None = None,
    rclone_binary: str | Path | None = None,
    runtime_seed: str | Path | None = None,
    runtime_seed_remote: str | None = None,
    runtime_seed_platform: str | None = None,
    allow_online_fallback: bool = False,
) -> AutoDLBootstrapArtifacts:
    """Build immutable release files and atomically activate their digest.

    The bundle contains only Topodrome source, non-secret site/model
    configuration, the locked dependency graph, and two vetted executables.
    Virtual environments, package caches, Ray state, and rclone VFS data are
    deliberately created later on AutoDL's ephemeral filesystem.
    """

    if not isinstance(allow_online_fallback, bool):
        raise ValueError("allow_online_fallback must be a boolean")

    root = Path(repository_root).expanduser().resolve()
    required_files = (root / "pyproject.toml", root / "uv.lock", root / "README.md")
    source_directory = root / "src"
    if (
        not all(path.is_file() and not path.is_symlink() for path in required_files)
        or not source_directory.is_dir()
        or source_directory.is_symlink()
    ):
        raise ValueError("repository_root is not a complete Topodrome checkout")

    site = _read_mapping(Path(site_config), "site configuration")
    models = _read_mapping(Path(model_config), "model configuration")
    _reject_embedded_secrets(site, "site configuration")
    _reject_embedded_secrets(models, "model configuration")
    _validate_site_config(site)
    _validate_model_config(models, site_id=str(site["site_id"]))

    uv_path = _executable(uv_binary, "uv")
    if runtime_seed is None and runtime_seed_platform is not None:
        raise ValueError("runtime_seed_platform requires a runtime_seed")
    runtime_platform = (
        _runtime_platform(runtime_seed_platform) if runtime_seed is not None else None
    )
    lock_digest = _sha256(root / "uv.lock")
    seed = _prepare_runtime_seed(
        runtime_seed,
        runtime_seed_remote=runtime_seed_remote,
        site_remote=str(_mapping(site["rclone"], "site rclone configuration")["remote"]),
        uv_lock_sha256=lock_digest,
        runtime_platform=runtime_platform,
        repository_root=root,
        uv_binary=uv_path,
        allow_online_fallback=allow_online_fallback,
    )

    normalized_site = dict(site)
    normalized_site["model_config"] = "config/models.yaml"
    rclone = dict(_mapping(normalized_site["rclone"], "site rclone configuration"))
    rclone["binary"] = "rclone"
    # AutoDL's global mirror is the scheduler's authoritative namespace.
    # Verify at Lab1; never download uploaded outputs back onto paid compute.
    rclone["verification_via_scheduler"] = True
    normalized_site["rclone"] = rclone

    rclone_path = _executable(rclone_binary, "rclone")
    members = _repository_members(root)
    site_bytes = yaml.safe_dump(normalized_site, sort_keys=True).encode()
    model_bytes = yaml.safe_dump(models, sort_keys=True).encode()
    members.extend(
        (
            _BundleMember("config/site.yaml", 0o600, content=site_bytes),
            _BundleMember("config/models.yaml", 0o600, content=model_bytes),
            _BundleMember("tools/uv", 0o700, source=uv_path),
            _BundleMember("tools/rclone", 0o700, source=rclone_path),
        )
    )
    _reject_bundle_credentials(members)
    manifest = {
        "format_version": _FORMAT_VERSION,
        "site_id": str(site["site_id"]),
        "uv_lock_sha256": lock_digest,
        "allow_online_fallback": bool(allow_online_fallback),
        "runtime_seed": (
            None
            if seed is None
            else {
                "format": "site-packages-tar-gz",
                "remote": seed.remote,
                "sha256": seed.sha256,
                "size_bytes": seed.size_bytes,
                "member_count": seed.member_count,
                "uncompressed_bytes": seed.uncompressed_bytes,
                "platform_key": seed.platform_key,
                "uv_lock_sha256": lock_digest,
                "requires_online_repair": seed.requires_online_repair,
            }
        ),
        "tools": {
            "uv_sha256": _sha256(uv_path),
            "rclone_sha256": _sha256(rclone_path),
        },
    }
    members.append(
        _BundleMember(
            "bundle-manifest.json",
            0o600,
            content=(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".release-",
        suffix=".tar.gz.tmp",
        dir=destination,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        _write_archive(temporary_path, members)
        digest = _sha256(temporary_path)
        archive = destination / f"release-{digest}.tar.gz"
        if archive.exists():
            if _sha256(archive) != digest:
                raise RuntimeError("existing immutable release archive failed verification")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, archive)
            archive.chmod(0o600)
            _fsync_directory(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    checksum = destination / f"release-{digest}.sha256"
    pointer = destination / "current"
    bootstrap = destination / "bootstrap.sh"
    _atomic_write(checksum, f"{digest}  {archive.name}\n".encode(), mode=0o600)
    _atomic_write(pointer, f"{digest}\n".encode(), mode=0o600)
    template = Path(__file__).with_name("autodl_bootstrap.sh").read_bytes()
    _atomic_write(bootstrap, template, mode=0o700)
    return AutoDLBootstrapArtifacts(
        bootstrap,
        archive,
        checksum,
        pointer,
        digest,
        runtime_seed_remote=None if seed is None else seed.remote,
    )


def _repository_members(root: Path) -> list[_BundleMember]:
    members = [
        _BundleMember("README.md", 0o600, source=root / "README.md"),
        _BundleMember("pyproject.toml", 0o600, source=root / "pyproject.toml"),
        _BundleMember("uv.lock", 0o600, source=root / "uv.lock"),
    ]
    source_root = (root / "src").resolve()
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"release source cannot contain symbolic links: {relative}")
        if not path.is_file() or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        members.append(_BundleMember(relative.as_posix(), 0o600, source=path))
    return members


def _prepare_runtime_seed(
    value: str | Path | None,
    *,
    runtime_seed_remote: str | None,
    site_remote: str,
    uv_lock_sha256: str,
    runtime_platform: _RuntimePlatform | None,
    repository_root: Path,
    uv_binary: Path,
    allow_online_fallback: bool,
) -> _RuntimeSeed | None:
    if value is None:
        if runtime_seed_remote is not None:
            raise ValueError("runtime_seed_remote requires a local runtime_seed for verification")
        return None

    if runtime_platform is None:
        raise ValueError("runtime_seed requires an explicit target platform identity")

    unresolved_path = Path(value).expanduser()
    if unresolved_path.is_symlink():
        raise ValueError("runtime_seed must be a regular non-symlink file")
    path = unresolved_path.resolve()
    if not path.is_file():
        raise ValueError("runtime_seed must be a regular non-symlink file")
    if not (path.name.endswith(".tar.gz") or path.name.endswith(".tgz")):
        raise ValueError("runtime_seed must use the site-packages .tar.gz format")

    inspection = _inspect_runtime_seed(path)
    uv_compatible = _runtime_seed_is_uv_compatible(
        inspection,
        repository_root=repository_root,
        uv_binary=uv_binary,
        runtime_platform=runtime_platform,
    )
    if not uv_compatible and not allow_online_fallback:
        raise ValueError(
            "runtime_seed is not synchronized with the frozen lock for target "
            f"{runtime_platform.uv_target}; rebuild it with uv for that platform or explicitly "
            "allow online fallback"
        )
    digest = _sha256(path)
    platform_key = runtime_platform.key
    remote_root = _validate_rclone_reference(site_remote, label="site rclone remote")
    remote_prefix = remote_root if remote_root.endswith(":") else f"{remote_root.rstrip('/')}/"
    default_remote = (
        f"{remote_prefix}bootstrap/runtime-seeds/{uv_lock_sha256}/"
        f"{platform_key}/runtime-seed-{digest}.tar.gz"
    )
    remote = _validate_rclone_reference(
        default_remote if runtime_seed_remote is None else runtime_seed_remote,
        label="runtime seed remote",
    )
    required_prefix = remote_prefix
    if not remote.startswith(required_prefix):
        raise ValueError("runtime seed remote must be inside the configured Lab1 rclone root")
    if not all(component in remote for component in (uv_lock_sha256, platform_key, digest)):
        raise ValueError(
            "runtime seed remote must be keyed by uv.lock, target platform, and seed digest"
        )
    return _RuntimeSeed(
        path=path,
        sha256=digest,
        size_bytes=path.stat().st_size,
        member_count=inspection.member_count,
        uncompressed_bytes=inspection.uncompressed_bytes,
        platform_key=platform_key,
        remote=remote,
        requires_online_repair=not uv_compatible,
    )


def _runtime_platform(value: str | None) -> _RuntimePlatform:
    if sys.platform != "linux":
        raise ValueError("AutoDL runtime seeds must be built on Linux")
    if value is not None:
        match = _RUNTIME_SEED_TARGET.fullmatch(value)
        if match is None:
            raise ValueError(
                "runtime_seed_platform must look like x86_64-manylinux_2_31"
            )
        machine = match.group("machine")
        libc_name = "glibc" if match.group("libc") == "manylinux" else "musl"
        libc_version = f"{int(match.group('major'))}.{int(match.group('minor'))}"
        uv_target = value
        if libc_name == "musl":
            uv_target = f"{machine}-unknown-linux-musl"
    else:
        machine = platform.machine().strip().casefold().replace("amd64", "x86_64")
        raw_libc_name, raw_libc_version = platform.libc_ver()
        libc_name = raw_libc_name.strip().casefold()
        if libc_name in {"gnu libc", "glibc"}:
            libc_name = "glibc"
        elif "musl" in libc_name:
            libc_name = "musl"
        version_match = _LIBC_VERSION.fullmatch(raw_libc_version.strip())
        if (
            machine not in {"x86_64", "aarch64"}
            or libc_name not in {"glibc", "musl"}
            or version_match is None
        ):
            raise ValueError(
                "AutoDL runtime seed platform must identify Linux architecture and libc version"
            )
        libc_version = (
            f"{int(version_match.group('major'))}.{int(version_match.group('minor'))}"
        )
        if libc_name == "glibc":
            major, minor = libc_version.split(".", 1)
            uv_target = f"{machine}-manylinux_{major}_{minor}"
        else:
            uv_target = f"{machine}-unknown-linux-musl"

    key = (
        f"cpython-{sys.version_info.major}.{sys.version_info.minor}-linux-"
        f"{machine}-{libc_name}-{libc_version}"
    )
    return _RuntimePlatform(key=key, uv_target=uv_target)


def _runtime_seed_is_uv_compatible(
    inspection: _RuntimeSeedInspection,
    *,
    repository_root: Path,
    uv_binary: Path,
    runtime_platform: _RuntimePlatform,
) -> bool:
    """Ask uv whether the seed metadata exactly satisfies the target lock.

    Only ``*.dist-info/{METADATA,WHEEL}`` is staged. uv needs those files to
    audit versions and wheel tags, while avoiding a second multi-gigabyte seed
    extraction on the build host. ``--inexact`` deliberately ignores dev-only
    packages in the source environment; every agent/runtime dependency remains
    required by the frozen project selection.
    """

    with tempfile.TemporaryDirectory(prefix="topodrome-runtime-seed-check-") as temporary:
        temporary_root = Path(temporary)
        environment = temporary_root / "venv"
        cache = temporary_root / "uv-cache"
        clean_environment = {
            "HOME": os.environ.get("HOME", str(temporary_root)),
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "UV_CACHE_DIR": str(cache),
            "UV_NO_PROGRESS": "1",
            "UV_PROJECT_ENVIRONMENT": str(environment),
            "UV_PYTHON_DOWNLOADS": "never",
        }
        created = subprocess.run(  # noqa: S603 - executable was resolved and validated above
            [
                str(uv_binary),
                "venv",
                "--offline",
                "--no-python-downloads",
                "--python",
                sys.executable,
                str(environment),
            ],
            cwd=repository_root,
            env=clean_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            raise ValueError("uv could not create the runtime-seed verification environment")

        purelib = (
            environment
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        purelib.mkdir(parents=True, exist_ok=True)
        for relative_name, content in inspection.distribution_metadata:
            destination = purelib / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

        checked = subprocess.run(  # noqa: S603 - executable was resolved and validated above
            [
                str(uv_binary),
                "sync",
                "--check",
                "--inexact",
                "--frozen",
                "--no-dev",
                "--extra",
                "agent",
                "--extra",
                "runtime",
                "--no-install-project",
                "--offline",
                "--no-python-downloads",
                "--python",
                str(environment / "bin" / "python"),
                "--python-platform",
                runtime_platform.uv_target,
            ],
            cwd=repository_root,
            env=clean_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        return checked.returncode == 0


def _inspect_runtime_seed(path: Path) -> _RuntimeSeedInspection:
    names: set[str] = set()
    member_count = 0
    uncompressed_bytes = 0
    metadata_bytes = 0
    distribution_metadata: list[tuple[str, bytes]] = []
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                logical = PurePosixPath(member.name)
                if (
                    logical.is_absolute()
                    or not logical.parts
                    or logical.parts[0] != "site-packages"
                    or ".." in logical.parts
                ):
                    raise ValueError("runtime_seed may contain only a relative site-packages/ tree")
                normalized = logical.as_posix()
                if normalized in names:
                    raise ValueError("runtime_seed contains duplicate archive members")
                names.add(normalized)
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError("runtime_seed contains an unsupported archive member")
                if member.issym() or member.islnk():
                    link = PurePosixPath(member.linkname)
                    if link.is_absolute() or ".." in link.parts or not link.parts:
                        raise ValueError("runtime_seed contains an unsafe archive link")
                    target = logical.parent / link if member.issym() else link
                    if not target.parts or target.parts[0] != "site-packages":
                        raise ValueError("runtime_seed archive link escapes site-packages")
                if member.mode & 0o7000:
                    raise ValueError("runtime_seed archive members may not use special mode bits")
                member_count += 1
                if member.isfile():
                    uncompressed_bytes += member.size
                    if (
                        len(logical.parts) == 3
                        and logical.parts[1].endswith(".dist-info")
                        and logical.parts[2] in _SEED_METADATA_FILES
                    ):
                        if member.size > _MAX_SEED_METADATA_FILE_BYTES:
                            raise ValueError("runtime_seed distribution metadata file is too large")
                        metadata_bytes += member.size
                        if metadata_bytes > _MAX_SEED_METADATA_BYTES:
                            raise ValueError("runtime_seed distribution metadata is too large")
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError("runtime_seed distribution metadata is unreadable")
                        content = source.read()
                        if len(content) != member.size:
                            raise ValueError("runtime_seed distribution metadata is truncated")
                        relative_name = PurePosixPath(*logical.parts[1:]).as_posix()
                        distribution_metadata.append((relative_name, content))
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("runtime_seed is not a readable site-packages .tar.gz archive") from exc
    if member_count == 0 or "site-packages" not in names:
        raise ValueError("runtime_seed must contain a site-packages/ root directory")
    if not distribution_metadata:
        raise ValueError("runtime_seed must contain installed distribution metadata")
    return _RuntimeSeedInspection(
        member_count=member_count,
        uncompressed_bytes=uncompressed_bytes,
        distribution_metadata=tuple(distribution_metadata),
    )


def _validate_rclone_reference(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(character in value for character in "\r\n\0\\?#@")
    ):
        raise ValueError(f"{label} must be a credential-free named rclone reference")
    remote_name, separator, raw_path = value.partition(":")
    logical_path = PurePosixPath(raw_path) if raw_path else None
    if (
        not separator
        or not _RCLONE_REMOTE_NAME.fullmatch(remote_name)
        or (
            raw_path
            and (
                not _RCLONE_REMOTE_PATH.fullmatch(raw_path)
                or logical_path is None
                or logical_path.is_absolute()
                or ".." in logical_path.parts
            )
        )
    ):
        raise ValueError(f"{label} must be a credential-free named rclone reference")
    return f"{remote_name}:{logical_path.as_posix() if logical_path is not None else ''}"


def _reject_bundle_credentials(members: Sequence[_BundleMember]) -> None:
    for member in members:
        suffix = PurePosixPath(member.name).suffix.casefold()
        if suffix not in _TEXT_BUNDLE_SUFFIXES and member.name not in {"uv.lock"}:
            continue
        try:
            with member.open() as handle:
                content = handle.read().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"release text member is unreadable: {member.name}") from exc
        if _CREDENTIAL_VALUE.search(content) or _CREDENTIAL_URL.search(content):
            raise ValueError(f"release bundle contains credential-like material: {member.name}")


def _write_archive(path: Path, members: Sequence[_BundleMember]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for member in sorted(members, key=lambda value: value.name):
                    logical = PurePosixPath(member.name)
                    if logical.is_absolute() or ".." in logical.parts:
                        raise ValueError(f"unsafe release member: {member.name}")
                    info = tarfile.TarInfo(logical.as_posix())
                    info.size = member.size
                    info.mode = member.mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    with member.open() as content:
                        archive.addfile(info, content)
        raw.flush()
        os.fsync(raw.fileno())


def _validate_site_config(site: Mapping[str, Any]) -> None:
    if site.get("site_kind") != "autodl":
        raise ValueError("AutoDL bootstrap requires site_kind: autodl")
    site_id = site.get("site_id")
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("AutoDL site configuration requires a site_id")
    control_url = _safe_control_url(site.get("scheduler_url"))
    _require_under(site.get("persistent_root"), _PERSISTENT_ROOT, "persistent_root")
    _require_under(site.get("scratch_root"), _EPHEMERAL_ROOT, "scratch_root")
    _require_under(site.get("spool_path"), _PERSISTENT_ROOT, "spool_path")

    rclone = _mapping(site.get("rclone"), "site rclone configuration")
    rclone_mode = rclone.get("mode", "mount")
    if rclone_mode not in {"mount", "mirror"}:
        raise ValueError("rclone mode must be mount or mirror")
    mountpoint = _absolute_path(rclone.get("mountpoint"), "rclone mountpoint")
    global_root = _absolute_path(site.get("global_root"), "global_root")
    if mountpoint != global_root:
        raise ValueError("global_root must equal the managed rclone mountpoint")
    if rclone_mode == "mirror":
        _require_under(str(global_root), _EPHEMERAL_ROOT, "mirror global_root")
    _require_under(rclone.get("cache_dir"), _EPHEMERAL_ROOT, "rclone cache_dir")
    remote = rclone.get("remote")
    if not isinstance(remote, str):
        raise ValueError("rclone remote must be configured without embedded credentials")
    _validate_rclone_reference(remote, label="rclone remote")
    _validate_control_tunnel(site.get("ssh_local_forward"), control_url=control_url)


def _validate_model_config(models: Mapping[str, Any], *, site_id: str) -> None:
    definitions = _mapping(models.get("models"), "model catalog")
    if not definitions:
        raise ValueError("model catalog must contain at least one model")
    found = False
    for alias, raw_definition in definitions.items():
        definition = _mapping(raw_definition, f"model {alias!r}")
        sites = _mapping(definition.get("sites"), f"model {alias!r} sites")
        if site_id not in sites:
            continue
        found = True
        location = _mapping(sites[site_id], f"model {alias!r} AutoDL location")
        _require_under(
            location.get("persistent_path"),
            _PERSISTENT_ROOT,
            f"model {alias!r} persistent_path",
        )
        # Regional-only loading is valid on small system disks. Staging is
        # opt-in through local_path and must remain on the ephemeral root.
        if location.get("local_path") is not None:
            _require_under(
                location["local_path"],
                _EPHEMERAL_ROOT,
                f"model {alias!r} local_path",
            )
        if location.get("origin") is not None:
            origin = location.get("origin")
            if not isinstance(origin, str) or not _HF_REPOSITORY.fullmatch(origin):
                raise ValueError(
                    f"model {alias!r} origin must be a credential-free Hugging Face repository"
                )
            endpoint = urlsplit(str(location.get("endpoint", "")))
            try:
                endpoint_port = endpoint.port
            except ValueError:
                endpoint_port = -1
            if (
                endpoint.scheme != "https"
                or endpoint.hostname != "hf-mirror.com"
                or endpoint_port not in {None, 443}
                or endpoint.path not in {"", "/"}
                or endpoint.username is not None
                or endpoint.password is not None
                or endpoint.query
                or endpoint.fragment
            ):
                raise ValueError(f"model {alias!r} must acquire through https://hf-mirror.com")
            excludes = location.get("exclude", ())
            if not isinstance(excludes, Sequence) or isinstance(excludes, (str, bytes)):
                raise ValueError(f"model {alias!r} exclude must be a sequence")
            if "original/*" not in excludes:
                raise ValueError(f"model {alias!r} must exclude the redundant original/* payload")
    if not found:
        raise ValueError(f"model catalog has no capability for AutoDL site {site_id!r}")


def _safe_control_url(raw: object) -> SplitResult:
    if not isinstance(raw, str):
        raise ValueError("scheduler_url is required")
    parsed = urlsplit(raw)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if (
        parsed.scheme not in {"http", "https", "ws", "wss"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port == -1
        or (parsed.hostname or "").endswith(".invalid")
    ):
        raise ValueError("scheduler_url must be a credential-free deployable control URL")
    plain_control = parsed.scheme in {"http", "ws"}
    loopback = (parsed.hostname or "").strip("[]").casefold() in {
        "127.0.0.1",
        "::1",
        "localhost",
    }
    if plain_control and not loopback:
        raise ValueError("plaintext scheduler_url must use an explicit loopback address")
    return parsed


def _validate_control_tunnel(value: object, *, control_url: SplitResult) -> None:
    """Validate the secret-free AutoDL-to-Lab1 control route.

    A loopback scheduler origin is useful only when the site agent owns a
    matching outbound SSH local forward.  Validate this while building the
    immutable release so a paid worker cannot boot into a configuration that
    can never report its heartbeat.  Key material remains outside the bundle;
    the configuration contains protected paths only.
    """

    if value is None:
        if control_url.scheme in {"http", "ws"}:
            raise ValueError("loopback scheduler_url requires ssh_local_forward")
        return
    if control_url.scheme not in {"http", "ws"}:
        raise ValueError("ssh_local_forward requires a loopback plaintext scheduler_url")
    tunnel = _mapping(value, "site SSH local-forward configuration")
    local_host = tunnel.get("local_host", "127.0.0.1")
    local_port = tunnel.get("local_port")
    if local_host != control_url.hostname or local_port != control_url.port:
        raise ValueError("ssh_local_forward local endpoint must match scheduler_url exactly")
    if tunnel.get("target_host", "127.0.0.1") not in {"127.0.0.1", "::1"}:
        raise ValueError("ssh_local_forward target_host must be loopback")
    for field in ("local_port", "target_port", "remote_port"):
        raw_port = tunnel.get(field, 22 if field == "remote_port" else None)
        if (
            isinstance(raw_port, bool)
            or not isinstance(raw_port, int)
            or not 1 <= raw_port <= 65535
        ):
            raise ValueError(f"ssh_local_forward {field} must be between 1 and 65535")
    remote_host = tunnel.get("remote_host")
    if not isinstance(remote_host, str) or not _SSH_HOST.fullmatch(remote_host):
        raise ValueError("ssh_local_forward remote_host is invalid")
    remote_user = tunnel.get("remote_user")
    if not isinstance(remote_user, str) or not _SSH_USER.fullmatch(remote_user):
        raise ValueError("ssh_local_forward remote_user is invalid")
    _require_under(
        tunnel.get("identity_file"),
        _PERSISTENT_ROOT,
        "ssh_local_forward identity_file",
    )
    _require_under(
        tunnel.get("known_hosts_file"),
        _PERSISTENT_ROOT,
        "ssh_local_forward known_hosts_file",
    )
    _require_under(
        tunnel.get("state_directory"),
        _EPHEMERAL_ROOT,
        "ssh_local_forward state_directory",
    )
    ssh_executable = _absolute_path(
        tunnel.get("ssh_executable", "/usr/bin/ssh"),
        "ssh_local_forward ssh_executable",
    )
    if ssh_executable == PurePosixPath("/"):
        raise ValueError("ssh_local_forward ssh_executable must name an executable")


def _reject_embedded_secrets(value: object, label: str, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            if _SENSITIVE_KEY.search(key):
                raise ValueError(f"{label} embeds a credential at {'.'.join(child_path)}")
            _reject_embedded_secrets(child, label, path=child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_embedded_secrets(child, label, path=(*path, str(index)))
    elif isinstance(value, str) and _CREDENTIAL_VALUE.search(value):
        raise ValueError(f"{label} contains a credential-like value at {'.'.join(path)}")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _absolute_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be an absolute path")
    path = PurePosixPath(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _require_under(value: object, root: PurePosixPath, label: str) -> PurePosixPath:
    path = _absolute_path(value, label)
    if path == root or root not in path.parents:
        raise ValueError(f"{label} must be below {root}")
    return path


def _executable(value: str | Path | None, name: str) -> Path:
    candidate = shutil.which(name) if value is None else str(value)
    if candidate is None:
        raise FileNotFoundError(f"{name} is required to build an offline bootstrap artifact")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{name} executable is unavailable: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a secret-free AutoDL bootstrap release")
    parser.add_argument("--site-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--uv-binary", type=Path)
    parser.add_argument("--rclone-binary", type=Path)
    parser.add_argument(
        "--runtime-seed",
        type=Path,
        help="verified site-packages .tar.gz staged separately on the Lab1 rclone remote",
    )
    parser.add_argument(
        "--runtime-seed-remote",
        help="credential-free Lab1 rclone object; defaults to a lock/platform/hash keyed path",
    )
    parser.add_argument(
        "--runtime-seed-platform",
        help=(
            "exact target wheel platform, for example x86_64-manylinux_2_31; "
            "defaults to the build host"
        ),
    )
    parser.add_argument(
        "--allow-online-fallback",
        action="store_true",
        help="allow a seeded bootstrap to use package indexes if offline reconciliation fails",
    )
    arguments = parser.parse_args(argv)
    result = build_autodl_bootstrap(
        repository_root=arguments.repository_root,
        site_config=arguments.site_config,
        model_config=arguments.model_config,
        output_dir=arguments.output_dir,
        uv_binary=arguments.uv_binary,
        rclone_binary=arguments.rclone_binary,
        runtime_seed=arguments.runtime_seed,
        runtime_seed_remote=arguments.runtime_seed_remote,
        runtime_seed_platform=arguments.runtime_seed_platform,
        allow_online_fallback=arguments.allow_online_fallback,
    )
    print(f"built AutoDL bootstrap release {result.digest}")
    print(result.bootstrap_script)
    if result.runtime_seed_remote is not None:
        print(f"runtime seed remote: {result.runtime_seed_remote}")


if __name__ == "__main__":  # pragma: no cover - console entrypoint
    main()
