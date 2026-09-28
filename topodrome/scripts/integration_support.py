"""Credential-safe helpers shared by real-environment integration harnesses.

Nothing in this module opts a caller into remote or paid mutations.  Test entry
points must check their explicit opt-in variable before constructing a client.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, Self, cast
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from pydantic import SecretStr

from topodrome.config import SecretBundle, load_secret_bundle
from topodrome.jupyter.terminal import (
    TerminalChannel,
    send_terminal_command,
    synchronize_terminal,
)
from topodrome.logging import redact

DEFAULT_SECRET_FILE = Path("/home/xiangkun/.config/topodrome/secrets.env")
REMOTE_JUPYTER_ACK = "RUN_REMOTE_JUPYTER_TESTS"
LAB1_GPU_ACK = "RUN_LAB1_GPU_TESTS"

_RUN_ID = re.compile(r"^[0-9a-f]{12}$")
_TERMINAL_NAME = re.compile(r"^topodrome_it_[0-9a-f]{12}_[0-9a-f]{8}$")
_SAFE_REMOTE_PATH = re.compile(r"^topodrome_it/[0-9a-f]{12}(?:/[A-Za-z0-9_.-]+)?$")
_SENSITIVE_ARCHIVE_NAME = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|secrets?(?:\..*)?|credentials?(?:\..*)?)$",
    re.IGNORECASE,
)


class IntegrationHarnessError(RuntimeError):
    """A deliberately credential-free external-harness failure."""


def secret_file_from_environment() -> Path:
    return Path(os.environ.get("TOPODROME_SECRET_FILE", str(DEFAULT_SECRET_FILE)))


def integration_secrets() -> SecretBundle:
    """Load secrets only from a mode-checked file outside the repository."""

    return load_secret_bundle(secret_file_from_environment())


def opted_in(variable: str, acknowledgement: str) -> bool:
    return os.environ.get(variable) == acknowledgement


def state_directory() -> Path:
    root = Path(
        os.environ.get(
            "TOPODROME_INTEGRATION_STATE_DIR",
            str(Path.home() / ".local" / "state" / "topodrome" / "integration"),
        )
    ).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def new_run_id() -> str:
    return uuid4().hex[:12]


def require_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run ID must be exactly twelve lowercase hexadecimal characters")
    return run_id


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("paid-run checkpoint values must be strings")
    return value


def jupyter_base_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("Jupyter base URL must be HTTP(S)")
    if parts.username or parts.password:
        raise ValueError("Jupyter credentials must not be embedded in the URL")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _endpoint_key(base_url: str) -> str:
    return hashlib.sha256(jupyter_base_url(base_url).encode("utf-8")).hexdigest()[:24]


def terminal_name(run_id: str) -> str:
    return f"topodrome_it_{require_run_id(run_id)}_{uuid4().hex[:8]}"


def _safe_terminal_name(name: str) -> bool:
    return bool(_TERMINAL_NAME.fullmatch(name))


class TerminalRegistry:
    """Persistent allow-list for terminals created by this harness.

    Cleanup requires both membership in this registry and the restrictive name
    pattern.  Merely resembling a Topodrome terminal is never sufficient.
    """

    def __init__(self, base_url: str, *, root: Path | None = None) -> None:
        self._endpoint = _endpoint_key(base_url)
        self._root = root or state_directory()
        self._path = self._root / "jupyter-terminals.json"
        self._lock_path = self._root / "jupyter-terminals.lock"

    def names(self) -> frozenset[str]:
        with self._locked():
            payload = self._read()
            values = payload.get(self._endpoint, [])
            return frozenset(name for name in values if _safe_terminal_name(name))

    def register(self, name: str) -> None:
        if not _safe_terminal_name(name):
            raise ValueError("refusing to register an unsafe terminal name")
        with self._locked():
            payload = self._read()
            values = set(payload.get(self._endpoint, []))
            values.add(name)
            payload[self._endpoint] = sorted(values)
            self._write(payload)

    def unregister(self, name: str) -> None:
        if not _safe_terminal_name(name):
            raise ValueError("refusing to unregister an unsafe terminal name")
        with self._locked():
            payload = self._read()
            values = set(payload.get(self._endpoint, []))
            values.discard(name)
            if values:
                payload[self._endpoint] = sorted(values)
            else:
                payload.pop(self._endpoint, None)
            self._write(payload)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                yield
        finally:
            # fdopen closes the descriptor; this branch handles construction errors.
            with suppress(OSError):
                os.close(descriptor)

    def _read(self) -> dict[str, list[str]]:
        if not self._path.exists():
            return {}
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise IntegrationHarnessError("terminal cleanup registry is unreadable") from None
        if not isinstance(value, dict):
            raise IntegrationHarnessError("terminal cleanup registry is invalid")
        result: dict[str, list[str]] = {}
        for key, names in value.items():
            if isinstance(key, str) and isinstance(names, list):
                result[key] = [name for name in names if isinstance(name, str)]
        return result

    def _write(self, value: Mapping[str, Sequence[str]]) -> None:
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(self._path)


@dataclass(frozen=True, slots=True)
class TerminalResult:
    name: str
    output: str


@dataclass(frozen=True, slots=True)
class SSHCommandResult:
    output: str
    exit_status: int


class ManagedSSHClient:
    """Credential-safe SSH/SFTP access to one disposable AutoDL worker.

    AutoDL's ``jupyter_domain`` currently exposes AutoPanel rather than a
    Jupyter Server API.  The provider snapshot also supplies a short-lived SSH
    endpoint, which is the reliable automation surface for the paid smoke.
    All file mutations are confined to one run-specific directory.
    """

    def __init__(
        self,
        host: str,
        port: int,
        password: str | SecretStr,
        *,
        run_id: str,
        username: str = "root",
        max_output_chars: int = 131_072,
    ) -> None:
        if not host.strip() or any(character.isspace() for character in host):
            raise ValueError("SSH host must be non-empty and contain no whitespace")
        if not 1 <= port <= 65535:
            raise ValueError("SSH port is outside the valid range")
        if username != "root":
            raise ValueError("the AutoDL integration harness only supports its root account")
        secret = password if isinstance(password, SecretStr) else SecretStr(password)
        if not secret.get_secret_value():
            raise ValueError("SSH password must not be empty")
        self.host = host
        self.port = port
        self.username = username
        self.run_id = require_run_id(run_id)
        self.remote_root = f"/root/autodl-tmp/topodrome_it/{self.run_id}"
        self._password = secret
        self._max_output_chars = max_output_chars
        self._connection: Any = None
        self._sftp: Any = None

    def __repr__(self) -> str:
        return f"ManagedSSHClient(host={self.host!r}, port={self.port!r}, run_id={self.run_id!r})"

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            import asyncssh

            async with asyncio.timeout(60):
                self._connection = await asyncssh.connect(
                    self.host,
                    port=self.port,
                    username=self.username,
                    password=self._password.get_secret_value(),
                    known_hosts=None,
                    login_timeout=30,
                )
                self._sftp = await self._connection.start_sftp_client()
                await self._sftp.makedirs(self.remote_root, exist_ok=True)
                await self._sftp.chmod(self.remote_root, 0o700)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            await self.aclose()
            raise IntegrationHarnessError("AutoDL SSH connection failed") from None

    async def aclose(self) -> None:
        connection, self._connection = self._connection, None
        self._sftp = None
        if connection is not None:
            connection.close()
            with suppress(Exception):
                await connection.wait_closed()

    async def put_file(self, name: str, content: bytes, *, mode: int = 0o600) -> str:
        path = self._child_path(name)
        if self._sftp is None:
            raise IntegrationHarnessError("AutoDL SSH client is not connected")
        try:
            # Proxy SSH links can be slow and AsyncSSH's single huge write can
            # otherwise sit below the harness watchdog without a useful stage.
            timeout = min(1200.0, max(60.0, len(content) / 32768.0))
            async with asyncio.timeout(timeout):
                async with self._sftp.open(path, "wb") as handle:
                    for offset in range(0, len(content), 1024 * 1024):
                        await handle.write(content[offset : offset + 1024 * 1024])
                await self._sftp.chmod(path, mode)
        except Exception:
            raise IntegrationHarnessError("AutoDL SFTP upload failed") from None
        return path

    async def put_path(
        self,
        local_path: Path,
        name: str,
        *,
        mode: int = 0o600,
    ) -> str:
        """Stream one local regular file into the managed remote directory.

        Unlike :meth:`put_file`, this method never materializes the whole file in
        memory.  It is used for multi-gigabyte runtime seeds copied from Lab1 to
        an AutoDL worker's replaceable local disk.
        """

        path = self._child_path(name)
        if self._sftp is None:
            raise IntegrationHarnessError("AutoDL SSH client is not connected")
        if mode < 0 or mode & ~0o777:
            raise ValueError("remote file mode must contain only permission bits")
        try:
            metadata = await asyncio.to_thread(local_path.stat)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError
        except OSError:
            raise IntegrationHarnessError("local SFTP upload source is unavailable") from None
        try:
            timeout = min(1200.0, max(60.0, metadata.st_size / 32768.0))
            async with asyncio.timeout(timeout):
                source = await asyncio.to_thread(local_path.open, "rb")
                try:
                    async with self._sftp.open(path, "wb") as target:
                        while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
                            await target.write(chunk)
                finally:
                    await asyncio.to_thread(source.close)
                await self._sftp.chmod(path, mode)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise IntegrationHarnessError("AutoDL SFTP upload failed") from None
        return path

    async def run(self, command: str, *, timeout_seconds: float) -> SSHCommandResult:
        if self._connection is None:
            raise IntegrationHarnessError("AutoDL SSH client is not connected")
        if not command.strip() or "\0" in command or timeout_seconds <= 0:
            raise ValueError("SSH command and timeout must be valid")
        try:
            result = await self._connection.run(
                command,
                check=False,
                timeout=timeout_seconds,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise IntegrationHarnessError("AutoDL SSH command failed") from None
        combined = f"{result.stdout or ''}\n{result.stderr or ''}"
        safe = str(redact(combined, (self._password.get_secret_value(),)))
        if len(safe) > self._max_output_chars:
            safe = safe[-self._max_output_chars :]
        return SSHCommandResult(output=safe, exit_status=int(result.exit_status or 0))

    async def cleanup(self) -> None:
        if self._sftp is None:
            return
        try:
            await self._sftp.rmtree(self.remote_root, ignore_errors=True)
        except Exception:
            raise IntegrationHarnessError("AutoDL SFTP cleanup failed") from None

    def _child_path(self, name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
            raise ValueError("remote filename is outside the managed run directory")
        return f"{self.remote_root}/{name}"


class ManagedJupyterClient:
    """Small named-terminal and Contents API client for integration tests."""

    def __init__(
        self,
        base_url: str,
        token: str | SecretStr,
        *,
        registry: TerminalRegistry | None = None,
        timeout_seconds: float = 30,
        max_output_chars: int = 131_072,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        secret = token if isinstance(token, SecretStr) else SecretStr(token)
        if not secret.get_secret_value():
            raise ValueError("Jupyter token must not be empty")
        self.base_url = jupyter_base_url(base_url)
        self._token = secret
        self._registry = registry or TerminalRegistry(self.base_url)
        self._timeout = timeout_seconds
        self._max_output_chars = max_output_chars
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"token {secret.get_secret_value()}"},
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"ManagedJupyterClient(base_url={self.base_url!r})"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def list_terminals(self) -> frozenset[str]:
        response = await self._request("GET", "/api/terminals")
        try:
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError
            return frozenset(
                str(item["name"]) for item in payload if isinstance(item, dict) and "name" in item
            )
        except (TypeError, ValueError):
            raise IntegrationHarnessError("Jupyter returned an invalid terminal list") from None

    async def cleanup_registered_terminals(self) -> tuple[str, ...]:
        """Delete only safe names that this endpoint's registry proves we created."""

        registered = self._registry.names()
        existing = await self.list_terminals()
        removed: list[str] = []
        for name in sorted(registered & existing):
            if not _safe_terminal_name(name):
                continue
            await self._delete_registered_terminal(name)
            removed.append(name)
        for missing in registered - existing:
            self._registry.unregister(missing)
        return tuple(removed)

    async def run_terminal(
        self,
        command: str,
        *,
        ready_marker: str,
        run_id: str,
        timeout_seconds: float,
    ) -> TerminalResult:
        if not command.strip() or "\0" in command:
            raise ValueError("terminal command must be non-empty and contain no NUL")
        if not ready_marker or ready_marker in command:
            raise ValueError("ready marker must be non-empty and absent from command text")
        if timeout_seconds <= 0:
            raise ValueError("terminal timeout must be positive")

        name = terminal_name(run_id)
        self._registry.register(name)  # Journal ownership before the ambiguous POST.
        await self._create_named_terminal(name)
        connection: Any = None
        output = ""
        try:
            try:
                import websockets
            except ImportError:
                raise IntegrationHarnessError(
                    "terminal integration requires the jupyter extra"
                ) from None
            connect = cast(Any, websockets.connect)
            connection = await connect(
                self._websocket_url(name),
                additional_headers={"Authorization": f"token {self._token.get_secret_value()}"},
                origin=self.base_url,
                open_timeout=self._timeout,
            )
            try:
                async with asyncio.timeout(timeout_seconds):
                    channel = cast(TerminalChannel, connection)
                    await synchronize_terminal(channel)
                    await send_terminal_command(channel, command)
                    marker_window = ""
                    while ready_marker not in marker_window:
                        value = _terminal_text(await connection.recv())
                        output += value
                        if len(output) > self._max_output_chars:
                            output = output[-self._max_output_chars :]
                        marker_window += value
                        if ready_marker not in marker_window:
                            overlap = len(ready_marker) - 1
                            marker_window = marker_window[-overlap:] if overlap else ""
            except TimeoutError:
                raise IntegrationHarnessError("Jupyter terminal command timed out") from None
            except IntegrationHarnessError:
                raise
            except Exception:
                raise IntegrationHarnessError("Jupyter terminal command failed") from None
            safe = redact(str(output), (self._token.get_secret_value(),))
            return TerminalResult(name=name, output=str(safe))
        finally:
            if connection is not None:
                with suppress(Exception):
                    await connection.close()
            with suppress(IntegrationHarnessError):
                await self._delete_registered_terminal(name)

    async def put_file(self, relative_path: str, content: bytes) -> None:
        path = _validate_remote_path(relative_path, require_file=True)
        await self._request(
            "PUT",
            f"/api/contents/{_quote_remote_path(path)}",
            json_body={
                "type": "file",
                "format": "base64",
                "content": base64.b64encode(content).decode("ascii"),
            },
        )

    async def delete_path(self, relative_path: str) -> None:
        path = _validate_remote_path(relative_path, require_file=False)
        await self._request("DELETE", f"/api/contents/{_quote_remote_path(path)}")

    async def _create_named_terminal(self, name: str) -> None:
        if not _safe_terminal_name(name) or name not in self._registry.names():
            raise IntegrationHarnessError("terminal name is not registered for creation")
        response = await self._request("POST", "/api/terminals", json_body={"name": name})
        try:
            payload = response.json()
            returned = payload["name"] if isinstance(payload, dict) else None
        except (TypeError, ValueError):
            returned = None
        if returned != name:
            # Keep the requested safe name journaled: POST outcome is ambiguous.
            raise IntegrationHarnessError("Jupyter did not honor the managed terminal name")

    async def _delete_registered_terminal(self, name: str) -> None:
        if not _safe_terminal_name(name) or name not in self._registry.names():
            raise IntegrationHarnessError("refusing to delete an unregistered terminal")
        await self._request("DELETE", f"/api/terminals/{quote(name, safe='')}")
        self._registry.unregister(name)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._http.request(
                method,
                f"{self.base_url}{path}",
                json=None if json_body is None else dict(json_body),
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            raise IntegrationHarnessError("Jupyter API request failed") from None

    def _websocket_url(self, terminal: str) -> str:
        parts = urlsplit(self.base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        path = f"{parts.path}/terminals/websocket/{quote(terminal, safe='')}"
        return urlunsplit((scheme, parts.netloc, path, "", ""))


def _terminal_text(message: str | bytes) -> str:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    try:
        value = json.loads(message)
    except json.JSONDecodeError:
        return ""
    if (
        not isinstance(value, list)
        or len(value) < 2
        or value[0] not in {"stdout", "stderr"}
        or not isinstance(value[1], str)
    ):
        return ""
    return value[1]


def python_terminal_command(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""


def decode_result(output: str, prefix: str) -> Mapping[str, Any]:
    values: list[Mapping[str, Any]] = []
    for line in output.replace("\r", "\n").splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        encoded = line[position + len(prefix) :].strip()
        try:
            decoded = base64.urlsafe_b64decode(encoded.encode("ascii"))
            value = json.loads(decoded)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values.append(value)
    if not values:
        raise IntegrationHarnessError("terminal result marker was missing or invalid")
    return values[-1]


def create_source_archive(repository_root: Path) -> bytes:
    """Build a small credential-filtered source archive for a disposable worker."""

    root = repository_root.resolve()
    required = (
        root / ".python-version",
        root / "README.md",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "src",
        root / "configs" / "models.example.yaml",
        root / "scripts" / "remote_model_smoke.py",
    )
    missing = tuple(str(path.relative_to(root)) for path in required if not path.exists())
    if missing:
        raise IntegrationHarnessError("source archive inputs are incomplete")

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for source in required:
            archive.add(
                source,
                arcname=str(source.relative_to(root)),
                recursive=True,
                filter=_archive_filter,
            )
    return stream.getvalue()


def create_managed_python_archive(executable: Path) -> bytes:
    """Archive one relocatable uv-managed CPython distribution.

    Some approved provider images expose only an end-of-life system Python and
    cannot reach uv's managed-Python download host.  The integration harness
    may therefore copy the *current locked interpreter distribution* alongside
    the uv binary.  The caller supplies an already resolved executable so this
    helper never searches arbitrary roots or follows a virtualenv directory.
    """

    resolved = executable.resolve(strict=True)
    if resolved.parent.name != "bin" or not resolved.name.startswith("python3.11"):
        raise IntegrationHarnessError("integration Python is not a CPython 3.11 distribution")
    root = resolved.parent.parent
    if root == Path(root.anchor) or not (root / "lib" / "python3.11").is_dir():
        raise IntegrationHarnessError("integration Python distribution is incomplete")

    total_size = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )
    if total_size > 512 * 1024 * 1024:
        raise IntegrationHarnessError("integration Python distribution exceeds safety limit")

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz", compresslevel=6) as archive:
        for source in sorted(root.iterdir()):
            archive.add(source, arcname=source.name, recursive=True)
    return stream.getvalue()


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = info.name.replace("\\", "/")
    components = set(PurePosixPath(name).parts)
    if components & {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}:
        return None
    if _SENSITIVE_ARCHIVE_NAME.search(name):
        return None
    return info


def _validate_remote_path(value: str, *, require_file: bool) -> str:
    value = value.strip().strip("/")
    if not _SAFE_REMOTE_PATH.fullmatch(value):
        raise ValueError("remote integration path is outside the managed run directory")
    path = PurePosixPath(value)
    if require_file and len(path.parts) != 3:
        raise ValueError("remote file path must name a file inside the managed run directory")
    return value


def _quote_remote_path(value: str) -> str:
    return "/".join(quote(component, safe="") for component in PurePosixPath(value).parts)


def secret_env_bytes(hf_token: SecretStr) -> bytes:
    value = hf_token.get_secret_value()
    if not value or "\n" in value or "\r" in value or "\0" in value:
        raise IntegrationHarnessError("Hugging Face token has an invalid format")
    return f"TOPODROME_HF_TOKEN={value}\n".encode()


__all__ = [
    "DEFAULT_SECRET_FILE",
    "LAB1_GPU_ACK",
    "REMOTE_JUPYTER_ACK",
    "IntegrationHarnessError",
    "ManagedJupyterClient",
    "ManagedSSHClient",
    "SSHCommandResult",
    "TerminalRegistry",
    "TerminalResult",
    "create_managed_python_archive",
    "create_source_archive",
    "decode_result",
    "integration_secrets",
    "jupyter_base_url",
    "new_run_id",
    "opted_in",
    "python_terminal_command",
    "require_run_id",
    "secret_env_bytes",
    "secret_file_from_environment",
    "state_directory",
]
