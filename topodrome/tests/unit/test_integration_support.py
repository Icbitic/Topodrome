from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import shlex
import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts.integration_support import (
    IntegrationHarnessError,
    ManagedJupyterClient,
    ManagedSSHClient,
    TerminalRegistry,
    create_managed_python_archive,
    create_source_archive,
)

pytestmark = pytest.mark.unit


async def test_terminal_cleanup_requires_safe_name_and_endpoint_registry(tmp_path: Path) -> None:
    base_url = "https://jupyter.invalid"
    registry = TerminalRegistry(base_url, root=tmp_path)
    owned = "topodrome_it_012345abcdef_1234abcd"
    unowned = "topodrome_it_012345abcdef_deadbeef"
    legacy = "topodrome-probe-123"
    registry.register(owned)
    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[{"name": owned}, {"name": unowned}, {"name": legacy}],
            )
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(204)
        raise AssertionError(request.method)

    async with ManagedJupyterClient(
        base_url,
        "test-token",
        registry=registry,
        transport=httpx.MockTransport(handler),
    ) as client:
        removed = await client.cleanup_registered_terminals()

    assert removed == (owned,)
    assert deleted == [f"/api/terminals/{owned}"]
    assert registry.names() == frozenset()


async def test_managed_terminal_waits_for_shell_and_chunks_long_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_marker = "TOPODROME_TEST_COMMAND_READY"

    class FakeConnection:
        def __init__(self) -> None:
            self.messages: asyncio.Queue[str] = asyncio.Queue()
            self.sent: list[str] = []
            self.closed = False

        async def send(self, message: str) -> None:
            self.sent.append(message)
            value = json.loads(message)[1]
            if value.startswith("stty -echo"):
                quoted = re.findall(r"'([^']*)'", value)
                await self.messages.put(json.dumps(["stdout", quoted[1] + quoted[2]]))
            elif value.startswith("printf '%s%s"):
                quoted = re.findall(r"'([^']*)'", value)
                await self.messages.put(json.dumps(["stdout", quoted[1] + quoted[2]]))
            elif value.startswith(')"; _topodrome_status='):
                await self.messages.put(json.dumps(["stdout", ready_marker]))

        async def recv(self) -> str:
            return await self.messages.get()

        async def close(self) -> None:
            self.closed = True

    connection = FakeConnection()

    async def connect(*_: object, **__: object) -> FakeConnection:
        return connection

    import websockets

    monkeypatch.setattr(websockets, "connect", connect)
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if request.method == "POST":
            return httpx.Response(201, json=json.loads(request.content))
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(request.method)

    command = "printf integration # " + ("x" * 12_000)
    registry = TerminalRegistry("https://jupyter.invalid", root=tmp_path)
    async with ManagedJupyterClient(
        "https://jupyter.invalid",
        "test-token",
        registry=registry,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.run_terminal(
            command,
            ready_marker=ready_marker,
            run_id="012345abcdef",
            timeout_seconds=1,
        )

    values = [json.loads(message)[1] for message in connection.sent]
    assert values[0].startswith("stty -echo")
    delimiter_match = re.search(r"<<'([^']+)'", values[2])
    assert delimiter_match is not None
    delimiter_index = values.index(f"{delimiter_match.group(1)}\n")
    encoded = "".join(value.rstrip("\n") for value in values[3:delimiter_index])
    assert base64.b64decode(encoded, validate=True) == f"{command}\n".encode()
    assert max(len(value.encode()) for value in values) < 2_048
    assert result.output == ready_marker
    assert requests == ["POST", "DELETE"]
    assert connection.closed
    assert registry.names() == frozenset()


def test_terminal_registry_rejects_non_harness_names(tmp_path: Path) -> None:
    registry = TerminalRegistry("https://jupyter.invalid", root=tmp_path)

    with pytest.raises(ValueError, match="unsafe terminal"):
        registry.register("topodrome-probe-123")


def test_source_archive_excludes_secret_files(tmp_path: Path) -> None:
    (tmp_path / ".python-version").write_text("3.11\n")
    (tmp_path / "README.md").write_text("# Test\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n")
    (tmp_path / "src" / ".env").write_text("TOKEN=never-archive\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "models.example.yaml").write_text("models: {}\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "remote_model_smoke.py").write_text("print('ok')\n")

    payload = create_source_archive(tmp_path)

    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
        assert "src/module.py" in names
        assert "src/.env" not in names
        assert all("never-archive" not in json.dumps(member.pax_headers) for member in archive)


def test_managed_python_archive_is_confined_to_distribution_root(tmp_path: Path) -> None:
    distribution = tmp_path / "cpython"
    executable = distribution / "bin" / "python3.11"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    executable.chmod(0o700)
    standard_library = distribution / "lib" / "python3.11"
    standard_library.mkdir(parents=True)
    (standard_library / "marker.py").write_text("VALUE = 1\n")
    outside = tmp_path / "outside-secret"
    outside.write_text("never archive me")

    payload = create_managed_python_archive(executable)

    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "bin/python3.11" in names
    assert "lib/python3.11/marker.py" in names
    assert all("outside-secret" not in name for name in names)


def test_managed_ssh_client_confines_paths_and_redacts_password() -> None:
    password = "ssh-password-canary"  # noqa: S105 - redaction canary
    client = ManagedSSHClient(
        "worker.invalid",
        22022,
        password,
        run_id="012345abcdef",
    )

    assert password not in repr(client)
    assert client.remote_root == "/root/autodl-tmp/topodrome_it/012345abcdef"
    assert client._child_path("source.tar.gz").startswith(client.remote_root + "/")
    with pytest.raises(ValueError, match="outside"):
        client._child_path("../secrets.env")


async def test_managed_ssh_client_streams_local_path_in_bounded_chunks(
    tmp_path: Path,
) -> None:
    payload = (b"topodrome-runtime-seed" * 150_000) + b"tail"
    source = tmp_path / "runtime.tar.zst"
    source.write_bytes(payload)

    class Connection:
        async def run(self, command, *, stdin, **kwargs):
            parts = shlex.split(command)
            chunks = []
            digest = hashlib.sha256()
            while chunk := await stdin.read(kwargs["bufsize"]):
                chunks.append(len(chunk))
                digest.update(chunk)
            self.chunks = chunks
            self.digest = digest.hexdigest()
            self.root, self.name, size, expected, self.mode = parts[-5:]
            assert self.digest == expected
            return SimpleNamespace(stdout=f"TOPODROME_SSH_FILE_OK {size} {expected}\n".encode())

    client = ManagedSSHClient(
        "worker.invalid",
        22022,
        "test-password",
        run_id="012345abcdef",
    )
    connection = Connection()
    client._connection = connection

    remote = await client.put_path(source, "runtime.tar.zst", mode=0o640)

    assert remote == f"{client.remote_root}/runtime.tar.zst"
    assert connection.root == client.remote_root
    assert connection.name == "runtime.tar.zst"
    assert connection.mode == "640"
    assert connection.digest == hashlib.sha256(payload).hexdigest()
    assert len(connection.chunks) > 1
    assert max(connection.chunks) <= 256 * 1024


async def test_managed_ssh_client_put_path_rejects_invalid_inputs(tmp_path: Path) -> None:
    client = ManagedSSHClient(
        "worker.invalid",
        22022,
        "test-password",
        run_id="012345abcdef",
    )
    client._connection = object()
    source = tmp_path / "runtime.tar.zst"
    source.write_bytes(b"seed")

    with pytest.raises(ValueError, match="outside"):
        await client.put_path(source, "../runtime.tar.zst")
    with pytest.raises(ValueError, match="permission bits"):
        await client.put_path(source, "runtime.tar.zst", mode=0o1640)
    with pytest.raises(IntegrationHarnessError, match="source is unavailable"):
        await client.put_path(tmp_path / "missing.tar.zst", "runtime.tar.zst")
