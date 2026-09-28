from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.client.config import DEFAULT_URL, client_connection, save_client_connection


def _private_environment(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True)
    path.parent.chmod(0o700)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def test_client_uses_owner_only_default_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "config"
    environment = config_root / "topodrome" / "client.env"
    _private_environment(
        environment,
        "TOPODROME_URL=http://127.0.0.1:9000/base\nTOPODROME_API_TOKEN=file-token\n",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))

    assert client_connection(None, None) == ("http://127.0.0.1:9000/base", "file-token")


def test_explicit_and_process_environment_take_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "private" / "client.env"
    _private_environment(
        environment,
        "TOPODROME_URL=http://127.0.0.1:9000\nTOPODROME_API_TOKEN=file-token\n",
    )
    monkeypatch.setenv("TOPODROME_CLIENT_ENV_FILE", str(environment))
    monkeypatch.setenv("TOPODROME_URL", "http://127.0.0.1:9001")
    monkeypatch.setenv("TOPODROME_API_TOKEN", "environment-token")

    assert client_connection("http://127.0.0.1:9002/", "explicit-token") == (
        "http://127.0.0.1:9002",
        "explicit-token",
    )
    assert client_connection(None, None) == (
        "http://127.0.0.1:9001",
        "environment-token",
    )


def test_client_rejects_overly_permissive_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "private" / "client.env"
    _private_environment(environment, "TOPODROME_API_TOKEN=file-token\n")
    environment.chmod(0o644)
    monkeypatch.setenv("TOPODROME_CLIENT_ENV_FILE", str(environment))

    with pytest.raises(PermissionError, match="owner-only"):
        client_connection(None, None)


def test_client_defaults_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPODROME_URL", raising=False)
    monkeypatch.delenv("TOPODROME_API_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/definitely/missing/topodrome-test-config")

    assert client_connection(None, None) == (DEFAULT_URL, None)


def test_save_client_connection_is_private_and_immediately_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
    monkeypatch.delenv("TOPODROME_URL", raising=False)
    monkeypatch.delenv("TOPODROME_API_TOKEN", raising=False)

    path = save_client_connection("http://127.0.0.1:8042/", "user-token")

    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert client_connection(None, None) == ("http://127.0.0.1:8042", "user-token")


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/socket",
        "http://user:password@127.0.0.1:8042",
        "http://127.0.0.1:8042?token=secret",
    ],
)
def test_save_client_connection_rejects_unsafe_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    with pytest.raises(ValueError, match="credential-free"):
        save_client_connection(url, "token")
