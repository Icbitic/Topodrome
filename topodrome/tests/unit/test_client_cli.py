from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from pydantic import BaseModel
from typer.testing import CliRunner

from topodrome.client import cli
from topodrome.domain.specs import KernelRequest, ModelShardedParallelism


class _FakeClient:
    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def system_status(self) -> dict[str, Any]:
        return {"schema_version": 1, "sites": []}


class _PrintableModel(BaseModel):
    value: str


def test_autodl_manual_release_controls(monkeypatch: Any) -> None:
    calls = []

    class ManualClient(_FakeClient):
        def release_autodl(self, site: str, *, force: bool = False) -> dict[str, Any]:
            values = {"site_id": site, "force": force, "state": "release_pending"}
            calls.append(values)
            return values

        def autodl_control(self) -> dict[str, Any]:
            return {"mode": "manual_ssh", "idle_timeout_seconds": 300}

        def autodl_instances(self) -> list:
            return []

        def autodl_operations(self) -> list:
            return []

    monkeypatch.setattr(cli, "Client", ManualClient)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["autodl", "release", "--site", "worker"]).exit_code == 0
    assert calls[-1] == {"site_id": "worker", "force": False, "state": "release_pending"}
    assert (
        runner.invoke(cli.app, ["autodl", "release", "--site", "worker", "--force"]).exit_code == 0
    )
    assert calls[-1]["force"] is True
    assert json.loads(runner.invoke(cli.app, ["autodl"]).stdout)["control"]["mode"] == "manual_ssh"
    for args in (["off"], ["on", "--one-gpu", "5"], ["rent"], ["release"]):
        assert runner.invoke(cli.app, ["autodl", *args]).exit_code != 0
    assert len(calls) == 2


def test_autodl_registration_without_ssh_dependency(monkeypatch: Any) -> None:
    import builtins
    import sys

    original_import = builtins.__import__

    def without_asyncssh(name, *args, **kwargs):
        if name == "asyncssh" or name.startswith("asyncssh."):
            raise ModuleNotFoundError("No module named 'asyncssh'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_asyncssh)
    # Force a fresh import without replacing class identities held by other
    # tests; monkeypatch restores the original module afterwards.
    monkeypatch.delitem(sys.modules, "topodrome.deployment.manual_autodl", raising=False)
    calls = []

    class RegistrationClient(_FakeClient):
        def register_autodl(self, ssh, password):
            calls.append((ssh, password))
            return {"state": "starting"}

    monkeypatch.setattr(cli, "Client", RegistrationClient)
    command = "ssh -p 23505 root@connect.bjb2.seetacloud.com"
    result = CliRunner().invoke(
        cli.app, ["autodl", "register"], input=command + "\nsynthetic-password\n"
    )
    assert result.exit_code == 0, result.output
    assert calls == [(command, "synthetic-password")]
    assert "synthetic-password" not in result.output


def test_print_recursively_serializes_models_in_collections(capsys: Any) -> None:
    cli._print([_PrintableModel(value="structured")])

    assert json.loads(capsys.readouterr().out) == [{"value": "structured"}]


class _KernelHandle:
    def status(self) -> dict[str, str]:
        return {"state": "requested"}


class _KernelClient(_FakeClient):
    request: KernelRequest | None = None

    def kernel(self, request: KernelRequest) -> _KernelHandle:
        type(self).request = request
        return _KernelHandle()


class _KernelLifecycleClient(_FakeClient):
    status_session_id: str | None = None
    stop_session_id: str | None = None

    def kernel_status(self, session_id: str) -> dict[str, str]:
        type(self).status_session_id = session_id
        return {"session_id": session_id, "state": "running"}

    def stop_kernel(self, session_id: str) -> dict[str, str]:
        type(self).stop_session_id = session_id
        return {"session_id": session_id, "state": "stopping"}


class _PolicyClient(_FakeClient):
    called = False

    def reload_policy(self) -> dict[str, Any]:
        type(self).called = True
        return {
            "status": "draining",
            "generation": 7,
            "loaded_at": "2026-09-09T01:00:00+00:00",
            "pending_generation": 8,
            "pending_loaded_at": "2026-09-09T01:01:00+00:00",
            "blocker_count": 3,
        }


class _ReservationClient(_FakeClient):
    request: Any = None
    idempotency_key: str | None = None
    canceled_id: str | None = None

    def reserve_gpu(self, request: Any, *, idempotency_key: str | None = None) -> dict[str, Any]:
        type(self).request = request
        type(self).idempotency_key = idempotency_key
        return {"reservation_id": "reservation-1", "state": "ready"}

    def gpu_reservations(self, *, limit: int, active_only: bool) -> list[dict[str, Any]]:
        return [{"reservation_id": "reservation-1", "active_only": active_only, "limit": limit}]

    def gpu_reservation(self, reservation_id: str) -> dict[str, Any]:
        return {"reservation_id": reservation_id, "state": "pending"}

    def cancel_gpu_reservation(self, reservation_id: str) -> dict[str, Any]:
        type(self).canceled_id = reservation_id
        return {"reservation_id": reservation_id, "state": "canceled"}


def test_extend_reservation_command_reads_then_compare_and_sets(monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    expiry = datetime(2026, 9, 28, 12, tzinfo=UTC)
    calls = []

    class ExtensionClient(_ReservationClient):
        def gpu_reservation(self, reservation_id):
            calls.append(("read", reservation_id))
            return SimpleNamespace(expires_at=expiry)

        def extend_gpu_reservation(self, reservation_id, request):
            calls.append(("extend", reservation_id, request))
            return {"reservation_id": reservation_id}

    monkeypatch.setattr(cli, "Client", ExtensionClient)
    result = CliRunner().invoke(
        cli.app, ["gpu-reservation-extend", "reservation-1", "--seconds", "1800"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0] == ("read", "reservation-1")
    assert calls[1][0:2] == ("extend", "reservation-1")
    assert calls[1][2].expected_expires_at == expiry and calls[1][2].additional_seconds == 1800


def test_top_command_wires_refresh_options_to_read_only_snapshot(
    monkeypatch: Any,
) -> None:
    observed: dict[str, object] = {}

    def watch(
        fetch: Callable[[], Mapping[str, Any]],
        *,
        interval_seconds: float,
        once: bool,
        output: TextIO,
        color: bool | None,
    ) -> None:
        observed.update(
            snapshot=fetch(),
            interval_seconds=interval_seconds,
            once=once,
            color=color,
        )

    monkeypatch.setattr(cli, "Client", _FakeClient)
    monkeypatch.setattr(cli, "watch_dashboard", watch)

    result = CliRunner().invoke(
        cli.app,
        ["top", "--once", "--interval", "0.5", "--no-color"],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "snapshot": {"schema_version": 1, "sites": []},
        "interval_seconds": 0.5,
        "once": True,
        "color": False,
    }


def test_login_prompts_without_echo_and_installs_user_credential(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    observed: dict[str, str] = {}

    def save(url: str, token: str) -> Any:
        observed.update(url=url, token=token)
        return tmp_path / "client.env"

    monkeypatch.setattr(cli, "save_client_connection", save)
    result = CliRunner().invoke(
        cli.app,
        ["login", "--url", "http://127.0.0.1:8042"],
        input="private-user-token\n",
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "url": "http://127.0.0.1:8042",
        "token": "private-user-token",
    }
    assert "private-user-token" not in result.output


def test_top_once_returns_nonzero_and_redacts_fetch_failure(monkeypatch: Any) -> None:
    secret = "must-not-render"  # noqa: S105 - redaction canary

    class FailingClient(_FakeClient):
        def system_status(self) -> dict[str, Any]:
            raise RuntimeError(f"https://scheduler.invalid/?token={secret}")

    monkeypatch.setattr(cli, "Client", FailingClient)

    result = CliRunner().invoke(cli.app, ["top", "--once", "--no-color"])

    assert result.exit_code == 1
    assert result.output == "TOPODROME  status refresh failed (RuntimeError)\n"
    assert secret not in result.output


def test_kernel_cli_preserves_single_gpu_default(monkeypatch: Any) -> None:
    _KernelClient.request = None
    monkeypatch.setattr(cli, "Client", _KernelClient)

    result = CliRunner().invoke(cli.app, ["kernel", "--model", "model-a"])

    assert result.exit_code == 0, result.output
    request = _KernelClient.request
    assert request is not None
    assert request.resources.gpu == 1
    assert request.model.parallelism.mode == "single"


def test_kernel_cli_constructs_explicit_model_sharded_gang(monkeypatch: Any) -> None:
    _KernelClient.request = None
    monkeypatch.setattr(cli, "Client", _KernelClient)

    result = CliRunner().invoke(
        cli.app,
        [
            "kernel",
            "--model",
            "model-a",
            "--gpus",
            "2",
            "--min-vram-gb",
            "12",
            "--max-memory-per-gpu",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    request = _KernelClient.request
    assert request is not None
    assert request.resources.gpu == 2
    assert isinstance(request.model.parallelism, ModelShardedParallelism)
    assert request.model.parallelism.degree == 2
    assert request.model.parallelism.engine == "transformers_device_map"
    assert request.model.parallelism.strategy == "balanced"
    assert request.model.parallelism.max_memory_gb_per_gpu == 10


def test_kernel_cli_requires_sharding_memory_for_multiple_gpus(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _KernelClient)

    result = CliRunner().invoke(
        cli.app,
        ["kernel", "--model", "model-a", "--gpus", "2"],
    )

    assert result.exit_code != 0
    assert "--max-memory-per-gpu is required" in result.output


def test_kernel_cli_rejects_sharding_memory_for_one_gpu(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _KernelClient)

    result = CliRunner().invoke(
        cli.app,
        [
            "kernel",
            "--model",
            "model-a",
            "--max-memory-per-gpu",
            "10",
        ],
    )

    assert result.exit_code != 0
    assert "requires --gpus greater than 1" in result.output


def test_kernel_cli_help_documents_multi_gpu_model_sharding() -> None:
    result = CliRunner().invoke(cli.app, ["kernel", "--help"])

    assert result.exit_code == 0, result.output
    assert "--gpus" in result.output
    assert "--max-memory-per-gpu" in result.output


def test_kernel_status_cli_prints_json_and_uses_kernel_session(
    monkeypatch: Any,
) -> None:
    _KernelLifecycleClient.status_session_id = None
    monkeypatch.setattr(cli, "Client", _KernelLifecycleClient)

    result = CliRunner().invoke(cli.app, ["kernel-status", "kernel-1"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "session_id": "kernel-1",
        "state": "running",
    }
    assert _KernelLifecycleClient.status_session_id == "kernel-1"


def test_kernel_stop_cli_prints_json_and_uses_kernel_session(
    monkeypatch: Any,
) -> None:
    _KernelLifecycleClient.stop_session_id = None
    monkeypatch.setattr(cli, "Client", _KernelLifecycleClient)

    result = CliRunner().invoke(cli.app, ["kernel-stop", "kernel-1"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "session_id": "kernel-1",
        "state": "stopping",
    }
    assert _KernelLifecycleClient.stop_session_id == "kernel-1"


def test_root_help_documents_kernel_lifecycle_commands() -> None:
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "kernel-status" in result.output
    assert "kernel-stop" in result.output


def test_policy_reload_cli_prints_staged_and_active_status(monkeypatch: Any) -> None:
    _PolicyClient.called = False
    monkeypatch.setattr(cli, "Client", _PolicyClient)

    result = CliRunner().invoke(cli.app, ["policy-reload"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "status": "draining",
        "generation": 7,
        "loaded_at": "2026-09-09T01:00:00+00:00",
        "pending_generation": 8,
        "pending_loaded_at": "2026-09-09T01:01:00+00:00",
        "blocker_count": 3,
    }
    assert _PolicyClient.called


def test_policy_reload_cli_redacts_api_failure(monkeypatch: Any) -> None:
    secret = "must-not-render"  # noqa: S105 - redaction canary

    class FailingPolicyClient(_FakeClient):
        def reload_policy(self) -> dict[str, Any]:
            raise RuntimeError(f"https://scheduler.invalid/?token={secret}")

    monkeypatch.setattr(cli, "Client", FailingPolicyClient)

    result = CliRunner().invoke(cli.app, ["policy-reload"])

    assert result.exit_code == 1
    assert result.output == "[topodrome] policy reload failed (RuntimeError)\n"
    assert secret not in result.output


def test_gpu_reserve_cli_builds_exact_index_request(monkeypatch: Any) -> None:
    _ReservationClient.request = None
    _ReservationClient.idempotency_key = None
    monkeypatch.setattr(cli, "Client", _ReservationClient)

    result = CliRunner().invoke(
        cli.app,
        [
            "gpu-reserve",
            "lab1",
            "--gpu-index",
            "3",
            "--duration-seconds",
            "900",
            "--reason",
            "maintenance",
            "--idempotency-key",
            "reserve-1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "ready"
    assert _ReservationClient.request.site_id == "lab1"
    assert _ReservationClient.request.gpu_index == 3
    assert _ReservationClient.request.gpu_uuid is None
    assert _ReservationClient.request.duration_seconds == 900
    assert _ReservationClient.idempotency_key == "reserve-1"


def test_gpu_reserve_cli_rejects_ambiguous_target(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _ReservationClient)
    result = CliRunner().invoke(
        cli.app,
        [
            "gpu-reserve",
            "lab1",
            "--gpu-index",
            "0",
            "--gpu-uuid",
            "GPU-A",
            "--duration-seconds",
            "60",
            "--reason",
            "invalid",
        ],
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_gpu_reservation_audit_and_cancel_commands(monkeypatch: Any) -> None:
    _ReservationClient.canceled_id = None
    monkeypatch.setattr(cli, "Client", _ReservationClient)

    listed = CliRunner().invoke(
        cli.app,
        ["gpu-reservations", "--active-only", "--limit", "7"],
    )
    detail = CliRunner().invoke(cli.app, ["gpu-reservation", "reservation-1"])
    canceled = CliRunner().invoke(
        cli.app,
        ["gpu-reservation-cancel", "reservation-1"],
    )

    assert listed.exit_code == detail.exit_code == canceled.exit_code == 0
    assert json.loads(listed.output)[0]["active_only"] is True
    assert json.loads(detail.output)["state"] == "pending"
    assert json.loads(canceled.output)["state"] == "canceled"
    assert _ReservationClient.canceled_id == "reservation-1"
