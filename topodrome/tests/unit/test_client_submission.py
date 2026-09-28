from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from topodrome.client import AsyncClient, Client
from topodrome.client import source as client_source_module
from topodrome.domain.specs import GPUReservationExtensionRequest, JobSpec
from topodrome.source_limits import LEGACY_SOURCE_LIMITS, SourceLimits

pytestmark = pytest.mark.unit


def _job(source_ref: str | None = None) -> JobSpec:
    return JobSpec(
        entrypoint="experiment.py",
        source_ref=source_ref,
        model="meta-llama-3-8b-instruct",
        resources={"gpu": 1, "min_vram_gb": 15},
    )


def _summary(spec: dict[str, Any]) -> dict[str, Any]:
    now = datetime(2026, 9, 8, tzinfo=UTC).isoformat()
    return {
        "job_id": "job_test",
        "state": "queued",
        "spec": spec,
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
    }


def _reservation_summary(*, state: str = "ready") -> dict[str, Any]:
    now = datetime(2026, 9, 12, tzinfo=UTC).isoformat()
    return {
        "reservation_id": "reservation_test",
        "site_id": "lab1",
        "gpu_uuid": "GPU-A",
        "gpu_index": 0,
        "requested_by": "@operator",
        "reason": "maintenance",
        "duration_seconds": 600,
        "state": state,
        "created_at": now,
        "expires_at": datetime(2026, 9, 12, 0, 10, tzinfo=UTC).isoformat(),
        "canceled_at": now if state == "canceled" else None,
        "eviction_requested_at": None,
    }


def test_sync_submit_uploads_source_before_idempotent_job(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment.py").write_text("def run(context):\n    return context.job_id\n")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/source-limits":
            return httpx.Response(200, json=LEGACY_SOURCE_LIMITS.model_dump(mode="json"))
        body = json.loads(request.content)
        if request.url.path == "/api/v1/sources":
            assert base64.b64decode(body["files"]["experiment.py"]).startswith(b"def run")
            return httpx.Response(201, json={"source_ref": "sha256:source"})
        assert request.url.path == "/api/v1/jobs"
        assert request.headers["Idempotency-Key"] == "submission-1"
        assert body["source_ref"] == "sha256:source"
        return httpx.Response(202, json=_summary(body))

    client = Client(transport=httpx.MockTransport(handler))
    handle = client.submit(_job(), source_dir=source, idempotency_key="submission-1")

    assert handle.job_id == "job_test"
    assert [request.url.path for request in requests] == [
        "/api/v1/source-limits",
        "/api/v1/sources",
        "/api/v1/jobs",
    ]
    client.close()


async def test_async_submit_uploads_source_and_preserves_idempotency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment.py").write_text("def run(context):\n    return context.attempt_id\n")
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/source-limits":
            return httpx.Response(200, json=LEGACY_SOURCE_LIMITS.model_dump(mode="json"))
        body = json.loads(request.content)
        if request.url.path == "/api/v1/sources":
            return httpx.Response(201, json={"source_ref": "sha256:async-source"})
        assert request.headers["Idempotency-Key"] == "submission-2"
        assert body["source_ref"] == "sha256:async-source"
        return httpx.Response(202, json=_summary(body))

    async with AsyncClient(transport=httpx.MockTransport(handler)) as client:
        summary = await client.submit(
            _job(),
            source_dir=source,
            idempotency_key="submission-2",
        )

    assert summary.job_id == "job_test"
    assert [request.url.path for request in requests] == [
        "/api/v1/source-limits",
        "/api/v1/sources",
        "/api/v1/jobs",
    ]


def test_submission_without_source_is_rejected_before_network() -> None:
    client = Client(transport=httpx.MockTransport(lambda _request: pytest.fail("network used")))
    with pytest.raises(ValueError, match="source_dir or a previously uploaded source_ref"):
        client.submit(_job())
    client.close()


async def test_async_submission_without_source_is_rejected_before_network() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("network used")

    async with AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="source_dir or a previously uploaded source_ref"):
            await client.submit(_job())


def test_sync_system_status_uses_bounded_status_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/status"
        return httpx.Response(200, json={"schema_version": 1, "sites": []})

    with Client(transport=httpx.MockTransport(handler)) as client:
        assert client.system_status() == {"schema_version": 1, "sites": []}


async def test_async_system_status_uses_bounded_status_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/status"
        return httpx.Response(200, json={"schema_version": 1, "gpus": []})

    async with AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await client.system_status() == {"schema_version": 1, "gpus": []}


def test_sync_gpu_reservation_client_preserves_target_idempotency_and_cancel() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/gpu-reservations":
            assert request.method == "POST"
            assert request.headers["Idempotency-Key"] == "reservation-key"
            assert json.loads(request.content) == {
                "site_id": "lab1",
                "gpu_uuid": "GPU-A",
                "gpu_index": None,
                "duration_seconds": 600,
                "reason": "maintenance",
                "force": False,
            }
            return httpx.Response(201, json=_reservation_summary())
        assert request.url.path == "/api/v1/gpu-reservations/reservation_test/cancel"
        assert request.method == "POST"
        return httpx.Response(200, json=_reservation_summary(state="canceled"))

    with Client(transport=httpx.MockTransport(handler)) as client:
        created = client.reserve_gpu(
            site_id="lab1",
            gpu_uuid="GPU-A",
            duration_seconds=600,
            reason="maintenance",
            idempotency_key="reservation-key",
        )
        canceled = client.cancel_gpu_reservation(created.reservation_id)

    assert created.state.value == "ready"
    assert canceled.state.value == "canceled"
    assert [request.url.path for request in requests] == [
        "/api/v1/gpu-reservations",
        "/api/v1/gpu-reservations/reservation_test/cancel",
    ]


async def test_async_gpu_reservation_client_lists_and_reads_audit_records() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/gpu-reservations":
            assert str(request.url.params) == "limit=7&active_only=true"
            return httpx.Response(200, json=[_reservation_summary()])
        assert request.url.path == "/api/v1/gpu-reservations/reservation_test"
        return httpx.Response(200, json=_reservation_summary())

    async with AsyncClient(transport=httpx.MockTransport(handler)) as client:
        listed = await client.gpu_reservations(limit=7, active_only=True)
        detail = await client.gpu_reservation("reservation_test")

    assert listed == [detail]


def test_sync_extension_sends_exact_expected_expiry():
    extension = GPUReservationExtensionRequest(
        expected_expires_at="2026-09-28T12:00:00+00:00", additional_seconds=1800
    )

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/gpu-reservations/reservation_test/extend"
        assert json.loads(request.content) == extension.model_dump(mode="json")
        return httpx.Response(200, json=_reservation_summary())

    with Client(transport=httpx.MockTransport(handler)) as client:
        assert client.extend_gpu_reservation("reservation_test", extension).state.value == "ready"


def _policy_reload_response(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert request.url.path == "/api/v1/policy/reload"
    assert request.content == b""
    return httpx.Response(
        200,
        json={
            "status": "draining",
            "generation": 4,
            "loaded_at": "2026-09-09T01:00:00+00:00",
            "pending_generation": 5,
            "pending_loaded_at": "2026-09-09T01:01:00+00:00",
            "blocker_count": 2,
        },
    )


def test_sync_client_stages_policy_reload_without_a_request_body() -> None:
    with Client(transport=httpx.MockTransport(_policy_reload_response)) as client:
        result = client.reload_policy()

    assert result["status"] == "draining"
    assert result["generation"] == 4
    assert result["pending_generation"] == 5
    assert result["blocker_count"] == 2


async def test_async_client_stages_policy_reload_without_a_request_body() -> None:
    async with AsyncClient(transport=httpx.MockTransport(_policy_reload_response)) as client:
        result = await client.reload_policy()

    assert result["status"] == "draining"
    assert result["generation"] == 4
    assert result["pending_generation"] == 5
    assert result["blocker_count"] == 2


def _live_log_response(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/v1/jobs/job-1/attempts/attempt-1/logs"
    assert request.url.params["offset"] == "3"
    assert request.url.params["anchor_sha256"] == "a" * 64
    return httpx.Response(
        200,
        content=b"next",
        headers={
            "X-Topodrome-Log-Offset": "3",
            "X-Topodrome-Log-Next-Offset": "7",
            "X-Topodrome-Log-Size": "7",
            "X-Topodrome-Log-Reset": "false",
            "X-Topodrome-Log-Complete": "true",
            "X-Topodrome-Log-Anchor-SHA256": "b" * 64,
        },
    )


def test_sync_client_reads_completed_attempt_log_range() -> None:
    with Client(transport=httpx.MockTransport(_live_log_response)) as client:
        chunk = client.attempt_log_chunk(
            "job-1",
            "attempt-1",
            offset=3,
            anchor_sha256="a" * 64,
        )

    assert chunk.data == b"next"
    assert chunk.next_offset == 7
    assert chunk.complete


async def test_async_client_reads_completed_attempt_log_range() -> None:
    async with AsyncClient(transport=httpx.MockTransport(_live_log_response)) as client:
        chunk = await client.attempt_log_chunk(
            "job-1",
            "attempt-1",
            offset=3,
            anchor_sha256="a" * 64,
        )

    assert chunk.data == b"next"
    assert chunk.next_offset == 7
    assert chunk.complete


def test_source_dir_cannot_silently_replace_an_existing_source_ref(tmp_path: Path) -> None:
    client = Client(transport=httpx.MockTransport(lambda _request: pytest.fail("network used")))

    with pytest.raises(ValueError, match="source_dir or source_ref"):
        client.submit(_job("sha256:existing"), source_dir=tmp_path)
    client.close()


def test_source_upload_rejects_symlinks_and_empty_trees(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/source-limits"
        return httpx.Response(200, json=LEGACY_SOURCE_LIMITS.model_dump(mode="json"))

    client = Client(transport=httpx.MockTransport(handler))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no publishable files"):
        client.upload_source(empty)

    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "outside.py"
    target.write_text("secret = True\n")
    (source / "linked.py").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        client.upload_source(source)
    client.close()


def test_client_source_reader_mirrors_file_count_and_decoded_size_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    source.mkdir()
    (source / "one.bin").write_bytes(b"123")
    (source / "two.bin").write_bytes(b"12")

    with pytest.raises(ValueError, match="per-file"):
        client_source_module.read_source_files(
            source,
            limits=SourceLimits(
                max_files=2,
                max_path_chars=20,
                max_file_bytes=2,
                max_total_bytes=10,
            ),
        )

    with pytest.raises(ValueError, match="total decoded-size"):
        client_source_module.read_source_files(
            source,
            limits=SourceLimits(
                max_files=2,
                max_path_chars=20,
                max_file_bytes=3,
                max_total_bytes=4,
            ),
        )

    with pytest.raises(ValueError, match="file limit"):
        client_source_module.read_source_files(
            source,
            limits=SourceLimits(
                max_files=1,
                max_path_chars=20,
                max_file_bytes=3,
                max_total_bytes=10,
            ),
        )


def test_sync_source_limit_discovery_is_bounded_and_legacy_404_falls_back(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-limits"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"12345")
    calls: list[str] = []

    def strict_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/api/v1/source-limits"
        return httpx.Response(
            200,
            json=SourceLimits(
                max_files=2,
                max_path_chars=20,
                max_file_bytes=4,
                max_total_bytes=10,
            ).model_dump(mode="json"),
        )

    with Client(transport=httpx.MockTransport(strict_handler)) as client:
        with pytest.raises(ValueError, match="per-file"):
            client.upload_source(source)
    assert calls == ["/api/v1/source-limits"]

    def legacy_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/source-limits":
            return httpx.Response(404)
        assert request.url.path == "/api/v1/sources"
        return httpx.Response(201, json={"source_ref": "sha256:legacy"})

    with Client(transport=httpx.MockTransport(legacy_handler)) as client:
        assert client.upload_source(source) == "sha256:legacy"
