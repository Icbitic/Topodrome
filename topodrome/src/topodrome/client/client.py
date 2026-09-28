from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from topodrome.domain.specs import (
    GPUReservationExtensionRequest,
    GPUReservationRequest,
    GPUReservationSummary,
    JobSpec,
    JobSummary,
    KernelOperationRequest,
    KernelOperationResult,
    KernelRequest,
    KernelSummary,
)
from topodrome.scheduler.assets import (
    AssetPublishRequest,
    AssetSummary,
    AssetVisibility,
)
from topodrome.source_limits import LEGACY_SOURCE_LIMITS, SourceLimits

from .config import client_connection
from .source import read_source_files


@dataclass(frozen=True, slots=True)
class AttemptLogChunk:
    data: bytes
    offset: int
    next_offset: int
    size_bytes: int
    reset: bool
    anchor_sha256: str | None
    complete: bool


class Client:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url, credential = client_connection(base_url, token)
        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def submit(
        self,
        spec: JobSpec | None = None,
        *,
        source_dir: str | Path | None = None,
        idempotency_key: str | None = None,
        **job: Any,
    ) -> JobHandle:
        value = spec or JobSpec.model_validate(job)
        if source_dir is not None:
            if value.source_ref is not None:
                raise ValueError("provide source_dir or source_ref, not both")
            value = value.model_copy(update={"source_ref": self.upload_source(source_dir)})
        elif value.source_ref is None:
            raise ValueError(
                "batch submission requires source_dir or a previously uploaded source_ref"
            )
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        response = self._http.post(
            "/api/v1/jobs", json=value.model_dump(mode="json"), headers=headers
        )
        _raise(response)
        summary = JobSummary.model_validate(response.json())
        return JobHandle(client=self, job_id=summary.job_id)

    def upload_source(self, root: str | Path) -> str:
        files = read_source_files(root, limits=self.source_limits())
        response = self._http.post("/api/v1/sources", json={"files": files})
        _raise(response)
        return str(response.json()["source_ref"])

    def submit_array(self, spec=None, *, source_dir=None, idempotency_key=None, **values):
        from topodrome.domain.arrays import JobArraySpec

        request = (
            spec if isinstance(spec, JobArraySpec) else JobArraySpec.model_validate(spec or values)
        )
        if source_dir is not None:
            if request.job.source_ref is not None:
                raise ValueError("provide source_dir or source_ref, not both")
            request = request.model_copy(
                update={
                    "job": request.job.model_copy(
                        update={"source_ref": self.upload_source(source_dir)}
                    )
                }
            )
        if request.job.source_ref is None:
            raise ValueError("array submission requires source_dir or source_ref")
        response = self._http.post(
            "/api/v1/arrays",
            json=request.model_dump(mode="json"),
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        _raise(response)
        return response.json()

    def array(self, array_id: str, *, after=-1, limit=100):
        response = self._http.get(
            f"/api/v1/arrays/{array_id}", params={"after": after, "limit": limit}
        )
        _raise(response)
        return response.json()

    def arrays(self):
        response = self._http.get("/api/v1/arrays")
        _raise(response)
        return response.json()

    def update_array(self, array_id: str, *, max_parallel=None, held=None):
        from topodrome.domain.arrays import ArrayUpdate

        update = ArrayUpdate(max_parallel=max_parallel, held=held)
        response = self._http.patch(
            f"/api/v1/arrays/{array_id}", json=update.model_dump(exclude_none=True)
        )
        _raise(response)
        return response.json()

    def cancel_array(self, array_id: str):
        response = self._http.post(f"/api/v1/arrays/{array_id}/cancel")
        _raise(response)
        return response.json()

    def source_limits(self) -> SourceLimits:
        """Read the active server admission bounds, with legacy-server fallback."""

        response = self._http.get("/api/v1/source-limits")
        if response.status_code == 404:
            return LEGACY_SOURCE_LIMITS
        _raise(response)
        return SourceLimits.model_validate(response.json())

    def publish_asset(
        self,
        request: AssetPublishRequest | None = None,
        **values: Any,
    ) -> AssetSummary:
        """Give committed job output an immutable, reusable asset name."""

        value = request or AssetPublishRequest.model_validate(values)
        response = self._http.post(
            "/api/v1/assets",
            json=value.model_dump(mode="json"),
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    def upload_asset(
        self,
        path: str | Path,
        *,
        name: str,
        version: str,
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        """Publish a local file/tree directly, with no producer job or GPU."""
        from topodrome.asset_upload import CONTENT_TYPE
        from topodrome.client.asset_upload import prepare_asset_upload, upload_chunks
        from topodrome.scheduler.assets import AssetUploadRequest

        value = AssetUploadRequest(name=name, version=version, visibility=visibility)
        header, source = prepare_asset_upload(path)
        response = self._http.post(
            f"/api/v1/asset-uploads/{value.name}/{value.version}",
            params={"visibility": value.visibility},
            headers={"Content-Type": CONTENT_TYPE},
            content=upload_chunks(header, source),
            timeout=600,
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    def assets(
        self,
        *,
        visibility: AssetVisibility | None = None,
        limit: int = 100,
        tags: tuple[str, ...] = (),
    ) -> list[AssetSummary]:
        params: dict[str, str | int | list[str]] = {"limit": limit}
        if visibility is not None:
            params["visibility"] = visibility
        if tags:
            from topodrome.scheduler.assets import normalize_tags

            params["tag"] = list(normalize_tags(tags))
        response = self._http.get("/api/v1/assets", params=params)
        _raise(response)
        return [AssetSummary.model_validate(item) for item in response.json()]

    def delete_asset(self, name: str, version: str, *, visibility: AssetVisibility = "private"):
        from topodrome.scheduler.assets import AssetDeleteResult, AssetUploadRequest

        value = AssetUploadRequest(name=name, version=version, visibility=visibility)
        response = self._http.delete(
            f"/api/v1/assets/{value.name}/{value.version}",
            params={"visibility": value.visibility},
        )
        _raise(response)
        return AssetDeleteResult.model_validate(response.json())

    def update_asset_tags(
        self,
        name: str,
        version: str,
        *,
        add=(),
        remove=(),
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        from topodrome.scheduler.assets import AssetTagUpdate, AssetUploadRequest

        value = AssetUploadRequest(name=name, version=version, visibility=visibility)
        update = AssetTagUpdate(add=add, remove=remove)
        response = self._http.patch(
            f"/api/v1/assets/{value.name}/{value.version}/tags",
            params={"visibility": value.visibility},
            json=update.model_dump(mode="json"),
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    def download_asset(
        self,
        name: str,
        version: str,
        destination: str | Path,
        *,
        visibility: AssetVisibility = "private",
    ) -> Path:
        """Download verified editable files to a new local destination; no GPU."""
        from topodrome.client.asset_download import CHUNK_BYTES, AssetDownload
        from topodrome.scheduler.assets import AssetUploadRequest

        coordinates = AssetUploadRequest(name=name, version=version, visibility=visibility)
        asset = self.asset(coordinates.name, coordinates.version, visibility=visibility)
        with AssetDownload(asset, destination) as download:
            for index, item in enumerate(asset.files):
                with self._http.stream(
                    "GET",
                    f"/api/v1/assets/{name}/{version}/files/{index}",
                    params={"visibility": visibility},
                    headers={"Accept-Encoding": "identity"},
                    timeout=600,
                ) as response:
                    if not response.is_success:
                        response.read()
                    _raise(response)
                    count = 0
                    digest = hashlib.sha256()
                    with download.open_file(item) as output:
                        for block in response.iter_bytes(chunk_size=CHUNK_BYTES):
                            count += len(block)
                            if count > item.size_bytes:
                                raise ValueError("asset download exceeds manifest size")
                            output.write(block)
                            digest.update(block)
                        download.verify_file(output, item, count, digest.hexdigest())
            return download.commit()

    def asset(
        self,
        name: str,
        version: str,
        *,
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        response = self._http.get(
            f"/api/v1/assets/{name}/{version}",
            params={"visibility": visibility},
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    def status(self, job_id: str) -> JobSummary:
        response = self._http.get(f"/api/v1/jobs/{job_id}")
        _raise(response)
        return JobSummary.model_validate(response.json())

    def attempts(self, job_id: str) -> list[dict[str, Any]]:
        response = self._http.get(f"/api/v1/jobs/{job_id}/attempts")
        _raise(response)
        return list(response.json())

    def logs(self, job_id: str, *, tail_bytes: int = 131072) -> str:
        response = self._http.get(f"/api/v1/jobs/{job_id}/logs", params={"tail_bytes": tail_bytes})
        _raise(response)
        return response.text

    def attempt_log_chunk(
        self,
        job_id: str,
        attempt_id: str,
        *,
        offset: int = 0,
        limit_bytes: int = 262144,
        anchor_sha256: str | None = None,
    ) -> AttemptLogChunk:
        params: dict[str, str | int] = {
            "offset": offset,
            "limit_bytes": limit_bytes,
        }
        if anchor_sha256 is not None:
            params["anchor_sha256"] = anchor_sha256
        response = self._http.get(
            f"/api/v1/jobs/{job_id}/attempts/{attempt_id}/logs",
            params=params,
        )
        _raise(response)
        return _parse_attempt_log_chunk(response)

    def cancel(self, job_id: str) -> JobSummary:
        response = self._http.post(f"/api/v1/jobs/{job_id}/cancel")
        _raise(response)
        return JobSummary.model_validate(response.json()["job"])

    def wait(
        self, job_id: str, *, poll_seconds: float = 1, timeout: float | None = None
    ) -> JobSummary:
        started = time.monotonic()
        while True:
            result = self.status(job_id)
            if result.state.value in {"succeeded", "failed", "canceled"}:
                return result
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"job {job_id} did not finish within {timeout} seconds")
            time.sleep(poll_seconds)

    def kernel(self, request: KernelRequest | None = None, **values: Any) -> KernelHandle:
        value = request or KernelRequest.model_validate(values)
        response = self._http.post("/api/v1/kernels", json=value.model_dump(mode="json"))
        _raise(response)
        summary = KernelSummary.model_validate(response.json())
        return KernelHandle(client=self, session_id=summary.session_id)

    def kernel_status(self, session_id: str) -> KernelSummary:
        response = self._http.get(f"/api/v1/kernels/{session_id}")
        _raise(response)
        return KernelSummary.model_validate(response.json())

    def stop_kernel(self, session_id: str) -> KernelSummary:
        response = self._http.delete(f"/api/v1/kernels/{session_id}")
        _raise(response)
        return KernelSummary.model_validate(response.json())

    def kernel_connect_url(self, session_id: str) -> str:
        response = self._http.get(f"/api/v1/kernels/{session_id}/connect")
        _raise(response)
        return str(response.json()["url"])

    def kernel_notebook_server(self, session_id: str) -> dict[str, Any]:
        """Start or recover the stock Jupyter Server for an IDE session."""

        response = self._http.post(f"/api/v1/kernels/{session_id}/notebook-server")
        _raise(response)
        return dict(response.json())

    def kernel_notebook_server_info(self, session_id: str) -> dict[str, Any]:
        response = self._http.get(f"/api/v1/kernels/{session_id}/notebook-server")
        _raise(response)
        return dict(response.json())

    def stop_kernel_notebook_server(
        self,
        session_id: str,
        *,
        stop_kernel: bool = True,
    ) -> dict[str, Any]:
        response = self._http.delete(
            f"/api/v1/kernels/{session_id}/notebook-server",
            params={"stop_kernel": stop_kernel},
        )
        _raise(response)
        return dict(response.json())

    def execute_kernel(
        self,
        session_id: str,
        operation: KernelOperationRequest | str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> KernelOperationResult:
        request = (
            operation
            if isinstance(operation, KernelOperationRequest)
            else KernelOperationRequest(operation=operation, arguments=arguments or {})
        )
        response = self._http.post(
            self.kernel_connect_url(session_id),
            json=request.model_dump(mode="json"),
        )
        _raise(response)
        return KernelOperationResult.model_validate(response.json())

    def sites(self) -> list[dict[str, Any]]:
        return self._list("/api/v1/sites")

    def gpus(self) -> list[dict[str, Any]]:
        return self._list("/api/v1/gpus")

    def models(self) -> list[dict[str, Any]]:
        return self._list("/api/v1/models")

    def reserve_gpu(
        self,
        reservation: GPUReservationRequest | None = None,
        *,
        idempotency_key: str | None = None,
        **values: Any,
    ) -> GPUReservationSummary:
        value = reservation or GPUReservationRequest.model_validate(values)
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        response = self._http.post(
            "/api/v1/gpu-reservations",
            json=value.model_dump(mode="json"),
            headers=headers,
        )
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    def gpu_reservations(
        self,
        *,
        limit: int = 100,
        active_only: bool = False,
    ) -> list[GPUReservationSummary]:
        response = self._http.get(
            "/api/v1/gpu-reservations",
            params={"limit": limit, "active_only": active_only},
        )
        _raise(response)
        return [GPUReservationSummary.model_validate(row) for row in response.json()]

    def gpu_reservation(self, reservation_id: str) -> GPUReservationSummary:
        response = self._http.get(f"/api/v1/gpu-reservations/{reservation_id}")
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    def cancel_gpu_reservation(self, reservation_id: str) -> GPUReservationSummary:
        response = self._http.post(f"/api/v1/gpu-reservations/{reservation_id}/cancel")
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    def extend_gpu_reservation(
        self,
        reservation_id: str,
        extension: GPUReservationExtensionRequest,
    ) -> GPUReservationSummary:
        response = self._http.post(
            f"/api/v1/gpu-reservations/{reservation_id}/extend",
            json=extension.model_dump(mode="json"),
        )
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    def system_status(self) -> dict[str, Any]:
        """Fetch one sanitized, read-only operator snapshot."""

        response = self._http.get("/api/v1/status")
        _raise(response)
        return dict(response.json())

    def reload_policy(self) -> dict[str, Any]:
        """Validate and stage the configured scheduling policy."""

        response = self._http.post("/api/v1/policy/reload")
        _raise(response)
        return dict(response.json())

    def storage_gc(self, *, apply: bool = False) -> dict[str, Any]:
        response = self._http.post("/api/v1/storage/gc", params={"apply": apply}, timeout=120)
        _raise(response)
        return dict(response.json())

    def access_users(self) -> list[str]:
        """List Unix accounts currently allowed to use this scheduler."""

        response = self._http.get("/api/v1/access/users")
        _raise(response)
        return [str(value) for value in response.json()["users"]]

    def grant_access_user(self, username: str) -> dict[str, str]:
        """Grant or rotate a user's credential and return it exactly once."""

        response = self._http.post(f"/api/v1/access/users/{username}")
        _raise(response)
        return {str(key): str(value) for key, value in response.json().items()}

    def revoke_access_user(self, username: str) -> dict[str, Any]:
        response = self._http.delete(f"/api/v1/access/users/{username}")
        _raise(response)
        return dict(response.json())

    def register_autodl(self, ssh: str, password: str) -> dict[str, Any]:
        response = self._http.post(
            "/api/v1/providers/autodl/register", json={"ssh": ssh, "password": password}, timeout=90
        )
        _raise(response)
        return response.json()

    def unregister_autodl(self, site_id: str) -> dict[str, Any]:
        response = self._http.delete(f"/api/v1/providers/autodl/registrations/{site_id}")
        _raise(response)
        return response.json()

    def release_autodl(self, site_id: str, *, force: bool = False) -> dict[str, Any]:
        response = self._http.post(
            f"/api/v1/providers/autodl/registrations/{site_id}/release", json={"force": force}
        )
        _raise(response)
        return response.json()

    def autodl_control(self) -> dict[str, Any]:
        response = self._http.get("/api/v1/providers/autodl/control")
        _raise(response)
        return dict(response.json())

    def autodl_instances(self) -> list[dict[str, Any]]:
        return self._list("/api/v1/providers/autodl/instances")

    def autodl_operations(self) -> list[dict[str, Any]]:
        return self._list("/api/v1/providers/autodl/operations")

    def _list(self, path: str) -> list[dict[str, Any]]:
        response = self._http.get(path)
        _raise(response)
        return list(response.json())


@dataclass(frozen=True, slots=True)
class JobHandle:
    client: Client
    job_id: str

    def status(self) -> JobSummary:
        return self.client.status(self.job_id)

    def attempts(self) -> list[dict[str, Any]]:
        return self.client.attempts(self.job_id)

    def logs(self, *, tail_bytes: int = 131072) -> str:
        return self.client.logs(self.job_id, tail_bytes=tail_bytes)

    def cancel(self) -> JobSummary:
        return self.client.cancel(self.job_id)

    def wait(self, *, poll_seconds: float = 1, timeout: float | None = None) -> JobSummary:
        return self.client.wait(self.job_id, poll_seconds=poll_seconds, timeout=timeout)


@dataclass(frozen=True, slots=True)
class KernelHandle:
    client: Client
    session_id: str

    def status(self) -> KernelSummary:
        return self.client.kernel_status(self.session_id)

    def connect_url(self) -> str:
        return self.client.kernel_connect_url(self.session_id)

    def notebook_server(self) -> dict[str, Any]:
        return self.client.kernel_notebook_server(self.session_id)

    def notebook_server_info(self) -> dict[str, Any]:
        return self.client.kernel_notebook_server_info(self.session_id)

    def stop_notebook_server(self, *, stop_kernel: bool = True) -> dict[str, Any]:
        return self.client.stop_kernel_notebook_server(
            self.session_id,
            stop_kernel=stop_kernel,
        )

    def execute(
        self,
        operation: KernelOperationRequest | str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> KernelOperationResult:
        return self.client.execute_kernel(
            self.session_id,
            operation,
            arguments=arguments,
        )

    def stop(self) -> KernelSummary:
        return self.client.stop_kernel(self.session_id)


def _raise(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Topodrome API returned {response.status_code}: {detail}") from exc


def _parse_attempt_log_chunk(response: httpx.Response) -> AttemptLogChunk:
    reset_value = response.headers.get("x-topodrome-log-reset")
    complete_value = response.headers.get("x-topodrome-log-complete")
    if reset_value not in {"true", "false"} or complete_value not in {
        "true",
        "false",
    }:
        raise RuntimeError("Topodrome API returned invalid live-log metadata")
    anchor = response.headers.get("x-topodrome-log-anchor-sha256")
    if anchor is not None and (
        len(anchor) != 64 or any(value not in "0123456789abcdef" for value in anchor)
    ):
        raise RuntimeError("Topodrome API returned invalid live-log metadata")
    start_offset = _log_header(response, "x-topodrome-log-offset")
    next_offset = _log_header(response, "x-topodrome-log-next-offset")
    size_bytes = _log_header(response, "x-topodrome-log-size")
    if (
        next_offset < start_offset
        or next_offset > size_bytes
        or next_offset - start_offset != len(response.content)
    ):
        raise RuntimeError("Topodrome API returned invalid live-log metadata")
    return AttemptLogChunk(
        data=response.content,
        offset=start_offset,
        next_offset=next_offset,
        size_bytes=size_bytes,
        reset=reset_value == "true",
        anchor_sha256=anchor,
        complete=complete_value == "true",
    )


def _log_header(response: httpx.Response, name: str) -> int:
    try:
        value = int(response.headers[name])
    except (KeyError, ValueError):
        raise RuntimeError("Topodrome API returned invalid live-log metadata") from None
    if value < 0:
        raise RuntimeError("Topodrome API returned invalid live-log metadata")
    return value
