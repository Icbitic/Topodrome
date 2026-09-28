# Public parity with the synchronous SDK intentionally names the user-facing
# deadline `timeout`; the implementation below does use asyncio.timeout.
# ruff: noqa: ASYNC109

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from functools import partial
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

from .client import AttemptLogChunk, _parse_attempt_log_chunk
from .config import client_connection
from .source import read_source_files


class AsyncClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url, credential = client_connection(base_url, token)
        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def submit(
        self,
        spec: JobSpec | None = None,
        *,
        source_dir: str | Path | None = None,
        idempotency_key: str | None = None,
        **job: Any,
    ) -> JobSummary:
        value = spec or JobSpec.model_validate(job)
        if source_dir is not None:
            if value.source_ref is not None:
                raise ValueError("provide source_dir or source_ref, not both")
            source_ref = await self.upload_source(source_dir)
            value = value.model_copy(update={"source_ref": source_ref})
        elif value.source_ref is None:
            raise ValueError(
                "batch submission requires source_dir or a previously uploaded source_ref"
            )
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        response = await self._http.post(
            "/api/v1/jobs", json=value.model_dump(mode="json"), headers=headers
        )
        _raise(response)
        return JobSummary.model_validate(response.json())

    async def upload_source(self, root: str | Path) -> str:
        limits = await self.source_limits()
        files = await asyncio.to_thread(read_source_files, root, limits=limits)
        response = await self._http.post("/api/v1/sources", json={"files": files})
        _raise(response)
        return str(response.json()["source_ref"])

    async def submit_array(self, spec=None, *, source_dir=None, idempotency_key=None, **values):
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
                        update={"source_ref": await self.upload_source(source_dir)}
                    )
                }
            )
        if request.job.source_ref is None:
            raise ValueError("array submission requires source_dir or source_ref")
        response = await self._http.post(
            "/api/v1/arrays",
            json=request.model_dump(mode="json"),
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        _raise(response)
        return response.json()

    async def array(self, array_id: str, *, after=-1, limit=100):
        response = await self._http.get(
            f"/api/v1/arrays/{array_id}", params={"after": after, "limit": limit}
        )
        _raise(response)
        return response.json()

    async def arrays(self):
        response = await self._http.get("/api/v1/arrays")
        _raise(response)
        return response.json()

    async def update_array(self, array_id: str, *, max_parallel=None, held=None):
        from topodrome.domain.arrays import ArrayUpdate

        update = ArrayUpdate(max_parallel=max_parallel, held=held)
        response = await self._http.patch(
            f"/api/v1/arrays/{array_id}", json=update.model_dump(exclude_none=True)
        )
        _raise(response)
        return response.json()

    async def cancel_array(self, array_id: str):
        response = await self._http.post(f"/api/v1/arrays/{array_id}/cancel")
        _raise(response)
        return response.json()

    async def source_limits(self) -> SourceLimits:
        """Read the active server admission bounds, with legacy-server fallback."""

        response = await self._http.get("/api/v1/source-limits")
        if response.status_code == 404:
            return LEGACY_SOURCE_LIMITS
        _raise(response)
        return SourceLimits.model_validate(response.json())

    async def publish_asset(
        self,
        request: AssetPublishRequest | None = None,
        **values: Any,
    ) -> AssetSummary:
        """Give committed job output an immutable, reusable asset name."""

        value = request or AssetPublishRequest.model_validate(values)
        response = await self._http.post(
            "/api/v1/assets",
            json=value.model_dump(mode="json"),
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    async def upload_asset(
        self,
        path: str | Path,
        *,
        name: str,
        version: str,
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        """Stream a local file/tree as a named asset, without a GPU job."""
        from topodrome.asset_upload import CONTENT_TYPE
        from topodrome.client.asset_upload import prepare_asset_upload, upload_chunks
        from topodrome.scheduler.assets import AssetUploadRequest

        value = AssetUploadRequest(name=name, version=version, visibility=visibility)
        header, source = await asyncio.to_thread(prepare_asset_upload, path)
        chunks = upload_chunks(header, source)

        async def body() -> AsyncIterator[bytes]:
            try:
                while True:
                    pending = asyncio.create_task(asyncio.to_thread(next, chunks, None))
                    try:
                        block = await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        await pending
                        raise
                    if block is None:
                        break
                    yield block
            finally:
                chunks.close()

        response = await self._http.post(
            f"/api/v1/asset-uploads/{value.name}/{value.version}",
            params={"visibility": value.visibility},
            headers={"Content-Type": CONTENT_TYPE},
            content=body(),
            timeout=600,
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    async def assets(
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
        response = await self._http.get("/api/v1/assets", params=params)
        _raise(response)
        return [AssetSummary.model_validate(item) for item in response.json()]

    async def delete_asset(
        self, name: str, version: str, *, visibility: AssetVisibility = "private"
    ):
        from topodrome.scheduler.assets import AssetDeleteResult, AssetUploadRequest

        value = AssetUploadRequest(name=name, version=version, visibility=visibility)
        response = await self._http.delete(
            f"/api/v1/assets/{value.name}/{value.version}",
            params={"visibility": value.visibility},
        )
        _raise(response)
        return AssetDeleteResult.model_validate(response.json())

    async def update_asset_tags(
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
        response = await self._http.patch(
            f"/api/v1/assets/{value.name}/{value.version}/tags",
            params={"visibility": value.visibility},
            json=update.model_dump(mode="json"),
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    async def download_asset(
        self,
        name: str,
        version: str,
        destination: str | Path,
        *,
        visibility: AssetVisibility = "private",
    ) -> Path:
        """Stream verified asset files to a new local destination, without a job."""
        from topodrome.client.asset_download import CHUNK_BYTES, AssetDownload, download_io
        from topodrome.scheduler.assets import AssetUploadRequest

        coordinates = AssetUploadRequest(name=name, version=version, visibility=visibility)
        asset = await self.asset(coordinates.name, coordinates.version, visibility=visibility)
        with AssetDownload(asset, destination) as download:
            for index, item in enumerate(asset.files):
                async with self._http.stream(
                    "GET",
                    f"/api/v1/assets/{name}/{version}/files/{index}",
                    params={"visibility": visibility},
                    headers={"Accept-Encoding": "identity"},
                    timeout=600,
                ) as response:
                    if not response.is_success:
                        await response.aread()
                    _raise(response)
                    count = 0
                    digest = hashlib.sha256()
                    with download.open_file(item) as output:
                        async for block in response.aiter_bytes(chunk_size=CHUNK_BYTES):
                            count += len(block)
                            if count > item.size_bytes:
                                raise ValueError("asset download exceeds manifest size")
                            await download_io(partial(output.write, block))
                            digest.update(block)
                        await download_io(
                            partial(download.verify_file, output, item, count, digest.hexdigest())
                        )
            return download.commit()

    async def asset(
        self,
        name: str,
        version: str,
        *,
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        response = await self._http.get(
            f"/api/v1/assets/{name}/{version}",
            params={"visibility": visibility},
        )
        _raise(response)
        return AssetSummary.model_validate(response.json())

    async def status(self, job_id: str) -> JobSummary:
        response = await self._http.get(f"/api/v1/jobs/{job_id}")
        _raise(response)
        return JobSummary.model_validate(response.json())

    async def attempts(self, job_id: str) -> list[dict[str, Any]]:
        response = await self._http.get(f"/api/v1/jobs/{job_id}/attempts")
        _raise(response)
        return list(response.json())

    async def logs(self, job_id: str, *, tail_bytes: int = 131072) -> str:
        response = await self._http.get(
            f"/api/v1/jobs/{job_id}/logs", params={"tail_bytes": tail_bytes}
        )
        _raise(response)
        return response.text

    async def attempt_log_chunk(
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
        response = await self._http.get(
            f"/api/v1/jobs/{job_id}/attempts/{attempt_id}/logs",
            params=params,
        )
        _raise(response)
        return _parse_attempt_log_chunk(response)

    async def cancel(self, job_id: str) -> JobSummary:
        response = await self._http.post(f"/api/v1/jobs/{job_id}/cancel")
        _raise(response)
        return JobSummary.model_validate(response.json()["job"])

    async def wait(
        self, job_id: str, *, poll_seconds: float = 1, timeout: float | None = None
    ) -> JobSummary:
        async def poll() -> JobSummary:
            while True:
                result = await self.status(job_id)
                if result.state.value in {"succeeded", "failed", "canceled"}:
                    return result
                await asyncio.sleep(poll_seconds)

        if timeout is None:
            return await poll()
        try:
            async with asyncio.timeout(timeout):
                return await poll()
        except TimeoutError as exc:
            raise TimeoutError(f"job {job_id} did not finish within {timeout} seconds") from exc

    async def kernel(
        self,
        request: KernelRequest | None = None,
        **values: Any,
    ) -> KernelSummary:
        value = request or KernelRequest.model_validate(values)
        response = await self._http.post("/api/v1/kernels", json=value.model_dump(mode="json"))
        _raise(response)
        return KernelSummary.model_validate(response.json())

    async def kernel_status(self, session_id: str) -> KernelSummary:
        response = await self._http.get(f"/api/v1/kernels/{session_id}")
        _raise(response)
        return KernelSummary.model_validate(response.json())

    async def stop_kernel(self, session_id: str) -> KernelSummary:
        response = await self._http.delete(f"/api/v1/kernels/{session_id}")
        _raise(response)
        return KernelSummary.model_validate(response.json())

    async def kernel_connect_url(self, session_id: str) -> str:
        response = await self._http.get(f"/api/v1/kernels/{session_id}/connect")
        _raise(response)
        return str(response.json()["url"])

    async def kernel_notebook_server(self, session_id: str) -> dict[str, Any]:
        """Start or recover the stock Jupyter Server for an IDE session."""

        response = await self._http.post(f"/api/v1/kernels/{session_id}/notebook-server")
        _raise(response)
        return dict(response.json())

    async def kernel_notebook_server_info(self, session_id: str) -> dict[str, Any]:
        response = await self._http.get(f"/api/v1/kernels/{session_id}/notebook-server")
        _raise(response)
        return dict(response.json())

    async def stop_kernel_notebook_server(
        self,
        session_id: str,
        *,
        stop_kernel: bool = True,
    ) -> dict[str, Any]:
        response = await self._http.delete(
            f"/api/v1/kernels/{session_id}/notebook-server",
            params={"stop_kernel": stop_kernel},
        )
        _raise(response)
        return dict(response.json())

    async def execute_kernel(
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
        url = await self.kernel_connect_url(session_id)
        response = await self._http.post(url, json=request.model_dump(mode="json"))
        _raise(response)
        return KernelOperationResult.model_validate(response.json())

    async def sites(self) -> list[dict[str, Any]]:
        return await self._list("/api/v1/sites")

    async def gpus(self) -> list[dict[str, Any]]:
        return await self._list("/api/v1/gpus")

    async def models(self) -> list[dict[str, Any]]:
        return await self._list("/api/v1/models")

    async def reserve_gpu(
        self,
        reservation: GPUReservationRequest | None = None,
        *,
        idempotency_key: str | None = None,
        **values: Any,
    ) -> GPUReservationSummary:
        value = reservation or GPUReservationRequest.model_validate(values)
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        response = await self._http.post(
            "/api/v1/gpu-reservations",
            json=value.model_dump(mode="json"),
            headers=headers,
        )
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    async def gpu_reservations(
        self,
        *,
        limit: int = 100,
        active_only: bool = False,
    ) -> list[GPUReservationSummary]:
        response = await self._http.get(
            "/api/v1/gpu-reservations",
            params={"limit": limit, "active_only": active_only},
        )
        _raise(response)
        return [GPUReservationSummary.model_validate(row) for row in response.json()]

    async def gpu_reservation(self, reservation_id: str) -> GPUReservationSummary:
        response = await self._http.get(f"/api/v1/gpu-reservations/{reservation_id}")
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    async def cancel_gpu_reservation(
        self,
        reservation_id: str,
    ) -> GPUReservationSummary:
        response = await self._http.post(f"/api/v1/gpu-reservations/{reservation_id}/cancel")
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    async def extend_gpu_reservation(
        self,
        reservation_id: str,
        extension: GPUReservationExtensionRequest,
    ) -> GPUReservationSummary:
        response = await self._http.post(
            f"/api/v1/gpu-reservations/{reservation_id}/extend",
            json=extension.model_dump(mode="json"),
        )
        _raise(response)
        return GPUReservationSummary.model_validate(response.json())

    async def system_status(self) -> dict[str, Any]:
        """Fetch one sanitized, read-only operator snapshot."""

        response = await self._http.get("/api/v1/status")
        _raise(response)
        return dict(response.json())

    async def reload_policy(self) -> dict[str, Any]:
        """Validate and stage the configured scheduling policy."""

        response = await self._http.post("/api/v1/policy/reload")
        _raise(response)
        return dict(response.json())

    async def storage_gc(self, *, apply: bool = False) -> dict[str, Any]:
        response = await self._http.post(
            "/api/v1/storage/gc",
            params={"apply": apply},
            timeout=120,
        )
        _raise(response)
        return dict(response.json())

    async def access_users(self) -> list[str]:
        """List Unix accounts currently allowed to use this scheduler."""

        response = await self._http.get("/api/v1/access/users")
        _raise(response)
        return [str(value) for value in response.json()["users"]]

    async def grant_access_user(self, username: str) -> dict[str, str]:
        """Grant or rotate a user's credential and return it exactly once."""

        response = await self._http.post(f"/api/v1/access/users/{username}")
        _raise(response)
        return {str(key): str(value) for key, value in response.json().items()}

    async def revoke_access_user(self, username: str) -> dict[str, Any]:
        response = await self._http.delete(f"/api/v1/access/users/{username}")
        _raise(response)
        return dict(response.json())

    async def unregister_autodl(self, site_id: str) -> dict[str, Any]:
        response = await self._http.delete(f"/api/v1/providers/autodl/registrations/{site_id}")
        _raise(response)
        return response.json()

    async def release_autodl(self, site_id: str, *, force: bool = False) -> dict[str, Any]:
        response = await self._http.post(
            f"/api/v1/providers/autodl/registrations/{site_id}/release", json={"force": force}
        )
        _raise(response)
        return response.json()

    async def register_autodl(self, ssh: str, password: str) -> dict[str, Any]:
        response = await self._http.post(
            "/api/v1/providers/autodl/register", json={"ssh": ssh, "password": password}, timeout=90
        )
        _raise(response)
        return response.json()

    async def autodl_control(self) -> dict[str, Any]:
        response = await self._http.get("/api/v1/providers/autodl/control")
        _raise(response)
        return dict(response.json())

    async def autodl_instances(self) -> list[dict[str, Any]]:
        return await self._list("/api/v1/providers/autodl/instances")

    async def autodl_operations(self) -> list[dict[str, Any]]:
        return await self._list("/api/v1/providers/autodl/operations")

    async def _list(self, path: str) -> list[dict[str, Any]]:
        response = await self._http.get(path)
        _raise(response)
        return list(response.json())


def _raise(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Topodrome API returned {response.status_code}: {detail}") from exc
