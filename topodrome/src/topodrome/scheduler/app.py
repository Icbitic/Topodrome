from __future__ import annotations

import asyncio
import hashlib
import hmac
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from topodrome.config import ModelCatalog, SchedulerSettings
from topodrome.db.session import CURRENT_SCHEMA_REVISION, Database
from topodrome.deployment.manual_autodl import ManualRegistration, ManualRelease
from topodrome.domain.arrays import ArrayUpdate, JobArraySpec
from topodrome.domain.enums import KernelMode
from topodrome.domain.errors import ConflictError, NotFoundError, StaleFenceError, TopodromeError
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.logs import LOG_READ_LIMIT_BYTES
from topodrome.domain.specs import (
    GPUReservationExtensionRequest,
    GPUReservationRequest,
    GPUReservationSummary,
    HealthStatus,
    JobSpec,
    JobSummary,
    KernelOperationRequest,
    KernelOperationResult,
    KernelRequest,
    KernelSummary,
    contains_sensitive_key,
)
from topodrome.jupyter.protocol import (
    JUPYTER_WEBSOCKET_PROTOCOL,
    JupyterChannelSession,
    JupyterProtocolError,
    browser_client_html,
    decode_websocket_message,
    encode_websocket_message,
)
from topodrome.protocol.auth import extract_bearer, token_matches
from topodrome.protocol.user_access import UserCredentialStore, UserPrincipal
from topodrome.scheduler.assets import (
    AssetDeleteResult,
    AssetPublishRequest,
    AssetSummary,
    AssetTagUpdate,
    AssetVisibility,
    normalize_tags,
)
from topodrome.scheduler.autodl import build_autodl_orchestrator
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.jupyter_bootstrap import build_jupyter_bootstrap_controller
from topodrome.scheduler.managed_jupyter import build_managed_jupyter_controller
from topodrome.scheduler.manual_autodl import ManualAutoDLController
from topodrome.scheduler.service import SchedulerService
from topodrome.scheduler.source_store import (
    SourceAdmissionError,
    SourceBundle,
    SourceResult,
)
from topodrome.scheduler.status import system_status_snapshot
from topodrome.source_limits import SourceLimits


class CancelResponse(BaseModel):
    job: JobSummary


class UserGrantResponse(BaseModel):
    username: str
    token: str = Field(repr=False)


class UserListResponse(BaseModel):
    users: tuple[str, ...]


def create_app(
    settings: SchedulerSettings | None = None,
    *,
    database: Database | None = None,
    service: SchedulerService | None = None,
    run_dispatcher: bool = True,
    settings_path: Path | None = None,
) -> FastAPI:
    settings = settings or SchedulerSettings()
    database = database or Database(settings.database_dsn)
    user_credentials = (
        UserCredentialStore(
            settings.user_access.credential_store_path,
            require_local_account=settings.user_access.require_local_account,
        )
        if settings.user_access.enabled and settings.user_access.credential_store_path is not None
        else None
    )
    owner_access_authorizer = (
        user_credentials.is_whitelisted if user_credentials is not None else None
    )
    owner_access_revision_provider = (
        user_credentials.authorization_revision if user_credentials is not None else None
    )
    if service is None:
        connections = AgentConnectionRegistry()
        model_catalog = (
            ModelCatalog.from_yaml(settings.model_config_path)
            if settings.model_config_path is not None
            else None
        )
        provider_controller = build_autodl_orchestrator(
            settings,
            database,
            connections,
        )
        jupyter_bootstrap_controller = build_jupyter_bootstrap_controller(
            settings,
            database,
            connections,
        )
        service = SchedulerService(
            database=database,
            settings=settings,
            connections=connections,
            model_catalog=model_catalog,
            provider_controller=provider_controller,
            jupyter_bootstrap_controller=jupyter_bootstrap_controller,
            owner_access_authorizer=owner_access_authorizer,
            owner_access_revision_provider=owner_access_revision_provider,
            settings_path=settings_path,
        )
        service.set_owner_access_authorizer(
            owner_access_authorizer,
            owner_access_revision_provider,
        )
        service.managed_jupyter_controller = build_managed_jupyter_controller(
            settings,
            service,
        )
    else:
        service.set_owner_access_authorizer(
            owner_access_authorizer,
            owner_access_revision_provider,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        storage_task: asyncio.Task[None] | None = None
        if run_dispatcher:
            task = asyncio.create_task(service.run_background(), name="topodrome-dispatcher")
            storage_task = asyncio.create_task(
                service.run_storage_maintenance(), name="topodrome-storage-maintenance"
            )
        yield
        service.stop_background()
        if task:
            await task
        if storage_task:
            await storage_task
        try:
            if service.managed_jupyter_controller is not None:
                await service.managed_jupyter_controller.close()
        finally:
            try:
                if service.jupyter_bootstrap_controller is not None:
                    await service.jupyter_bootstrap_controller.close()
            finally:
                try:
                    if service.provider_controller is not None:
                        await service.provider_controller.close()
                finally:
                    await database.close()

    app = FastAPI(title="Topodrome", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.database = database
    app.state.service = service
    app.state.user_credentials = user_credentials
    app.include_router(_api_router())
    kernel_connections: dict[str, int] = {}
    kernel_channel_senders: dict[str, dict[str, Callable[[Any], Awaitable[None]]]] = {}
    kernel_shell_locks: dict[str, asyncio.Lock] = {}
    kernel_connections_lock = asyncio.Lock()

    @app.get("/api/v1/kernel-relay/{session_id}")
    async def kernel_relay_info(
        request: Request,
        session_id: str,
        token: str | None = Query(default=None, min_length=1),
        access: str | None = Query(default=None, min_length=1),
    ) -> Any:
        try:
            credential, source = _kernel_credential(
                request,
                session_id=session_id,
                token=token,
                legacy_access=access,
            )
            info = await service.relay_info(session_id, credential, touch=False)
        except StaleFenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="kernel access credential is invalid or expired",
            ) from exc
        if "text/html" in request.headers.get("accept", ""):
            if source in {"query", "legacy-query"}:
                response = RedirectResponse(
                    url=f"/api/v1/kernel-relay/{session_id}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
                response.set_cookie(
                    _kernel_cookie_name(session_id),
                    credential,
                    max_age=max(
                        1,
                        int(
                            (
                                _aware_datetime(info["access_expires_at"]) - datetime.now(UTC)
                            ).total_seconds()
                        ),
                    ),
                    httponly=True,
                    secure=settings.public_url.lower().startswith("https://"),
                    samesite="lax",
                    path=f"/api/v1/kernel-relay/{session_id}",
                )
                return _secure_kernel_response(response)
            return _secure_kernel_response(HTMLResponse(browser_client_html()))
        return _secure_kernel_response(JSONResponse(_jsonable(info)))

    @app.post(
        "/api/v1/kernel-relay/{session_id}",
        response_model=KernelOperationResult,
    )
    async def kernel_relay_execute(
        request: Request,
        session_id: str,
        operation: KernelOperationRequest,
        token: str | None = Query(default=None, min_length=1),
        access: str | None = Query(default=None, min_length=1),
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> KernelOperationResult:
        try:
            credential, source = _kernel_credential(
                request,
                session_id=session_id,
                token=token,
                legacy_access=access,
                allow_cookie=False,
            )
            del source
            return await service.execute_kernel(
                session_id,
                credential,
                operation,
                request_id=idempotency_key,
            )
        except StaleFenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="kernel access credential is invalid or expired",
            ) from exc

    @app.get("/api/v1/kernel-relay/{session_id}/gateway/api")
    async def gateway_version(request: Request, session_id: str) -> Any:
        credential, _ = _kernel_credential(request, session_id=session_id)
        await _relay_info_or_forbidden(service, session_id, credential)
        return _secure_kernel_response(JSONResponse({"version": "1.0", "protocol_version": "5.4"}))

    @app.get("/api/v1/kernel-relay/{session_id}/gateway/api/kernelspecs")
    async def gateway_kernelspecs(request: Request, session_id: str) -> Any:
        credential, _ = _kernel_credential(request, session_id=session_id)
        info = await _relay_info_or_forbidden(service, session_id, credential)
        return _secure_kernel_response(JSONResponse(_kernel_specs(info)))

    @app.get("/api/v1/kernel-relay/{session_id}/gateway/api/kernelspecs/{kernel_name}")
    async def gateway_kernelspec(
        request: Request,
        session_id: str,
        kernel_name: str,
    ) -> Any:
        if kernel_name != "topodrome":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="kernelspec not found"
            )
        credential, _ = _kernel_credential(request, session_id=session_id)
        info = await _relay_info_or_forbidden(service, session_id, credential)
        return _secure_kernel_response(
            JSONResponse(_kernel_specs(info)["kernelspecs"]["topodrome"])
        )

    @app.get("/api/v1/kernel-relay/{session_id}/gateway/api/kernels")
    async def gateway_list_kernels(request: Request, session_id: str) -> Any:
        credential, _ = _kernel_credential(request, session_id=session_id)
        info = await _relay_info_or_forbidden(service, session_id, credential)
        async with kernel_connections_lock:
            connections = kernel_connections.get(session_id, 0)
        return _secure_kernel_response(
            JSONResponse([_gateway_kernel_model(info, connections=connections)])
        )

    @app.post(
        "/api/v1/kernel-relay/{session_id}/gateway/api/kernels",
        status_code=status.HTTP_201_CREATED,
    )
    async def gateway_attach_kernel(
        request: Request,
        session_id: str,
        payload: Annotated[dict[str, Any], Body()],
    ) -> Any:
        credential, _ = _kernel_credential(request, session_id=session_id, allow_cookie=False)
        environment = payload.get("env", {})
        environment_is_safe = (
            isinstance(environment, dict)
            and not contains_sensitive_key(environment)
            and all(
                isinstance(key, str) and key.startswith("KERNEL_") and isinstance(value, str)
                for key, value in environment.items()
            )
            and sum(len(key) + len(value) for key, value in environment.items()) <= 16_384
        )
        if (
            payload.get("name", "topodrome") != "topodrome"
            or set(payload) - {"name", "path", "env"}
            or not environment_is_safe
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "the scoped gateway can only attach its Topodrome kernel with "
                    "non-sensitive standard Jupyter environment hints"
                ),
            )
        info = await _relay_info_or_forbidden(service, session_id, credential)
        async with kernel_connections_lock:
            connections = kernel_connections.get(session_id, 0)
        response = JSONResponse(
            _gateway_kernel_model(info, connections=connections),
            status_code=status.HTTP_201_CREATED,
            headers={
                "Location": (
                    f"/api/v1/kernel-relay/{session_id}/gateway/api/kernels/"
                    f"{_gateway_kernel_id(session_id)}"
                )
            },
        )
        return _secure_kernel_response(response)

    @app.get("/api/v1/kernel-relay/{session_id}/gateway/api/kernels/{kernel_id}")
    async def gateway_get_kernel(
        request: Request,
        session_id: str,
        kernel_id: str,
    ) -> Any:
        _require_gateway_kernel(session_id, kernel_id)
        credential, _ = _kernel_credential(request, session_id=session_id)
        info = await _relay_info_or_forbidden(service, session_id, credential)
        async with kernel_connections_lock:
            connections = kernel_connections.get(session_id, 0)
        return _secure_kernel_response(
            JSONResponse(_gateway_kernel_model(info, connections=connections))
        )

    @app.delete(
        "/api/v1/kernel-relay/{session_id}/gateway/api/kernels/{kernel_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
    )
    async def gateway_delete_kernel(
        request: Request,
        session_id: str,
        kernel_id: str,
    ) -> Response:
        _require_gateway_kernel(session_id, kernel_id)
        credential, _ = _kernel_credential(request, session_id=session_id, allow_cookie=False)
        await _relay_info_or_forbidden(service, session_id, credential)
        await service.stop_kernel(session_id)
        return _secure_kernel_response(Response(status_code=204))

    @app.post(
        "/api/v1/kernel-relay/{session_id}/gateway/api/kernels/{kernel_id}/interrupt",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
    )
    async def gateway_interrupt_kernel(
        request: Request,
        session_id: str,
        kernel_id: str,
    ) -> Response:
        _require_gateway_kernel(session_id, kernel_id)
        credential, _ = _kernel_credential(request, session_id=session_id, allow_cookie=False)
        await service.interrupt_kernel(session_id, credential)
        return _secure_kernel_response(Response(status_code=204))

    @app.post("/api/v1/kernel-relay/{session_id}/gateway/api/kernels/{kernel_id}/restart")
    async def gateway_restart_kernel(
        request: Request,
        session_id: str,
        kernel_id: str,
    ) -> Any:
        _require_gateway_kernel(session_id, kernel_id)
        credential, _ = _kernel_credential(request, session_id=session_id, allow_cookie=False)
        await _relay_info_or_forbidden(service, session_id, credential)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="live model state is not migratable; create a replacement kernel",
        )

    async def kernel_channels(websocket: WebSocket, session_id: str, kernel_id: str) -> None:
        if not _gateway_kernel_matches(session_id, kernel_id):
            await websocket.close(code=4404, reason="kernel is outside this gateway scope")
            return
        try:
            credential, source = _kernel_websocket_credential(websocket, session_id)
            if source == "cookie" and not _origin_matches(
                websocket.headers.get("origin"), settings.public_url
            ):
                await websocket.close(code=4403, reason="kernel origin rejected")
                return
            info = await service.relay_info(session_id, credential, touch=False)
        except (StaleFenceError, ConflictError, ValueError):
            await websocket.close(code=4403, reason="kernel credential rejected")
            return
        offered = {
            item.strip()
            for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        }
        subprotocol = JUPYTER_WEBSOCKET_PROTOCOL if JUPYTER_WEBSOCKET_PROTOCOL in offered else None
        await websocket.accept(subprotocol=subprotocol)
        connection_id = uuid.uuid4().hex
        send_lock = asyncio.Lock()
        request_tasks: set[asyncio.Task[None]] = set()

        async def send_local(message: Any) -> None:
            # Recheck the capability immediately before every response so a
            # rotated generation cannot receive IOPub from another client in
            # the short interval before its polling loop observes revocation.
            await service.relay_info(session_id, credential, touch=False)
            encoded = encode_websocket_message(
                message,
                subprotocol=subprotocol,
                maximum_bytes=settings.kernel_max_output_bytes,
            )
            async with send_lock:
                if isinstance(encoded, bytes):
                    await websocket.send_bytes(encoded)
                else:
                    await websocket.send_text(encoded)

        async def send_message(message: Any) -> None:
            # Shell/control replies are private to their requesting client.
            # IOPub is the kernel's broadcast channel and must reach every
            # attached notebook client, even when they negotiated different
            # WebSocket framing versions.
            if message.channel != "iopub":
                await send_local(message)
                return
            async with kernel_connections_lock:
                recipients = tuple(kernel_channel_senders.get(session_id, {}).values())
            await asyncio.gather(
                *(recipient(message) for recipient in recipients),
                return_exceptions=True,
            )

        async with kernel_connections_lock:
            senders = kernel_channel_senders.setdefault(session_id, {})
            senders[connection_id] = send_local
            kernel_connections[session_id] = len(senders)
            shell_lock = kernel_shell_locks.setdefault(session_id, asyncio.Lock())
        channel = JupyterChannelSession(
            backend=service,
            kernel_session=session_id,
            access=credential,
            mode=KernelMode(str(info["mode"])),
            maximum_output_bytes=settings.kernel_max_output_bytes,
            shell_lock=shell_lock,
        )

        async def process_message(message: Any) -> None:
            should_close = await channel.handle(message, send_message)
            if should_close:
                await websocket.close(code=1000)

        try:
            while True:
                # Revalidation enforces session stop, token expiry, and access
                # generation rotation on every client frame.
                await service.relay_info(session_id, credential, touch=False)
                remaining = (
                    _aware_datetime(info["access_expires_at"]) - datetime.now(UTC)
                ).total_seconds()
                if remaining <= 0:
                    await websocket.close(code=4403, reason="kernel credential expired")
                    break
                try:
                    incoming = await asyncio.wait_for(
                        websocket.receive(), timeout=min(remaining, 1.0)
                    )
                except TimeoutError:
                    # Polling makes a stopped session or rotated generation
                    # revoke an otherwise idle established socket promptly.
                    continue
                if incoming["type"] == "websocket.disconnect":
                    break
                await service.relay_info(session_id, credential, touch=False)
                raw = incoming.get("bytes")
                if raw is None:
                    raw = incoming.get("text")
                if not isinstance(raw, (str, bytes)):
                    raise JupyterProtocolError("empty Jupyter WebSocket frame")
                message = decode_websocket_message(
                    raw,
                    subprotocol=subprotocol,
                    maximum_bytes=settings.kernel_max_request_bytes,
                )
                task = asyncio.create_task(
                    process_message(message),
                    name=f"topodrome-jupyter-{message.msg_id}",
                )
                request_tasks.add(task)
                task.add_done_callback(request_tasks.discard)
        except TimeoutError:
            await websocket.close(code=4403, reason="kernel credential expired")
        except (JupyterProtocolError, StaleFenceError, ConflictError):
            await websocket.close(code=4403, reason="kernel channel rejected")
        except WebSocketDisconnect:
            pass
        finally:
            for task in request_tasks:
                task.cancel()
            if request_tasks:
                await asyncio.gather(*request_tasks, return_exceptions=True)
            async with kernel_connections_lock:
                remaining_senders = kernel_channel_senders.get(session_id)
                if remaining_senders is not None:
                    remaining_senders.pop(connection_id, None)
                if remaining_senders:
                    kernel_connections[session_id] = len(remaining_senders)
                else:
                    kernel_connections.pop(session_id, None)
                    kernel_channel_senders.pop(session_id, None)
                    kernel_shell_locks.pop(session_id, None)

    async def direct_kernel_channels(websocket: WebSocket, session_id: str) -> None:
        await kernel_channels(websocket, session_id, session_id)

    app.add_api_websocket_route(
        "/api/v1/kernel-relay/{session_id}/channels",
        direct_kernel_channels,
    )
    app.add_api_websocket_route(
        "/api/v1/kernel-relay/{session_id}/gateway/api/kernels/{kernel_id}/channels",
        kernel_channels,
    )

    @app.get("/health/live", response_model=HealthStatus)
    async def live() -> HealthStatus:
        return HealthStatus(status="ok")

    @app.get("/health/ready", response_model=HealthStatus, status_code=200)
    async def ready(request: Request) -> HealthStatus:
        expected_revision = None if settings.insecure_development_mode else CURRENT_SCHEMA_REVISION
        available = await request.app.state.database.ready(
            expected_schema_revision=expected_revision
        )
        if not available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable or schema migration required",
            )
        return HealthStatus(status="ok", database=True)

    def enrollment_token(site_id: str) -> str | None:
        configured = settings.agent_tokens.get(site_id)
        expected = configured.get_secret_value() if configured else None
        manual_enrollment = getattr(service.provider_controller, "enrollment_token", None)
        if expected is None and manual_enrollment is not None:
            expected = manual_enrollment(site_id)
        return expected

    from topodrome.scheduler.storage_digest import digest_router

    app.include_router(digest_router(settings.global_root, enrollment_token))

    @app.websocket("/api/v1/agents/control")
    async def agent_control(
        websocket: WebSocket,
        site_id: str = Query(min_length=1),
        session_id: str = Query(min_length=1),
    ) -> None:
        expected = enrollment_token(site_id)
        provided = extract_bearer(websocket.headers.get("authorization"))
        if (expected is None and not settings.allow_unenrolled_agents) or not token_matches(
            provided, expected
        ):
            await websocket.close(code=4403, reason="site enrollment rejected")
            return
        await websocket.accept()

        async def close_replaced_connection() -> None:
            await websocket.close(code=4409, reason="site connection was superseded")

        connection = AgentConnection(
            site_id=site_id,
            session_id=session_id,
            send=websocket.send_json,
            close=close_replaced_connection,
        )
        await service.connections.register(connection)
        try:
            # Enrollment can be revoked while registration waits for the site's lock.
            current_token = enrollment_token(site_id)
            if (
                current_token is None and not settings.allow_unenrolled_agents
            ) or not token_matches(provided, current_token):
                await websocket.close(code=4403, reason="site enrollment rejected")
                return
            while True:
                envelope = AgentEnvelope.model_validate(await websocket.receive_json())
                if envelope.site_id != site_id or envelope.agent_session_id != session_id:
                    await websocket.close(code=4403, reason="agent identity mismatch")
                    return
                current, reply = await service.connections.run_if_current(
                    connection,
                    partial(
                        service.handle_agent_envelope,
                        envelope,
                        connection_generation=connection.generation,
                    ),
                )
                if not current or reply is None:
                    await websocket.close(code=4409, reason="site connection was superseded")
                    return
                async with connection.send_lock:
                    if not await service.connections.is_current(connection):
                        return
                    await websocket.send_json(reply.model_dump(mode="json"))
        except WebSocketDisconnect:
            pass
        except TopodromeError as exc:
            # HTTP exception handlers cannot emit a response after a WebSocket
            # has been accepted.  Reject an invalid/fenced stream in protocol
            # space and let the agent reconnect without crashing Starlette's
            # response machinery.
            await websocket.close(code=4400, reason=exc.code)
        finally:
            await service.connections.unregister(
                site_id,
                session_id,
                generation=connection.generation,
            )

    @app.exception_handler(TopodromeError)
    async def domain_error(_: Request, exc: TopodromeError) -> Any:
        from fastapi.responses import JSONResponse

        status_code = (
            status.HTTP_409_CONFLICT
            if isinstance(exc, ConflictError)
            else status.HTTP_400_BAD_REQUEST
        )
        if isinstance(exc, NotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        return JSONResponse(status_code=status_code, content={"code": exc.code, "detail": str(exc)})

    return app


def _api_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1", dependencies=[Depends(_authorize_user)])
    asset_upload_slots = asyncio.Semaphore(2)

    @router.post("/asset-uploads/{name}/{version}", response_model=AssetSummary)
    async def upload_asset(
        request: Request,
        response: Response,
        name: str,
        version: str,
        visibility: Annotated[AssetVisibility, Query()] = "private",
    ) -> AssetSummary:
        from topodrome.asset_upload import CONTENT_TYPE
        from topodrome.scheduler.asset_upload import receive_asset
        from topodrome.scheduler.assets import AssetUploadRequest

        if request.headers.get("content-type") != CONTENT_TYPE:
            raise HTTPException(415, "expected Topodrome streaming asset format")
        try:
            publication = AssetUploadRequest(name=name, version=version, visibility=visibility)
        except ValueError:
            raise HTTPException(422, "invalid asset name or version") from None
        async with asset_upload_slots, asyncio.timeout(6 * 60 * 60):
            result = await receive_asset(
                _service(request).assets,
                publication,
                _principal(request).username,
                request.stream(),
            )
        response.status_code = 201 if result.created else 200
        response.headers["Cache-Control"] = "no-store"
        return result.asset

    @router.get("/source-limits", response_model=SourceLimits)
    async def source_limits(request: Request, response: Response) -> SourceLimits:
        response.headers["Cache-Control"] = "no-store"
        return _service(request).source_limits()

    @router.post("/sources", response_model=SourceResult, status_code=status.HTTP_201_CREATED)
    async def upload_source(request: Request, bundle: SourceBundle) -> SourceResult:
        try:
            return await asyncio.to_thread(_service(request).sources.put, bundle)
        except SourceAdmissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from None

    @router.post("/arrays", status_code=status.HTTP_202_ACCEPTED)
    async def submit_array(
        request: Request,
        spec: JobArraySpec,
        idempotency_key: Annotated[
            str | None, Header(alias="Idempotency-Key", min_length=1, max_length=200)
        ] = None,
    ) -> dict[str, Any]:
        return await _service(request).submit_array(
            spec,
            idempotency_key=idempotency_key,
            owner_username=_principal(request).username,
            is_admin=_principal(request).is_admin,
        )

    @router.get("/arrays")
    async def list_job_arrays(
        request: Request, response: Response, limit: int = Query(default=50, ge=1, le=100)
    ):
        from .arrays import list_arrays

        response.headers["Cache-Control"] = "no-store"
        async with request.app.state.database.session() as session:
            return await list_arrays(session, owner_username=_owner_scope(request), limit=limit)

    @router.get("/arrays/{array_id}")
    async def job_array(
        request: Request,
        response: Response,
        array_id: str,
        after: int = Query(default=-1, ge=-1),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        from .arrays import array_detail

        response.headers["Cache-Control"] = "no-store"
        return await array_detail(
            _service(request),
            array_id,
            owner_username=_owner_scope(request),
            after=after,
            limit=limit,
        )

    @router.patch("/arrays/{array_id}")
    async def change_job_array(request: Request, array_id: str, update: ArrayUpdate):
        from .arrays import update_array

        return await update_array(
            _service(request), array_id, update, owner_username=_owner_scope(request)
        )

    @router.post("/arrays/{array_id}/cancel")
    async def cancel_job_array(request: Request, array_id: str):
        from .arrays import cancel_array

        return await cancel_array(_service(request), array_id, owner_username=_owner_scope(request))

    @router.post("/jobs", response_model=JobSummary, status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(
        request: Request,
        spec: JobSpec,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", min_length=1, max_length=200),
        ] = None,
    ) -> JobSummary:
        return await _service(request).submit(
            spec,
            idempotency_key=idempotency_key,
            owner_username=_principal(request).username,
            is_admin=_principal(request).is_admin,
        )

    @router.post("/assets", response_model=AssetSummary, status_code=status.HTTP_201_CREATED)
    async def publish_asset(
        request: Request,
        response: Response,
        publication: AssetPublishRequest,
    ) -> AssetSummary:
        principal = _principal(request)
        result = await _service(request).publish_asset(
            publication,
            owner_username=principal.username,
            is_admin=principal.is_admin,
        )
        response.status_code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
        response.headers["Cache-Control"] = "no-store"
        return result.asset

    @router.get("/assets", response_model=list[AssetSummary])
    async def list_assets(
        request: Request,
        response: Response,
        visibility: Annotated[AssetVisibility | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        tag: Annotated[list[str] | None, Query(max_length=32)] = None,
    ) -> list[AssetSummary]:
        response.headers["Cache-Control"] = "no-store"
        try:
            tags = normalize_tags(tag or ())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return await _service(request).list_assets(
            owner_username=_principal(request).username,
            visibility=visibility,
            limit=limit,
            tags=tags,
        )

    @router.delete("/assets/{name}/{version}", response_model=AssetDeleteResult)
    async def delete_asset(
        request: Request,
        response: Response,
        name: str,
        version: str,
        visibility: Annotated[AssetVisibility, Query()] = "private",
    ) -> AssetDeleteResult:
        response.headers["Cache-Control"] = "no-store"
        principal = _principal(request)
        return await _service(request).delete_asset(
            name,
            version,
            owner_username=principal.username,
            visibility=visibility,
            is_admin=principal.is_admin,
        )

    @router.patch("/assets/{name}/{version}/tags", response_model=AssetSummary)
    async def update_asset_tags(
        request: Request,
        response: Response,
        name: str,
        version: str,
        update: AssetTagUpdate,
        visibility: Annotated[AssetVisibility, Query()] = "private",
    ) -> AssetSummary:
        response.headers["Cache-Control"] = "no-store"
        principal = _principal(request)
        return await _service(request).update_asset_tags(
            name,
            version,
            update,
            owner_username=principal.username,
            visibility=visibility,
            is_admin=principal.is_admin,
        )

    @router.get("/assets/{name}/{version}", response_model=AssetSummary)
    async def asset_detail(
        request: Request,
        response: Response,
        name: str,
        version: str,
        visibility: Annotated[AssetVisibility, Query()] = "private",
    ) -> AssetSummary:
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).asset(
            name,
            version,
            owner_username=_principal(request).username,
            visibility=visibility,
        )

    @router.get("/assets/{name}/{version}/files/{member_index}")
    async def download_asset_file(
        request: Request,
        name: str,
        version: str,
        member_index: int,
        visibility: Annotated[AssetVisibility, Query()] = "private",
    ) -> Response:
        from topodrome.scheduler.asset_download import AssetFileResponse, open_member

        catalog = _service(request).assets
        summary = await asyncio.to_thread(
            catalog.get,
            name,
            version,
            requester_username=_principal(request).username,
            visibility=visibility,
        )
        pin = await asyncio.to_thread(
            catalog.resolve,
            summary.ref,
            requester_username=_principal(request).username,
        )
        if not 0 <= member_index < len(pin.members):
            raise NotFoundError("asset file does not exist")
        member = pin.members[member_index]
        opening = asyncio.create_task(asyncio.to_thread(open_member, catalog.global_root, member))
        try:
            handle = await asyncio.shield(opening)
        except asyncio.CancelledError:
            handle = await opening
            handle.close()
            raise
        return AssetFileResponse(handle, member)

    @router.get("/jobs/{job_id}", response_model=JobSummary)
    async def job_status(request: Request, job_id: str) -> JobSummary:
        return await _service(request).status(
            job_id,
            owner_username=_owner_scope(request),
        )

    @router.get("/jobs/{job_id}/attempts")
    async def job_attempts(request: Request, job_id: str) -> list[dict[str, Any]]:
        return await _service(request).attempts(
            job_id,
            owner_username=_owner_scope(request),
        )

    @router.get("/jobs/{job_id}/attempts/{attempt_id}/logs")
    async def attempt_log_range(
        request: Request,
        job_id: str,
        attempt_id: str,
        offset: int = Query(default=0, ge=0),
        limit_bytes: int = Query(default=262144, ge=1, le=LOG_READ_LIMIT_BYTES),
        anchor_sha256: str | None = Query(default=None, pattern=r"^[0-9a-f]{64}$"),
    ) -> Response:
        service = _service(request)
        attempts = await service.attempts(
            job_id,
            owner_username=_owner_scope(request),
        )
        if not any(item["attempt_id"] == attempt_id for item in attempts):
            raise NotFoundError("attempt does not belong to this job")
        attempts_root: Path = request.app.state.settings.global_root / "jobs" / job_id / "attempts"
        chunk = await asyncio.to_thread(
            service.live_logs.read_range,
            job_id,
            attempt_id,
            attempts_root=attempts_root,
            offset=offset,
            limit_bytes=limit_bytes,
            anchor_sha256=anchor_sha256,
        )
        headers = {
            "Cache-Control": "no-store",
            "X-Topodrome-Log-Offset": str(chunk.offset),
            "X-Topodrome-Log-Next-Offset": str(chunk.next_offset),
            "X-Topodrome-Log-Size": str(chunk.size_bytes),
            "X-Topodrome-Log-Reset": "true" if chunk.reset else "false",
            "X-Topodrome-Log-Complete": "true" if chunk.complete else "false",
        }
        if chunk.anchor_sha256 is not None:
            headers["X-Topodrome-Log-Anchor-SHA256"] = chunk.anchor_sha256
        return Response(
            content=chunk.data,
            media_type="application/octet-stream",
            headers=headers,
        )

    @router.post("/jobs/{job_id}/cancel", response_model=CancelResponse)
    async def cancel_job(request: Request, job_id: str) -> CancelResponse:
        return CancelResponse(
            job=await _service(request).cancel(
                job_id,
                owner_username=_owner_scope(request),
            )
        )

    @router.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
    async def job_logs(
        request: Request,
        response: Response,
        job_id: str,
        tail_bytes: int = Query(default=131072, ge=1, le=10_000_000),
    ) -> str:
        service = _service(request)
        attempts = await service.attempts(
            job_id,
            owner_username=_owner_scope(request),
        )
        root: Path = request.app.state.settings.global_root / "jobs" / job_id / "attempts"
        data = await asyncio.to_thread(
            service.live_logs.read_tail,
            job_id,
            [
                item["attempt_id"]
                for item in sorted(attempts, key=lambda item: item["attempt_number"])
            ],
            attempts_root=root,
            limit_bytes=tail_bytes,
        )
        response.headers["Cache-Control"] = "no-store"
        return data.decode("utf-8", errors="replace")

    @router.get("/sites")
    async def sites(request: Request) -> list[dict[str, Any]]:
        async with request.app.state.database.session() as session:
            rows = await request.app.state.service.repository.list_sites(session)
            return [
                {
                    "site_id": row.id,
                    "display_name": row.display_name,
                    "kind": row.kind.value,
                    "state": row.state.value,
                    "heartbeat_at": row.heartbeat_at,
                    "storage_available": row.storage_available,
                    "queue_depth": row.queue_depth,
                    "reliability": row.reliability,
                    "gpu_safety_margin_bytes": row.gpu_safety_margin_bytes,
                    "region": row.region,
                    "provider_instance_id": row.provider_instance_id,
                    "activity": {
                        "job_running": row.job_running,
                        "assignment_queued_locally": row.assignment_queued_locally,
                        "interactive_kernel_active": row.interactive_kernel_active,
                        "artifact_flush_pending": row.artifact_flush_pending,
                        "staging_operation_active": row.staging_operation_active,
                    },
                }
                for row in rows
            ]

    @router.get("/gpus")
    async def gpus(request: Request) -> list[dict[str, Any]]:
        async with request.app.state.database.session() as session:
            rows = await request.app.state.service.repository.list_sites(session)
            operator_reserved = (
                await request.app.state.service.repository.active_gpu_reservation_keys(session)
            )
            return [
                {
                    "site_id": site.id,
                    "uuid": gpu.uuid,
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_bytes": gpu.total_memory_bytes,
                    "free_memory_bytes": gpu.free_memory_bytes,
                    "utilization_percent": gpu.utilization_percent,
                    "foreign_process_count": gpu.foreign_process_count,
                    "externally_reserved": gpu.externally_reserved,
                    "operator_reserved": (site.id, gpu.uuid) in operator_reserved,
                    "healthy": gpu.healthy,
                    "observed_at": gpu.observed_at,
                }
                for site in rows
                for gpu in site.gpus
            ]

    @router.post(
        "/gpu-reservations",
        response_model=GPUReservationSummary,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_gpu_reservation(
        request: Request,
        response: Response,
        reservation: GPUReservationRequest,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", max_length=200),
        ] = None,
    ) -> GPUReservationSummary:
        """Reserve one exact physical GPU; operator credentials are required."""

        principal = _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).reserve_gpu(
            reservation,
            requested_by=principal.username,
            idempotency_key=idempotency_key,
        )

    @router.get("/gpu-reservations", response_model=list[GPUReservationSummary])
    async def list_gpu_reservations(
        request: Request,
        response: Response,
        limit: int = Query(default=100, ge=1, le=1000),
        active_only: bool = Query(default=False),
    ) -> list[GPUReservationSummary]:
        """Enumerate reservation audit records for an operator."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).list_gpu_reservations(
            limit=limit,
            active_only=active_only,
        )

    @router.get(
        "/gpu-reservations/{reservation_id}",
        response_model=GPUReservationSummary,
    )
    async def get_gpu_reservation(
        request: Request,
        response: Response,
        reservation_id: str,
    ) -> GPUReservationSummary:
        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).get_gpu_reservation(reservation_id)

    @router.post(
        "/gpu-reservations/{reservation_id}/cancel",
        response_model=GPUReservationSummary,
    )
    async def cancel_gpu_reservation(
        request: Request,
        response: Response,
        reservation_id: str,
    ) -> GPUReservationSummary:
        """Cancel idempotently; already-running work is never signaled."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).cancel_gpu_reservation(reservation_id)

    @router.post(
        "/gpu-reservations/{reservation_id}/extend",
        response_model=GPUReservationSummary,
    )
    async def extend_gpu_reservation(
        request: Request,
        response: Response,
        reservation_id: str,
        extension: GPUReservationExtensionRequest,
    ) -> GPUReservationSummary:
        """Extend in place; compare-and-set prevents stale edits or double extension."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).extend_gpu_reservation(reservation_id, extension)

    @router.get("/models")
    async def models(request: Request) -> list[dict[str, Any]]:
        async with request.app.state.database.session() as session:
            rows = await request.app.state.service.repository.list_sites(session)
            return [
                {
                    "site_id": site.id,
                    "alias": model.alias,
                    "revision": model.revision,
                    "backend": model.backend,
                    "backend_version": model.backend_version,
                    "dtype": model.dtype,
                    "quantization": model.quantization or None,
                    "adapter": model.adapter or None,
                    "availability": model.availability.value,
                    "gpu_uuid": model.gpu_uuid or None,
                    "verified_at": model.verified_at,
                }
                for site in rows
                for model in site.models
            ]

    @router.get("/status")
    async def system_status(request: Request, response: Response) -> dict[str, Any]:
        """Return a bounded snapshot with owner-scoped jobs and kernels."""

        response.headers["Cache-Control"] = "no-store"
        async with request.app.state.database.session() as session:
            snapshot = await system_status_snapshot(
                session,
                owner_username=_owner_scope(request),
                heartbeat_ttl_seconds=request.app.state.settings.heartbeat_ttl_seconds,
                starvation_seconds=_service(request)
                .scheduling_policy_snapshot()
                .settings.starvation_seconds,
            )
        service = _service(request)
        service.process_telemetry.decorate(
            snapshot["gpus"],
            identities=await service.connections.identities(),
            now=datetime.now(UTC),
            ttl_seconds=service.settings.heartbeat_ttl_seconds,
        )
        snapshot["policy"] = await service.policy_status()
        for job in snapshot["jobs"]:
            job["progress"] = (
                await asyncio.to_thread(
                    _service(request).live_logs.read_progress, job["job_id"], job["attempt_id"]
                )
                if job.get("attempt_id")
                and job["state"] in {"running", "canceling", "finalizing", "assigning"}
                else job.get("progress")
            )
        return snapshot

    @router.post("/storage/gc")
    async def storage_gc(
        request: Request,
        response: Response,
        apply: bool = False,
    ) -> dict[str, object]:
        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).collect_storage(apply=apply)

    @router.post("/policy/reload")
    async def reload_policy(request: Request, response: Response) -> dict[str, object]:
        """Atomically reload the validated, data-only scheduling policy."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        return await _service(request).reload_scheduling_policy()

    @router.get("/access/users", response_model=UserListResponse)
    async def list_access_users(request: Request, response: Response) -> UserListResponse:
        """List whitelisted local accounts without exposing credential material."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        store = _user_store(request)
        users = await asyncio.to_thread(store.list_users)
        return UserListResponse(users=users)

    @router.post("/access/users/{username}", response_model=UserGrantResponse)
    async def grant_access_user(
        request: Request,
        response: Response,
        username: str,
    ) -> UserGrantResponse:
        """Grant or rotate one local user's token, returned exactly once."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            granted = await asyncio.to_thread(_user_store(request).grant, username)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        # A grant for an existing user is a credential rotation. Revoke every
        # previously issued relay/managed-server capability before exposing
        # the replacement bearer, without stopping the underlying kernels.
        await _service(request).invalidate_owner_access(granted.username)
        return UserGrantResponse(username=granted.username, token=granted.token)

    @router.delete("/access/users/{username}")
    async def revoke_access_user(
        request: Request,
        response: Response,
        username: str,
    ) -> dict[str, object]:
        """Revoke one user immediately; repeated revocation is idempotent."""

        _require_admin(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            removed = await asyncio.to_thread(_user_store(request).revoke, username)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        # Repeated revocation also retries exact managed-child cleanup. This is
        # intentionally independent of whether the whitelist row existed.
        await _service(request).invalidate_owner_access(username.strip())
        return {"username": username, "revoked": removed}

    @router.get("/providers/autodl/control")
    async def autodl_control(request: Request) -> dict[str, Any]:
        _require_admin(request)
        controller = _service(request).provider_controller
        if controller is None:
            return {"configured": False, "enabled": False}
        return controller.control_status()

    @router.post("/providers/autodl/register", status_code=202)
    async def register_manual_autodl(request: Request, registration: ManualRegistration):
        _require_admin(request)
        controller = _service(request).provider_controller
        if not isinstance(controller, ManualAutoDLController):
            raise HTTPException(status_code=409, detail="Manual SSH AutoDL is not configured")
        try:
            return await controller.register(registration)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except Exception:
            raise HTTPException(
                status_code=502, detail="SSH inspection failed; check connection and credentials"
            ) from None

    @router.delete("/providers/autodl/registrations/{site_id}")
    async def unregister_manual_autodl(request: Request, site_id: str):
        _require_admin(request)
        controller = _service(request).provider_controller
        if not isinstance(controller, ManualAutoDLController):
            raise HTTPException(status_code=409, detail="Manual SSH AutoDL is not configured")
        try:
            return await controller.unregister(site_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @router.post("/providers/autodl/registrations/{site_id}/release", status_code=202)
    async def release_manual_autodl(request: Request, site_id: str, release: ManualRelease):
        _require_admin(request)
        service = _service(request)
        controller = service.provider_controller
        if not isinstance(controller, ManualAutoDLController):
            raise HTTPException(status_code=409, detail="Manual SSH AutoDL is not configured")
        try:
            async with service._dispatch_reload_lock:
                result = await controller.request_release(site_id, force=release.force)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        # Durable intent is enough for an immediate response. The background
        # reservation and provider reconcilers carry out preemption and shutdown.
        return result

    @router.get("/providers/autodl/instances")
    async def autodl_instances(request: Request) -> list[dict[str, Any]]:
        _require_admin(request)
        return await _service(request).provider_instances()

    @router.get("/providers/autodl/operations")
    async def autodl_operations(request: Request) -> list[dict[str, Any]]:
        _require_admin(request)
        return await _service(request).provider_operations()

    @router.post("/kernels", response_model=KernelSummary, status_code=status.HTTP_202_ACCEPTED)
    async def create_kernel(request: Request, kernel: KernelRequest) -> KernelSummary:
        return await _service(request).create_kernel(
            kernel,
            owner_username=_principal(request).username,
        )

    @router.get("/kernels/{session_id}", response_model=KernelSummary)
    async def kernel_status(request: Request, session_id: str) -> KernelSummary:
        return await _service(request).get_kernel(
            session_id,
            owner_username=_owner_scope(request),
        )

    @router.delete("/kernels/{session_id}", response_model=KernelSummary)
    async def stop_kernel(request: Request, session_id: str) -> KernelSummary:
        return await _service(request).stop_kernel(
            session_id,
            owner_username=_owner_scope(request),
        )

    @router.get("/kernels/{session_id}/connect")
    async def connect_kernel(
        request: Request,
        response: Response,
        session_id: str,
    ) -> dict[str, Any]:
        _secure_kernel_response(response)
        service = _service(request)
        url = await service.connect_kernel_url(
            session_id,
            owner_username=_owner_scope(request),
        )
        parsed = urlsplit(url)
        token = parse_qs(parsed.query)["token"][0]
        info = await service.relay_info(session_id, token, touch=False)
        gateway_path = f"{parsed.path}/gateway"
        gateway_url = urlunsplit((parsed.scheme, parsed.netloc, gateway_path, "", ""))
        websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
        websocket_path = f"{parsed.path}/channels"
        websocket_url = urlunsplit(
            (
                websocket_scheme,
                parsed.netloc,
                websocket_path,
                urlencode({"token": token}),
                "",
            )
        )
        return {
            "url": url,
            "base_url": gateway_url,
            "ws_url": gateway_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1),
            "websocket_url": websocket_url,
            "token": token,
            # Jupyter Server 2.x only registers its native channel handler for
            # UUID-shaped kernel identifiers. Keep Topodrome's durable session
            # ID private to the scoped base URL and expose a deterministic alias.
            "kernel_id": _gateway_kernel_id(session_id),
            "expires_at": info["access_expires_at"],
        }

    @router.post("/kernels/{session_id}/notebook-server")
    async def start_notebook_server(
        request: Request,
        response: Response,
        session_id: str,
    ) -> dict[str, Any]:
        _secure_kernel_response(response)
        return await _service(request).ensure_managed_jupyter(
            session_id,
            owner_username=_owner_scope(request),
        )

    @router.get("/kernels/{session_id}/notebook-server")
    async def notebook_server_status(
        request: Request,
        response: Response,
        session_id: str,
    ) -> dict[str, Any]:
        _secure_kernel_response(response)
        return await _service(request).managed_jupyter_info(
            session_id,
            owner_username=_owner_scope(request),
        )

    @router.delete("/kernels/{session_id}/notebook-server")
    async def stop_notebook_server(
        request: Request,
        response: Response,
        session_id: str,
        stop_kernel: bool = Query(default=True),
    ) -> dict[str, Any]:
        _secure_kernel_response(response)
        return await _service(request).stop_managed_jupyter(
            session_id,
            stop_kernel=stop_kernel,
            owner_username=_owner_scope(request),
        )

    return router


async def _authorize_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected_secret = request.app.state.settings.api_token
    provided = extract_bearer(authorization)
    if expected_secret is not None and provided is not None:
        expected = expected_secret.get_secret_value()
        if hmac.compare_digest(provided, expected):
            request.state.topodrome_principal = UserPrincipal("@operator", is_admin=True)
            return
    store: UserCredentialStore | None = request.app.state.user_credentials
    if store is not None and provided is not None:
        try:
            principal = await asyncio.to_thread(store.authenticate, provided)
        except (OSError, ValueError):
            principal = None
        if principal is not None:
            request.state.topodrome_principal = principal
            return
    if expected_secret is None and store is None:
        request.state.topodrome_principal = UserPrincipal("@operator", is_admin=True)
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")


def _principal(request: Request) -> UserPrincipal:
    principal = getattr(request.state, "topodrome_principal", None)
    if not isinstance(principal, UserPrincipal):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
    return principal


def _require_admin(request: Request) -> UserPrincipal:
    principal = _principal(request)
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="operator access required",
        )
    return principal


def _owner_scope(request: Request) -> str | None:
    """Return a row filter for a user, while retaining operator-wide access."""

    principal = _principal(request)
    return None if principal.is_admin else principal.username


def _user_store(request: Request) -> UserCredentialStore:
    store = request.app.state.user_credentials
    if not isinstance(store, UserCredentialStore):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="multi-user access is disabled",
        )
    return store


def _service(request: Request) -> SchedulerService:
    return cast(SchedulerService, request.app.state.service)


def _kernel_cookie_name(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    return f"topodrome_kernel_{digest}"


def _kernel_credential(
    request: Request,
    *,
    session_id: str,
    token: str | None = None,
    legacy_access: str | None = None,
    allow_cookie: bool = True,
) -> tuple[str, str]:
    query_token = token or request.query_params.get("token")
    if query_token:
        return query_token, "query"
    old_query = legacy_access or request.query_params.get("access")
    if old_query:
        return old_query, "legacy-query"
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, separator, value = authorization.partition(" ")
        if separator and scheme.lower() in {"bearer", "token"} and value:
            return value, "header"
    if allow_cookie:
        cookie = request.cookies.get(_kernel_cookie_name(session_id))
        if cookie:
            return cookie, "cookie"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="kernel access credential is required",
    )


def _kernel_websocket_credential(
    websocket: WebSocket,
    session_id: str,
) -> tuple[str, str]:
    token = websocket.query_params.get("token")
    if token:
        return token, "query"
    legacy = websocket.query_params.get("access")
    if legacy:
        return legacy, "legacy-query"
    authorization = websocket.headers.get("authorization")
    if authorization:
        scheme, separator, value = authorization.partition(" ")
        if separator and scheme.lower() in {"bearer", "token"} and value:
            return value, "header"
    cookie = websocket.cookies.get(_kernel_cookie_name(session_id))
    if cookie:
        return cookie, "cookie"
    raise ValueError("kernel access credential is required")


async def _relay_info_or_forbidden(
    service: SchedulerService,
    session_id: str,
    credential: str,
) -> dict[str, Any]:
    try:
        return await service.relay_info(session_id, credential, touch=False)
    except (StaleFenceError, ConflictError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="kernel access credential is invalid or expired",
        ) from exc


def _kernel_specs(info: dict[str, Any]) -> dict[str, Any]:
    mode = str(info["mode"])
    display_name = "Topodrome raw model" if mode == "raw-model" else "Topodrome service"
    return {
        "default": "topodrome",
        "kernelspecs": {
            "topodrome": {
                "name": "topodrome",
                "spec": {
                    # Declarative only: the scoped remote gateway owns the
                    # already-running kernel and never launches this argv.
                    "argv": ["topodrome-kernel", "-f", "{connection_file}"],
                    "display_name": display_name,
                    "language": "python" if mode == "raw-model" else "topodrome-service",
                    "interrupt_mode": "message",
                    "env": {},
                    "metadata": {"topodrome_mode": mode},
                },
                "resources": {},
            }
        },
    }


def _gateway_kernel_model(info: dict[str, Any], *, connections: int) -> dict[str, Any]:
    return {
        "id": _gateway_kernel_id(str(info["session_id"])),
        "name": "topodrome",
        "last_activity": _jsonable(info["expires_at"]),
        "execution_state": "idle",
        "connections": connections,
    }


def _require_gateway_kernel(session_id: str, kernel_id: str) -> None:
    # Accept the durable ID for backwards compatibility with the original
    # scoped gateway descriptor, but only advertise the UUID alias expected by
    # stock Jupyter Server's native WebSocket route.
    if not _gateway_kernel_matches(session_id, kernel_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="kernel not found")


def _gateway_kernel_matches(session_id: str, kernel_id: str) -> bool:
    return hmac.compare_digest(session_id, kernel_id) or hmac.compare_digest(
        _gateway_kernel_id(session_id), kernel_id
    )


def _gateway_kernel_id(session_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"topodrome-kernel:{session_id}"))


def _origin_matches(origin: str | None, public_url: str) -> bool:
    if origin is None:
        return False
    actual = urlsplit(origin)
    expected = urlsplit(public_url)
    return (actual.scheme.lower(), actual.netloc.lower()) == (
        expected.scheme.lower(),
        expected.netloc.lower(),
    )


def _secure_kernel_response(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
        "connect-src 'self' ws: wss:"
    )
    return response


def _aware_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _aware_datetime(value).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
