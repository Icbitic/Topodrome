"""Concrete bridge from site assignments to the reusable model runtime."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import threading
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from topodrome.domain.enums import KernelMode, Portability
from topodrome.domain.specs import JobSpec, ModelShardedParallelism
from topodrome.logging import redact
from topodrome.runtime import (
    ActorKey,
    ArtifactRegistration,
    AsyncRuntimeExecutionService,
    CheckpointRecord,
    CheckpointRestore,
    ExecutionTransport,
    ModelLoader,
    ModelRegistration,
    ModelRegistry,
    PreparedActor,
    TransformersLoader,
    invoke_experiment,
)
from topodrome.runtime.environment import configure_ray_driver_environment
from topodrome.source_layout import (
    SOURCE_BUNDLE_FORMAT_VERSION,
    SOURCE_BUNDLE_MANIFEST_NAME,
    SOURCE_BUNDLE_METADATA_DIRECTORY,
)
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.assets import AssetResolver
from topodrome.storage.base import StorageAdapter, StorageUnavailableError
from topodrome.storage.checkpoint_retention import pin_latest_checkpoint, rotate_checkpoints
from topodrome.storage.progress import (
    cancellable_transfers,
    check_transfer_cancelled,
    observe_transfers,
    reuse_verified_outputs,
    transfer_authority,
)

from .admission import ActorAllocation
from .agent import Assignment, ExecutionOutcome, PreparedRuntime
from .regional import RegionalStateStore, is_regional_reference
from .transfer_progress import TransferLogReporter
from .workspace import AttemptWorkspace

_CHECKPOINT_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ModelAcquirer(Protocol):
    def __call__(self, registration: ModelRegistration) -> ModelRegistration: ...


SourceResolver = Callable[[str], Path]
LoaderFactory = Callable[[ModelRegistration], ModelLoader]
WorkingDirectoryPublisher = Callable[[Path, Path], str]


class RuntimePreparationCleanupError(RuntimeError):
    """Canceled preparation left a runtime whose disposal is unconfirmed."""

    cleanup_unconfirmed = True


def _publish_ray_working_directory(directory: Path, scratch: Path) -> str:
    """Upload a verified local tree and return Ray's immutable content URI.

    Ray rejects local ``working_dir`` paths on task/actor runtime environments.
    Its driver-side packager is therefore invoked after the site has resolved
    the content-addressed reference, while the temporary archive is kept on the
    attempt's site-local scratch filesystem.
    """

    try:
        from ray._private.runtime_env.working_dir import (
            upload_working_dir_if_needed,
        )
    except ImportError as exc:  # pragma: no cover - version/environment specific
        raise RuntimeError("Ray working-directory packaging is unavailable") from exc
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        scratch.chmod(0o700)
    except OSError:
        pass
    runtime_env: dict[str, Any] = {"working_dir": str(directory)}
    published = upload_working_dir_if_needed(
        runtime_env,
        include_gitignore=False,
        scratch_dir=str(scratch),
    )
    reference = published.get("working_dir")
    if not isinstance(reference, str) or not reference.startswith("gcs://"):
        raise RuntimeError("Ray did not publish the local working directory")
    return reference


@dataclass(frozen=True, slots=True)
class _RuntimeHandle:
    prepared: PreparedActor
    owner: object


@dataclass(frozen=True, slots=True)
class _ExecutionReport:
    artifacts: tuple[ArtifactRegistration, ...]
    checkpoint_ref: str | None
    output_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CheckpointWriter:
    directory: Path
    logical_prefix: str
    keep: int = 2

    def __call__(self, checkpoint: CheckpointRecord) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        name_digest = hashlib.sha256(checkpoint.name.encode()).hexdigest()[:12]
        target = self.directory / f"{checkpoint.sequence:08d}-{name_digest}.pkl"
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("xb") as handle:
                pickle.dump(checkpoint.value, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(self.directory)
        finally:
            temporary.unlink(missing_ok=True)
        reference = f"{self.logical_prefix}/{target.name}"
        digest = _sha256_file(target)
        # Retention failure must never invalidate an already durable snapshot.
        with suppress(OSError, ValueError):
            rotate_checkpoints(self.directory, keep=self.keep)
        return _checkpoint_reference(reference, digest)


def _append_runtime_failure(path: Path, error: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write("\n[Topodrome] runtime failure\n")
        stream.write(redact("".join(traceback.format_exception(error))))
        stream.flush()
        os.fsync(stream.fileno())


def _execute_and_collect(
    context: Any,
    *,
    entrypoint: str,
    experiment_kwargs: Mapping[str, Any],
    log_path: str,
) -> _ExecutionReport:
    """Run inside the actor, durably stage stdio, and return compact metadata."""

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # ``redirect_stdout`` changes process-global streams. Ray gives every model
    # actor its own process; the lock also keeps the local fallback correct when
    # tests run independent actors concurrently in one process.
    with (
        _STDIO_CAPTURE_LOCK,
        target.open("ab", buffering=0) as raw,
        io.TextIOWrapper(raw, encoding="utf-8", write_through=True) as stream,
    ):
        try:
            target.chmod(0o600)
        except OSError:
            pass
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                invoke_experiment(entrypoint, context, experiment_kwargs)
        except BaseException as exc:
            stream.write(f"\n[Topodrome] experiment raised {type(exc).__name__}\n")
            stream.write(redact("".join(traceback.format_exception(exc))))
            if not isinstance(exc, Exception):
                raise RuntimeError(
                    f"experiment terminated with {type(exc).__name__}: {exc}"
                ) from exc
            raise
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return _ExecutionReport(tuple(context.artifacts), context.latest_checkpoint_ref)


_SERVICE_KERNEL_OPERATIONS = frozenset({"describe", "generate", "tokenize"})
_RAW_KERNEL_OPERATIONS = _SERVICE_KERNEL_OPERATIONS | frozenset(
    {"call", "exec", "inspect", "jupyter_execute"}
)
_RAW_KERNEL_NAMESPACES: dict[tuple[int, str], dict[str, Any]] = {}
_STDIO_CAPTURE_LOCK = threading.RLock()


def _execute_kernel_operation(
    context: Any,
    *,
    session_id: str,
    mode: str,
    operation: str,
    arguments: Mapping[str, Any],
) -> Any:
    """Execute one JSON-shaped operation inside the serialized model actor."""

    kernel_mode = KernelMode(mode)
    allowed = (
        _RAW_KERNEL_OPERATIONS
        if kernel_mode is KernelMode.RAW_MODEL
        else _SERVICE_KERNEL_OPERATIONS
    )
    if operation not in allowed:
        raise ValueError(f"operation {operation!r} is unavailable in {kernel_mode.value} mode")
    if operation == "describe":
        return {
            "mode": kernel_mode.value,
            "model_type": type(context.model).__qualname__,
            "tokenizer_type": (
                None if context.tokenizer is None else type(context.tokenizer).__qualname__
            ),
            "device": str(context.device),
        }
    if operation == "tokenize":
        if context.tokenizer is None:
            raise RuntimeError("the model actor has no tokenizer")
        text = str(arguments["text"])
        options = _mapping(arguments.get("options", {}), "tokenizer options")
        return _portable_result(context.tokenizer(text, **dict(options)))
    if operation == "generate":
        if context.tokenizer is None:
            raise RuntimeError("the model actor has no tokenizer")
        prompt = str(arguments["prompt"])
        generation = _mapping(arguments.get("generation", {}), "generation options")
        encoded = context.tokenizer(prompt, return_tensors="pt")
        model_inputs = {
            key: value.to(context.device) if hasattr(value, "to") else value
            for key, value in dict(encoded).items()
        }
        output = context.model.generate(**model_inputs, **dict(generation))
        first = output[0] if hasattr(output, "__getitem__") else output
        return {"text": str(context.tokenizer.decode(first, skip_special_tokens=True))}
    if operation == "inspect":
        target = _resolve_kernel_target(context, str(arguments.get("target", "model")))
        return _portable_result(target)
    if operation == "call":
        target = _resolve_kernel_target(context, str(arguments["target"]))
        if not callable(target):
            raise ValueError("raw kernel call target is not callable")
        positional = arguments.get("args", ())
        keywords = _mapping(arguments.get("kwargs", {}), "call keyword arguments")
        if not isinstance(positional, Sequence) or isinstance(positional, (str, bytes)):
            raise ValueError("call args must be a sequence")
        return _portable_result(target(*tuple(positional), **dict(keywords)))
    if operation == "exec":
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("raw kernel exec requires non-empty code")
        namespace = _RAW_KERNEL_NAMESPACES.setdefault(
            (id(context.model), session_id),
            {
                "model": context.model,
                "tokenizer": context.tokenizer,
                "device": context.device,
            },
        )
        exec(compile(code, f"<topodrome-kernel-{session_id}>", "exec"), namespace)  # noqa: S102
        expression = arguments.get("result")
        if expression is None:
            return None
        if not isinstance(expression, str):
            raise ValueError("raw kernel result expression must be a string")
        return _portable_result(eval(expression, namespace))  # noqa: S307
    if operation == "jupyter_execute":
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return {
                "status": "ok",
                "stdout": "",
                "stderr": "",
                "result": None,
                "truncated": False,
            }
        try:
            output_limit = min(
                4_194_304,
                max(4096, int(arguments.get("_output_limit_bytes", 262_144))),
            )
        except (TypeError, ValueError):
            output_limit = 262_144
        stdout = _BoundedTextWriter(maximum_bytes=max(1024, output_limit // 4))
        stderr = _BoundedTextWriter(maximum_bytes=max(1024, output_limit // 4))
        namespace = _RAW_KERNEL_NAMESPACES.setdefault(
            (id(context.model), session_id),
            {
                "__name__": "__topodrome_kernel__",
                "model": context.model,
                "tokenizer": context.tokenizer,
                "device": context.device,
            },
        )
        result_name = "__topodrome_cell_result__"
        try:
            tree = ast.parse(code, filename="<topodrome-cell>", mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                final = tree.body[-1]
                tree.body[-1] = ast.Assign(
                    targets=[ast.Name(id=result_name, ctx=ast.Store())],
                    value=final.value,
                )
                ast.fix_missing_locations(tree)
            namespace.pop(result_name, None)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile(tree, "<topodrome-cell>", "exec"), namespace)  # noqa: S102
            result = _portable_result(namespace.pop(result_name, None))
            return {
                "status": "ok",
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "result": result,
                "truncated": stdout.truncated or stderr.truncated,
            }
        except Exception as exc:
            namespace.pop(result_name, None)
            value = str(exc)
            if len(value) > 1000:
                value = f"{value[:1000]}... <truncated>"
            return {
                "status": "error",
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "ename": type(exc).__name__,
                "evalue": value,
                "traceback": [],
                "truncated": stdout.truncated or stderr.truncated,
            }
    raise AssertionError("unreachable kernel operation")


class _BoundedTextWriter(io.TextIOBase):
    """A text stream that discards bytes after a hard UTF-8 boundary."""

    def __init__(self, *, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self._chunks: list[bytes] = []
        self._size = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        encoded = str(value).encode("utf-8", errors="replace")
        remaining = self.maximum_bytes - self._size
        if remaining > 0:
            chunk = encoded[:remaining]
            self._chunks.append(chunk)
            self._size += len(chunk)
        if len(encoded) > max(0, remaining):
            self.truncated = True
        return len(value)

    def getvalue(self) -> str:
        value = b"".join(self._chunks).decode("utf-8", errors="ignore")
        if self.truncated:
            value += "\n... <Topodrome output truncated>"
        return value


class RuntimeExecutionBackend:
    """Load exact single- or multi-GPU model runtimes and commit durable outputs."""

    @property
    def storage_authoritative(self) -> bool:
        from topodrome.storage.local import LocalStorage
        return isinstance(self.storage, LocalStorage)

    def __init__(
        self,
        *,
        site_id: str,
        registry: ModelRegistry,
        runtime: AsyncRuntimeExecutionService,
        storage: StorageAdapter,
        source_resolver: SourceResolver | None = None,
        asset_resolver: AssetResolver | None = None,
        acquirer: ModelAcquirer | None = None,
        loader_factory: LoaderFactory = TransformersLoader,
        use_ray: bool = True,
        ray_address: str | None = None,
        ray_namespace: str = "topodrome",
        regional_state: RegionalStateStore | None = None,
        working_dir_publisher: WorkingDirectoryPublisher = _publish_ray_working_directory,
        ray_connect_timeout_seconds: float = 30.0,
        ray_control_timeout_seconds: float = 120.0,
        finalization_concurrency: int = 2,
        checkpoint_keep_snapshots: int = 2,
        finalization_journal=None,
    ) -> None:
        if ray_connect_timeout_seconds <= 0 or ray_control_timeout_seconds <= 0:
            raise ValueError("Ray connection and control timeouts must be positive")
        if finalization_concurrency < 1:
            raise ValueError("finalization concurrency must be positive")
        if checkpoint_keep_snapshots < 2:
            raise ValueError("checkpoint retention must keep at least two snapshots")
        self.checkpoint_keep_snapshots = checkpoint_keep_snapshots
        self.site_id = site_id
        self.registry = registry
        self.runtime = runtime
        self.storage = storage
        self.source_resolver = source_resolver
        self.asset_resolver = asset_resolver or AssetResolver(storage)
        self.acquirer = acquirer
        self.loader_factory = loader_factory
        self.use_ray = use_ray
        self.ray_address = ray_address
        self.ray_namespace = ray_namespace
        self.regional_state = regional_state
        self.working_dir_publisher = working_dir_publisher
        self.ray_connect_timeout_seconds = ray_connect_timeout_seconds
        self.ray_control_timeout_seconds = ray_control_timeout_seconds
        self._owner = object()
        self._ray_connected = False
        self._ray_lock = threading.Lock()
        self._prepared: dict[str, tuple[PreparedActor, str]] = {}
        self._prepared_lock = threading.RLock()
        self._confirmed_stale_actor_ids: set[str] = set()
        self._finalization_slots = asyncio.Semaphore(finalization_concurrency)
        self._finalization_cancellations: dict[str, threading.Event] = {}
        self.finalization_journal = finalization_journal
        self.finalization_authority = lambda attempt_id: True

    async def prepare(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> PreparedRuntime:
        job_spec = JobSpec.model_validate(assignment.raw_spec)
        if any(not asset.pinned for asset in job_spec.assets.values()):
            raise RuntimeError("site assignments must contain only scheduler-pinned assets")
        gpu_uuids = assignment.lease.gpu_uuids
        try:
            gpu_indices = tuple(str(int(assignment.gpu_indices[uuid])) for uuid in gpu_uuids)
        except KeyError as exc:
            raise RuntimeError("physical GPU index was not preserved after admission") from exc

        model = _mapping(assignment.raw_spec.get("model"), "assignment model")
        registration = self._resolve_registration(model)
        if registration.key.quantization != _optional_text(model.get("quantization")):
            raise RuntimeError("registered model has an incompatible quantization mode")
        if registration.key.adapter != _optional_text(model.get("adapter")):
            raise RuntimeError("registered model has an incompatible adapter identity")
        if self.acquirer is not None:
            acquired = await _to_thread(self.acquirer, registration)
            if acquired.key != registration.key:
                raise RuntimeError("model acquisition changed the exact model identity")
            registration = acquired
            self.registry.register(registration, replace=True)
        loader = self.loader_factory(registration)
        if self.use_ray:
            await asyncio.wait_for(
                _to_thread(self._ensure_ray_driver),
                timeout=self.ray_connect_timeout_seconds,
            )
        keys = tuple(
            ActorKey(
                self.site_id,
                gpu_uuid,
                registration.key,
                runtime_fingerprint=job_spec.runtime_fingerprint,
            )
            for gpu_uuid in gpu_uuids
        )
        working_dir = await asyncio.wait_for(
            _to_thread(
                self._resolve_runtime_working_dir,
                job_spec.runtime.working_dir_ref,
                workspace.scratch_dir / "ray-runtime-env",
            ),
            timeout=self.ray_control_timeout_seconds,
        )
        runtime_options: dict[str, Any] = {
            "ray_namespace": self.ray_namespace,
            "num_cpus": job_spec.resources.cpu,
            "memory_bytes": round(job_spec.resources.memory_gb * 1024**3),
            "pip": job_spec.runtime.pip,
            "env": job_spec.runtime.env,
            "working_dir": working_dir,
        }
        parallelism = job_spec.model.parallelism
        if isinstance(parallelism, ModelShardedParallelism):
            if not self.use_ray:
                raise RuntimeError("model-sharded inference requires the site-local Ray runtime")
            replica_key = ActorKey(
                self.site_id,
                gpu_uuids[0],
                registration.key,
                runtime_fingerprint=job_spec.runtime_fingerprint,
                replica_gpu_uuids=gpu_uuids,
            )
            prepared = await self._await_runtime_preparation(
                self.runtime.prepare_sharded(
                    replica_key,
                    loader,
                    cuda_visible_devices=gpu_indices,
                    strategy=parallelism.strategy,
                    max_memory_bytes_per_gpu=round(parallelism.max_memory_gb_per_gpu * 1024**3),
                    **runtime_options,
                )
            )
        elif len(keys) == 1:
            prepared = await self._await_runtime_preparation(
                self.runtime.prepare(
                    keys[0],
                    loader,
                    use_ray=self.use_ray,
                    cuda_visible_device=gpu_indices[0],
                    **runtime_options,
                )
            )
        else:
            if not self.use_ray:
                raise RuntimeError("multi-GPU execution requires the site-local Ray runtime")
            prepared = await self._await_runtime_preparation(
                self.runtime.prepare_group(
                    keys,
                    loader,
                    cuda_visible_devices=gpu_indices,
                    **runtime_options,
                )
            )
        with self._prepared_lock:
            self._prepared = {
                actor_id: value
                for actor_id, value in self._prepared.items()
                if not _prepared_gpu_uuids(value[0]) & set(gpu_uuids)
                or actor_id == prepared.actor_id
            }
            self._prepared[prepared.actor_id] = (prepared, assignment.model_key)
        return PreparedRuntime(
            actor_id=prepared.actor_id,
            model_key=assignment.model_key,
            reused=prepared.reused,
            opaque=_RuntimeHandle(prepared, self._owner),
            process_ids=prepared.worker_pids,
        )

    async def shutdown_driver(self) -> bool:
        """Disconnect this process from Ray before its owned head is stopped.

        ``ray.shutdown`` disconnects a driver that joined an existing cluster;
        it does not stop that cluster.  Keeping this operation separate from
        the site-local head lifecycle is what lets an adopted/foreign head
        remain untouched while preventing an owned raylet from terminating an
        attached Topodrome driver during service shutdown.
        """

        if not self.use_ray:
            return False
        return await asyncio.wait_for(
            _to_thread(self._shutdown_ray_driver),
            timeout=self.ray_control_timeout_seconds,
        )

    async def _await_runtime_preparation(
        self,
        operation: Any,
    ) -> PreparedActor:
        """Do not abandon a worker thread that may already own GPU resources."""

        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as canceled:
            try:
                prepared = await task
            except Exception as exc:
                if bool(getattr(exc, "cleanup_unconfirmed", False)):
                    raise
                raise canceled from exc
            try:
                await self.runtime.evict(prepared, force=True)
            except Exception as exc:
                raise RuntimePreparationCleanupError(
                    "canceled runtime preparation could not be disposed"
                ) from exc
            raise canceled

    async def execute_with_finalization(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        on_compute_finished: Callable[[bool], Awaitable[bool]],
    ) -> ExecutionOutcome:
        return await self.execute(
            prepared,
            assignment,
            workspace,
            on_compute_finished=on_compute_finished,
        )

    def cancel_finalization(self, attempt_id: str) -> None:
        event = self._finalization_cancellations.get(attempt_id)
        if event is not None:
            event.set()

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        *,
        on_compute_finished: Callable[[bool], Awaitable[bool]] | None = None,
    ) -> ExecutionOutcome:
        runtime_handle = self._handle(prepared)
        source_root = await _to_thread(self._materialize_source, assignment, workspace)
        committed_inputs = await _to_thread(self._resolve_assets, assignment)
        checkpoint_restore = await _to_thread(self._load_checkpoint, assignment)
        checkpoint_directory, checkpoint_prefix = await _to_thread(
            self._checkpoint_target,
            assignment,
            workspace,
        )
        try:
            report = await self.runtime.execute(
                runtime_handle,
                _execute_and_collect,
                job_id=assignment.job_id,
                attempt_id=assignment.attempt_id,
                kwargs={
                    "entrypoint": assignment.entrypoint,
                    "experiment_kwargs": dict(assignment.kwargs),
                    "log_path": str(workspace.log_dir / "attempt.log"),
                },
                workspace=source_root,
                committed_inputs=committed_inputs,
                scratch=workspace.scratch_dir,
                checkpoint_handler=_CheckpointWriter(
                    checkpoint_directory,
                    checkpoint_prefix,
                    keep=self.checkpoint_keep_snapshots,
                ),
                checkpoint_interval_seconds=_checkpoint_interval(assignment),
                checkpoint_restore=checkpoint_restore,
            )
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                # The worker may have died outside Python (OOM/SIGKILL). Save
                # the driver's failure diagnostic locally before any recovery
                # upload. The agent's live publisher/finalizer owns transport;
                # a slow checkpoint must never hide the reason for death.
                with suppress(Exception):
                    await _to_thread(
                        _append_runtime_failure, workspace.log_dir / "attempt.log", exc
                    )
            with self._prepared_lock:
                self._prepared.pop(prepared.actor_id, None)
            raise
        if not isinstance(report, _ExecutionReport):
            raise RuntimeError("model actor returned an invalid execution report")
        parallelism_mode = JobSpec.model_validate(assignment.raw_spec).model.parallelism.mode
        reusable = (
            assignment.lease.allow_actor_reuse
            and parallelism_mode in {"single", "model_sharded"}
            and await self.runtime.is_live(runtime_handle, require_idle=True)
        )
        if self.finalization_journal is not None:
            report = await _to_thread(self._snapshot_completion, assignment, workspace, report)
        return await self.finalize(assignment, workspace, report, reusable, on_compute_finished)

    async def finalize(
        self, assignment, workspace, report, reusable=False, on_compute_finished=None
    ):
        canceled = threading.Event()
        self._finalization_cancellations[assignment.attempt_id] = canceled
        try:
            if on_compute_finished is not None:
                await on_compute_finished(reusable)
            TransferLogReporter(workspace.log_dir / "attempt.log")("waiting-io")
            # Once the callback releases the GPU, nothing below may inspect,
            # evict or otherwise touch its actor: another job can own it now.
            while True:
                if canceled.is_set():
                    raise RuntimeError("finalization canceled")
                try:
                    await asyncio.wait_for(self._finalization_slots.acquire(), timeout=1)
                    break
                except TimeoutError:
                    continue
            try:
                with cancellable_transfers(canceled), transfer_authority(
                    lambda: self.finalization_authority(assignment.attempt_id)
                ):
                    operation = asyncio.create_task(
                        self._retry_finalization(assignment, workspace, report, reusable, canceled)
                    )
                    try:
                        return await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        canceled.set()
                        # to_thread cancellation cannot stop filesystem writes.
                        # Wait for the actual operation before scratch cleanup.
                        with suppress(Exception):
                            await operation
                        raise
            finally:
                self._finalization_slots.release()
        except Exception as exc:
            with suppress(Exception):
                await _to_thread(_append_runtime_failure, workspace.log_dir / "attempt.log", exc)
            raise
        finally:
            self._finalization_cancellations.pop(assignment.attempt_id, None)

    async def _retry_finalization(self, assignment, workspace, report, reusable, canceled):
        delay = 1
        while True:
            try:
                if self.finalization_journal is not None:
                    with reuse_verified_outputs():
                        return await self._finalize_outputs(assignment, workspace, report, reusable)
                return await self._finalize_outputs(assignment, workspace, report, reusable)
            except (StorageUnavailableError, TimeoutError, OSError):
                if self.finalization_journal is None:
                    raise
                TransferLogReporter(workspace.log_dir / "attempt.log")("upload-retry")
                # Retry transport only. No actor is reachable from this loop.
                for _ in range(delay * 5):
                    if canceled.is_set():
                        from topodrome.storage.progress import TransferCancelled
                        raise TransferCancelled("transfer canceled") from None
                    await asyncio.sleep(0.2)
                delay = min(30, delay * 2)

    def _snapshot_completion(self, assignment, workspace, report):
        store = AttemptArtifactStore(
            self.storage, job_id=assignment.job_id, attempt_id=assignment.attempt_id
        )
        paths = set(assignment.raw_spec.get("required_outputs", ()))
        for registration in report.artifacts:
            source = registration.source.resolve()
            if not source.is_file():
                if registration.required:
                    raise RuntimeError("required artifact missing at compute completion")
                continue
            target = self.storage.resolve(f"{store.staging_prefix}/{registration.logical_path}")
            if source != target.resolve():
                store.stage_file(source, registration.logical_path)
            paths.add(registration.logical_path)
        # fsync payloads and directories before the compute receipt. Scratch
        # is not a durable output location and may be removed after restart.
        for path in paths:
            target = self.storage.resolve(f"{store.staging_prefix}/{path}")
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            _fsync_directory(target.parent)
        self.finalization_journal.save({
            "version": 1, "job_id": assignment.job_id, "attempt_id": assignment.attempt_id,
            "output_paths": sorted(paths), "checkpoint_ref": report.checkpoint_ref,
        })
        return replace(report, artifacts=(), output_paths=tuple(sorted(paths)))

    async def resume_finalization(self, assignment, workspace, receipt):
        report = _ExecutionReport((), receipt.get("checkpoint_ref"), tuple(receipt["output_paths"]))
        return await self.finalize(assignment, workspace, report)

    async def _finalize_outputs(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        report: _ExecutionReport,
        reusable: bool,
    ) -> ExecutionOutcome:
        check_transfer_cancelled()
        manifest = await _to_thread(
            self._commit_artifacts,
            assignment,
            workspace,
            report.artifacts,
            report.output_paths,
        )
        artifacts = tuple(
            {
                "logical_path": artifact.logical_path,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
                "durable": True,
            }
            for artifact in manifest.artifacts
        )
        checkpoint_ref = await _to_thread(self._finalize_checkpoints, assignment, workspace)
        checkpoint_ref = checkpoint_ref or report.checkpoint_ref or assignment.checkpoint_ref
        # Do not write progress while flushing this same log: it must remain
        # immutable between local hashing and authoritative verification.
        TransferLogReporter(workspace.log_dir / "attempt.log")("logs")
        log = await _to_thread(self._verify_log, assignment, workspace)
        check_transfer_cancelled()
        return ExecutionOutcome(
            succeeded=True,
            keep_warm=reusable,
            artifacts_verified=True,
            metadata={
                "artifacts": artifacts,
                "checkpoint_ref": checkpoint_ref,
                **({"log": log} if log is not None else {}),
            },
        )

    async def verify_attempt_log(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> Mapping[str, object] | None:
        """Flush and independently verify the complete attempt log."""

        return cast(
            Mapping[str, object] | None,
            await _to_thread(self._verify_log, assignment, workspace),
        )

    async def execute_kernel(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        *,
        operation_id: str,
        session_id: str,
        mode: KernelMode,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        """Run one operation through the actor's serialized execution queue."""

        allowed = (
            _RAW_KERNEL_OPERATIONS if mode is KernelMode.RAW_MODEL else _SERVICE_KERNEL_OPERATIONS
        )
        if operation not in allowed:
            raise ValueError(f"operation {operation!r} is unavailable in {mode.value} mode")
        runtime_handle = self._handle(prepared)
        return await self.runtime.execute(
            runtime_handle,
            _execute_kernel_operation,
            job_id=assignment.job_id,
            attempt_id=f"{assignment.attempt_id}:{operation_id}",
            kwargs={
                "session_id": session_id,
                "mode": mode.value,
                "operation": operation,
                "arguments": dict(arguments),
            },
            workspace=workspace.staging_dir,
            scratch=workspace.scratch_dir,
        )

    async def request_preemption(self, attempt_id: str) -> None:
        await self.runtime.request_preemption(attempt_id)

    async def cancel(self, attempt_id: str) -> None:
        await self.runtime.cancel(attempt_id)

    async def evict(self, prepared: PreparedRuntime) -> None:
        handle = self._handle(prepared)
        await self.runtime.evict(handle, force=True)
        with self._prepared_lock:
            self._prepared.pop(prepared.actor_id, None)

    async def evict_warm_allocation(
        self,
        allocation: ActorAllocation,
    ) -> tuple[int, ...]:
        """Compatibility entry point for an exact primary warm allocation."""

        with self._prepared_lock:
            value = self._prepared.get(allocation.actor_id)
        if value is None:
            raise RuntimeError("tracked warm allocation has no live runtime handle")
        handle, model_key = value
        if (
            model_key != allocation.model_key
            or handle.actor_id != allocation.actor_id
            or len(handle.key.replica_gpu_uuids) != 1
            or allocation.replica_rank != 0
            or allocation.gpu_uuid != handle.key.replica_gpu_uuids[0]
            or handle.transport is ExecutionTransport.RAY_GROUP
        ):
            raise RuntimeError("tracked warm allocation identity is inconsistent")
        await self.runtime.evict(handle, force=True)
        with self._prepared_lock:
            if self._prepared.get(allocation.actor_id) == value:
                self._prepared.pop(allocation.actor_id, None)
        return handle.worker_pids

    async def evict_warm_replica(
        self,
        allocations: Sequence[ActorAllocation],
    ) -> tuple[int, ...]:
        """Dispose only a complete rank-ordered, idle owned model replica."""

        ordered = tuple(allocations)
        if (
            not ordered
            or len({item.actor_id for item in ordered}) != 1
            or len({item.model_key for item in ordered}) != 1
            or tuple(item.replica_rank for item in ordered) != tuple(range(len(ordered)))
            or len({item.gpu_uuid for item in ordered}) != len(ordered)
            or any(item.state not in {"idle", "stopping"} for item in ordered)
        ):
            raise RuntimeError("warm eviction requires one complete ordered replica")
        actor_id = ordered[0].actor_id
        with self._prepared_lock:
            value = self._prepared.get(actor_id)
        if value is None:
            raise RuntimeError("tracked warm replica has no live runtime handle")
        handle, model_key = value
        expected_gpu_uuids = tuple(item.gpu_uuid for item in ordered)
        if (
            model_key != ordered[0].model_key
            or handle.actor_id != actor_id
            or handle.key.replica_gpu_uuids != expected_gpu_uuids
            or handle.transport is ExecutionTransport.RAY_GROUP
        ):
            raise RuntimeError("tracked warm replica identity is inconsistent")
        if not await self.runtime.is_live(handle, require_idle=True):
            raise RuntimeError("tracked warm replica is not idle and live")
        with self._prepared_lock:
            if self._prepared.get(actor_id) != value:
                raise RuntimeError("tracked warm replica changed during eviction validation")
        stopped = await self.runtime.evict(handle, force=True)
        if stopped is not True:
            raise RuntimeError("exact warm replica disposal was not confirmed")
        with self._prepared_lock:
            if self._prepared.get(actor_id) == value:
                self._prepared.pop(actor_id, None)
        return handle.worker_pids

    async def force_stop(self, prepared: PreparedRuntime) -> bool:
        """Immediately kill only the exact Ray handle prepared by this backend."""

        handle = self._handle(prepared)
        stopped = await self.runtime.force_stop(handle)
        if stopped:
            with self._prepared_lock:
                self._prepared.pop(prepared.actor_id, None)
        return stopped

    async def latest_checkpoint(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> str | None:
        latest = await _to_thread(self._verify_checkpoints, assignment, workspace)
        return latest or assignment.checkpoint_ref

    async def reconnect_warm(
        self,
        allocation: ActorAllocation,
        gpu_index: int,
    ) -> PreparedRuntime | None:
        """Rebuild process-local bookkeeping around one exact detached actor."""

        if not self.use_ray:
            return None
        try:
            identity = _mapping(json.loads(allocation.model_key), "persisted actor identity")
            model, runtime_fingerprint = _decode_actor_identity(identity)
            parsed = JobSpec.model_validate(
                {
                    "entrypoint": "recovery.py",
                    "model": model,
                    "resources": {
                        "gpu": 1,
                        "min_vram_gb": 0,
                    },
                }
            )
            if isinstance(parsed.model.parallelism, ModelShardedParallelism):
                return None
            registration = self._resolve_registration(model)
        except (KeyError, TypeError, ValueError):
            return None
        await asyncio.wait_for(
            _to_thread(self._ensure_ray_driver),
            timeout=self.ray_connect_timeout_seconds,
        )
        prepared = await self.runtime.reconnect(
            ActorKey(
                self.site_id,
                allocation.gpu_uuid,
                registration.key,
                runtime_fingerprint=runtime_fingerprint,
            ),
            actor_id=allocation.actor_id,
            cuda_visible_device=str(gpu_index),
            ray_namespace=self.ray_namespace,
        )
        if prepared is None:
            with self._prepared_lock:
                self._confirmed_stale_actor_ids.add(allocation.actor_id)
            return None
        with self._prepared_lock:
            self._prepared[prepared.actor_id] = (prepared, allocation.model_key)
        return PreparedRuntime(
            actor_id=prepared.actor_id,
            model_key=allocation.model_key,
            reused=True,
            opaque=_RuntimeHandle(prepared, self._owner),
            process_ids=prepared.worker_pids,
        )

    async def reconnect_warm_group(
        self,
        allocations: Sequence[ActorAllocation],
        gpu_indices: Mapping[str, int],
    ) -> PreparedRuntime | None:
        """Reconnect one exact clean sharded replica after an agent restart."""

        if not self.use_ray or len(allocations) < 2:
            return None
        ordered = tuple(sorted(allocations, key=lambda item: item.replica_rank))
        if (
            len({item.actor_id for item in ordered}) != 1
            or len({item.model_key for item in ordered}) != 1
            or len({item.replica_rank for item in ordered}) != len(ordered)
            or tuple(item.replica_rank for item in ordered) != tuple(range(len(ordered)))
        ):
            return None
        try:
            identity = _mapping(
                json.loads(ordered[0].model_key),
                "persisted replica identity",
            )
            model, runtime_fingerprint = _decode_actor_identity(identity)
            parallelism = _mapping(model.get("parallelism"), "model parallelism")
            if parallelism.get("mode") != "model_sharded" or int(
                parallelism.get("degree", 0)
            ) != len(ordered):
                return None
            registration = self._resolve_registration(model)
            gpu_uuids = tuple(item.gpu_uuid for item in ordered)
            visible = ",".join(str(int(gpu_indices[uuid])) for uuid in gpu_uuids)
        except (KeyError, TypeError, ValueError):
            return None
        await asyncio.wait_for(
            _to_thread(self._ensure_ray_driver),
            timeout=self.ray_connect_timeout_seconds,
        )
        prepared = await self.runtime.reconnect(
            ActorKey(
                self.site_id,
                gpu_uuids[0],
                registration.key,
                runtime_fingerprint=runtime_fingerprint,
                replica_gpu_uuids=gpu_uuids,
            ),
            actor_id=ordered[0].actor_id,
            cuda_visible_device=visible,
            ray_namespace=self.ray_namespace,
        )
        if prepared is None:
            with self._prepared_lock:
                self._confirmed_stale_actor_ids.add(ordered[0].actor_id)
            return None
        with self._prepared_lock:
            self._prepared[prepared.actor_id] = (prepared, ordered[0].model_key)
        return PreparedRuntime(
            actor_id=prepared.actor_id,
            model_key=ordered[0].model_key,
            reused=True,
            opaque=_RuntimeHandle(prepared, self._owner),
            process_ids=prepared.worker_pids,
        )

    def model_capabilities(self) -> tuple[Mapping[str, Any], ...]:
        """Return shared-schema capability payloads without site paths."""

        with self._prepared_lock:
            tracked = tuple(self._prepared.values())
        warm_actors = sorted(
            (
                (prepared, serialized_model_key)
                for prepared, serialized_model_key in tracked
                if prepared.transport is not ExecutionTransport.RAY_GROUP
                and self.runtime.core.is_live(prepared, require_idle=True)
            ),
            key=lambda value: (value[0].key.model, value[0].key.gpu_uuid),
        )
        capabilities: list[Mapping[str, Any]] = []
        for capability in self.registry.public_capabilities():
            value = capability.key.as_dict()
            value.update(
                {
                    "availability": capability.availability.value,
                    "gpu_uuid": None,
                    "runtime_fingerprint": None,
                }
            )
            capabilities.append(value)
        for prepared, serialized_model_key in warm_actors:
            for gpu_uuid in prepared.key.replica_gpu_uuids:
                warm_value: dict[str, Any] = dict(prepared.key.model.as_dict())
                warm_value.update(
                    {
                        "availability": "warm",
                        "gpu_uuid": gpu_uuid,
                        "runtime_fingerprint": prepared.key.runtime_fingerprint,
                        "replica_id": prepared.actor_id,
                        "replica_gpu_uuids": list(prepared.key.replica_gpu_uuids),
                        "replica_key_hash": hashlib.sha256(
                            serialized_model_key.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                capabilities.append(warm_value)
        return tuple(capabilities)

    async def stage_inputs(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> None:
        """Fetch assigned assets before this job prepares or acquires a model actor."""

        del workspace
        await _to_thread(self._resolve_assets, assignment)

    def prefetch_checkpoint(self, reference):
        path, sha256 = _parse_checkpoint_reference(reference)
        if is_regional_reference(path):
            if self.regional_state is None:
                raise RuntimeError("regional checkpoint storage unavailable")
            resolved = self.regional_state.resolve(path)
        else:
            resolved = self.storage.resolve(path)
        if _sha256_file(resolved) != sha256:
            raise RuntimeError("checkpoint digest does not match its immutable reference")
        return resolved

    def cached_assets(self) -> tuple[str, ...]:
        """Expose complete content-addressed asset cache entries."""

        return self.asset_resolver.cached_assets()

    def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
        """Confirm persistent allocation state against this live runtime."""

        with self._prepared_lock:
            value = self._prepared.get(allocation.actor_id)
        return bool(
            value is not None
            and allocation.gpu_uuid in _prepared_gpu_uuids(value[0])
            and value[1] == allocation.model_key
            and self.runtime.core.is_live(value[0], require_idle=True)
        )

    async def reconcile_warm_liveness(self) -> tuple[str, ...]:
        """Dispose stale tracked actors before admission or advertisement.

        Actor IDs are returned only after exact disposal has been confirmed, so
        the site authority may then remove the corresponding persistent rows.
        """

        with self._prepared_lock:
            candidates = tuple(self._prepared.items())

        async def reconcile_one(
            actor_id: str,
            value: tuple[PreparedActor, str],
        ) -> str | None:
            prepared, _model_key = value
            if await self.runtime.is_live(prepared):
                return None
            try:
                await self.runtime.evict(prepared, force=True)
            except Exception:
                # An ambiguous control-plane failure is not disposal proof.
                return None
            with self._prepared_lock:
                if self._prepared.get(actor_id) == value:
                    self._prepared.pop(actor_id, None)
                    return actor_id
            return None

        results = await asyncio.gather(
            *(reconcile_one(actor_id, value) for actor_id, value in candidates)
        )
        return tuple(actor_id for actor_id in results if actor_id is not None)

    def stale_warm_cleanup_confirmed(self, actor_id: str) -> bool:
        """Consume proof that reconnect found no live owned actor."""

        with self._prepared_lock:
            if actor_id not in self._confirmed_stale_actor_ids:
                return False
            self._confirmed_stale_actor_ids.remove(actor_id)
            return True

    def _handle(self, prepared: PreparedRuntime) -> PreparedActor:
        value = prepared.opaque
        if not isinstance(value, _RuntimeHandle) or value.owner is not self._owner:
            raise RuntimeError("prepared runtime belongs to another execution backend")
        return value.prepared

    def _resolve_registration(self, model: Mapping[str, Any]) -> ModelRegistration:
        return self.registry.resolve(
            str(model["alias"]),
            revision=_configured_wildcard(model.get("revision")),
            backend=_optional_text(model.get("backend")),
            backend_version=_optional_text(model.get("backend_version")),
            dtype=_optional_text(model.get("dtype")),
            quantization=_optional_text(model.get("quantization")),
            adapter=_optional_text(model.get("adapter")),
        )

    def _resolve_runtime_working_dir(
        self,
        reference: str | None,
        scratch: Path,
    ) -> str | None:
        if reference is None:
            return None
        parsed = urlsplit(reference)
        if parsed.scheme in {"http", "https", "s3", "gs"}:
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("runtime working directory URI must be public and immutable")
            return reference
        if self.source_resolver is None:
            raise RuntimeError("runtime working directory cannot be resolved at this site")
        directory = self.source_resolver(reference).resolve()
        if not directory.is_dir():
            raise RuntimeError("runtime working directory is unavailable")
        if not self.use_ray:
            return str(directory)
        return self.working_dir_publisher(directory, scratch)

    def _checkpoint_target(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> tuple[Path, str]:
        spec = JobSpec.model_validate(assignment.raw_spec)
        logical = f"jobs/{assignment.job_id}/attempts/{assignment.attempt_id}/checkpoints"
        if spec.portability is not Portability.REGION_AFFINE:
            return workspace.checkpoint_dir, logical
        if self.regional_state is None or not self.regional_state.healthcheck():
            raise RuntimeError("region-affine checkpoint storage is unavailable")
        directory = self.regional_state.checkpoint_directory(
            assignment.job_id,
            assignment.attempt_id,
        )
        return directory, self.regional_state.reference_for(directory)

    def _ensure_ray_driver(self) -> None:
        if self._ray_connected:
            return
        with self._ray_lock:
            if self._ray_connected:
                return
            # Ray 2.58 detects ``uv run`` and otherwise creates a nested,
            # incomplete worker environment from the current project.  Site
            # dependencies are already provisioned by Topodrome's pinned uv
            # environment, so workers must inherit that interpreter directly.
            configure_ray_driver_environment()
            try:
                import ray
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Ray execution requires the optional runtime extra") from exc
            if not ray.is_initialized():
                ray.init(
                    address=self.ray_address or "auto",
                    namespace=self.ray_namespace,
                    ignore_reinit_error=True,
                )
            self._ray_connected = True

    def _shutdown_ray_driver(self) -> bool:
        with self._ray_lock:
            if not self._ray_connected:
                return False
            configure_ray_driver_environment()
            try:
                import ray
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Ray execution requires the optional runtime extra") from exc
            if ray.is_initialized():
                # This driver always joins the separately managed site-local
                # head by explicit address. Ray documents this call as a local
                # disconnect in that mode; it never stops the cluster.
                ray.shutdown()
            self._ray_connected = False
            return True

    def _materialize_source(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> Path:
        source_ref = assignment.raw_spec.get("source_ref")
        if source_ref is None:
            return workspace.staging_dir
        if not isinstance(source_ref, str) or self.source_resolver is None:
            raise RuntimeError("assignment source cannot be resolved at this site")
        source = self.source_resolver(source_ref).resolve()
        current_manifest = source / SOURCE_BUNDLE_METADATA_DIRECTORY / SOURCE_BUNDLE_MANIFEST_NAME
        current_layout = current_manifest.is_file()
        manifest_path = current_manifest if current_layout else source / SOURCE_BUNDLE_MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("assignment source manifest is invalid") from exc
        if not isinstance(files, Mapping):
            raise RuntimeError("assignment source manifest has no file map")
        format_version = manifest.get("format_version")
        if (current_layout and format_version != SOURCE_BUNDLE_FORMAT_VERSION) or (
            not current_layout and format_version is not None
        ):
            raise RuntimeError("assignment source manifest has an invalid format")
        manifest_reference = manifest.get("source_ref")
        if manifest_reference is not None and manifest_reference != source_ref:
            raise RuntimeError("assignment source manifest has a different identity")
        destination_root = workspace.staging_dir.resolve()
        for raw_name, raw_digest in files.items():
            name = str(raw_name)
            logical = PurePosixPath(name)
            if (
                logical.is_absolute()
                or ".." in logical.parts
                or name in {"", "."}
                or name != logical.as_posix()
                or (current_layout and logical.parts[0] == SOURCE_BUNDLE_METADATA_DIRECTORY)
            ):
                raise RuntimeError("assignment source contains an unsafe logical path")
            source_file = (source / Path(*logical.parts)).resolve()
            if not source_file.is_relative_to(source) or not source_file.is_file():
                raise RuntimeError("assignment source file is unavailable")
            digest = _sha256_file(source_file)
            if digest != str(raw_digest):
                raise RuntimeError("assignment source failed content verification")
            destination = (destination_root / Path(*logical.parts)).resolve()
            if not destination.is_relative_to(destination_root):
                raise RuntimeError("assignment source escaped its attempt workspace")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        return destination_root

    def _commit_artifacts(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        registrations: Sequence[ArtifactRegistration],
        output_paths: Sequence[str] = (),
    ) -> Any:
        progress = TransferLogReporter(workspace.log_dir / "attempt.log")
        progress("staging")
        store = AttemptArtifactStore(
            self.storage,
            job_id=assignment.job_id,
            attempt_id=assignment.attempt_id,
        )
        paths: set[str] = set(output_paths)
        raw_required = assignment.raw_spec.get("required_outputs", ())
        if not isinstance(raw_required, Sequence) or isinstance(raw_required, (str, bytes)):
            raise RuntimeError("required_outputs must be a sequence")
        paths.update(str(path) for path in raw_required)
        for registration in registrations:
            source = registration.source.resolve()
            if not source.is_file():
                if registration.required:
                    raise RuntimeError(
                        f"registered artifact is missing: {registration.logical_path}"
                    )
                continue
            logical_path = registration.logical_path
            target = self.storage.resolve(f"{store.staging_prefix}/{logical_path}")
            if source != target.resolve():
                store.stage_file(source, logical_path)
            paths.add(logical_path)
        return store.commit(paths, progress=progress)

    def _resolve_assets(self, assignment: Assignment) -> Mapping[str, Path]:
        raw_assets = assignment.raw_spec.get("assets", {})
        if not isinstance(raw_assets, Mapping):
            raise RuntimeError("assets must map input aliases to pinned references")
        resolved: dict[str, Path] = {}
        for raw_alias, raw_reference in raw_assets.items():
            alias = str(raw_alias)
            reference: Any = raw_reference
            if isinstance(raw_reference, Mapping):
                reference = dict(raw_reference)
            asset = self.asset_resolver.resolve(reference)
            resolved[alias] = asset.local_path
        return resolved

    def _finalize_checkpoints(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> str | None:
        progress = TransferLogReporter(workspace.log_dir / "attempt.log")
        progress("checkpoint")
        with observe_transfers(
            lambda stage, current, total: progress("ckpt-" + stage, current, total)
        ):
            return self._verify_checkpoints(assignment, workspace)

    def _verify_checkpoints(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> str | None:
        spec = JobSpec.model_validate(assignment.raw_spec)
        regional = spec.portability is Portability.REGION_AFFINE
        if regional:
            if self.regional_state is None or not self.regional_state.healthcheck():
                raise RuntimeError("region-affine checkpoint storage is unavailable")
            checkpoint_directory = self.regional_state.checkpoint_directory(
                assignment.job_id,
                assignment.attempt_id,
            )
        else:
            checkpoint_directory = workspace.checkpoint_dir
        selected_checkpoint = pin_latest_checkpoint(checkpoint_directory)
        latest: str | None = None
        # Checkpoints are self-contained recovery snapshots. Older snapshots
        # are not prerequisites for the newest one; uploading all of them can
        # turn completed remote work into days of redundant network traffic.
        for checkpoint in (() if selected_checkpoint is None else (selected_checkpoint,)):
            if regional:
                assert self.regional_state is not None
                storage_reference = self.regional_state.reference_for(checkpoint)
                verified_sha256 = self.regional_state.flush_and_verify(
                    storage_reference
                ).digest.sha256
            else:
                storage_reference = (
                    f"jobs/{assignment.job_id}/attempts/{assignment.attempt_id}/"
                    f"checkpoints/{checkpoint.name}"
                )
                verified_sha256 = self.storage.flush_and_verify(storage_reference).digest.sha256
            latest = _checkpoint_reference(
                storage_reference,
                verified_sha256,
            )
        return latest

    def _verify_log(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> Mapping[str, object] | None:
        log_path = workspace.log_dir / "attempt.log"
        if not log_path.is_file():
            return None
        logical_path = f"jobs/{assignment.job_id}/attempts/{assignment.attempt_id}/logs/attempt.log"
        verified = self.storage.flush_and_verify(logical_path)
        return {
            "logical_path": logical_path,
            "start_offset": 0,
            "end_offset": verified.digest.size_bytes,
            "sha256": verified.digest.sha256,
        }

    def _load_checkpoint(self, assignment: Assignment) -> CheckpointRestore | None:
        reference = assignment.checkpoint_ref
        if reference is None:
            return None
        storage_reference, reference_sha256 = _parse_checkpoint_reference(reference)
        spec = JobSpec.model_validate(assignment.raw_spec)
        regional = is_regional_reference(storage_reference)
        if regional != (spec.portability is Portability.REGION_AFFINE):
            raise ValueError("checkpoint storage does not match the job portability policy")
        logical_reference = (
            storage_reference.removeprefix("regional://") if regional else storage_reference
        )
        _validate_checkpoint_reference(logical_reference, assignment)
        # This is an immutable input, not a newly produced output. Its producer
        # already established durability; verify the pinned local bytes without
        # uploading the checkpoint back to its authority on every restoration.
        local_path = self.prefetch_checkpoint(reference)
        # Send only inert metadata. The actor rehashes the shared site-local
        # file before unpickling under its leased CUDA visibility.
        return CheckpointRestore(
            reference, serialized_path=str(local_path.resolve()), sha256=reference_sha256
        )


def source_resolver_for_storage(storage: StorageAdapter) -> SourceResolver:
    def resolve(source_ref: str) -> Path:
        algorithm, separator, digest = source_ref.partition(":")
        if algorithm != "sha256" or not separator or len(digest) != 64:
            raise ValueError("invalid source reference")
        return storage.resolve(f"projects/.topodrome-sources/{digest}")

    return resolve


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            check_transfer_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_reference(storage_reference: str, sha256: str) -> str:
    """Bind an opaque checkpoint path to the bytes approved for restoration."""

    if not _CHECKPOINT_DIGEST.fullmatch(sha256):
        raise ValueError("checkpoint digest must be a lowercase SHA-256 value")
    if "#" in storage_reference or not storage_reference:
        raise ValueError("checkpoint storage reference is invalid")
    return f"{storage_reference}#sha256={sha256}"


def _parse_checkpoint_reference(reference: str) -> tuple[str, str]:
    storage_reference, marker, sha256 = reference.rpartition("#sha256=")
    if (
        marker != "#sha256="
        or not storage_reference
        or "#" in storage_reference
        or not _CHECKPOINT_DIGEST.fullmatch(sha256)
    ):
        raise ValueError("checkpoint reference requires an immutable SHA-256 digest")
    return storage_reference, sha256


async def _to_thread(function: Callable[..., Any], /, *args: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(function, *args)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _resolve_kernel_target(context: Any, path: str) -> Any:
    parts = path.split(".")
    roots = {
        "model": context.model,
        "tokenizer": context.tokenizer,
        "device": context.device,
    }
    if not parts or parts[0] not in roots:
        raise ValueError("kernel target must begin with model, tokenizer, or device")
    target = roots[parts[0]]
    for part in parts[1:]:
        if not part or part.startswith("__"):
            raise ValueError("dunder kernel targets are not allowed")
        target = getattr(target, part)
    return target


def _portable_result(value: Any, *, depth: int = 0) -> Any:
    """Bound results to JSON-safe metadata instead of leaking live objects."""

    if depth > 8:
        return "<maximum result depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _portable_result(item, depth=depth + 1)
            for key, item in list(value.items())[:4096]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_portable_result(item, depth=depth + 1) for item in list(value)[:4096]]
    numel = getattr(value, "numel", None)
    if callable(numel):
        count = int(numel())
        if count <= 4096 and hasattr(value, "tolist"):
            return _portable_result(value.tolist(), depth=depth + 1)
        return {
            "type": type(value).__qualname__,
            "shape": [int(item) for item in getattr(value, "shape", ())],
            "dtype": str(getattr(value, "dtype", "unknown")),
            "device": str(getattr(value, "device", "unknown")),
            "numel": count,
        }
    representation = repr(value)
    return {
        "type": type(value).__qualname__,
        "repr": representation[:1000],
    }


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _decode_actor_identity(
    identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    raw_model = identity.get("model")
    if raw_model is None:
        return identity, "legacy-runtime"
    model = _mapping(raw_model, "persisted actor model")
    runtime_fingerprint = str(identity.get("runtime_fingerprint") or "")
    if not runtime_fingerprint:
        raise ValueError("persisted actor identity has no runtime fingerprint")
    return model, runtime_fingerprint


def _prepared_gpu_uuids(prepared: PreparedActor) -> set[str]:
    """Return every physical slot held by a prepared runtime process/group."""

    members: set[str] = set()
    for key in prepared.group_keys or (prepared.key,):
        members.update(key.replica_gpu_uuids)
    return members


def _configured_wildcard(value: Any) -> str | None:
    text = _optional_text(value)
    return None if text in {None, "configured"} else text


def _checkpoint_interval(assignment: Assignment) -> float | None:
    raw = assignment.raw_spec.get("checkpoint")
    if raw is None:
        return None
    policy = _mapping(raw, "checkpoint policy")
    value = policy.get("interval_seconds")
    if value is None:
        return None
    interval = float(value)
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    return interval


def _validate_checkpoint_reference(reference: str, assignment: Assignment) -> None:
    logical = PurePosixPath(reference)
    parts = logical.parts
    if (
        len(parts) != 6
        or parts[0] != "jobs"
        or parts[1] != assignment.job_id
        or parts[2] != "attempts"
        or parts[3] == assignment.attempt_id
        or parts[4] != "checkpoints"
        or not parts[5].endswith(".pkl")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("checkpoint reference is outside the job's prior attempts")


__all__ = ["RuntimeExecutionBackend", "source_resolver_for_storage"]
