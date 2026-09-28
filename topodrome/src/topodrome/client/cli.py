from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import typer
from pydantic import BaseModel

from topodrome.client.client import Client
from topodrome.client.config import DEFAULT_URL, save_client_connection
from topodrome.client.dashboard import DashboardRefreshError, watch_dashboard
from topodrome.client.log_follow import follow_job
from topodrome.client.yaml_spec import load_job_file
from topodrome.domain.specs import (
    GPUReservationExtensionRequest,
    GPUReservationRequest,
    KernelRequest,
    ModelRequirement,
    ModelShardedParallelism,
    ResourceRequirements,
)
from topodrome.scheduler.assets import AssetPublishRequest, AssetTagUpdate, normalize_tags

app = typer.Typer(no_args_is_help=True, help="Submit and inspect Topodrome workloads.")


@app.command("storage-gc")
def storage_gc(
    apply: Annotated[
        bool, typer.Option("--apply", help="Delete eligible files in one bounded pass.")
    ] = False,
) -> None:
    """Operator-only storage retention audit; dry-run unless --apply is given."""
    with Client() as client:
        _print(client.storage_gc(apply=apply))


def _print(value: object, *, err: bool = False) -> None:
    typer.echo(
        json.dumps(_json_compatible(value), indent=2, sort_keys=True, default=str),
        err=err,
    )


def _json_compatible(value: object) -> object:
    """Recursively convert SDK models so collection commands emit real JSON."""

    if isinstance(value, BaseModel):
        return _json_compatible(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _with_loopback_tunnel_guidance(result: dict[str, Any]) -> dict[str, Any]:
    """Explain how a workstation IDE reaches a loopback-only stock server."""

    rendered = dict(result)
    if rendered.get("state") != "running":
        return rendered
    url = rendered.get("url")
    if not isinstance(url, str):
        return rendered
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return rendered
    try:
        port = parsed.port
    except ValueError:
        return rendered
    if port is None:
        return rendered
    local_url = f"{parsed.scheme}://127.0.0.1:{port}"
    rendered["ide_connection"] = {
        "local_url": local_url,
        "ssh_tunnel": f"ssh -N -L {port}:127.0.0.1:{port} lab1",
        "note": (
            "Start the SSH tunnel on the IDE workstation, then configure the IDE "
            "with local_url and the separately returned token."
        ),
    }
    return rendered


@app.command("login")
def login(
    url: Annotated[
        str,
        typer.Option("--url", help="Scheduler HTTP(S) origin."),
    ] = DEFAULT_URL,
) -> None:
    """Install this Unix user's whitelisted token in an owner-only file."""

    token = typer.prompt("Topodrome token", hide_input=True, confirmation_prompt=False)
    try:
        path = save_client_connection(url, token)
    except (OSError, ValueError) as exc:
        typer.echo(f"[topodrome] login failed ({type(exc).__name__})", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Saved Topodrome access for this user in {path}")


@app.command()
def submit(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    array: Annotated[
        str | None, typer.Option("--array", help="Array indices: 0-99 or 1,3,7.")
    ] = None,
    max_parallel: Annotated[int, typer.Option(min=1, max=1000)] = 1,
    idempotency_key: Annotated[
        str | None, typer.Option(help="Safe replay key for this submission.")
    ] = None,
    notify_codex: Annotated[
        str | None, typer.Option(help="Notify this local Codex session UUID on final job outcome.")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Job display name.")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Job description.")
    ] = None,
    estimated_runtime_seconds: Annotated[
        float | None,
        typer.Option(
            "--estimated-runtime-seconds",
            metavar="SECONDS",
            help="Positive runtime estimate; shorter jobs go first at equal priority.",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            help="Follow live attempt output until the job reaches a terminal state.",
        ),
    ] = False,
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            min=0.1,
            max=60,
            help="Seconds between status/log checks while using --wait.",
        ),
    ] = 0.5,
    log_grace: Annotated[
        float,
        typer.Option(
            "--log-grace",
            min=0,
            max=3600,
            help=(
                "Seconds to drain verified final logs after terminal job state; "
                "late logs remain available through `topodrome logs`."
            ),
        ),
    ] = 10.0,
) -> None:
    try:
        if notify_codex is not None:
            from .codex_bridge import open_store, preflight, state_root

            notify_codex = preflight(notify_codex)
            open_store(state_root()).close()
        spec, source_dir = load_job_file(manifest)
        metadata = {
            key: value
            for key, value in (
                ("name", name),
                ("description", description),
                ("estimated_runtime_seconds", estimated_runtime_seconds),
            )
            if value is not None
        }
        if metadata:
            spec = type(spec).model_validate({**spec.model_dump(), **metadata})
        if array is not None:
            from topodrome.domain.arrays import JobArraySpec

            from .arrays import parse_indices

            _submit_array_cli(
                JobArraySpec(job=spec, elements=parse_indices(array), max_parallel=max_parallel),
                source_dir,
                idempotency_key=idempotency_key,
                notify_codex=notify_codex,
                wait=wait,
                poll_seconds=poll_interval,
            )
            return
        if max_parallel != 1:
            raise ValueError("--max-parallel requires --array")
        with Client() as client:
            handle = client.submit(spec, source_dir=source_dir, idempotency_key=idempotency_key)
            if notify_codex is not None:
                try:
                    from .codex_bridge import start_bridge, watch_job

                    watch_job(
                        state_root(),
                        job_id=handle.job_id,
                        base_url=client.base_url,
                        thread_id=notify_codex,
                    )
                    start_bridge(state_root())
                except Exception as exc:
                    typer.echo(
                        f"[topodrome] job {handle.job_id} submitted; notification setup failed "
                        f"({type(exc).__name__}). Do not resubmit the job; use codex-notify.",
                        err=True,
                    )
            if not wait:
                _print(handle.status())
                return
            typer.echo(
                f"[topodrome] submitted {handle.job_id}; following output "
                "(Ctrl-C stops following only)",
                err=True,
            )
            try:
                summary = follow_job(
                    client,
                    handle.job_id,
                    log_output=typer.get_text_stream("stdout"),
                    status_output=typer.get_text_stream("stderr"),
                    poll_seconds=poll_interval,
                    terminal_log_grace_seconds=log_grace,
                )
            except KeyboardInterrupt:
                typer.echo(
                    f"\n[topodrome] stopped following {handle.job_id}; job continues",
                    err=True,
                )
                raise typer.Exit(130) from None
            except Exception as exc:
                typer.echo(
                    f"[topodrome] output follow failed ({type(exc).__name__}); "
                    f"job {handle.job_id} continues",
                    err=True,
                )
                raise typer.Exit(1) from None
            # Keep machine-captured workload output on stdout. Lifecycle
            # notices and the final structured summary use stderr.
            _print(summary, err=True)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(
            f"[topodrome] submission failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(1) from None


@app.command()
def status(job_id: str) -> None:
    with Client() as client:
        _print(client.status(job_id))


def _submit_array_cli(
    spec, source_dir, *, idempotency_key=None, notify_codex=None, wait=False, poll_seconds=2
):
    from uuid import uuid4

    from .arrays import follow_array

    key = idempotency_key or f"array-{uuid4().hex}"
    typer.echo(f"[topodrome] array replay key: --idempotency-key {key}", err=True)
    with Client() as client:
        result = client.submit_array(spec, source_dir=source_dir, idempotency_key=key)
        array_id = result["array_id"]
        typer.echo(f"[topodrome] submitted {array_id}", err=True)
        if notify_codex:
            from .codex_bridge import start_bridge, state_root, watch_job

            try:
                watch_job(
                    state_root(), job_id=array_id, base_url=client.base_url, thread_id=notify_codex
                )
                start_bridge(state_root())
            except Exception as exc:
                typer.echo(
                    f"[topodrome] {array_id} submitted; notification setup failed "
                    f"({type(exc).__name__}); use codex-notify, do not resubmit",
                    err=True,
                )
        if wait:
            try:
                result = follow_array(
                    client,
                    array_id,
                    output=typer.get_text_stream("stderr"),
                    poll_seconds=poll_seconds,
                )
            except KeyboardInterrupt:
                typer.echo(f"[topodrome] detached; {array_id} continues", err=True)
                raise typer.Exit(130) from None
            except Exception as exc:
                typer.echo(
                    f"[topodrome] follow failed ({type(exc).__name__}); "
                    f"{array_id} continues; inspect with `topodrome array {array_id}`",
                    err=True,
                )
                raise typer.Exit(1) from None
        _print(result)
        if wait and result["state"] != "succeeded":
            raise typer.Exit(1)


@app.command("submit-array")
def submit_array_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    idempotency_key: str | None = None,
    notify_codex: str | None = None,
    wait: bool = False,
):
    """Submit a parameterized array manifest, sharing source and immutable assets."""
    from .arrays import load_array_file

    if notify_codex:
        from .codex_bridge import preflight

        notify_codex = preflight(notify_codex)
    try:
        spec, source_dir = load_array_file(manifest)
        _submit_array_cli(
            spec, source_dir, idempotency_key=idempotency_key, notify_codex=notify_codex, wait=wait
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from None
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(
            f"[topodrome] submission failed ({type(exc).__name__}); "
            "if uncertain, retry with the same replay key and unchanged manifest",
            err=True,
        )
        raise typer.Exit(1) from None


@app.command("array")
def array_status(array_id: str, after: int = -1, limit: int = 100):
    """Show aggregate state and a page of elements (use job IDs for logs/cancel)."""
    with Client() as client:
        _print(client.array(array_id, after=after, limit=limit))


@app.command("arrays")
def arrays_status():
    with Client() as client:
        _print(client.arrays())


@app.command("array-set")
def array_set(
    array_id: str,
    max_parallel: Annotated[int | None, typer.Option(min=1, max=1000)] = None,
    held: Annotated[bool | None, typer.Option("--hold/--resume")] = None,
):
    """Change dispatch allowance; existing work is never interrupted."""
    with Client() as client:
        _print(client.update_array(array_id, max_parallel=max_parallel, held=held))


@app.command("array-cancel")
def array_cancel(array_id: str):
    """Cancel all unfinished elements; preserve completed results."""
    with Client() as client:
        _print(client.cancel_array(array_id))


@app.command()
def logs(job_id: str, tail_bytes: int = 131072) -> None:
    with Client() as client:
        typer.echo(client.logs(job_id, tail_bytes=tail_bytes), nl=False)


@app.command()
def cancel(job_id: str) -> None:
    with Client() as client:
        _print(client.cancel(job_id))


@app.command("asset-publish")
def asset_publish(
    name: str,
    version: str,
    job_id: Annotated[str, typer.Option("--job", help="Succeeded producer job ID.")],
    path: Annotated[
        str,
        typer.Option(
            "--path",
            help="Committed output file or directory prefix to publish.",
        ),
    ],
    shared: Annotated[
        bool,
        typer.Option("--shared", help="Make the immutable asset visible to every user."),
    ] = False,
) -> None:
    """Publish committed output as a named asset without copying its bytes."""

    request = AssetPublishRequest(
        name=name,
        version=version,
        job_id=job_id,
        path=path,
        visibility="shared" if shared else "private",
    )
    with Client() as client:
        _print(client.publish_asset(request))


@app.command("asset-upload")
def asset_upload(
    name: str,
    version: str,
    path: Path,
    shared: Annotated[bool, typer.Option("--shared")] = False,
) -> None:
    """Upload a local file or directory as a new immutable asset; no GPU job."""
    with Client() as client:
        _print(
            client.upload_asset(
                path,
                name=name,
                version=version,
                visibility="shared" if shared else "private",
            )
        )


@app.command("asset-download")
def asset_download(
    name: str,
    version: str,
    destination: Path,
    shared: Annotated[bool, typer.Option("--shared")] = False,
) -> None:
    """Download asset files to a new editable local path; no GPU or model."""
    with Client() as client:
        path = client.download_asset(
            name,
            version,
            destination,
            visibility="shared" if shared else "private",
        )
        _print({"path": str(path)})


@app.command("assets")
def assets(
    visibility: Annotated[
        str | None,
        typer.Option("--visibility", help="Filter by private or shared visibility."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 100,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Require each tag (repeatable).")
    ] = None,
) -> None:
    """List reusable assets visible to the current user."""

    if visibility not in {None, "private", "shared"}:
        raise typer.BadParameter("--visibility must be private or shared")
    try:
        tags = normalize_tags(tag or ())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    with Client() as client:
        _print(client.assets(visibility=visibility, limit=limit, tags=tags))


@app.command("asset-delete")
def asset_delete(
    name: str,
    version: str,
    shared: Annotated[bool, typer.Option("--shared")] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Confirm removal of this exact reference.")
    ] = False,
) -> None:
    """Delete one asset reference; pinned job inputs and shared bytes remain protected."""
    visibility = "shared" if shared else "private"
    if not yes:
        typer.confirm(
            f"Delete asset://{visibility}/{name}@{version}? "
            "New consumers lose this reference; existing job pins remain protected.",
            abort=True,
        )
    with Client() as client:
        _print(client.delete_asset(name, version, visibility=visibility))


@app.command("asset-tags")
def asset_tags(
    name: str,
    version: str,
    add: Annotated[list[str] | None, typer.Option("--add", help="Add a tag (repeatable).")] = None,
    remove: Annotated[
        list[str] | None, typer.Option("--remove", help="Remove a tag (repeatable).")
    ] = None,
    shared: Annotated[bool, typer.Option("--shared")] = False,
) -> None:
    """Read or edit labels without changing an asset's immutable content."""
    try:
        update = AssetTagUpdate(add=add or (), remove=remove or ())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    with Client() as client:
        visibility = "shared" if shared else "private"
        result = (
            client.update_asset_tags(
                name, version, add=update.add, remove=update.remove, visibility=visibility
            )
            if update.add or update.remove
            else client.asset(name, version, visibility=visibility)
        )
        _print(result)


@app.command("asset")
def asset(
    name: str,
    version: str,
    shared: Annotated[
        bool,
        typer.Option("--shared", help="Read from the shared rather than private namespace."),
    ] = False,
) -> None:
    """Show one immutable asset version."""

    with Client() as client:
        _print(
            client.asset(
                name,
                version,
                visibility="shared" if shared else "private",
            )
        )


@app.command()
def kernel(
    model: Annotated[str, typer.Option("--model")],
    mode: Annotated[str, typer.Option("--mode")] = "raw-model",
    min_vram_gb: Annotated[float, typer.Option("--min-vram-gb")] = 0,
    gpus: Annotated[
        int,
        typer.Option(
            "--gpus",
            min=1,
            max=64,
            help="Physical GPU count for the kernel's atomic lease.",
        ),
    ] = 1,
    max_memory_per_gpu: Annotated[
        float | None,
        typer.Option(
            "--max-memory-per-gpu",
            min=0.001,
            help=(
                "Transformers model-sharding cap in GiB per GPU; required when "
                "--gpus is greater than 1."
            ),
        ),
    ] = None,
) -> None:
    if gpus == 1 and max_memory_per_gpu is not None:
        raise typer.BadParameter("--max-memory-per-gpu requires --gpus greater than 1")
    if gpus > 1 and max_memory_per_gpu is None:
        raise typer.BadParameter("--max-memory-per-gpu is required when --gpus is greater than 1")
    if min_vram_gb > 0 and max_memory_per_gpu is not None and max_memory_per_gpu > min_vram_gb:
        raise typer.BadParameter("--max-memory-per-gpu cannot exceed --min-vram-gb")
    parallelism = (
        ModelShardedParallelism(
            mode="model_sharded",
            degree=gpus,
            engine="transformers_device_map",
            strategy="balanced",
            max_memory_gb_per_gpu=max_memory_per_gpu,
        )
        if max_memory_per_gpu is not None
        else None
    )
    model_requirement = (
        ModelRequirement(alias=model, parallelism=parallelism)
        if parallelism is not None
        else ModelRequirement(alias=model)
    )
    request = KernelRequest(
        model=model_requirement,
        resources=ResourceRequirements(gpu=gpus, min_vram_gb=min_vram_gb),
        mode=mode,
    )
    with Client() as client:
        handle = client.kernel(request)
        _print(handle.status())


@app.command("kernel-status")
def kernel_status(session_id: str) -> None:
    """Show one interactive kernel session."""

    with Client() as client:
        _print(client.kernel_status(session_id))


@app.command("kernel-stop")
def kernel_stop(session_id: str) -> None:
    """Stop one interactive kernel session."""

    with Client() as client:
        _print(client.stop_kernel(session_id))


@app.command("notebook-server")
def notebook_server(
    session_id: str,
    status_only: Annotated[bool, typer.Option("--status")] = False,
    stop: Annotated[bool, typer.Option("--stop")] = False,
    keep_kernel: Annotated[bool, typer.Option("--keep-kernel")] = False,
) -> None:
    """Print an IDE-ready stock Jupyter Server URL and token."""

    if status_only and stop:
        raise typer.BadParameter("--status and --stop are mutually exclusive")
    if keep_kernel and not stop:
        raise typer.BadParameter("--keep-kernel requires --stop")
    with Client() as client:
        if stop:
            result = client.stop_kernel_notebook_server(
                session_id,
                stop_kernel=not keep_kernel,
            )
        elif status_only:
            result = client.kernel_notebook_server_info(session_id)
        else:
            result = client.kernel_notebook_server(session_id)
        _print(_with_loopback_tunnel_guidance(result))


@app.command()
def sites() -> None:
    with Client() as client:
        _print(client.sites())


@app.command()
def gpus() -> None:
    with Client() as client:
        _print(client.gpus())


@app.command()
def models() -> None:
    with Client() as client:
        _print(client.models())


@app.command("gpu-reserve")
def gpu_reserve(
    site_id: str,
    duration_seconds: Annotated[
        int,
        typer.Option("--duration-seconds", min=1, max=31_536_000),
    ],
    reason: Annotated[str, typer.Option("--reason")],
    gpu_uuid: Annotated[str | None, typer.Option("--gpu-uuid")] = None,
    gpu_index: Annotated[int | None, typer.Option("--gpu-index", min=0)] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Evict owned active work; retry batch jobs. Uncheckpointed progress may be lost.",
        ),
    ] = False,
    idempotency_key: Annotated[
        str | None,
        typer.Option("--idempotency-key"),
    ] = None,
) -> None:
    """Reserve one exact physical GPU for a bounded maintenance window."""

    if (gpu_uuid is None) == (gpu_index is None):
        raise typer.BadParameter("provide exactly one of --gpu-uuid or --gpu-index")
    if force:
        typer.echo(
            "[topodrome] Forced reservation: batch work will retry; uncheckpointed "
            "progress may be lost. Interactive sessions will stop.",
            err=True,
        )
    request = GPUReservationRequest(
        site_id=site_id,
        gpu_uuid=gpu_uuid,
        gpu_index=gpu_index,
        duration_seconds=duration_seconds,
        reason=reason,
        force=force,
    )
    with Client() as client:
        _print(client.reserve_gpu(request, idempotency_key=idempotency_key))


@app.command("gpu-reservations")
def gpu_reservations(
    active_only: Annotated[bool, typer.Option("--active-only")] = False,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 100,
) -> None:
    """List durable GPU reservation audit records."""

    with Client() as client:
        _print(client.gpu_reservations(limit=limit, active_only=active_only))


@app.command("gpu-reservation")
def gpu_reservation(reservation_id: str) -> None:
    """Show one durable GPU reservation."""

    with Client() as client:
        _print(client.gpu_reservation(reservation_id))


@app.command("gpu-reservation-cancel")
def gpu_reservation_cancel(reservation_id: str) -> None:
    """Cancel a GPU reservation idempotently."""

    with Client() as client:
        _print(client.cancel_gpu_reservation(reservation_id))


@app.command("gpu-reservation-extend")
def gpu_reservation_extend(
    reservation_id: str,
    additional_seconds: Annotated[int, typer.Option("--seconds", min=1, max=31_536_000)],
) -> None:
    """Add seconds to an active reservation's expiry without releasing the GPU."""

    with Client() as client:
        current = client.gpu_reservation(reservation_id)
        extension = GPUReservationExtensionRequest(
            expected_expires_at=current.expires_at,
            additional_seconds=additional_seconds,
        )
        _print(client.extend_gpu_reservation(reservation_id, extension))


@app.command("top")
def top_status(
    interval: Annotated[
        float,
        typer.Option("--interval", min=0.1, help="Refresh interval in seconds."),
    ] = 2.0,
    once: Annotated[
        bool,
        typer.Option("--once", help="Render one snapshot and exit."),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable ANSI colors."),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Use the legacy text dashboard instead of the workbench."),
    ] = False,
) -> None:
    """Open the interactive workbench, or print one snapshot with --once."""

    output = typer.get_text_stream("stdout")
    if not once and not plain and output.isatty():
        try:
            from .workbench import run_workbench
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {"textual", "rich"}:
                typer.echo(
                    "Workbench requires topodrome[ui]; use --plain or --once meanwhile.", err=True
                )
                raise typer.Exit(code=1) from None
            raise
        run_workbench(interval_seconds=interval, no_color=no_color)
        return
    try:
        with Client() as client:
            watch_dashboard(
                client.system_status,
                interval_seconds=interval,
                once=once or (not plain and not output.isatty()),
                output=output,
                color=False if no_color else None,
            )
    except DashboardRefreshError:
        raise typer.Exit(code=1) from None


@app.command("policy-reload")
def policy_reload() -> None:
    """Validate and stage a scheduling-policy reload."""

    try:
        with Client() as client:
            _print(client.reload_policy())
    except Exception as exc:
        typer.echo(
            f"[topodrome] policy reload failed ({type(exc).__name__})",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command("user-list")
def user_list() -> None:
    """List local Unix accounts whitelisted for Topodrome."""

    with Client() as client:
        _print({"users": client.access_users()})


@app.command("user-grant")
def user_grant(username: str) -> None:
    """Grant or rotate one user's API token; the token is shown only once."""

    with Client() as client:
        _print(client.grant_access_user(username))


@app.command("user-revoke")
def user_revoke(username: str) -> None:
    """Remove a local Unix account from the Topodrome whitelist."""

    with Client() as client:
        _print(client.revoke_access_user(username))


@app.command("codex-notify")
def codex_notify(
    job_id: list[str],
    thread: Annotated[str, typer.Option(help="Exact Codex session UUID on this machine.")],
) -> None:
    """Notify once for one job or a group: all succeed, any fail, or any cancel."""
    from .codex_bridge import preflight, start_bridge, state_root, watch_group, watch_job

    thread = preflight(thread)
    with Client() as client:
        if len(set(job_id)) > 1 and any(member.startswith("array_") for member in job_id):
            raise typer.BadParameter("subscribe to each array separately")
        for member in set(job_id):
            if member.startswith("array_"):
                client.array(member, limit=1)
            else:
                client.status(member)  # Authorize before persisting anything.
        if len(set(job_id)) > 1:
            notification_id = watch_group(
                state_root(), job_ids=job_id, base_url=client.base_url, thread_id=thread
            )
        else:
            notification_id = job_id[0]
            watch_job(
                state_root(), job_id=notification_id, base_url=client.base_url, thread_id=thread
            )
    start_bridge(state_root())
    _print(
        {
            "notification_id": notification_id,
            "job_ids": job_id,
            "codex_thread": thread,
            "notification": "registered",
        }
    )


@app.command("codex-unsubscribe")
def codex_unsubscribe(notification_id: str) -> None:
    """Stop a local job/group notification, without canceling any jobs."""
    from .codex_bridge import state_root, unsubscribe

    try:
        result = unsubscribe(state_root(), notification_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print(result)
    if result["delivery_may_already_be_queued"]:
        typer.echo("An in-flight or already queued message cannot be recalled.", err=True)


@app.command("codex-bridge")
def codex_bridge(
    status: Annotated[
        bool, typer.Option(help="Show this user's notification delivery state.")
    ] = False,
    once: Annotated[bool, typer.Option(help="Poll once instead of running continuously.")] = False,
    retry_uncertain: Annotated[
        str | None,
        typer.Option(help="Explicitly retry uncertain delivery; may duplicate a message."),
    ] = None,
) -> None:
    """Run or inspect the user-local notification bridge, never the scheduler."""
    from . import codex_bridge as bridge

    if sum((status, once, retry_uncertain is not None)) > 1:
        raise typer.BadParameter("choose only one bridge action")
    root = bridge.state_root()
    if retry_uncertain:
        bridge.retry_uncertain(root, retry_uncertain)
        bridge.start_bridge(root)
    elif status:
        _print(bridge.status(root))
    else:
        bridge.run_bridge(root, once=once)


@app.command("autodl")
def autodl_status(
    action: Annotated[
        str, typer.Argument(help="status, register, release, or unregister")
    ] = "status",
    site: Annotated[
        str | None, typer.Option(help="Exact manual AutoDL site ID to release or unregister")
    ] = None,
    ssh: Annotated[str | None, typer.Option(help="Quoted SSH connection command")] = None,
    history: Annotated[bool, typer.Option(help="Include provider operation audit history")] = False,
    force: Annotated[
        bool,
        typer.Option(
            help="Release: preempt jobs for retry elsewhere; uncheckpointed work may be lost."
        ),
    ] = False,
) -> None:
    """Register, inspect, or unregister manually rented AutoDL instances."""

    if action not in {"status", "register", "release", "unregister"}:
        raise typer.BadParameter("expected status, register, release, or unregister")
    if (action in {"release", "unregister"}) != (site is not None):
        raise typer.BadParameter(
            "release/unregister require --site; --site is only for these actions"
        )
    if force and action != "release":
        raise typer.BadParameter("--force requires autodl release")
    if ssh is not None and action != "register":
        raise typer.BadParameter("--ssh requires autodl register")
    with Client() as client:
        if action == "release":
            _print(client.release_autodl(site, force=force))
            return
        if action == "unregister":
            _print(client.unregister_autodl(site))
            return
        if action == "register":
            from topodrome.deployment.manual_autodl import parse_ssh

            command = ssh or typer.prompt("SSH command")
            try:
                parse_ssh(command)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from None
            password = typer.prompt("SSH password", hide_input=True)
            _print(client.register_autodl(command, password))
            return
        result = {"control": client.autodl_control(), "instances": client.autodl_instances()}
        if history:
            result["operations"] = client.autodl_operations()
        _print(result)


if __name__ == "__main__":
    app()
