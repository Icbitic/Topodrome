"""Create an explicitly labeled, no-GPU fixture for transport benchmarking.

Requires an idle scheduler and explicit opt-in. All bytes/metadata belong to one
fresh synthetic job; no permanent assets or credentials are created. The public
download API pins its verified test files for the benchmark consumer.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import func, select

from topodrome.client.config import client_connection
from topodrome.config import SchedulerSettings, _read_yaml
from topodrome.db.models import (
    ArtifactRow,
    AttemptRow,
    InputStageRow,
    JobRow,
    KernelSessionRow,
    LeaseRow,
    SiteRow,
)
from topodrome.db.session import Database
from topodrome.deployment.jupyter_agent import _read_protected_file
from topodrome.domain.enums import ArtifactState, AttemptState, JobState
from topodrome.domain.specs import JobSpec
from topodrome.protocol.user_access import UserCredentialStore
from topodrome.scheduler.source_store import SourceBundle, SourceStore
from topodrome.storage.artifacts import CommitManifest, CommittedArtifact
from topodrome.storage.local import fsync_and_digest


async def publish_fixture(db, manifest, site_id, username, log_attempts):
    """Commit parents before children, including on FK-enforcing databases."""
    async with db.transaction() as session:
        existing = await session.get(JobRow, manifest.job_id)
        if existing is not None:
            attempts = (
                await session.scalars(
                    select(AttemptRow).where(AttemptRow.job_id == manifest.job_id)
                )
            ).all()
            artifacts = (
                await session.scalars(
                    select(ArtifactRow).where(ArtifactRow.job_id == manifest.job_id)
                )
            ).all()
            if (
                existing.owner_username != username
                or existing.state != JobState.SUCCEEDED
                or existing.spec.get("model", {}).get("repo_id") != "test/no-gpu-test-fixture"
                or {item.id for item in attempts} != {manifest.attempt_id, *log_attempts}
                or any(
                    item.site_id != site_id or item.state != AttemptState.SUCCEEDED
                    for item in attempts
                )
                or {(item.logical_path, item.size_bytes, item.sha256) for item in artifacts}
                != {
                    (item.logical_path, item.size_bytes, item.sha256) for item in manifest.artifacts
                }
            ):
                raise ValueError("fixture already exists with different metadata")
            return  # Safe recovery after metadata commit but before pin creation.
        session.add(
            JobRow(
                id=manifest.job_id,
                owner_username=username,
                state=JobState.SUCCEEDED,
                phase="succeeded",
                spec=JobSpec(
                    entrypoint="synthetic_transport_fixture.py",
                    model="test/no-gpu-test-fixture",
                    name="TEST ONLY: transport benchmark, no GPU job",
                ).model_dump(mode="json"),
            )
        )
        await session.flush()
        for number, identity in enumerate([manifest.attempt_id, *log_attempts], start=1):
            session.add(
                AttemptRow(
                    id=identity,
                    job_id=manifest.job_id,
                    site_id=site_id,
                    number=number,
                    state=AttemptState.SUCCEEDED,
                    completed_at=manifest.committed_at,
                )
            )
        await session.flush()
        for member in manifest.artifacts:
            session.add(
                ArtifactRow(
                    job_id=manifest.job_id,
                    attempt_id=manifest.attempt_id,
                    logical_path=member.logical_path,
                    size_bytes=member.size_bytes,
                    sha256=member.sha256,
                    state=ArtifactState.COMMITTED,
                    committed_at=manifest.committed_at,
                )
            )


def write_receipt(state, receipt):
    path = state / "preparation.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    path.chmod(0o600)
    fsync_and_digest(path)
    directory = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


async def prepare(args):
    url, token = client_connection(None, None)
    if args.scheduler_env_file:
        _read_protected_file(args.scheduler_env_file, description="benchmark scheduler environment")
    settings = SchedulerSettings(
        _env_file=args.scheduler_env_file, **_read_yaml(args.scheduler_config)
    )
    if token is None:
        raise ValueError("private client credential required")
    if hmac.compare_digest(token, settings.api_token.get_secret_value()):
        username = "@operator"
    else:
        principal = UserCredentialStore(settings.user_access.credential_store_path).authenticate(
            token
        )
        if principal is None:
            raise ValueError("client credential not recognized")
        username = principal.username
    db = Database(settings.database_dsn)
    state = args.state_dir.resolve()
    state.mkdir(mode=0o700, exist_ok=bool(args.recover_job))
    if (state / "descriptor.json").exists():
        raise ValueError("fixture is already prepared; use its existing descriptor")
    job_id = args.recover_job or "job_transportbench_" + uuid4().hex[:24]
    if not re.fullmatch(r"job_transportbench_[a-f0-9]{24}", job_id):
        raise ValueError("recovery requires an exact benchmark-owned job")
    attempt_id = "attempt_" + uuid4().hex
    timestamp = datetime.now(UTC)
    try:
        async with db.session() as session:
            gates = [
                (JobRow, ["succeeded", "failed", "canceled"]),
                (LeaseRow, ["released", "expired", "rejected"]),
                (KernelSessionRow, ["stopped", "failed", "expired"]),
                (InputStageRow, ["consumed", "canceled", "failed", "expired"]),
            ]
            for model, terminal in gates:
                if await session.scalar(
                    select(func.count())
                    .select_from(model)
                    .where(
                        model.state.not_in(terminal),
                    )
                ):
                    raise ValueError("scheduler is not idle")
            site = await session.get(SiteRow, args.site_id)
            if site is None or str(site.state) != "ready":
                raise ValueError("benchmark requires an already enrolled ready site")
        root = settings.global_root
        receipt = {"job_id": job_id, "global_root": str(root), "site_id": args.site_id}
        previous = {}
        if (state / "preparation.json").exists():
            previous = json.loads((state / "preparation.json").read_text())
            if any(previous.get(key) != value for key, value in receipt.items()):
                raise ValueError("preparation receipt belongs to another fixture")
        receipt.update(previous)
        write_receipt(state, receipt)
        if shutil.disk_usage(root).free < (args.bulk_gib + 12) * 1024**3:
            raise ValueError("insufficient benchmark disk headroom")
        publication = root / "jobs" / job_id / "committed" / "manifest.json"
        if args.recover_job:
            manifest = CommitManifest.from_bytes(publication.read_bytes())
            if manifest.job_id != job_id or manifest.schema_version != 1:
                raise ValueError("recovery manifest is not the benchmark fixture")
            for member in manifest.artifacts:
                path = root / member.storage_path
                if not path.resolve().is_relative_to(root / "jobs" / job_id):
                    raise ValueError("recovery member escapes the fixture")
                observed = fsync_and_digest(path)
                if (observed.size_bytes, observed.sha256) != (member.size_bytes, member.sha256):
                    raise ValueError("recovery member failed independent verification")
            attempt_id, files = manifest.attempt_id, list(manifest.artifacts)
        else:
            manifest, files = create_files(root, job_id, attempt_id, timestamp, args.bulk_gib)
        receipt["attempt_id"] = attempt_id
        write_receipt(state, receipt)
        source = SourceStore(root).put(
            SourceBundle(
                files={
                    f"lib/module_{index}.py": base64.b64encode(
                        f"# transport fixture {job_id}\nVALUE = {index}\n".encode(),
                    ).decode()
                    for index in range(32)
                }
            )
        )
        log_attempts = previous.get("log_attempts") or ["attempt_" + uuid4().hex for _ in range(16)]
        receipt.update(source_ref=source.source_ref, log_attempts=log_attempts)
        write_receipt(state, receipt)
        await publish_fixture(db, manifest, args.site_id, username, log_attempts)
        pins = []
        async with httpx.AsyncClient(
            base_url=url, headers={"Authorization": "Bearer " + token}, trust_env=False, timeout=120
        ) as client:
            for member in files:
                response = await client.post(
                    "/api/v1/downloads", json={"reference": f"job://{job_id}/{member.logical_path}"}
                )
                response.raise_for_status()
                pins.append(response.json())
                receipt["download_ids"] = [item["download_id"] for item in pins]
                write_receipt(state, receipt)
        descriptor = {
            "job_id": job_id,
            "site_id": args.site_id,
            "attempt_id": attempt_id,
            "download_ids": [item["download_id"] for item in pins],
            "owner_username": username,
            "source_path": "projects/.topodrome-sources/"
            + source.source_ref.removeprefix("sha256:"),
            "files": [
                {
                    "logical_path": member.storage_path,
                    "size_bytes": member.size_bytes,
                    "sha256": member.sha256,
                    "http_path": f"/api/v1/downloads/{pin['download_id']}/files/0",
                }
                for member, pin in zip(files, pins, strict=True)
            ],
            "fresh_log_paths": [
                f"jobs/{job_id}/attempts/{item}/logs/attempt.log" for item in log_attempts
            ],
            "resume_output": f"jobs/{job_id}/attempts/{attempt_id}/benchmark-output/resume.bin",
        }
        path = state / "descriptor.json"
        path.write_text(json.dumps(descriptor, indent=2) + "\n")
        path.chmod(0o600)
        fsync_and_digest(path)
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "descriptor": str(path),
                    "bytes": sum(item.size_bytes for item in files),
                    "gpu_jobs": 0,
                }
            )
        )
    finally:
        await db.close()


def create_files(root, job_id, attempt_id, timestamp, bulk_gib):
    prefix = root / "jobs" / job_id / "attempts" / attempt_id / "committed"
    prefix.mkdir(mode=0o700, parents=True)
    files = []
    for name, size in (
        ("00-tiny.bin", 8192),
        ("01-medium.bin", 4 * 1024**2),
        ("02-bulk.bin", bulk_gib * 1024**3),
    ):
        path = prefix / name
        digest = hashlib.sha256()
        remaining = size
        block = os.urandom(1024**2)
        with path.open("xb") as output:
            while remaining:
                body = block[: min(len(block), remaining)]
                output.write(body)
                digest.update(body)
                remaining -= len(body)
            output.flush()
            os.fsync(output.fileno())
        observed = fsync_and_digest(path)
        if observed.sha256 != digest.hexdigest():
            raise ValueError("benchmark fixture failed independent verification")
        files.append(
            CommittedArtifact(name, path.relative_to(root).as_posix(), size, observed.sha256)
        )
    manifest = CommitManifest(job_id, attempt_id, tuple(files), timestamp)
    publication = root / "jobs" / job_id / "committed" / "manifest.json"
    publication.parent.mkdir(parents=True)
    publication.write_bytes(manifest.to_bytes())
    fsync_and_digest(publication)
    return manifest, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ack", choices=["RUN_TRANSPORT_BENCHMARK"], required=True)
    parser.add_argument("--scheduler-config", type=Path, required=True)
    parser.add_argument("--scheduler-env-file", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--bulk-gib", type=int, default=2)
    parser.add_argument("--recover-job", help="Reuse only an explicitly identified failed fixture")
    args = parser.parse_args()
    if not 1 <= args.bulk_gib <= 8:
        parser.error("benchmark payload must be between one and eight GiB")
    try:
        asyncio.run(prepare(args))
    except Exception as exc:
        locations = []
        trace = exc.__traceback__
        while trace:
            locations.append(
                {
                    "file": Path(trace.tb_frame.f_code.co_filename).name,
                    "line": trace.tb_lineno,
                    "function": trace.tb_frame.f_code.co_name,
                }
            )
            trace = trace.tb_next
        print(
            json.dumps({"benchmark_setup_error": type(exc).__name__, "locations": locations}),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
