"""SSH-only AutoDL transport. Passwords never enter commands or public state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator


def parse_ssh(command: str) -> tuple[str, int, str]:
    """Accept connection coordinates, never arbitrary SSH options or a shell."""
    parts = shlex.split(command)
    if len(parts) != 4 or parts[:2] != ["ssh", "-p"]:
        raise ValueError("expected: ssh -p PORT root@HOST")
    if not parts[2].isdigit() or not 1 <= int(parts[2]) <= 65535:
        raise ValueError("invalid SSH port")
    target = re.fullmatch(r"root@([a-zA-Z0-9][a-zA-Z0-9.-]*)", parts[3])
    if target is None:
        raise ValueError("expected root@HOST, without additional commands")
    host = target[1].lower()
    if not host.endswith(".seetacloud.com"):
        raise ValueError("manual AutoDL registration requires a seetacloud.com host")
    return host, int(parts[2]), "root"


class ManualRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    ssh: str
    password: SecretStr

    @field_validator("ssh")
    @classmethod
    def validate_ssh(cls, value: str) -> str:
        parse_ssh(value)
        return value


class ManualRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force: bool = False


def private_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or path.parent.stat().st_mode & 0o077:
        raise PermissionError("manual AutoDL state directory must be private")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".manual-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def private_read(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        stat = os.fstat(stream.fileno())
        if stat.st_uid != os.geteuid() or stat.st_mode & 0o077 or stat.st_size > 65536:
            raise PermissionError("invalid manual AutoDL private state")
        return json.load(stream)


def connect(record: dict):
    # Parsing registration input is also used by the lightweight CLI. Only
    # the scheduler opens SSH connections and needs this optional dependency.
    import asyncssh

    # Trust on first use is limited to explicit registration. Every later
    # bootstrap/shutdown reconnect pins this exact server public key.
    key = record.get("host_key")
    known = ([asyncssh.import_public_key(key)], [], []) if key else None
    return asyncssh.connect(
        record["host"],
        port=record["port"],
        username="root",
        password=record["password"],
        known_hosts=known,
        client_keys=[],
        agent_path=None,
        login_timeout=20,
    )


async def inspect(request: ManualRegistration) -> dict:
    host, port, _ = parse_ssh(request.ssh)
    record = {"host": host, "port": port, "password": request.password.get_secret_value()}
    async with connect(record) as connection:
        record["host_key"] = connection.get_server_host_key().export_public_key().decode()
        result = await connection.run(
            "test -x /usr/bin/shutdown && test -d /root/autodl-fs && "
            "test -d /root/autodl-tmp && hostname && uname -m && "
            "getconf GNU_LIBC_VERSION && "
            "nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits",
            check=True,
            timeout=30,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 4 or lines[1] != "x86_64":
            raise ValueError("unsupported AutoDL runtime or no visible GPUs")
        record.update(hostname=lines[0], architecture=lines[1], libc=lines[2], gpus=lines[3:])
        record["site_id"] = (
            "autodl-manual-" + hashlib.sha256(f"{host}:{port}:{lines[0]}".encode()).hexdigest()[:24]
        )
    return record


async def install(record: dict, bundle: Path) -> None:
    digest = (bundle / "current").read_text().strip()
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid manual AutoDL release pointer")
    # Per-registration release pointer: never rewrite another instance's
    # shared bootstrap, secrets, model cache, or running environment.
    remote = "/root/autodl-fs/topodrome/manual/" + record["site_id"]
    async with connect(record) as c:
        await c.run(
            "test -f /root/autodl-fs/topodrome/secrets/site.env && "
            "test -f /root/autodl-fs/topodrome/secrets/rclone.conf && "
            f"mkdir -p {remote} && chmod 700 {remote}",
            check=True,
            timeout=30,
        )
        # Refuse to repurpose a machine whose existing bootstrap owns the
        # runtime lock. The bootstrap itself acquires that same lifetime lock.
        await c.run(
            "mkdir -p /root/autodl-tmp/topodrome && "
            "flock -n /root/autodl-tmp/topodrome/bootstrap-agent.lock true && "
            "chmod 600 /root/autodl-tmp/topodrome/bootstrap-agent.lock",
            check=True,
            timeout=20,
        )
        python = await c.run(
            "find /root/autodl-fs/topodrome/python -maxdepth 3 -type f -path '*/bin/python3.11'",
            timeout=20,
        )
        candidates = python.stdout.strip().splitlines()
        interpreter = candidates[0] if len(candidates) == 1 else "3.11"
        if interpreter != "3.11" and not re.fullmatch(
            r"/root/autodl-fs/topodrome/python/[A-Za-z0-9._-]+/bin/python3\.11", interpreter
        ):
            raise ValueError("unsafe regional Python path")
        async with c.start_sftp_client() as sftp:
            for name in (f"release-{digest}.tar.gz", f"release-{digest}.sha256", "bootstrap.sh"):
                await sftp.put(str(bundle / name), f"{remote}/{name}.tmp")
                await sftp.posix_rename(f"{remote}/{name}.tmp", f"{remote}/{name}")
            await sftp.chmod(remote + "/bootstrap.sh", 0o700)
            async with sftp.open(remote + "/current.tmp", "w") as stream:
                await stream.write(digest + "\n")
            await sftp.posix_rename(remote + "/current.tmp", remote + "/current")
        # Install ownership marker before launching. Shutdown rechecks it along
        # with the pinned host key and container hostname, not just an SSH port.
        marker = shlex.quote(record["identity"])
        await c.run(
            "mkdir -p /root/autodl-tmp/topodrome && "
            f"printf %s {marker} > /root/autodl-tmp/topodrome/manual-identity && "
            "chmod 600 /root/autodl-tmp/topodrome/manual-identity",
            check=True,
            timeout=20,
        )
        await c.run(
            f"TOPODROME_AUTODL_MANUAL_SITE_ID={record['site_id']} "
            f"TOPODROME_AUTODL_PYTHON={shlex.quote(interpreter)} "
            f"TOPODROME_AUTODL_ARTIFACT_ROOT={remote} "
            f"nohup {remote}/bootstrap.sh "
            f">{remote}/bootstrap.log 2>&1 </dev/null &",
            check=True,
            timeout=30,
        )


class ShutdownPreflightBusy(RuntimeError):
    """No shutdown submitted: CUDA processes have not exited yet."""


async def shutdown(record: dict, *, before_submit=None) -> None:
    async with connect(record) as c:
        result = await c.run(
            "hostname; cat /root/autodl-tmp/topodrome/manual-identity",
            check=True,
            timeout=20,
        )
        if result.stdout.splitlines() != [record["hostname"], record["identity"]]:
            raise RuntimeError("manual AutoDL instance identity changed; shutdown refused")
        compute = await c.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits",
            check=True,
            timeout=20,
        )
        if compute.stdout.strip():
            raise ShutdownPreflightBusy("CUDA processes remain; shutdown not submitted")
        if before_submit is not None:
            await before_submit()
        # No shell options or user-supplied commands. A lost connection is
        # ambiguous, never proof of power-off and never blindly replayed.
        await asyncio.wait_for(c.run("/usr/bin/shutdown", check=True), timeout=30)
