"""Dependency-free HTTP runtime-seed acquisition before uv installs Topodrome.

Run this verified bundle member as a script. Credentials are read from the
permission-checked environment file in this process, never sourced into uv or
passed through argv. Only an owned forward is started/stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import sys
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit


def protected_token(path: Path) -> str:
    info = path.lstat()
    parent = path.parent.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError("unsafe bootstrap credentials")
    for line in path.read_text().splitlines():
        if line.startswith("TOPODROME_SITE_ENROLLMENT_TOKEN="):
            values = shlex.split(line.split("=", 1)[1], comments=True)
            if len(values) == 1 and values[0]:
                return values[0]
    raise ValueError("bootstrap enrollment identity missing")


def tunnel_for(raw: dict | None):
    if raw is None:
        return None
    # Import this stdlib-only module directly: package __init__ needs httpx,
    # which is precisely what the seed is about to install.
    module_path = Path(__file__).with_name("ssh_tunnel.py")
    spec = importlib.util.spec_from_file_location("topodrome_bootstrap_tunnel", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    configured = dict(raw)
    for name in ("identity_file", "known_hosts_file", "state_directory", "ssh_executable"):
        if name in configured:
            configured[name] = Path(configured[name])
    return module.SSHLocalForwardSupervisor(module.SSHLocalForwardConfig(**configured))


def fetch(manifest_path: Path, secret_file: Path, output: Path, cache: Path):
    manifest = json.loads(manifest_path.read_text())
    seed = manifest["runtime_seed"]
    route = manifest["http_storage"]
    site = os.environ.get("TOPODROME_AUTODL_MANUAL_SITE_ID") or manifest["site_id"]
    token = protected_token(secret_file)
    if site != manifest["site_id"]:
        if not re.fullmatch(r"autodl-manual-[a-f0-9]{24}", site):
            raise ValueError("invalid bootstrap site identity")
        token = hmac.new(
            token.encode(), ("topodrome-manual:" + site).encode(), hashlib.sha256
        ).hexdigest()
    endpoint = urlsplit(route["bulk_url"])
    if (
        endpoint.scheme not in {"http", "https"}
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or (
            endpoint.scheme == "http" and endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}
        )
    ):
        raise ValueError("unsafe bootstrap storage route")
    path = seed["logical_path"]
    if not re.fullmatch(r"bootstrap/runtime-seeds/[A-Za-z0-9._/-]+", path) or any(
        part in {"", ".", ".."} for part in path.split("/")
    ):
        raise ValueError("unsafe bootstrap seed path")
    partial = cache.with_name(cache.name + ".http-part")
    if cache.is_symlink() or partial.is_symlink():
        raise ValueError("unsafe bootstrap seed cache")
    cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tunnel = tunnel_for(route.get("bulk_ssh_local_forward"))
    connection = None
    try:
        if tunnel is not None:
            tunnel.ensure_running()
        origin = (
            endpoint.path.rstrip("/").removesuffix("/api/v1/agents/control")
            + "/api/v1/agents/storage"
        )
        for attempt in range(3):
            try:
                connection_type = (
                    http.client.HTTPSConnection
                    if endpoint.scheme == "https"
                    else http.client.HTTPConnection
                )
                connection = connection_type(endpoint.hostname, endpoint.port, timeout=30)
                query = urlencode({"site_id": site, "logical_path": path})
                headers = {"Authorization": "Bearer " + token, "Accept-Encoding": "identity"}
                connection.request("GET", origin + "/stat?" + query, headers=headers)
                response = connection.getresponse()
                body = response.read(4097)
                if response.status in {408, 429, 500, 502, 503, 504}:
                    raise ConnectionError("seed metadata temporarily unavailable")
                if response.status != 200 or len(body) > 4096:
                    raise ValueError("seed metadata rejected")
                metadata = json.loads(body)
                if (
                    metadata.get("IsDir") is not False
                    or metadata.get("Size") != seed["size_bytes"]
                    or not re.fullmatch(r"[a-f0-9]{64}", metadata.get("ID", ""))
                ):
                    raise ValueError("seed metadata mismatch")
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > seed["size_bytes"]:
                    partial.unlink()
                    offset = 0
                if offset < seed["size_bytes"]:
                    connection.request(
                        "GET",
                        origin + "/file?" + query,
                        headers={
                            **headers,
                            "Range": f"bytes={offset}-",
                            "If-Match": f'"{metadata["ID"]}"',
                        },
                    )
                    response = connection.getresponse()
                    if response.status in {408, 429, 500, 502, 503, 504}:
                        response.read(4097)
                        raise ConnectionError("seed transfer temporarily unavailable")
                    if (
                        response.status != 206
                        or response.getheader("ETag") != f'"{metadata["ID"]}"'
                        or response.getheader("Content-Range")
                        != f"bytes {offset}-{seed['size_bytes'] - 1}/{seed['size_bytes']}"
                        or response.getheader("Content-Length") != str(seed["size_bytes"] - offset)
                    ):
                        raise ValueError("seed range rejected")
                    with partial.open("ab", buffering=0) as stream:
                        try:
                            while content := response.read(256 * 1024):
                                if stream.tell() + len(content) > seed["size_bytes"]:
                                    raise ValueError("seed exceeded its pinned size")
                                stream.write(content)
                        finally:
                            os.fsync(stream.fileno())
                digest = hashlib.sha256()
                with partial.open("rb") as stream:
                    while content := stream.read(4 * 1024**2):
                        digest.update(content)
                if partial.stat().st_size != seed["size_bytes"]:
                    raise ConnectionError("seed stream truncated")
                if digest.hexdigest() != seed["sha256"]:
                    partial.unlink()
                    raise ValueError("seed verification failed")
                os.replace(partial, cache)
                directory = os.open(cache.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                shutil.copyfile(cache, output)
                return
            except (OSError, http.client.HTTPException):
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
            finally:
                if connection is not None:
                    connection.close()
    finally:
        if tunnel is not None:
            tunnel.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--secrets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    try:
        fetch(args.manifest, args.secrets, args.output, args.cache)
    except ValueError:
        print(
            "HTTP seed failed size or SHA-256 verification, identity or authorization",
            file=sys.stderr,
        )
        raise SystemExit(65) from None
    except Exception:
        # Network exceptions may include request metadata. Never print them.
        print("HTTP runtime-seed acquisition failed", file=sys.stderr)
        raise SystemExit(75) from None


if __name__ == "__main__":
    main()
