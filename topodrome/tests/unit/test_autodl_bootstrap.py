from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import yaml

from topodrome.deployment.autodl import build_autodl_bootstrap

pytestmark = pytest.mark.unit


def _site_config() -> dict[str, Any]:
    return {
        "site_id": "autodl",
        "display_name": "AutoDL",
        "scheduler_url": "https://scheduler.lab1.example",
        "site_kind": "autodl",
        "provider_instance_id": None,
        "global_root": "/mnt/topodrome-global",
        "scratch_root": "/root/autodl-tmp/topodrome",
        "persistent_root": "/root/autodl-fs/topodrome",
        "spool_path": "/root/autodl-fs/topodrome/agent-spool.sqlite3",
        "model_config": "/a/local/path/that/must/not/be-bundled.yaml",
        "rclone": {
            "mountpoint": "/mnt/topodrome-global",
            "remote": "lab1:topodrome",
            "cache_dir": "/root/autodl-tmp/rclone-cache",
            "binary": "/a/local/rclone",
            "vfs_cache_max_size_gb": 20,
            "min_free_space_gb": 10,
        },
    }


def _model_config() -> dict[str, Any]:
    return {
        "models": {
            "meta-llama-3-8b-instruct": {
                "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
                "backend": "transformers",
                "backend_version": "4.57.6",
                "dtype": "bfloat16",
                "sites": {
                    "autodl": {
                        "origin": "meta-llama/Meta-Llama-3-8B-Instruct",
                        "persistent_path": ("/root/autodl-fs/models/Meta-Llama-3-8B-Instruct"),
                        "local_path": "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
                        "endpoint": "https://hf-mirror.com",
                        "exclude": ["original/*"],
                    }
                },
            }
        }
    }


def _write_yaml(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    return path


def _tool(path: Path, body: str = "exit 0\n") -> Path:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}", encoding="utf-8")
    path.chmod(0o700)
    return path


def _build(
    tmp_path: Path,
    *,
    uv_body: str = "exit 0\n",
    rclone_body: str = "exit 0\n",
    runtime_seed: Path | None = None,
    runtime_seed_remote: str | None = None,
    runtime_seed_platform: str | None = None,
    allow_online_fallback: bool = False,
) -> Any:
    repository = Path(__file__).resolve().parents[2]
    site = _write_yaml(tmp_path / "site.yaml", _site_config())
    models = _write_yaml(tmp_path / "models.yaml", _model_config())
    uv = _tool(tmp_path / "uv", uv_body)
    rclone = _tool(tmp_path / "rclone", rclone_body)
    return build_autodl_bootstrap(
        repository_root=repository,
        site_config=site,
        model_config=models,
        output_dir=tmp_path / "persistent" / "bootstrap",
        uv_binary=uv,
        rclone_binary=rclone,
        runtime_seed=runtime_seed,
        runtime_seed_remote=runtime_seed_remote,
        runtime_seed_platform=runtime_seed_platform,
        allow_online_fallback=allow_online_fallback,
    )


def _runtime_seed(path: Path, *, member_name: str = "site-packages/seeded.py") -> Path:
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo("site-packages")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        payload = b"SEEDED = True\n"
        member = tarfile.TarInfo(member_name)
        member.mode = 0o644
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))
        metadata = b"Metadata-Version: 2.1\nName: seeded\nVersion: 1.0\n"
        metadata_member = tarfile.TarInfo("site-packages/seeded-1.0.dist-info/METADATA")
        metadata_member.mode = 0o644
        metadata_member.size = len(metadata)
        archive.addfile(metadata_member, BytesIO(metadata))
        wheel = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        wheel_member = tarfile.TarInfo("site-packages/seeded-1.0.dist-info/WHEEL")
        wheel_member.mode = 0o644
        wheel_member.size = len(wheel)
        archive.addfile(wheel_member, BytesIO(wheel))
    return path


def _bootstrap_environment(
    tmp_path: Path,
    artifacts: Any,
) -> tuple[dict[str, str], Path, Path, Path]:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    secrets = persistent / "secrets"
    secrets.mkdir(mode=0o700, exist_ok=True)
    rclone_config = secrets / "rclone.conf"
    rclone_config.write_text("[lab1]\ntype = sftp\nhost = lab1.internal\n", encoding="utf-8")
    rclone_config.chmod(0o600)
    result_path = tmp_path / "agent-result.txt"
    secret_file = secrets / "site.env"
    secret_file.write_text(
        "\n".join(
            (
                "TOPODROME_SITE_ENROLLMENT_TOKEN='enrollment-test-value'",
                f"TOPODROME_SITE_FENCE_VERIFICATION_KEY='{'f' * 32}'",
                "TOPODROME_HF_TOKEN='hf_test_value_not_a_real_token'",
                f"RCLONE_CONFIG='{rclone_config}'",
                f"TOPODROME_BOOTSTRAP_TEST_RESULT='{result_path}'",
                "",
            )
        ),
        encoding="utf-8",
    )
    secret_file.chmod(0o600)
    environment = {
        **os.environ,
        "TOPODROME_AUTODL_PERSISTENT_ROOT": str(persistent),
        "TOPODROME_AUTODL_EPHEMERAL_ROOT": str(ephemeral),
        "TOPODROME_AUTODL_ARTIFACT_ROOT": str(artifacts.release_archive.parent),
        "TOPODROME_SITE_SECRET_FILE": str(secret_file),
        # Prove that even credential-shaped ambient variables are stripped
        # before any bundled helper runs.  The protected site.env above is the
        # only source of the values that the final agent receives.
        "TOPODROME_SITE_ENROLLMENT_TOKEN": "ambient-value-must-not-leak",
        "TOPODROME_HF_TOKEN": "ambient-value-must-not-leak",
        # AutoDL-owned non-secret metadata is the one ambient input that the
        # sanitized bootstrap deliberately maps into generic site settings.
        "AutoDLContainerUUID": "autodl-pro-bootstrap-test",
        "AutoDLDataCenter": "beijingDC2",
    }
    return environment, persistent, ephemeral, result_path


def _replace_release_member(artifacts: Any, name: str, content: bytes) -> str:
    temporary = artifacts.release_archive.with_name("rewritten-release.tar.gz")
    with tarfile.open(artifacts.release_archive, "r:gz") as source:
        with tarfile.open(temporary, "w:gz") as destination:
            for source_member in source.getmembers():
                stream = source.extractfile(source_member)
                assert stream is not None
                payload = content if source_member.name == name else stream.read()
                member = tarfile.TarInfo(source_member.name)
                member.mode = source_member.mode
                member.size = len(payload)
                destination.addfile(member, BytesIO(payload))

    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    archive = artifacts.release_archive.parent / f"release-{digest}.tar.gz"
    temporary.replace(archive)
    (artifacts.release_archive.parent / f"release-{digest}.sha256").write_text(
        f"{digest}  {archive.name}\n",
        encoding="utf-8",
    )
    artifacts.current_pointer.write_text(f"{digest}\n", encoding="utf-8")
    return digest


def _seeded_uv_body(
    tmp_path: Path,
    *,
    agent_wait_file: Path | None = None,
    seed_check_status: int = 0,
) -> tuple[str, Path]:
    log = tmp_path / "uv-commands.log"
    agent_wait = ""
    if agent_wait_file is not None:
        agent_wait = f"""
    wait_path = Path({str(agent_wait_file)!r})
    while not wait_path.is_file():
        time.sleep(0.01)
"""
    body = f"""
[ -z "${{TOPODROME_SITE_ENROLLMENT_TOKEN:-}}" ]
[ -z "${{TOPODROME_SITE_FENCE_VERIFICATION_KEY:-}}" ]
[ -z "${{TOPODROME_HF_TOKEN:-}}" ]
[ -z "${{RCLONE_CONFIG:-}}" ]
case "${{1:-}}" in
  --version)
    exit 0
    ;;
  venv)
    for target do :; done
    {shlex.quote(sys.executable)} -m venv --without-pip "$target"
    ;;
  sync)
    case " $* " in
      *" --check "*) exit {seed_check_status} ;;
    esac
    printf '%s\n' "$*" >>{shlex.quote(str(log))}
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      {shlex.quote(sys.executable)} -m venv --without-pip "$UV_PROJECT_ENVIRONMENT"
    fi
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
    purelib="$(
      "$UV_PROJECT_ENVIRONMENT/bin/python" \
        -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
    )"
    mkdir -p "$purelib/topodrome/site_agent"
    : >"$purelib/topodrome/__init__.py"
    : >"$purelib/topodrome/site_agent/__init__.py"
    cat >"$purelib/topodrome/site_agent/app.py" <<'EOF'
import os
from pathlib import Path
import time


def main():
    Path(os.environ["TOPODROME_BOOTSTRAP_TEST_RESULT"]).write_text("seeded=1\\n")
{agent_wait}\
EOF
    cat >"$UV_PROJECT_ENVIRONMENT/bin/ray" <<EOF
#!$UV_PROJECT_ENVIRONMENT/bin/python
#!/bin/sh
exit 0
EOF
    chmod 700 "$UV_PROJECT_ENVIRONMENT/bin/ray"
    cat >"$UV_PROJECT_ENVIRONMENT/bin/topodrome-agent" <<EOF
#!$UV_PROJECT_ENVIRONMENT/bin/python
raise SystemExit("stale uv launcher was not replaced")
EOF
    chmod 700 "$UV_PROJECT_ENVIRONMENT/bin/topodrome-agent"
    ;;
  *)
    exit 90
    ;;
esac
"""
    return body, log


def _seeded_rclone_body(
    tmp_path: Path,
    *,
    copied_seed: Path,
) -> tuple[str, Path]:
    log = tmp_path / "rclone-commands.log"
    config = tmp_path / "persistent" / "secrets" / "rclone.conf"
    body = f"""
[ -z "${{TOPODROME_SITE_ENROLLMENT_TOKEN:-}}" ]
[ -z "${{TOPODROME_SITE_FENCE_VERIFICATION_KEY:-}}" ]
[ -z "${{TOPODROME_HF_TOKEN:-}}" ]
[ "$RCLONE_CONFIG" = {shlex.quote(str(config))} ]
case "${{1:-}}" in
  version)
    exit 0
    ;;
  copyto)
    printf '%s\n' "$*" >>{shlex.quote(str(log))}
    /bin/cp -- {shlex.quote(str(copied_seed))} "$3"
    ;;
  *)
    exit 91
    ;;
esac
"""
    return body, log


def test_builder_creates_immutable_secret_free_release(tmp_path: Path) -> None:
    artifacts = _build(tmp_path)

    assert artifacts.current_pointer.read_text().strip() == artifacts.digest
    assert artifacts.checksum_file.read_text().strip() == (
        f"{artifacts.digest}  {artifacts.release_archive.name}"
    )
    assert artifacts.bootstrap_script.stat().st_mode & 0o777 == 0o700
    with tarfile.open(artifacts.release_archive, "r:gz") as archive:
        names = set(archive.getnames())
        assert {
            "README.md",
            "pyproject.toml",
            "uv.lock",
            "config/site.yaml",
            "config/models.yaml",
            "tools/uv",
            "tools/rclone",
            "bundle-manifest.json",
        } <= names
        assert not any(".venv" in name or "__pycache__" in name for name in names)
        site = yaml.safe_load(archive.extractfile("config/site.yaml").read())
        assert site["model_config"] == "config/models.yaml"
        assert site["rclone"]["binary"] == "rclone"
        assert site["rclone"]["verification_via_scheduler"] is True
        manifest = json.loads(archive.extractfile("bundle-manifest.json").read())
        assert manifest["format_version"] == 2
        assert manifest["site_id"] == "autodl"


def test_builder_allows_regional_only_model_loading(tmp_path: Path) -> None:
    models = _model_config()
    del models["models"]["meta-llama-3-8b-instruct"]["sites"]["autodl"]["local_path"]
    artifacts = build_autodl_bootstrap(
        repository_root=Path(__file__).resolve().parents[2],
        site_config=_write_yaml(tmp_path / "site.yaml", _site_config()),
        model_config=_write_yaml(tmp_path / "models.yaml", models),
        output_dir=tmp_path / "output",
        uv_binary=_tool(tmp_path / "uv"),
        rclone_binary=_tool(tmp_path / "rclone"),
    )
    assert artifacts.release_archive.is_file()


def test_builder_rejects_credentials_and_non_ephemeral_runtime_paths(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    site_value = _site_config()
    site_value["enrollment_token"] = "must-not-enter-a-release"  # noqa: S105
    site = _write_yaml(tmp_path / "site.yaml", site_value)
    models = _write_yaml(tmp_path / "models.yaml", _model_config())
    uv = _tool(tmp_path / "uv")
    rclone = _tool(tmp_path / "rclone")

    with pytest.raises(ValueError, match="embeds a credential"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    site_value.pop("enrollment_token")
    site_value["scratch_root"] = "/root/autodl-fs/topodrome/venv"
    _write_yaml(site, site_value)
    with pytest.raises(ValueError, match="scratch_root must be below /root/autodl-tmp"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )


def test_builder_requires_known_rclone_mode_and_ephemeral_mirror(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    site_value = _site_config()
    site_value["rclone"]["mode"] = "unknown"
    site = _write_yaml(tmp_path / "site.yaml", site_value)
    models = _write_yaml(tmp_path / "models.yaml", _model_config())
    uv = _tool(tmp_path / "uv")
    rclone = _tool(tmp_path / "rclone")

    with pytest.raises(ValueError, match="rclone mode must be mount or mirror"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    site_value["rclone"]["mode"] = "mirror"
    _write_yaml(site, site_value)
    with pytest.raises(ValueError, match="mirror global_root must be below"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )


def test_beijing_b_autodl_example_builds_as_mirrored_tunneled_release(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    uv = _tool(tmp_path / "uv")
    rclone = _tool(tmp_path / "rclone")

    artifacts = build_autodl_bootstrap(
        repository_root=repository,
        site_config=repository / "configs/site.autodl.example.yaml",
        model_config=repository / "configs/models.example.yaml",
        output_dir=tmp_path / "output",
        uv_binary=uv,
        rclone_binary=rclone,
    )

    with tarfile.open(artifacts.release_archive, "r:gz") as archive:
        stream = archive.extractfile("config/site.yaml")
        assert stream is not None
        site = yaml.safe_load(stream.read())
    assert site["scheduler_url"] == "http://127.0.0.1:18042"
    assert site["rclone"]["mode"] == "mirror"
    assert site["rclone"]["remote"] == "lab1-global:"
    assert site["global_root"].startswith("/root/autodl-tmp/")
    assert site["ssh_local_forward"]["local_port"] == 18042
    assert site["ssh_local_forward"]["target_port"] == 8042
    assert site["ssh_local_forward"]["remote_port"] == 2201


def test_builder_requires_matching_ssh_route_for_loopback_scheduler(
    tmp_path: Path,
) -> None:
    site_value = _site_config()
    site_value["scheduler_url"] = "http://127.0.0.1:18042"
    site = _write_yaml(tmp_path / "site.yaml", site_value)
    models = _write_yaml(tmp_path / "models.yaml", _model_config())
    uv = _tool(tmp_path / "uv")
    rclone = _tool(tmp_path / "rclone")
    repository = Path(__file__).resolve().parents[2]

    with pytest.raises(ValueError, match="requires ssh_local_forward"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    site_value["ssh_local_forward"] = {
        "identity_file": "/root/autodl-fs/topodrome/secrets/control_ed25519",
        "known_hosts_file": "/root/autodl-fs/topodrome/secrets/lab1_known_hosts",
        "remote_host": "111.127.53.162",
        "remote_user": "xiangkun",
        "remote_port": 2201,
        "local_host": "127.0.0.1",
        "local_port": 18043,
        "target_host": "127.0.0.1",
        "target_port": 8042,
        "state_directory": "/root/autodl-tmp/topodrome/control-tunnel",
        "ssh_executable": "/usr/bin/ssh",
    }
    _write_yaml(site, site_value)
    with pytest.raises(ValueError, match="must match scheduler_url exactly"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    site_value["ssh_local_forward"]["local_port"] = 18042
    _write_yaml(site, site_value)
    artifacts = build_autodl_bootstrap(
        repository_root=repository,
        site_config=site,
        model_config=models,
        output_dir=tmp_path / "output",
        uv_binary=uv,
        rclone_binary=rclone,
    )
    assert artifacts.release_archive.is_file()


def test_bootstrap_rejects_malformed_provider_metadata_before_uv(
    tmp_path: Path,
) -> None:
    uv_log = tmp_path / "uv-called"
    artifacts = _build(
        tmp_path,
        uv_body=f"touch {shlex.quote(str(uv_log))}\n",
    )
    environment, _persistent, _ephemeral, _result = _bootstrap_environment(
        tmp_path,
        artifacts,
    )
    environment["AutoDLContainerUUID"] = "pro-invalid;payload"

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "container UUID has an unexpected format" in completed.stderr
    assert not uv_log.exists()


def test_bootstrap_builds_uv_environment_only_on_ephemeral_disk(tmp_path: Path) -> None:
    uv_body = r"""
[ -z "${TOPODROME_SITE_ENROLLMENT_TOKEN:-}" ]
[ -z "${TOPODROME_SITE_FENCE_VERIFICATION_KEY:-}" ]
[ -z "${TOPODROME_HF_TOKEN:-}" ]
if [ "${1:-}" = "--version" ]; then
  exit 0
fi
[ "${1:-}" = "sync" ]
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
cat >"$UV_PROJECT_ENVIRONMENT/bin/ray" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod 700 "$UV_PROJECT_ENVIRONMENT/bin/ray"
cat >"$UV_PROJECT_ENVIRONMENT/bin/topodrome-agent" <<'EOF'
#!/bin/sh
set -eu
{
  printf 'cwd=%s\n' "$PWD"
  printf 'venv=%s\n' "$UV_PROJECT_ENVIRONMENT"
  printf 'cache=%s\n' "$UV_CACHE_DIR"
  printf 'rclone=%s\n' "$RCLONE_CONFIG"
  printf 'ray=%s\n' "$(command -v ray)"
  printf 'args=%s\n' "$*"
  printf 'provider=%s\n' "$TOPODROME_SITE_PROVIDER_INSTANCE_ID"
  printf 'region=%s\n' "$TOPODROME_SITE_REGION"
} >"$TOPODROME_BOOTSTRAP_TEST_RESULT"
EOF
chmod 700 "$UV_PROJECT_ENVIRONMENT/bin/topodrome-agent"
"""
    artifacts = _build(tmp_path, uv_body=uv_body)
    environment, persistent, ephemeral, result_path = _bootstrap_environment(tmp_path, artifacts)
    rclone_config = persistent / "secrets" / "rclone.conf"
    interpreter_marker = tmp_path / "configured-bootstrap-python-used"
    bootstrap_python = tmp_path / "platform-python"
    bootstrap_python.write_text(
        "#!/bin/sh\n"
        f"touch {shlex.quote(str(interpreter_marker))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    bootstrap_python.chmod(0o700)
    environment["TOPODROME_AUTODL_PYTHON"] = str(bootstrap_python)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert interpreter_marker.is_file()
    values = dict(line.split("=", 1) for line in result_path.read_text().splitlines())
    assert Path(values["cwd"]).is_relative_to(ephemeral)
    assert Path(values["venv"]).is_relative_to(ephemeral)
    assert Path(values["cache"]).is_relative_to(ephemeral)
    assert values["rclone"] == str(rclone_config)
    assert Path(values["ray"]) == Path(values["venv"]) / "bin" / "ray"
    assert values["args"].startswith("--config ")
    assert values["provider"] == "pro-bootstrap-test"
    assert values["region"] == "beijingDC2"
    assert "enrollment-test-value" not in values["args"]
    assert not any(path.name in {".venv", "uv-cache"} for path in persistent.rglob("*"))


def test_seed_manifest_binds_lock_platform_and_content(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    artifacts = _build(tmp_path, runtime_seed=seed)

    with tarfile.open(artifacts.release_archive, "r:gz") as archive:
        manifest = json.loads(archive.extractfile("bundle-manifest.json").read())
        lock = archive.extractfile("uv.lock").read()

    configured = manifest["runtime_seed"]
    assert configured["format"] == "site-packages-tar-gz"
    assert configured["sha256"] == hashlib.sha256(seed.read_bytes()).hexdigest()
    assert configured["uv_lock_sha256"] == hashlib.sha256(lock).hexdigest()
    assert configured["platform_key"].startswith("cpython-3.11-linux-x86_64-glibc-")
    assert configured["platform_key"].rsplit("-", 1)[1].count(".") == 1
    assert configured["requires_online_repair"] is False
    assert configured["sha256"] in configured["remote"]
    assert configured["uv_lock_sha256"] in configured["remote"]
    assert configured["platform_key"] in configured["remote"]
    assert artifacts.runtime_seed_remote == configured["remote"]
    assert manifest["allow_online_fallback"] is False


def test_incompatible_target_seed_requires_explicit_online_repair(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, _uv_log = _seeded_uv_body(tmp_path, seed_check_status=42)

    with pytest.raises(ValueError, match="not synchronized with the frozen lock"):
        _build(
            tmp_path,
            uv_body=uv_body,
            runtime_seed=seed,
            runtime_seed_platform="x86_64-manylinux_2_31",
        )

    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        runtime_seed=seed,
        runtime_seed_platform="x86_64-manylinux_2_31",
        allow_online_fallback=True,
    )
    with tarfile.open(artifacts.release_archive, "r:gz") as archive:
        manifest = json.loads(archive.extractfile("bundle-manifest.json").read())

    configured = manifest["runtime_seed"]
    assert configured["platform_key"] == "cpython-3.11-linux-x86_64-glibc-2.31"
    assert configured["requires_online_repair"] is True
    assert configured["platform_key"] in configured["remote"]


def test_preflighted_seed_repair_uses_uv_online_only_with_opt_in(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, uv_log = _seeded_uv_body(tmp_path, seed_check_status=42)
    rclone_body, _rclone_log = _seeded_rclone_body(tmp_path, copied_seed=seed)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
        allow_online_fallback=True,
    )
    environment, _persistent, ephemeral, result = _bootstrap_environment(tmp_path, artifacts)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert result.read_text() == "seeded=1\n"
    sync_commands = uv_log.read_text().splitlines()
    assert len(sync_commands) == 1
    assert "--offline" not in sync_commands[0]
    release = ephemeral / "releases" / artifacts.digest
    for launcher_name in ("topodrome-agent", "ray"):
        launcher = (release / ".venv" / "bin" / launcher_name).read_text()
        assert launcher.startswith("#!/bin/sh\n")
        assert '$(dirname "$0")/python' in launcher
        assert ".venv-seed." not in launcher


def test_bootstrap_rejects_wrong_seed_abi_before_downloading_seed(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, _uv_log = _seeded_uv_body(tmp_path, seed_check_status=42)
    rclone_body, rclone_log = _seeded_rclone_body(tmp_path, copied_seed=seed)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
        runtime_seed_platform="x86_64-manylinux_2_99",
        allow_online_fallback=True,
    )
    environment, _persistent, _ephemeral, _result = _bootstrap_environment(tmp_path, artifacts)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "target platform does not match this worker" in completed.stderr
    assert not rclone_log.exists()


def test_legacy_seed_release_can_only_reuse_an_existing_uv_environment(
    tmp_path: Path,
) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, uv_log = _seeded_uv_body(tmp_path)
    rclone_body, rclone_log = _seeded_rclone_body(tmp_path, copied_seed=seed)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
    )
    with tarfile.open(artifacts.release_archive, "r:gz") as archive:
        manifest = json.loads(archive.extractfile("bundle-manifest.json").read())
    manifest["format_version"] = 1
    seed_manifest = manifest["runtime_seed"]
    seed_manifest["platform_key"] = seed_manifest["platform_key"].rsplit("-", 1)[0]
    seed_manifest.pop("requires_online_repair")
    legacy_digest = _replace_release_member(
        artifacts,
        "bundle-manifest.json",
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    environment, _persistent, ephemeral, result = _bootstrap_environment(tmp_path, artifacts)

    first = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert first.returncode != 0
    assert "legacy runtime seed may only reuse" in first.stderr
    assert not rclone_log.exists()

    release = ephemeral / "releases" / legacy_digest
    runtime = release / ".venv"
    subprocess.run(  # noqa: S603 - test-only interpreter creates the fake uv environment
        [sys.executable, "-m", "venv", "--without-pip", str(runtime)],
        check=True,
    )
    (runtime / ".topodrome-runtime-seed.json").write_text(
        json.dumps(
            {
                "seed_sha256": seed_manifest["sha256"],
                "uv_lock_sha256": manifest["uv_lock_sha256"],
                "platform_key": seed_manifest["platform_key"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    second = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 0, second.stderr
    assert result.read_text() == "seeded=1\n"
    assert "--offline" in uv_log.read_text()
    assert not rclone_log.exists()


def test_seeded_bootstrap_fetches_from_lab1_and_reconciles_offline(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, uv_log = _seeded_uv_body(tmp_path)
    rclone_body, rclone_log = _seeded_rclone_body(tmp_path, copied_seed=seed)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
    )
    environment, _persistent, ephemeral, result = _bootstrap_environment(tmp_path, artifacts)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert result.read_text() == "seeded=1\n"
    sync_commands = uv_log.read_text().splitlines()
    assert len(sync_commands) == 1
    assert sync_commands[0].startswith("sync --frozen --no-dev --extra agent --extra runtime")
    assert "--offline" in sync_commands[0]
    copy_command = rclone_log.read_text().strip()
    assert copy_command.startswith(f"copyto {artifacts.runtime_seed_remote} ")
    release = ephemeral / "releases" / artifacts.digest
    assert tuple(release.glob(".venv/lib/python*/site-packages/seeded.py"))
    assert not tuple(ephemeral.glob("tmp/runtime-seed.*.tar.gz"))

    cached = (
        _persistent
        / "runtime-seeds"
        / f"runtime-seed-{hashlib.sha256(seed.read_bytes()).hexdigest()}.tar.gz"
    )
    assert cached.read_bytes() == seed.read_bytes()
    # Simulate a newly allocated instance with empty local disk, same region.
    environment["TOPODROME_AUTODL_EPHEMERAL_ROOT"] = str(tmp_path / "replacement")
    before = rclone_log.read_text()
    replacement = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert replacement.returncode == 0, replacement.stderr
    assert rclone_log.read_text() == before


def test_seed_hash_mismatch_never_falls_back_to_network(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    corrupted = tmp_path / "corrupted-runtime-site-packages.tar.gz"
    body = bytearray(seed.read_bytes())
    body[-1] ^= 1
    corrupted.write_bytes(body)
    uv_body, uv_log = _seeded_uv_body(tmp_path)
    rclone_body, _ = _seeded_rclone_body(tmp_path, copied_seed=corrupted)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
        allow_online_fallback=True,
    )
    environment, _persistent, _ephemeral, _result = _bootstrap_environment(tmp_path, artifacts)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "failed size or SHA-256 verification" in completed.stderr
    assert not uv_log.exists()


def test_release_lock_mismatch_fails_before_seed_fetch_or_sync(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, uv_log = _seeded_uv_body(tmp_path)
    rclone_body, rclone_log = _seeded_rclone_body(tmp_path, copied_seed=seed)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
        allow_online_fallback=True,
    )
    _replace_release_member(artifacts, "uv.lock", b"tampered-lock\n")
    environment, _persistent, _ephemeral, _result = _bootstrap_environment(tmp_path, artifacts)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "release manifest does not match uv.lock" in completed.stderr
    assert not uv_log.exists()
    assert not rclone_log.exists()


def test_seed_fetch_uses_network_only_with_explicit_fallback(tmp_path: Path) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    uv_body, uv_log = _seeded_uv_body(tmp_path)
    rclone_body = r"""
case "${1:-}" in
  version) exit 0 ;;
  copyto) exit 42 ;;
  *) exit 91 ;;
esac
"""
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
        allow_online_fallback=True,
    )
    environment, _persistent, _ephemeral, result = _bootstrap_environment(tmp_path, artifacts)

    completed = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert result.read_text() == "seeded=1\n"
    sync_commands = uv_log.read_text().splitlines()
    assert len(sync_commands) == 1
    assert "--offline" not in sync_commands[0]

    repeated = subprocess.run(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert repeated.returncode == 0, repeated.stderr
    repeated_syncs = uv_log.read_text().splitlines()
    assert len(repeated_syncs) == 2
    assert "--offline" in repeated_syncs[1]


def test_concurrent_bootstraps_publish_one_valid_release_and_runtime(
    tmp_path: Path,
) -> None:
    seed = _runtime_seed(tmp_path / "runtime-site-packages.tar.gz")
    agent_exit_gate = tmp_path / "allow-agent-exit"
    uv_body, uv_log = _seeded_uv_body(tmp_path, agent_wait_file=agent_exit_gate)
    rclone_body, rclone_log = _seeded_rclone_body(tmp_path, copied_seed=seed)
    artifacts = _build(
        tmp_path,
        uv_body=uv_body,
        rclone_body=rclone_body,
        runtime_seed=seed,
    )
    environment, _persistent, ephemeral, result = _bootstrap_environment(tmp_path, artifacts)

    owner = subprocess.Popen(  # noqa: S603 - generated repository-owned executable
        [str(artifacts.bootstrap_script)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contenders: list[subprocess.Popen[str]] = []
    owner_stderr = ""
    try:
        deadline = time.monotonic() + 10
        while not result.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert result.exists()

        contenders = [
            subprocess.Popen(  # noqa: S603 - generated repository-owned executable
                [str(artifacts.bootstrap_script)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(3)
        ]
        completions = [process.communicate(timeout=10) for process in contenders]
        for process, (_stdout, stderr) in zip(contenders, completions, strict=True):
            assert process.returncode == 0, stderr
        assert owner.poll() is None
    finally:
        agent_exit_gate.touch()
        for contender in contenders:
            if contender.poll() is None:
                contender.terminate()
                contender.communicate(timeout=5)
        _stdout, owner_stderr = owner.communicate(timeout=10)
    assert owner.returncode == 0, owner_stderr

    assert result.read_text() == "seeded=1\n"
    assert len(uv_log.read_text().splitlines()) == 1
    assert len(rclone_log.read_text().splitlines()) == 1
    releases = ephemeral / "releases"
    release = releases / artifacts.digest
    assert (release / ".bundle-ready").read_text().strip() == artifacts.digest
    assert (release / ".venv/.topodrome-runtime-seed.json").is_file()
    assert not tuple(releases.glob(".stage.*"))
    assert not tuple(release.glob(".venv-seed.*"))
    assert not (release / artifacts.digest).exists()


def test_builder_rejects_unsafe_seed_and_credential_bearing_configuration(
    tmp_path: Path,
) -> None:
    unsafe = _runtime_seed(
        tmp_path / "unsafe-runtime-site-packages.tar.gz",
        member_name="../escape.py",
    )
    with pytest.raises(ValueError, match="only a relative site-packages"):
        _build(tmp_path, runtime_seed=unsafe)

    valid = _runtime_seed(tmp_path / "valid-runtime-site-packages.tar.gz")
    linked = tmp_path / "linked-runtime-site-packages.tar.gz"
    linked.symlink_to(valid)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _build(tmp_path, runtime_seed=linked)

    site = _site_config()
    site["fence_verification_key"] = None
    _write_yaml(tmp_path / "site.yaml", site)
    models = _write_yaml(tmp_path / "models.yaml", _model_config())
    uv = _tool(tmp_path / "uv")
    rclone = _tool(tmp_path / "rclone")
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="embeds a credential"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=tmp_path / "site.yaml",
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    site.pop("fence_verification_key")
    site["rclone"]["remote"] = ":sftp,host=lab1,user=root,pass=hidden:topodrome"
    _write_yaml(tmp_path / "site.yaml", site)
    with pytest.raises(ValueError, match="credential-free named rclone reference"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=tmp_path / "site.yaml",
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )


def test_builder_accepts_remote_root_and_places_seed_below_it(tmp_path: Path) -> None:
    site_value = _site_config()
    site_value["rclone"]["remote"] = "lab1-global:"
    site = _write_yaml(tmp_path / "site-root.yaml", site_value)
    models = _write_yaml(tmp_path / "models-root.yaml", _model_config())
    uv = _tool(tmp_path / "uv-root")
    rclone = _tool(tmp_path / "rclone-root")
    seed = _runtime_seed(tmp_path / "runtime-root.tar.gz")

    artifacts = build_autodl_bootstrap(
        repository_root=Path(__file__).resolve().parents[2],
        site_config=site,
        model_config=models,
        output_dir=tmp_path / "output-root",
        uv_binary=uv,
        rclone_binary=rclone,
        runtime_seed=seed,
    )

    assert artifacts.runtime_seed_remote is not None
    assert artifacts.runtime_seed_remote.startswith("lab1-global:bootstrap/runtime-seeds/")


def test_builder_rejects_credential_bearing_model_origin_and_repository_text(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    site = _write_yaml(tmp_path / "site.yaml", _site_config())
    model_value = _model_config()
    model_value["models"]["meta-llama-3-8b-instruct"]["sites"]["autodl"]["origin"] = (
        "https://user:password@hf-mirror.com/model"
    )
    models = _write_yaml(tmp_path / "models.yaml", model_value)
    uv = _tool(tmp_path / "uv")
    rclone = _tool(tmp_path / "rclone")
    with pytest.raises(ValueError, match="credential-free Hugging Face repository"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    model_value = _model_config()
    model_value["models"]["meta-llama-3-8b-instruct"]["sites"]["autodl"]["endpoint"] = (
        "https://hf-mirror.com/unapproved-proxy"
    )
    _write_yaml(models, model_value)
    with pytest.raises(ValueError, match=r"acquire through https://hf-mirror\.com"):
        build_autodl_bootstrap(
            repository_root=repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )

    copied_repository = tmp_path / "repository"
    copied_repository.mkdir()
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(repository / name, copied_repository / name)
    shutil.copytree(repository / "src", copied_repository / "src")
    (copied_repository / "src" / "topodrome" / "credential_leak.py").write_text(
        "VALUE = 'hf_" + "a" * 32 + "'\n",
        encoding="utf-8",
    )
    model_value = _model_config()
    _write_yaml(models, model_value)
    with pytest.raises(ValueError, match="credential-like material"):
        build_autodl_bootstrap(
            repository_root=copied_repository,
            site_config=site,
            model_config=models,
            output_dir=tmp_path / "output",
            uv_binary=uv,
            rclone_binary=rclone,
        )


def test_checked_in_bootstrap_is_valid_posix_shell() -> None:
    script = Path(__file__).resolve().parents[2] / "src/topodrome/deployment/autodl_bootstrap.sh"
    subprocess.run(["sh", "-n", str(script)], check=True)  # noqa: S603,S607
