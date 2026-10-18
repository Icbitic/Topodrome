#!/bin/sh
set -eu

# This script is copied to persistent AutoDL storage by
# topodrome-build-autodl-bootstrap. It never contains credentials. The protected
# environment remains separate from the release bundle. Storage uses HTTP only.
umask 077

fail() {
  printf '%s\n' "Topodrome AutoDL bootstrap: $1" >&2
  exit 1
}

# Start the entire verification/expansion phase from an allowlisted environment.
# Site and model credentials are loaded from the protected file only immediately
# before the agent is exec'd.  The private marker is an argv value rather than an
# inherited variable, so a normal caller cannot accidentally bypass sanitization.
if [ "${1:-}" != "--topodrome-sanitized-bootstrap" ]; then
  case "$0" in
    /*) bootstrap_self="$0" ;;
    */*) bootstrap_self="$(pwd)/$0" ;;
    *) bootstrap_self="$(command -v "$0")" ;;
  esac
  [ -n "$bootstrap_self" ] || fail "bootstrap executable could not be resolved"
  exec /usr/bin/env -i \
    HOME="${HOME:-/root}" \
    LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TOPODROME_AUTODL_PERSISTENT_ROOT="${TOPODROME_AUTODL_PERSISTENT_ROOT:-}" \
    TOPODROME_AUTODL_EPHEMERAL_ROOT="${TOPODROME_AUTODL_EPHEMERAL_ROOT:-}" \
    TOPODROME_AUTODL_ARTIFACT_ROOT="${TOPODROME_AUTODL_ARTIFACT_ROOT:-}" \
    TOPODROME_AUTODL_MANUAL_SITE_ID="${TOPODROME_AUTODL_MANUAL_SITE_ID:-}" \
    TOPODROME_SITE_SECRET_FILE="${TOPODROME_SITE_SECRET_FILE:-}" \
    TOPODROME_AUTODL_PYTHON="${TOPODROME_AUTODL_PYTHON:-}" \
    AutoDLContainerUUID="${AutoDLContainerUUID:-}" \
    AutoDLDataCenter="${AutoDLDataCenter:-}" \
    "$bootstrap_self" --topodrome-sanitized-bootstrap "$@"
fi
shift

# AutoDL base images do not consistently install Python on the sanitized
# system PATH.  Select one trusted platform location for the small bootstrap
# validators; uv still owns the project interpreter and environment below.
bootstrap_python=""
configured_python="${TOPODROME_AUTODL_PYTHON:-}"
case "$configured_python" in
  /*) bootstrap_python_candidates="$configured_python" ;;
  *) bootstrap_python_candidates="" ;;
esac
for candidate in $bootstrap_python_candidates \
  /usr/bin/python3 /usr/local/bin/python3 \
  /root/miniconda3/bin/python3 /root/miniconda3/bin/python \
  /opt/conda/bin/python3 /opt/conda/bin/python
do
  if [ -x "$candidate" ] \
    && "$candidate" -I -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)'
  then
    bootstrap_python="$candidate"
    break
  fi
done
[ -n "$bootstrap_python" ] || fail "a trusted Python 3 bootstrap interpreter is unavailable"

persistent_root="${TOPODROME_AUTODL_PERSISTENT_ROOT:-/root/autodl-fs/topodrome}"
ephemeral_root="${TOPODROME_AUTODL_EPHEMERAL_ROOT:-/root/autodl-tmp/topodrome}"
artifact_root="${TOPODROME_AUTODL_ARTIFACT_ROOT:-$persistent_root/bootstrap}"
secret_file="${TOPODROME_SITE_SECRET_FILE:-$persistent_root/secrets/site.env}"

# AutoDL exposes provider-owned, non-secret container identity metadata.  Keep
# it through the otherwise empty bootstrap environment, normalize the optional
# ``autodl-`` prefix seen in container IDs, and map it to the generic site-agent
# settings only after strict validation.  A missing value remains supported for
# older Pro images because the scheduler can associate its single managed site,
# but a malformed provider value must never be advertised.
provider_instance_id="${AutoDLContainerUUID:-}"
if [ -n "${TOPODROME_AUTODL_MANUAL_SITE_ID:-}" ]; then
  provider_instance_id=""
fi
case "$provider_instance_id" in
  autodl-pro-*) provider_instance_id="${provider_instance_id#autodl-}" ;;
esac
if [ -n "$provider_instance_id" ]; then
  case "$provider_instance_id" in
    pro-*) ;;
    *) fail "AutoDL container UUID has an unexpected format" ;;
  esac
  case "$provider_instance_id" in
    *[!A-Za-z0-9-]*) fail "AutoDL container UUID has an unexpected format" ;;
  esac
  [ "${#provider_instance_id}" -gt 4 ] && [ "${#provider_instance_id}" -le 120 ] \
    || fail "AutoDL container UUID has an unexpected format"
fi

provider_region="${AutoDLDataCenter:-}"
if [ -n "$provider_region" ]; then
  case "$provider_region" in
    [A-Za-z0-9]*) ;;
    *) fail "AutoDL data center has an unexpected format" ;;
  esac
  case "$provider_region" in
    *[!A-Za-z0-9-]*) fail "AutoDL data center has an unexpected format" ;;
  esac
  [ "${#provider_region}" -le 120 ] \
    || fail "AutoDL data center has an unexpected format"
fi

"$bootstrap_python" - "$persistent_root" "$ephemeral_root" <<'PY' || exit 1
from pathlib import Path
import sys

persistent = Path(sys.argv[1]).resolve(strict=False)
ephemeral = Path(sys.argv[2]).resolve(strict=False)
unsafe = {Path("/"), Path("/root"), Path("/root/autodl-fs"), Path("/root/autodl-tmp")}
if not persistent.is_absolute() or not ephemeral.is_absolute():
    raise SystemExit("bootstrap roots must be absolute")
if persistent in unsafe or ephemeral in unsafe:
    raise SystemExit("bootstrap roots must be scoped subdirectories")
if persistent == ephemeral or persistent in ephemeral.parents or ephemeral in persistent.parents:
    raise SystemExit("persistent and ephemeral bootstrap roots must not overlap")
PY

# A bootstrap invocation ultimately becomes the long-lived site agent.  Keep a
# single ownership lock across that exec so a concurrent provider retry cannot
# reconcile (and therefore rewrite) launchers in the live environment.  A
# contender exits successfully: the lock holder is already responsible for
# either completing bootstrap or running the agent, and a later retry can take
# over automatically after that process exits.  Python provides flock without
# depending on a host-installed util-linux `flock` executable.
if [ "${1:-}" != "--topodrome-runtime-lock-held" ]; then
  mkdir -p "$ephemeral_root"
  "$bootstrap_python" - "$ephemeral_root/bootstrap-agent.lock" "$0" "$@" <<'PY'
import errno
import fcntl
import os
from pathlib import Path
import stat
import sys

lock_path = Path(sys.argv[1])
bootstrap = sys.argv[2]
arguments = sys.argv[3:]
flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
try:
    descriptor = os.open(lock_path, flags, 0o600)
except OSError as exc:
    raise SystemExit(f"bootstrap ownership lock is unavailable: {exc}") from None

metadata = os.fstat(descriptor)
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != os.geteuid()
    or stat.S_IMODE(metadata.st_mode) & 0o077
):
    os.close(descriptor)
    raise SystemExit("bootstrap ownership lock is not a private regular file")
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as exc:
    os.close(descriptor)
    if exc.errno in {errno.EACCES, errno.EAGAIN}:
        raise SystemExit(0) from None
    raise SystemExit(f"bootstrap ownership lock failed: {exc}") from None

# The descriptor intentionally remains open through the shell and final agent
# exec.  Its lock is released by the kernel if bootstrap or the agent exits.
os.set_inheritable(descriptor, True)
os.execv(
    bootstrap,
    [
        bootstrap,
        "--topodrome-sanitized-bootstrap",
        "--topodrome-runtime-lock-held",
        *arguments,
    ],
)
PY
  exit $?
fi
shift

check_private_file() {
  "$bootstrap_python" - "$1" <<'PY'
import os
from pathlib import Path
import stat
import sys

path = Path(sys.argv[1])
try:
    metadata = path.lstat()
    parent = path.parent.stat()
except OSError as exc:
    raise SystemExit(f"protected file is unavailable: {exc}") from None
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("protected file must be a regular non-symlink")
if metadata.st_uid != os.geteuid() or parent.st_uid != os.geteuid():
    raise SystemExit("protected file and directory must be owned by the bootstrap user")
if stat.S_IMODE(metadata.st_mode) & 0o077:
    raise SystemExit("protected file must have mode 0600 or stricter")
if stat.S_IMODE(parent.st_mode) & 0o077:
    raise SystemExit("protected directory must have mode 0700 or stricter")
PY
}

# Validate protected inputs before doing expensive work, but do not source the
# credential-bearing site environment until release verification and dependency
# installation are complete.
check_private_file "$secret_file" || fail "site secret file failed its permission check"

pointer="$artifact_root/current"
[ -f "$pointer" ] && [ ! -L "$pointer" ] || fail "release pointer is unavailable"
release_digest="$(sed -n '1p' "$pointer")"
case "$release_digest" in
  *[!0-9a-f]*|'') fail "release pointer is invalid" ;;
esac
[ "${#release_digest}" -eq 64 ] || fail "release pointer is invalid"

archive="$artifact_root/release-$release_digest.tar.gz"
checksum="$artifact_root/release-$release_digest.sha256"
[ -f "$archive" ] && [ ! -L "$archive" ] || fail "release archive is unavailable"
[ -f "$checksum" ] && [ ! -L "$checksum" ] || fail "release checksum is unavailable"
(cd "$artifact_root" && sha256sum -c "$(basename "$checksum")" >/dev/null) \
  || fail "release archive failed verification"

releases_root="$ephemeral_root/releases"
release_root="$releases_root/$release_digest"
mkdir -p "$releases_root" "$ephemeral_root/uv-cache" "$ephemeral_root/python" \
  "$ephemeral_root/huggingface-cache" "$ephemeral_root/tmp"
if [ ! -f "$release_root/.bundle-ready" ]; then
  stage="$(mktemp -d "$releases_root/.stage.XXXXXXXX")"
  trap 'rm -rf -- "$stage"' EXIT HUP INT TERM
  "$bootstrap_python" - "$archive" "$stage" <<'PY' || fail "release archive extraction failed"
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()


def is_below(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle.getmembers():
        logical = PurePosixPath(member.name)
        if logical.is_absolute() or ".." in logical.parts or not member.isfile():
            raise SystemExit("release archive contains an unsafe member")
        target = (destination / Path(*logical.parts)).resolve()
        if not is_below(target, destination):
            raise SystemExit("release archive member escaped its destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = bundle.extractfile(member)
        if source is None:
            raise SystemExit("release archive member is unreadable")
        with source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
        target.chmod(member.mode & 0o700)
PY
  printf '%s\n' "$release_digest" >"$stage/.bundle-ready"
  if "$bootstrap_python" - "$stage" "$release_root" <<'PY'
import errno
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
try:
    os.rename(source, target)
except OSError as exc:
    if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SystemExit(17) from None
    raise
PY
  then
    :
  else
    activation_status=$?
    rm -rf -- "$stage"
    [ "$activation_status" -eq 17 ] \
      && [ "$(sed -n '1p' "$release_root/.bundle-ready" 2>/dev/null)" = "$release_digest" ] \
      || fail "release activation raced an invalid tree"
  fi
  trap - EXIT HUP INT TERM
fi
[ "$(sed -n '1p' "$release_root/.bundle-ready")" = "$release_digest" ] \
  || fail "expanded release failed verification"

UV_CACHE_DIR="$ephemeral_root/uv-cache"
UV_PROJECT_ENVIRONMENT="$release_root/.venv"
UV_PYTHON_INSTALL_DIR="$ephemeral_root/python"
XDG_CACHE_HOME="$ephemeral_root/cache"
HF_HOME="$ephemeral_root/huggingface-cache"
TMPDIR="$ephemeral_root/tmp"
runtime_path="$release_root/tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
bootstrap_home="${HOME:-/root}"
requested_python="${TOPODROME_AUTODL_PYTHON:-3.11}"

run_uv_for() {
  uv_environment="$1"
  shift
  /usr/bin/env -i \
    HOME="$bootstrap_home" \
    LANG=C.UTF-8 \
    PATH="$runtime_path" \
    TMPDIR="$TMPDIR" \
    UV_CACHE_DIR="$UV_CACHE_DIR" \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT="$uv_environment" \
    UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$release_root/tools/uv" "$@"
}

run_uv_for "$UV_PROJECT_ENVIRONMENT" --version >/dev/null
cd "$release_root"

bootstrap_plan="$(
  /usr/bin/env -i HOME="$bootstrap_home" LANG=C.UTF-8 \
    PATH=/usr/local/bin:/usr/bin:/bin \
    "$bootstrap_python" - "$release_root/bundle-manifest.json" "$release_root/uv.lock" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

manifest_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit("release manifest is unavailable or invalid") from None
lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
format_version = manifest.get("format_version")
if format_version not in {1, 2} or manifest.get("uv_lock_sha256") != lock_digest:
    raise SystemExit("release manifest does not match uv.lock")
fallback = manifest.get("allow_online_fallback", False)
if not isinstance(fallback, bool):
    raise SystemExit("release online-fallback policy is invalid")
seed = manifest.get("runtime_seed")
if seed is None:
    print("none - - 0 0 0 -", lock_digest, "1" if fallback else "0", "0")
    raise SystemExit(0)
if not isinstance(seed, dict):
    raise SystemExit("runtime seed manifest is invalid")
remote = seed.get("remote")
digest = seed.get("sha256")
platform_key = seed.get("platform_key")
requires_online_repair = seed.get("requires_online_repair", False)
integers = (seed.get("size_bytes"), seed.get("member_count"), seed.get("uncompressed_bytes"))
platform_pattern = (
    r"cpython-[0-9]+\.[0-9]+-linux-(?:x86_64|aarch64)-(?:glibc|musl)"
    if format_version == 1
    else (
        r"cpython-[0-9]+\.[0-9]+-linux-(?:x86_64|aarch64)-"
        r"(?:glibc|musl)-[0-9]+\.[0-9]+"
    )
)
if (
    seed.get("format") != "site-packages-tar-gz"
    or seed.get("uv_lock_sha256") != lock_digest
    or not isinstance(remote, str)
    or not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9_-]*:)?[A-Za-z0-9][A-Za-z0-9._/-]*", remote)
    or not isinstance(digest, str)
    or not re.fullmatch(r"[0-9a-f]{64}", digest)
    or not isinstance(platform_key, str)
    or not re.fullmatch(platform_pattern, platform_key)
    or not isinstance(requires_online_repair, bool)
    or (format_version == 1 and "requires_online_repair" in seed)
    or (requires_online_repair and not fallback)
    or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in integers)
    or integers[0] == 0
    or integers[1] == 0
):
    raise SystemExit("runtime seed manifest is invalid")
print(
    "legacy" if format_version == 1 else "seed",
    remote,
    digest,
    integers[0],
    integers[1],
    integers[2],
    platform_key,
    lock_digest,
    "1" if fallback else "0",
    "1" if requires_online_repair else "0",
)
PY
)" || fail "release runtime plan failed verification"
# Every field above is constrained to a whitespace-free alphabet.
# shellcheck disable=SC2086
set -- $bootstrap_plan
[ "$#" -eq 10 ] || fail "release runtime plan is invalid"
seed_mode="$1"
seed_remote="$2"
seed_sha256="$3"
seed_size="$4"
seed_members="$5"
seed_uncompressed="$6"
seed_platform="$7"
seed_lock_sha256="$8"
allow_online_fallback="$9"
seed_requires_online_repair="${10}"

# Reject a seed addressed to another OS ABI before paying the multi-gigabyte
# Lab1 transfer cost. The managed target interpreter is checked after uv creates
# the venv; this early check deliberately uses only host architecture and libc.
if [ "$seed_mode" = "seed" ]; then
  /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
    "$bootstrap_python" - "$seed_platform" <<'PY' \
    || fail "runtime seed target platform does not match this worker"
import platform
import re
import sys

expected = re.fullmatch(
    r"cpython-[0-9]+\.[0-9]+-linux-(x86_64|aarch64)-(glibc|musl)-([0-9]+\.[0-9]+)",
    sys.argv[1],
)
if expected is None:
    raise SystemExit(1)
machine = platform.machine().strip().casefold().replace("amd64", "x86_64")
raw_libc_name, raw_libc_version = platform.libc_ver()
libc_name = raw_libc_name.strip().casefold()
if libc_name in {"gnu libc", "glibc"}:
    libc_name = "glibc"
elif "musl" in libc_name:
    libc_name = "musl"
version = re.match(r"^([0-9]+)\.([0-9]+)", raw_libc_version.strip())
actual_version = "" if version is None else f"{int(version.group(1))}.{int(version.group(2))}"
actual = (machine, libc_name, actual_version)
# A manylinux target is a minimum glibc version (PEP 600), not an
# exact distribution release. uv still validates all locked wheels below.
compatible = actual == expected.groups()
if machine == expected[1] and libc_name == expected[2] == "glibc" and version:
    compatible = tuple(map(int, actual_version.split("."))) >= tuple(map(int, expected[3].split(".")))
raise SystemExit(0 if compatible else 1)
PY
fi

sync_runtime() {
  target_environment="$1"
  shift
  run_uv_for "$target_environment" sync --frozen --no-dev --extra agent --extra runtime \
    --python "$requested_python" "$@"
}

sync_seed_dependencies() {
  target_environment="$1"
  shift
  # A site-packages-only seed has neither console launchers nor an offline
  # build-backend cache.  Reconcile every locked third-party distribution with
  # uv, while the hash-verified release itself remains the project source.
  run_uv_for "$target_environment" sync --frozen --no-dev --extra agent --extra runtime \
    --no-install-project --python "$requested_python" "$@"
}

prepare_seed_launchers() {
  target_environment="$1"
  "$target_environment/bin/python" -I - \
    "$release_root/src" "$target_environment/bin" <<'PY'
import os
from pathlib import Path
import sys
import sysconfig
import tempfile

source = Path(sys.argv[1]).resolve()
binary_directory = Path(sys.argv[2]).resolve()
purelib = Path(sysconfig.get_path("purelib")).resolve()
environment = binary_directory.parent


def is_below(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if not source.is_dir() or not is_below(purelib, environment):
    raise SystemExit("seeded runtime paths are invalid")


def atomic_write(path: Path, content: str, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


atomic_write(purelib / "topodrome-source.pth", f"{source}\n", 0o600)
launchers = {
    "topodrome-agent": (
        "#!/bin/sh\n"
        'exec "$(dirname "$0")/python" -c '
        "'from topodrome.site_agent.app import main; main()' \"$@\"\n"
    ),
    "ray": (
        "#!/bin/sh\n"
        'exec "$(dirname "$0")/python" -c '
        "'from ray.scripts.scripts import main; main()' \"$@\"\n"
    ),
}
for name, content in launchers.items():
    path = binary_directory / name
    # uv-generated console scripts embed the temporary venv's absolute path in
    # their shebang. The environment is atomically renamed below, so these two
    # Topodrome-owned launchers must always be replaced with relocation-safe
    # wrappers even when uv already made them executable.
    atomic_write(path, content, 0o700)
PY
}

write_runtime_marker() {
  target_environment="$1"
  /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
    "$bootstrap_python" - "$target_environment/.topodrome-runtime-seed.json" \
    "$seed_sha256" "$seed_lock_sha256" "$seed_platform" \
    "$seed_requires_online_repair" "$seed_mode" <<'PY'
import json
import os
from pathlib import Path
import tempfile
import sys

path = Path(sys.argv[1])
marker = {
    "seed_sha256": sys.argv[2],
    "uv_lock_sha256": sys.argv[3],
    "platform_key": sys.argv[4],
}
if sys.argv[6] != "legacy":
    marker["requires_online_repair"] = bool(int(sys.argv[5]))
content = (
    json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
PY
}

seed_archive=""
seed_stage=""
cleanup_runtime_stage() {
  if [ -n "$seed_archive" ]; then
    rm -f -- "$seed_archive"
  fi
  if [ -n "$seed_stage" ]; then
    rm -rf -- "$seed_stage"
  fi
}
trap cleanup_runtime_stage EXIT
trap 'cleanup_runtime_stage; exit 1' HUP INT TERM

fetch_runtime_seed() {
  # Region-persistent, content-addressed cache survives instance replacement.
  # The caller verifies size and SHA-256 before extraction or publication.
  seed_cache_dir="$persistent_root/runtime-seeds"
  mkdir -p "$seed_cache_dir" || return 1
  seed_cached="$seed_cache_dir/runtime-seed-$seed_sha256.tar.gz"
  if [ -f "$seed_cached" ] && [ ! -L "$seed_cached" ]; then
    cp --reflink=auto -- "$seed_cached" "$seed_archive"
  else
    /usr/bin/env -i HOME="$bootstrap_home" LANG=C.UTF-8 PATH="$runtime_path" \
      TOPODROME_AUTODL_MANUAL_SITE_ID="${TOPODROME_AUTODL_MANUAL_SITE_ID:-}" \
      "$bootstrap_python" "$release_root/src/topodrome/deployment/bootstrap_http.py" \
      --manifest "$release_root/bundle-manifest.json" --secrets "$secret_file" \
      --output "$seed_archive" --cache "$seed_cached"
  fi
}

if [ "$seed_mode" = "none" ]; then
  # Backwards-compatible path: without a configured seed, package indexes are
  # the declared source of the locked environment.
  sync_runtime "$UV_PROJECT_ENVIRONMENT" \
    || fail "online uv synchronization failed"
else
  marker="$UV_PROJECT_ENVIRONMENT/.topodrome-runtime-seed.json"
  seed_reusable=0
  if [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
    if [ -f "$marker" ] && /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
      "$bootstrap_python" - "$marker" "$seed_sha256" "$seed_lock_sha256" \
      "$seed_platform" "$seed_requires_online_repair" "$seed_mode" <<'PY'
import json
from pathlib import Path
import sys

try:
    marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1) from None
expected = {
    "seed_sha256": sys.argv[2],
    "uv_lock_sha256": sys.argv[3],
    "platform_key": sys.argv[4],
}
if sys.argv[6] != "legacy":
    expected["requires_online_repair"] = bool(int(sys.argv[5]))
raise SystemExit(0 if marker == expected else 1)
PY
    then
      if sync_seed_dependencies "$UV_PROJECT_ENVIRONMENT" --offline \
        && prepare_seed_launchers "$UV_PROJECT_ENVIRONMENT"; then
        seed_reusable=1
      elif [ "$allow_online_fallback" -eq 1 ] \
        && sync_runtime "$UV_PROJECT_ENVIRONMENT" \
        && prepare_seed_launchers "$UV_PROJECT_ENVIRONMENT"; then
        seed_reusable=1
      else
        fail "existing seeded uv environment failed locked reconciliation"
      fi
    elif [ ! -f "$marker" ] && [ "$allow_online_fallback" -eq 1 ] \
      && sync_runtime "$UV_PROJECT_ENVIRONMENT" \
      && prepare_seed_launchers "$UV_PROJECT_ENVIRONMENT"; then
      # Repair a fully synchronized environment left between online fallback
      # and marker publication.  The frozen sync is the authority; the marker
      # records the release plan, not package provenance.
      write_runtime_marker "$UV_PROJECT_ENVIRONMENT" \
        || fail "online fallback marker could not be published"
      seed_reusable=1
    else
      fail "existing runtime environment has a mismatched seed identity"
    fi
  fi

  if [ "$seed_reusable" -eq 0 ]; then
    [ "$seed_mode" != "legacy" ] \
      || fail "legacy runtime seed may only reuse an already reconciled uv environment"
    seed_archive="$(mktemp "$TMPDIR/runtime-seed.XXXXXXXX.tar.gz")"
    fetch_status=0
    fetch_runtime_seed || fetch_status=$?
    if [ "$fetch_status" -ne 0 ]; then
      [ "$fetch_status" -ne 65 ] || fail "runtime seed failed size or SHA-256 verification"
      rm -f -- "$seed_archive"
      seed_archive=""
      if [ "$allow_online_fallback" -eq 1 ]; then
        sync_runtime "$UV_PROJECT_ENVIRONMENT" \
          || fail "runtime seed fetch and online uv fallback both failed"
        prepare_seed_launchers "$UV_PROJECT_ENVIRONMENT" \
          || fail "online fallback launchers could not be prepared"
        write_runtime_marker "$UV_PROJECT_ENVIRONMENT" \
          || fail "online fallback marker could not be published"
      else
        fail "runtime seed could not be fetched from Lab1"
      fi
    else
      /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
        "$bootstrap_python" - "$seed_archive" "$seed_sha256" "$seed_size" <<'PY' \
        || fail "runtime seed failed size or SHA-256 verification"
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_digest = sys.argv[2]
expected_size = int(sys.argv[3])
if path.stat().st_size != expected_size:
    raise SystemExit(1)
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
raise SystemExit(0 if digest.hexdigest() == expected_digest else 1)
PY

      if [ ! -f "$seed_cached" ]; then
        "$bootstrap_python" - "$seed_archive" "$seed_cached" <<'PY' \
          || fail "verified runtime seed could not be persisted in regional storage"
import os
from pathlib import Path
import shutil
import sys
import tempfile

source, target = map(Path, sys.argv[1:])
fd, temporary = tempfile.mkstemp(prefix=".seed-", dir=target.parent)
try:
    with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
        shutil.copyfileobj(input_file, output, 8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
finally:
    Path(temporary).unlink(missing_ok=True)
PY
      fi

      seed_stage="$(mktemp -d "$release_root/.venv-seed.XXXXXXXX")"
      if ! run_uv_for "$seed_stage" venv --offline --no-python-downloads \
        --allow-existing --python "$requested_python" "$seed_stage"; then
        [ "$allow_online_fallback" -eq 1 ] \
          || fail "uv could not create the offline runtime-seed environment"
        run_uv_for "$seed_stage" venv --allow-existing --python "$requested_python" "$seed_stage" \
          || fail "uv could not create the runtime-seed environment"
      fi

      "$seed_stage/bin/python" -I - "$seed_platform" <<'PY' \
        || fail "runtime seed platform does not match this worker"
import platform
import re
import sys

machine = platform.machine().strip().casefold().replace("amd64", "x86_64")
raw_libc_name, raw_libc_version = platform.libc_ver()
libc_name = raw_libc_name.strip().casefold()
if libc_name in {"gnu libc", "glibc"}:
    libc_name = "glibc"
elif "musl" in libc_name:
    libc_name = "musl"
version = re.match(r"^([0-9]+)\.([0-9]+)", raw_libc_version.strip())
libc_version = "" if version is None else f"{int(version.group(1))}.{int(version.group(2))}"
actual = (
    f"cpython-{sys.version_info.major}.{sys.version_info.minor}-linux-"
    f"{machine}-{libc_name}-{libc_version}"
)
compatible = actual == sys.argv[1]
expected_prefix, _, expected_version = sys.argv[1].rpartition("-")
actual_prefix, _, actual_version = actual.rpartition("-")
if expected_prefix == actual_prefix and libc_name == "glibc" and version:
    compatible = tuple(map(int, actual_version.split("."))) >= tuple(map(int, expected_version.split(".")))
raise SystemExit(0 if compatible else 1)
PY

      purelib="$("$seed_stage/bin/python" -I - <<'PY'
import sysconfig
print(sysconfig.get_path("purelib"))
PY
)" || fail "runtime seed target environment has no purelib path"
      /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
        "$bootstrap_python" - "$seed_archive" "$seed_stage" "$purelib" \
        "$seed_members" "$seed_uncompressed" <<'PY' \
        || fail "runtime seed archive failed safe extraction"
from pathlib import Path, PurePosixPath
import os
import sys
import tarfile

archive_path = Path(sys.argv[1])
environment = Path(sys.argv[2]).resolve()
purelib = Path(sys.argv[3]).resolve()
expected_members = int(sys.argv[4])
expected_bytes = int(sys.argv[5])
def is_below(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if purelib.name != "site-packages" or not is_below(purelib, environment):
    raise SystemExit("runtime seed purelib escaped its environment")

with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    names = set()
    total = 0
    for member in members:
        logical = PurePosixPath(member.name)
        if (
            logical.is_absolute()
            or not logical.parts
            or logical.parts[0] != "site-packages"
            or ".." in logical.parts
            or logical.as_posix() in names
        ):
            raise SystemExit("runtime seed member escaped site-packages")
        names.add(logical.as_posix())
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise SystemExit("runtime seed contains an unsupported member")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute() or ".." in link.parts or not link.parts:
                raise SystemExit("runtime seed contains an unsafe link")
            target = logical.parent / link if member.issym() else link
            if not target.parts or target.parts[0] != "site-packages":
                raise SystemExit("runtime seed link escaped site-packages")
        if member.mode & 0o7000:
            raise SystemExit("runtime seed contains special mode bits")
        member.uid = os.geteuid()
        member.gid = os.getegid()
        member.uname = ""
        member.gname = ""
        member.mode &= 0o755
        if member.isfile():
            total += member.size
    if len(members) != expected_members or total != expected_bytes:
        raise SystemExit("runtime seed archive metadata does not match its release manifest")
    archive.extractall(path=purelib.parent, members=members)  # noqa: S202

for candidate in purelib.rglob("*"):
    if candidate.is_symlink():
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            raise SystemExit("runtime seed contains an invalid symlink") from None
        if not is_below(resolved, purelib):
            raise SystemExit("runtime seed symlink escaped site-packages")
PY
      rm -f -- "$purelib/_editable_impl_topodrome.pth" "$purelib/topodrome-source.pth"
      rm -f -- "$seed_archive"
      seed_archive=""

      if [ "$seed_requires_online_repair" -eq 1 ]; then
        sync_runtime "$seed_stage" \
          || fail "target-specific online uv seed repair failed"
      elif ! sync_seed_dependencies "$seed_stage" --offline; then
        if [ "$allow_online_fallback" -eq 1 ]; then
          sync_runtime "$seed_stage" \
            || fail "offline and online uv reconciliation both failed"
        else
          fail "runtime seed did not satisfy the frozen uv lock offline"
        fi
      fi
      prepare_seed_launchers "$seed_stage" \
        || fail "seeded runtime launchers could not be prepared"
      write_runtime_marker "$seed_stage" \
        || fail "runtime seed marker could not be published"
      if /usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin \
        "$bootstrap_python" - "$seed_stage" "$UV_PROJECT_ENVIRONMENT" <<'PY'
import errno
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
try:
    os.rename(source, target)
except OSError as exc:
    if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SystemExit(17) from None
    raise
PY
      then
        seed_stage=""
      else
        publication_status=$?
        [ "$publication_status" -eq 17 ] \
          || fail "seeded runtime environment could not be published"
        rm -rf -- "$seed_stage"
        seed_stage=""
        [ -f "$UV_PROJECT_ENVIRONMENT/.topodrome-runtime-seed.json" ] \
          && sync_seed_dependencies "$UV_PROJECT_ENVIRONMENT" --offline \
          && prepare_seed_launchers "$UV_PROJECT_ENVIRONMENT" \
          || fail "concurrent seeded runtime publication was invalid"
      fi
    fi
  fi
fi

[ -x "$UV_PROJECT_ENVIRONMENT/bin/topodrome-agent" ] \
  || fail "uv environment has no Topodrome agent"
[ -x "$UV_PROJECT_ENVIRONMENT/bin/ray" ] \
  || fail "uv environment has no Ray executable"

# Only the long-lived site agent receives site/model credentials. Neither uv,
# archive helpers receive them. The seed helper reads only enrollment in-process.
set -a
# shellcheck disable=SC1090 -- the path was permission-checked above.
. "$secret_file"
set +a
: "${TOPODROME_SITE_ENROLLMENT_TOKEN:?site secret file lacks enrollment identity}"
: "${TOPODROME_SITE_FENCE_VERIFICATION_KEY:?site secret file lacks fence verification key}"
if [ -z "${TOPODROME_HF_TOKEN:-${HF_TOKEN:-}}" ]; then
  fail "site secret file lacks the Hugging Face acquisition credential"
fi

if [ -n "$provider_instance_id" ]; then
  TOPODROME_SITE_PROVIDER_INSTANCE_ID="$provider_instance_id"
  export TOPODROME_SITE_PROVIDER_INSTANCE_ID
fi
if [ -n "$provider_region" ]; then
  TOPODROME_SITE_REGION="$provider_region"
  export TOPODROME_SITE_REGION
fi

# Scheduler/provider and test-adapter secrets have no purpose on a site.
unset TOPODROME_AUTODL_TOKEN TOPODROME_TEST_JUPYTER_TOKEN TOPODROME_TEST_JUPYTER_URL || true

export PATH="$release_root/tools:$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR XDG_CACHE_HOME HF_HOME TMPDIR
export UV_NO_PROGRESS=1
trap - EXIT HUP INT TERM

# The foreground site agent owns independent control/bulk SSH forwards, the HTTP
# mirror, local Ray lifecycle, and clean signal handling. No credential appears in
# its arguments; worker environments are separately allowlisted by Topodrome.
exec "$UV_PROJECT_ENVIRONMENT/bin/topodrome-agent" --config "$release_root/config/site.yaml"
