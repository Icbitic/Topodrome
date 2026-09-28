from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from topodrome.scheduler.source_store import SourceAdmissionError, SourceBundle, SourceStore
from topodrome.source_layout import SOURCE_BUNDLE_METADATA_DIRECTORY
from topodrome.source_limits import HARD_MAX_SOURCE_FILES, SourceLimits

pytestmark = pytest.mark.unit


def _bundle() -> SourceBundle:
    return SourceBundle(
        files={
            "experiment.py": base64.b64encode(b"def run(context):\n    return 1\n").decode(),
            "helpers/value.txt": base64.b64encode(b"value").decode(),
        }
    )


def test_concurrent_identical_publication_is_atomic(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _index: store.put(_bundle()), range(16)))

    assert len({result.source_ref for result in results}) == 1
    published = store.resolve(results[0].source_ref)
    assert (published / SOURCE_BUNDLE_METADATA_DIRECTORY / "manifest.json").is_file()
    assert (published / "experiment.py").read_bytes().startswith(b"def run")
    assert not tuple(store.root.glob(".*.tmp-*"))


def test_existing_partial_source_is_rejected_and_preserved(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    first = store.put(_bundle())
    target = store.resolve(first.source_ref)
    (target / SOURCE_BUNDLE_METADATA_DIRECTORY / "manifest.json").unlink()
    marker = target / "do-not-delete"
    marker.write_text("preserved", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        store.put(_bundle())

    assert marker.read_text(encoding="utf-8") == "preserved"


def test_existing_source_body_is_reverified_before_reuse(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    first = store.put(_bundle())
    target = store.resolve(first.source_ref)
    (target / "experiment.py").write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed verification"):
        store.put(_bundle())

    with pytest.raises(RuntimeError, match="failed verification"):
        store.resolve(first.source_ref)


def test_user_manifest_filename_round_trips_and_reupload_is_idempotent(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)
    user_manifest = b'{"name":"user-project"}\n'
    bundle = SourceBundle(
        files={
            "experiment.py": base64.b64encode(b"def run(context): pass\n").decode(),
            "manifest.json": base64.b64encode(user_manifest).decode(),
            "nested/manifest.json": base64.b64encode(b"nested\n").decode(),
        }
    )

    first = store.put(bundle)
    second = store.put(bundle)
    published = store.resolve(first.source_ref)

    assert second == first
    assert (published / "manifest.json").read_bytes() == user_manifest
    assert (published / "nested/manifest.json").read_bytes() == b"nested\n"
    private_manifest = published / SOURCE_BUNDLE_METADATA_DIRECTORY / "manifest.json"
    assert private_manifest.is_file()
    assert json.loads(private_manifest.read_text(encoding="utf-8"))["format_version"] == 2


def test_source_reference_requires_exact_lowercase_sha256_and_existing_bundle(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)

    for invalid in (
        "",
        "sha256:short",
        f"sha256:{'A' * 64}",
        f"sha256:{'z' * 64}",
        f"sha256:{'../' + 'a' * 61}",
    ):
        with pytest.raises(ValueError, match="invalid source reference"):
            store.resolve(invalid)
    with pytest.raises(FileNotFoundError):
        store.resolve(f"sha256:{'a' * 64}")


def test_source_store_enforces_one_runtime_limit_snapshot_per_upload(tmp_path: Path) -> None:
    limits = [
        SourceLimits(max_files=2, max_path_chars=20, max_file_bytes=3, max_total_bytes=4),
        SourceLimits(max_files=2, max_path_chars=20, max_file_bytes=4, max_total_bytes=5),
    ]
    calls = 0

    def current_limits() -> SourceLimits:
        nonlocal calls
        value = limits[min(calls, len(limits) - 1)]
        calls += 1
        return value

    store = SourceStore(tmp_path, limits_provider=current_limits)
    bundle = SourceBundle(
        files={
            "one.bin": base64.b64encode(b"123").decode(),
            "two.bin": base64.b64encode(b"12").decode(),
        }
    )

    with pytest.raises(SourceAdmissionError, match="active total"):
        store.put(bundle)
    accepted = store.put(bundle)

    assert calls == 2
    assert store.resolve(accepted.source_ref).is_dir()


def test_lower_runtime_limit_does_not_invalidate_an_accepted_source(tmp_path: Path) -> None:
    active = SourceLimits(max_files=2, max_path_chars=20, max_file_bytes=5, max_total_bytes=5)
    store = SourceStore(tmp_path, limits_provider=lambda: active)
    result = store.put(
        SourceBundle(files={"source.bin": base64.b64encode(b"12345").decode()})
    )

    active = SourceLimits(max_files=2, max_path_chars=20, max_file_bytes=3, max_total_bytes=3)

    assert (store.resolve(result.source_ref) / "source.bin").read_bytes() == b"12345"


def test_source_bundle_enforces_file_count_limit() -> None:
    with pytest.raises(ValidationError, match=f"at most {HARD_MAX_SOURCE_FILES}"):
        SourceBundle(
            files={f"file-{index}": "" for index in range(HARD_MAX_SOURCE_FILES + 1)}
        )


@pytest.mark.parametrize(
    "paths, message",
    [
        ((".topodrome/manifest.json",), "reserved metadata namespace"),
        (("directory", "directory/file.py"), "overlap as a file and directory"),
        (("source.py", "./source.py"), "must be relative"),
    ],
)
def test_source_bundle_rejects_unrepresentable_or_reserved_paths(
    paths: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SourceBundle(files={path: base64.b64encode(b"body").decode() for path in paths})
