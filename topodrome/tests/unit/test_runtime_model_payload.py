from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from topodrome.runtime.model_payload import (
    ModelPayloadVerificationError,
    record_huggingface_local_dir_manifest,
    verify_transformers_payload,
)

pytestmark = pytest.mark.unit

REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"


def _git_blob_id(content: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _payload(root: Path, *, shard_content: bytes = b"weights") -> None:
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}
        ).encode(),
        "model-00001-of-00001.safetensors": shard_content,
    }
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    tree = {
        "format_version": 1,
        "files": {
            name: {
                "size": len(content),
                "blob_id": _git_blob_id(content),
                **(
                    {"lfs_sha256": hashlib.sha256(content).hexdigest()}
                    if name.endswith(".safetensors")
                    else {}
                ),
            }
            for name, content in files.items()
        },
    }
    manifest = root / ".cache" / "huggingface" / "trees" / f"{REVISION}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(tree), encoding="utf-8")


def test_verifies_complete_exact_revision_and_optional_lfs_hash(tmp_path: Path) -> None:
    _payload(tmp_path)

    result = verify_transformers_payload(
        tmp_path,
        revision=REVISION,
        verify_lfs_sha256=True,
    )

    assert result.revision == REVISION
    assert result.weight_files == ("model-00001-of-00001.safetensors",)
    assert result.weight_bytes == len(b"weights")
    assert len(result.identity_digest) == 64


def test_rejects_missing_indexed_shard(tmp_path: Path) -> None:
    _payload(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()

    with pytest.raises(ModelPayloadVerificationError, match="missing a required file"):
        verify_transformers_payload(tmp_path, revision=REVISION)


def test_rejects_wrong_revision_manifest_or_size(tmp_path: Path) -> None:
    _payload(tmp_path)

    with pytest.raises(ModelPayloadVerificationError, match="revision manifest"):
        verify_transformers_payload(tmp_path, revision="0" * 40)

    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"tampered-size")
    with pytest.raises(ModelPayloadVerificationError, match="configured Hub revision"):
        verify_transformers_payload(tmp_path, revision=REVISION)


def test_full_verification_rejects_same_size_weight_corruption(tmp_path: Path) -> None:
    _payload(tmp_path, shard_content=b"weights")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"changed")

    with pytest.raises(ModelPayloadVerificationError, match="LFS checksum"):
        verify_transformers_payload(
            tmp_path,
            revision=REVISION,
            verify_lfs_sha256=True,
        )


def test_external_manifest_attests_read_only_preseeded_tree(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _payload(model)
    internal = model / ".cache" / "huggingface" / "trees" / f"{REVISION}.json"
    external = tmp_path / "revision-manifest.json"
    internal.replace(external)
    external.chmod(0o600)

    result = verify_transformers_payload(
        model,
        revision=REVISION,
        verify_lfs_sha256=True,
        revision_manifest=external,
    )

    assert result.weight_files == ("model-00001-of-00001.safetensors",)


def test_external_manifest_fails_closed_for_symlink_or_writable_file(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _payload(model)
    internal = model / ".cache" / "huggingface" / "trees" / f"{REVISION}.json"
    external = tmp_path / "revision-manifest.json"
    internal.replace(external)
    link = tmp_path / "revision-manifest-link.json"
    link.symlink_to(external)

    with pytest.raises(ModelPayloadVerificationError, match="unavailable or unsafe"):
        verify_transformers_payload(
            model,
            revision=REVISION,
            verify_lfs_sha256=True,
            revision_manifest=link,
        )

    external.chmod(0o622)
    with pytest.raises(ModelPayloadVerificationError, match="unavailable or unsafe"):
        verify_transformers_payload(
            model,
            revision=REVISION,
            verify_lfs_sha256=True,
            revision_manifest=external,
        )


def test_rejects_shard_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "../escape.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelPayloadVerificationError, match="unsafe shard path"):
        verify_transformers_payload(tmp_path, revision="configured")


def test_records_atomic_offline_manifest_from_hub_local_dir_metadata(
    tmp_path: Path,
) -> None:
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors": b"weights",
    }
    metadata_root = tmp_path / ".cache" / "huggingface" / "download"
    for name, content in files.items():
        target = tmp_path / name
        target.write_bytes(content)
        metadata = metadata_root / f"{name}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        object_id = hashlib.sha256(content).hexdigest()
        metadata.write_text(f"{REVISION}\n{object_id}\n1234.5\n", encoding="utf-8")

    manifest = record_huggingface_local_dir_manifest(tmp_path, revision=REVISION)
    verified = verify_transformers_payload(
        tmp_path,
        revision=REVISION,
        verify_lfs_sha256=True,
    )

    assert manifest.name == f"{REVISION}.json"
    assert verified.weight_files == ("model.safetensors",)


def test_manifest_recorder_rejects_mixed_revision_metadata(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    metadata = tmp_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{'0' * 40}\n{'1' * 40}\n1234.5\n", encoding="utf-8")

    with pytest.raises(ModelPayloadVerificationError, match="different Hub revision"):
        record_huggingface_local_dir_manifest(tmp_path, revision=REVISION)
