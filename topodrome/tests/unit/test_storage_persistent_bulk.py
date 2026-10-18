from __future__ import annotations

import hashlib
import json

import pytest

from topodrome.storage.base import ObjectDigest, StorageUnavailableError, StorageVerificationError
from topodrome.storage.bulk import BulkTransportError
from topodrome.storage.http_mirror import HttpMirrorStorage
from topodrome.storage.progress import TransferCancelled

pytestmark = pytest.mark.unit


class Bulk:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.reads = []
        self.removes = []
        self.fail_after = None
        self.lose_rename_ack = False
        self.canceled = False
        self.digest_calls = []

    def healthcheck(self):
        return True

    def close(self):
        pass

    def stat(self, path):
        content = self.objects.get(path)
        if content is None:
            return None
        return {"IsDir": False, "Size": len(content), "ModTime": "stable"}

    def digest(self, path):
        self.digest_calls.append(path)
        content = self.objects.get(path)
        return (
            None
            if content is None
            else ObjectDigest(len(content), hashlib.sha256(content).hexdigest())
        )

    def read(self, path, output, offset, count, *, expected_identity=None):
        self.reads.append((path, offset, count))
        output.write(self.objects[path][offset : offset + count])

    def write(self, path, source, offset, size):
        self.writes.append((path, offset, size))
        body = source.read_bytes()
        end = size if self.fail_after is None else min(size, offset + self.fail_after)
        self.objects[path] = self.objects.get(path, b"")[:offset] + body[offset:end]
        if self.canceled:
            self.canceled = False
            raise TransferCancelled("transfer canceled")
        if self.fail_after is not None:
            self.fail_after = None
            raise StorageUnavailableError("connection dropped")

    def rename(self, source, destination):
        self.objects[destination] = self.objects.pop(source)
        if self.lose_rename_ack:
            self.lose_rename_ack = False
            raise StorageUnavailableError("ack lost")

    def remove(self, path):
        self.removes.append(path)
        self.objects.pop(path, None)

    def download_tree(self, path, destination):
        raise AssertionError("unexpected tree download")


def storage(tmp_path, bulk):
    return HttpMirrorStorage(
        tmp_path / "cache",
        "lab1:/global",
        authoritative_reader=bulk,
        bulk_transport=bulk,
        min_free_space_gb=1e-7,
    )


def prepare(mirror, body=b"0123456789"):
    path = "jobs/job/attempts/attempt/committed/output"
    local = mirror.root / path
    local.parent.mkdir(parents=True)
    local.write_bytes(body)
    return path


def test_upload_resumes_verified_prefix_across_adapter_restart(tmp_path):
    bulk = Bulk()
    first = storage(tmp_path, bulk)
    path = prepare(first)
    bulk.fail_after = 4
    with pytest.raises(StorageUnavailableError):
        first.flush_and_verify(path)
    assert list(first._upload_root.glob("*.json"))
    second = storage(tmp_path, bulk)
    result = second.flush_and_verify(path)
    assert result.digest.size_bytes == 10
    assert [offset for _, offset, _ in bulk.writes] == [0, 4]
    assert bulk.writes[0][0] == bulk.writes[1][0]
    assert bulk.objects == {path: b"0123456789"}
    assert not list(second._upload_root.glob("*.json"))


@pytest.mark.parametrize("corrupt", [b"xxxx", b"\0\0\0\0", b"x" * 20])
def test_bad_or_oversized_prefix_is_never_trusted(tmp_path, corrupt):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    bulk.fail_after = 4
    with pytest.raises(StorageUnavailableError):
        mirror.flush_and_verify(path)
    temporary = bulk.writes[0][0]
    bulk.objects[temporary] = corrupt
    mirror.flush_and_verify(path)
    assert [offset for _, offset, _ in bulk.writes] == [0, 0]
    assert bulk.removes == [temporary]
    assert bulk.objects[path] == b"0123456789"


@pytest.mark.parametrize("lost", ["write", "rename"])
def test_complete_transfer_with_lost_ack_is_not_resent(tmp_path, lost):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    if lost == "write":
        bulk.fail_after = 10
    else:
        bulk.lose_rename_ack = True
    with pytest.raises(StorageUnavailableError):
        mirror.flush_and_verify(path)
    mirror.flush_and_verify(path)
    assert len(bulk.writes) == 1
    assert bulk.objects == {path: b"0123456789"}


def test_cancellation_preserves_owned_prefix_for_safe_retry(tmp_path):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    bulk.canceled, bulk.fail_after = True, 4
    with pytest.raises(TransferCancelled):
        mirror.flush_and_verify(path)
    bulk.fail_after = None
    mirror.flush_and_verify(path)
    assert [offset for _, offset, _ in bulk.writes] == [0, 4]


def test_receipt_cannot_remove_an_unowned_remote_file(tmp_path):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    bulk.fail_after = 4
    with pytest.raises(StorageUnavailableError):
        mirror.flush_and_verify(path)
    receipt = next(mirror._upload_root.glob("*.json"))
    value = json.loads(receipt.read_bytes())
    value["temporary"] = "jobs/other/valuable-output"
    receipt.write_text(json.dumps(value))
    with pytest.raises(StorageVerificationError):
        mirror.flush_and_verify(path)
    assert bulk.removes == []


def test_source_change_discards_only_its_old_owned_prefix(tmp_path):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    bulk.fail_after = 4
    with pytest.raises(StorageUnavailableError):
        mirror.flush_and_verify(path)
    old = bulk.writes[0][0]
    (mirror.root / path).write_bytes(b"new-data")
    mirror.flush_and_verify(path)
    assert bulk.removes == [old]
    assert bulk.objects == {path: b"new-data"}


def test_pinned_bulk_checksum_is_checked_before_cache_visibility(tmp_path):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = "blobs/sha256/input"
    bulk.objects[path] = b"bad"
    with pytest.raises(StorageVerificationError):
        mirror.resolve_pinned(
            path, expected_size_bytes=3, expected_sha256=hashlib.sha256(b"good").hexdigest()
        )
    assert not (mirror.root / path).exists()
    bulk.objects[path] = b"yes"
    local = mirror.resolve_pinned(
        path, expected_size_bytes=3, expected_sha256=hashlib.sha256(b"yes").hexdigest()
    )
    assert local.read_bytes() == b"yes"


def test_success_has_two_authority_verifications_not_three(tmp_path):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    mirror.flush_and_verify(path)
    temporary = bulk.writes[0][0]
    assert bulk.digest_calls.count(temporary) == 1
    # One missing-target check, then one final durability/hash acknowledgement.
    assert bulk.digest_calls.count(path) == 2


def test_link_failure_resumes_inside_upload_without_recomputing(tmp_path):
    class Interrupted(Bulk):
        def write(self, *args):
            try:
                return super().write(*args)
            except StorageUnavailableError:
                raise BulkTransportError("ConnectionLost", retryable=True) from None

    bulk = Interrupted()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    bulk.fail_after = 4
    mirror.flush_and_verify(path)
    assert [offset for _, offset, _ in bulk.writes] == [0, 4]
    assert bulk.objects == {path: b"0123456789"}


def test_link_failure_resumes_private_download_inside_one_call(tmp_path):
    class Interrupted(Bulk):
        def read(self, path, output, offset, count, *, expected_identity=None):
            if not self.reads:
                self.reads.append((path, offset, count))
                output.write(self.objects[path][:4])
                raise BulkTransportError("ConnectionLost", retryable=True)
            return super().read(path, output, offset, count)

    bulk = Interrupted()
    mirror = storage(tmp_path, bulk)
    path = "jobs/job/attempts/attempt/committed/output"
    bulk.objects[path] = b"0123456789"
    result = mirror.resolve_pinned(
        path, expected_size_bytes=10, expected_sha256=hashlib.sha256(bulk.objects[path]).hexdigest()
    )
    assert [offset for _, offset, _ in bulk.reads] == [0, 4]
    assert result.read_bytes() == bulk.objects[path]


@pytest.mark.parametrize("retryable,attempts", [(False, 1), (True, 3)])
def test_transport_retry_is_bounded_and_never_downgrades_authentication(
    tmp_path, retryable, attempts
):
    bulk = Bulk()
    mirror = storage(tmp_path, bulk)
    path = prepare(mirror)
    calls = []

    def failure(*args):
        calls.append(args)
        raise BulkTransportError(
            "ConnectionLost" if retryable else "PermissionDenied", retryable=retryable
        )

    bulk.write = failure
    with pytest.raises(BulkTransportError):
        mirror.flush_and_verify(path)
    assert len(calls) == attempts
    assert not (mirror.root.parent / "unverified").exists()


def test_reconnects_share_one_transfer_deadline(tmp_path):
    from topodrome.storage.progress import TransferDeadlineExpired

    mirror = storage(tmp_path, Bulk())
    now = [0.0]
    mirror._monotonic = lambda: now[0]
    mirror._sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    mirror._transfer_timeout = 0.01
    calls = []

    def failure():
        calls.append(None)
        raise BulkTransportError("ConnectionLost", retryable=True)

    with pytest.raises(TransferDeadlineExpired):
        mirror._retry_bulk(failure)
    assert len(calls) == 1
