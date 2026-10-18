from __future__ import annotations

import json
import sys

import httpx
import pytest
import yaml

from topodrome.deployment.updater import UpgradeCommandFailed, UpgradeRunner
from topodrome.deployment.upgrade_store import UpgradeStore
from topodrome.domain.errors import ConflictError
from topodrome.domain.upgrades import UpgradeRequest

pytestmark = pytest.mark.unit


def plan_file(tmp_path, **changes):
    path = tmp_path / "release.yaml"
    plan = {
        "name": "test release",
        "sites": ["sqz"],
        "prepare": [{"argv": [sys.executable, "-c", "pass"]}],
        "apply": [{"argv": [sys.executable, "-c", "pass"]}],
        "verify": [{"argv": [sys.executable, "-c", "pass"]}],
    }
    plan.update(changes)
    path.write_text(yaml.safe_dump(plan))
    path.chmod(0o600)
    return path


def schedule(tmp_path, *, pause="targets", **changes):
    path = plan_file(tmp_path, **changes)
    store = UpgradeStore(tmp_path / "global")
    record = store.schedule(UpgradeRequest(plan_path=path, pause=pause, idempotency_key="release1"))
    return store, record


@pytest.mark.parametrize(
    "pause,expected",
    [
        ("none", (False, frozenset())),
        ("targets", (False, frozenset({"sqz"}))),
        ("all", (True, frozenset())),
    ],
)
def test_pause_only_after_preparation_and_always_during_apply(tmp_path, pause, expected):
    store, record = schedule(tmp_path, pause=pause)
    assert store.exclusions() == (False, frozenset())
    store.change(record.upgrade_id, expected={"queued"}, state="waiting")
    assert store.exclusions() == expected
    store.change(record.upgrade_id, expected={"waiting"}, state="applying", mutation_started=True)
    assert store.blocked("sqz")
    assert store.blocked("lab1") == (pause == "all")
    store.change(record.upgrade_id, expected={"applying"}, state="needs_attention")
    assert store.blocked("sqz")


def test_scheduler_target_fences_entire_cluster(tmp_path):
    store, record = schedule(tmp_path, scheduler=True, sites=[])
    store.change(record.upgrade_id, expected={"queued"}, state="waiting")
    assert store.exclusions() == (True, frozenset())


def test_frozen_release_idempotency_and_private_summary(tmp_path):
    member = tmp_path / "apply.py"
    member.write_text("print('SYNTHETIC_SECRET')")
    member.chmod(0o600)
    store, record = schedule(tmp_path, files=["apply.py"])
    request = UpgradeRequest(
        plan_path=tmp_path / "release.yaml", pause="targets", idempotency_key="release1"
    )
    assert store.schedule(request).upgrade_id == record.upgrade_id
    member.write_text("changed original")
    assert (store.release_path(record) / "apply.py").read_text() == "print('SYNTHETIC_SECRET')"
    with pytest.raises(ConflictError):
        store.schedule(request)
    assert "argv" not in json.dumps(record.summary())
    assert "SYNTHETIC_SECRET" not in json.dumps(record.summary())
    store.validate_release(record)
    (store.release_path(record) / "apply.py").write_text("tampered frozen input")
    with pytest.raises(ValueError, match="changed"):
        store.validate_release(record)


@pytest.mark.parametrize("unsafe", ["permissions", "symlink", "traversal"])
def test_rejects_unsafe_operator_inputs(tmp_path, unsafe):
    path = plan_file(tmp_path)
    if unsafe == "permissions":
        path.chmod(0o644)
    elif unsafe == "symlink":
        actual = tmp_path / "actual.yaml"
        path.rename(actual)
        path.symlink_to(actual)
    else:
        path = plan_file(tmp_path, files=["../outside"])
    with pytest.raises((ValueError, PermissionError)):
        UpgradeStore(tmp_path / "global").schedule(
            UpgradeRequest(plan_path=path, idempotency_key="bad")
        )


def test_only_one_pending_release_and_corrupt_journal_fails_closed(tmp_path):
    store, record = schedule(tmp_path)
    with pytest.raises(ConflictError):
        store.schedule(
            UpgradeRequest(plan_path=tmp_path / "release.yaml", idempotency_key="different")
        )
    assert UpgradeStore(tmp_path / "global").get(record.upgrade_id) == record
    store.path.write_text("not a valid journal")
    with pytest.raises(ValueError):
        store.exclusions()


def test_apply_never_runs_without_idle_claim(tmp_path):
    store, record = schedule(tmp_path)
    executed = []
    with httpx.Client(
        base_url="http://authority",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"claimed": False})
        ),
    ) as http:
        runner = UpgradeRunner(store, http, lambda step, _release: executed.append(step))
        assert not runner.run_once()
        assert len(executed) == 1  # Preparation only.
        assert store.get(record.upgrade_id).state == "waiting"
        assert not runner.run_once()
        assert len(executed) == 1


def test_interrupted_apply_never_replays_and_remains_fenced(tmp_path):
    store, record = schedule(tmp_path, pause="none")
    store.change(
        record.upgrade_id,
        expected={"queued"},
        state="applying",
        mutation_started=True,
        runner_id="dead-updater",
    )
    executed = []
    with httpx.Client(base_url="http://authority") as http:
        runner = UpgradeRunner(store, http, lambda step, _release: executed.append(step))
        assert not runner.run_once()
    assert not executed
    assert store.get(record.upgrade_id).state == "needs_attention"
    assert store.blocked("sqz")


def test_prepare_cancellation_wins_without_live_mutation(tmp_path):
    store, record = schedule(tmp_path)

    def cancel_during_prepare(_step, _release):
        store.change(record.upgrade_id, expected={"preparing"}, state="canceled")

    with httpx.Client(base_url="http://authority") as http:
        assert not UpgradeRunner(store, http, cancel_during_prepare).run_once()
    assert store.get(record.upgrade_id).state == "canceled"
    assert not store.blocked("sqz")


def test_runner_successful_claim_apply_verify_and_resume(tmp_path):
    store, record = schedule(tmp_path)
    executed = []

    def handler(request):
        runner_id = json.loads(request.content)["runner_id"]
        if request.url.path.endswith("/claim"):
            store.change(
                record.upgrade_id,
                expected={"waiting"},
                state="applying",
                mutation_started=True,
                runner_id=runner_id,
            )
            return httpx.Response(200, json={"claimed": True})
        store.change(record.upgrade_id, expected={"verifying"}, state="succeeded")
        return httpx.Response(200, json={"completed": True})

    with httpx.Client(base_url="http://authority", transport=httpx.MockTransport(handler)) as http:
        runner = UpgradeRunner(store, http, lambda step, _release: executed.append(step))
        assert runner.run_once()
        assert len(executed) == 3
        assert not runner.run_once()
    assert not store.blocked("sqz")


def test_failed_apply_rolls_back_without_clearing_fence_early(tmp_path):
    store, record = schedule(
        tmp_path,
        rollback=[{"argv": [sys.executable, "-c", "rollback"]}],
        verify_rollback=[{"argv": [sys.executable, "-c", "verify rollback"]}],
    )
    executed = []

    def execute(step, _release):
        executed.append(step)
        if len(executed) == 2:
            raise UpgradeCommandFailed("SYNTHETIC_SECRET")

    def handler(request):
        runner_id = json.loads(request.content)["runner_id"]
        if request.url.path.endswith("/claim"):
            store.change(
                record.upgrade_id,
                expected={"waiting"},
                state="applying",
                mutation_started=True,
                runner_id=runner_id,
            )
            return httpx.Response(200, json={"claimed": True})
        store.change(record.upgrade_id, expected={"verifying_rollback"}, state="rolled_back")
        return httpx.Response(200, json={"completed": True})

    with httpx.Client(base_url="http://authority", transport=httpx.MockTransport(handler)) as http:
        runner = UpgradeRunner(store, http, execute)
        assert not runner.run_once()
        assert store.blocked("sqz")
        assert store.current().state == "rolling_back"
        assert runner.run_once()
    assert len(executed) == 4
    assert not store.blocked("sqz")
    assert "SYNTHETIC_SECRET" not in store.path.read_text()


def test_single_external_runner_lock(tmp_path):
    store, _ = schedule(tmp_path)
    with httpx.Client(base_url="http://authority") as http:
        first, second = UpgradeRunner(store, http), UpgradeRunner(store, http)
        with first.exclusive():
            with pytest.raises(BlockingIOError), second.exclusive():
                pass
