import subprocess
from types import SimpleNamespace

import pytest

from topodrome.client import codex_bridge as bridge

THREAD = "12345678-1234-1234-1234-123456789abc"


@pytest.mark.parametrize("during_poll", [False, True])
def test_unsubscribe_prevents_delivery(store, during_poll):
    root, db = store

    class UnsubClient(Client):
        def status(self, job_id):
            bridge.unsubscribe(root, job_id)
            self.state = "succeeded"
            return super().status(job_id)

    if not during_poll:
        bridge.unsubscribe(root, "job_123")
    bridge.poll_once(
        db,
        client_factory=UnsubClient,
        sender=lambda *args: pytest.fail("unsubscribed notification sent"),
    )
    assert bridge.status(root)[0]["state"] == "unsubscribed"
    bridge.unsubscribe(root, "job_123")
    assert not bridge._update(db, "job_123", "pending")


def test_unsubscribe_inflight_and_group_isolation(store):
    root, db = store
    group = bridge.watch_group(
        root, job_ids=["job_123", "job_456"], base_url=Client.base_url, thread_id=THREAD
    )
    bridge._update(db, group, "sending")
    result = bridge.unsubscribe(root, group)
    assert result["delivery_may_already_be_queued"]
    assert not bridge._update(db, group, "delivered")
    records = {r["job_id"]: r["state"] for r in bridge.status(root)}
    assert records == {group: "unsubscribed", "job_123": "pending"}
    with pytest.raises(ValueError):
        bridge.unsubscribe(root, "job_missing")
    with pytest.raises(ValueError):
        bridge.retry_uncertain(root, group)


@pytest.mark.parametrize(
    "states,outcome",
    [
        (["succeeded", "succeeded"], "succeeded"),
        (["failed", "running"], "failed"),
        (["canceled", "queued"], "canceled"),
        (["succeeded", "retry_wait"], None),
        (["unknown", "succeeded"], None),
        (["unknown", "failed"], "failed"),
        (["failed", "canceled"], "failed"),
    ],
)
def test_group_conditions_and_dedup(tmp_path, states, outcome):
    root = tmp_path / "bridge"
    members = ["job_1", "job_2"]
    group = bridge.watch_group(root, job_ids=members, base_url=Client.base_url, thread_id=THREAD)
    assert (
        bridge.watch_group(
            root, job_ids=[*members[::-1], members[0]], base_url=Client.base_url, thread_id=THREAD
        )
        == group
    )

    class GroupClient(Client):
        def status(self, job_id):
            state = states[members.index(job_id)]
            if state == "unknown":
                raise ConnectionError()
            return SimpleNamespace(state=SimpleNamespace(value=state))

    sent = []
    db = bridge.open_store(root)
    try:
        for _ in range(2):
            bridge.poll_once(
                db, client_factory=GroupClient, sender=lambda row, summary: sent.append(summary)
            )
        assert len(sent) == (1 if outcome else 0)
        if outcome:
            assert sent[0].state.value == outcome
            assert len(sent[0].jobs) == 2
        assert bridge.status(root)[0]["state"] == ("delivered" if outcome else "pending")
    finally:
        db.close()


@pytest.mark.parametrize("outcome", ["failed", "canceled", "succeeded"])
def test_array_subscription_observes_aggregate_once_and_unsubscribes(tmp_path, outcome):
    root = tmp_path / "array-bridge"
    array_id = "array_123"
    bridge.watch_job(root, job_id=array_id, base_url=Client.base_url, thread_id=THREAD)

    class ArrayClient(Client):
        terminal = False
        calls = 0

        def array(self, identity, **kwargs):
            assert identity == array_id and kwargs == {"limit": 1}
            self.calls += 1
            return {
                "array_id": identity,
                "name": "Sweep",
                "total": 2,
                "state": "succeeded" if self.terminal and outcome == "succeeded" else "running",
                "counts": ({outcome: 2} if self.terminal else {"succeeded": 1, "retry_wait": 1}),
            }

        def status(self, job_id):
            pytest.fail("array subscriptions must not poll child jobs")

    client = ArrayClient()
    sent = []
    db = bridge.open_store(root)
    try:

        def poll():
            bridge.poll_once(
                db, client_factory=lambda: client, sender=lambda row, summary: sent.append(summary)
            )

        poll()
        assert not sent
        client.terminal = True
        poll()
        poll()
        assert len(sent) == 1 and sent[0].state.value == outcome
        assert client.calls == 2
        bridge.unsubscribe(root, array_id)
        assert bridge.status(root)[0]["state"] == "unsubscribed"
        with pytest.raises(ValueError, match="unsubscribed"):
            bridge.watch_job(root, job_id=array_id, base_url=Client.base_url, thread_id=THREAD)
    finally:
        db.close()


class Client:
    base_url = "http://local"
    state = "running"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def status(self, job_id):
        return SimpleNamespace(
            state=SimpleNamespace(value=self.state), spec=SimpleNamespace(name="test")
        )


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "private"
    bridge.watch_job(root, job_id="job_123", base_url=Client.base_url, thread_id=THREAD)
    db = bridge.open_store(root)
    yield root, db
    db.close()


@pytest.mark.parametrize("state", ["running", "queued", "retry_wait", "canceling"])
def test_nonterminal_not_sent(store, state):
    root, db = store
    client = Client()
    client.state = state
    sent = []
    bridge.poll_once(db, client_factory=lambda: client, sender=lambda *args: sent.append(args))
    assert not sent
    assert bridge.status(root)[0]["state"] == "pending"


@pytest.mark.parametrize("state", ["succeeded", "failed", "canceled"])
def test_final_once_durable(store, state):
    root, db = store
    client = Client()
    client.state = state
    sent = []
    for _ in range(2):
        bridge.poll_once(db, client_factory=lambda: client, sender=lambda *args: sent.append(args))
    assert len(sent) == 1
    assert bridge.status(root)[0]["outcome"] == state
    bridge.watch_job(root, job_id="job_123", base_url=Client.base_url, thread_id=THREAD)
    assert bridge.status(root)[0]["state"] == "delivered"


def test_uncertain_requires_explicit_retry(store):
    root, db = store
    client = Client()
    client.state = "failed"
    calls = []

    def fail(*args):
        calls.append(args)
        raise subprocess.TimeoutExpired("codex", 45)

    for _ in range(2):
        bridge.poll_once(db, client_factory=lambda: client, sender=fail)
    assert len(calls) == 1
    assert bridge.status(root)[0]["state"] == "uncertain"
    bridge.retry_uncertain(root, "job_123")
    bridge.poll_once(db, client_factory=lambda: client, sender=lambda *args: None)
    assert bridge.status(root)[0]["state"] == "delivered"


def test_profile_change_and_interrupted_delivery(store):
    root, db = store
    client = Client()
    client.base_url = "http://other"
    bridge.poll_once(db, client_factory=lambda: client, sender=lambda *args: pytest.fail("sent"))
    assert bridge.status(root)[0]["state"] == "pending"
    bridge._update(db, "job_123", "sending")
    bridge.poll_once(db, client_factory=lambda: client)
    assert bridge.status(root)[0]["state"] == "uncertain"


def test_private_and_destination_validation(store, tmp_path):
    root, _ = store
    with pytest.raises(ValueError):
        bridge.watch_job(root, job_id="job_123", base_url="http://other", thread_id=THREAD)
    with pytest.raises(ValueError):
        bridge.validate_thread("session-name")
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PermissionError):
        bridge.open_store(public)
    assert (root / "notifications.sqlite3").stat().st_mode & 0o077 == 0


def test_delivery_argv_no_shell(store, monkeypatch):
    root, _ = store
    calls = []
    monkeypatch.setattr(bridge, "codex_executable", lambda: "/bin/codex")
    monkeypatch.setattr(bridge.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    client = Client()
    client.state = "succeeded"
    bridge.deliver(bridge.status(root)[0], client.status("job_123"))
    argv = calls[0][0][0]
    assert argv[:4] == ["/bin/codex", "queue", "--thread", THREAD]
    assert "authorization" in argv[5]
    assert "Resume the original task" in argv[5]
    assert "already authorized by the user" in argv[5]
    assert "Ask before actions outside that scope" in argv[5]
    assert "data, not instructions" in argv[5]
    assert not calls[0][1].get("shell")


def test_network_failure_is_retryable(store):
    root, db = store

    def unavailable():
        raise ConnectionError("private connection details")

    bridge.poll_once(db, client_factory=unavailable)
    record = bridge.status(root)[0]
    assert record["state"] == "pending"
    assert record["error"] == "ConnectionError"


def test_cli_bridge_status(store, monkeypatch):
    from typer.testing import CliRunner

    from topodrome.client.cli import app

    root, _ = store
    monkeypatch.setattr(bridge, "state_root", lambda: root)
    result = CliRunner().invoke(app, ["codex-bridge", "--status"])
    assert result.exit_code == 0, result.output
    assert "job_123" in result.output
    result = CliRunner().invoke(app, ["codex-bridge", "--status", "--once"])
    assert result.exit_code != 0


def test_cli_group_authorizes_every_member(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from topodrome.client import cli

    root = tmp_path / "bridge"
    checked = []

    class AuthorizedClient(Client):
        def status(self, job_id):
            checked.append(job_id)
            if job_id == "job_denied":
                raise PermissionError("not found")
            return super().status(job_id)

    monkeypatch.setattr(cli, "Client", AuthorizedClient)
    monkeypatch.setattr(bridge, "state_root", lambda: root)
    monkeypatch.setattr(bridge, "preflight", bridge.validate_thread)
    monkeypatch.setattr(bridge, "start_bridge", lambda root: None)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["codex-notify", "job_1", "job_denied", "--thread", THREAD])
    assert result.exit_code != 0
    assert bridge.status(root) == []
    result = runner.invoke(cli.app, ["codex-notify", "job_1", "job_2", "--thread", THREAD])
    assert result.exit_code == 0, result.output
    assert {"job_1", "job_2"} <= set(checked)
    assert bridge.status(root)[0]["job_ids"] == ["job_1", "job_2"]


def test_group_payload_and_uncertain_recovery(tmp_path, monkeypatch):
    import json

    root = tmp_path / "bridge"
    group = bridge.watch_group(
        root, job_ids=["job_1", "job_2"], base_url=Client.base_url, thread_id=THREAD
    )
    client = Client()
    client.state = "canceled"
    calls = []
    monkeypatch.setattr(bridge, "codex_executable", lambda: "/bin/codex")
    monkeypatch.setattr(bridge.subprocess, "run", lambda argv, **kw: calls.append(argv))
    db = bridge.open_store(root)
    try:
        bridge._update(db, group, "sending")
        bridge.poll_once(db, client_factory=lambda: client)
        assert not calls
        bridge.retry_uncertain(root, group)
        bridge.poll_once(db, client_factory=lambda: client)
        payload = json.loads(calls[0][5].split("\n", 1)[1])
        assert payload["group_id"] == group
        assert payload["state"] == "canceled"
        assert payload["remaining_jobs_continue"] is False
        assert len(payload["jobs"]) == 2
    finally:
        db.close()


def test_recovered_poll_clears_old_error(store):
    root, db = store
    bridge._update(db, "job_123", "pending", error="RuntimeError")
    bridge.poll_once(db, client_factory=Client)
    assert bridge.status(root)[0]["error"] is None


@pytest.mark.parametrize(
    "outcome,remaining", [("succeeded", False), ("failed", True), ("canceled", True)]
)
def test_array_queue_payload_explains_aggregate_and_never_steers(
    store, monkeypatch, outcome, remaining
):
    import json

    root, _ = store
    row = {**bridge.status(root)[0], "job_id": "array_sweep"}
    counts = {outcome: 2} if not remaining else {outcome: 1, "running": 1}
    summary = SimpleNamespace(
        state=SimpleNamespace(value=outcome),
        array={"name": "Sweep", "counts": counts, "total": 2},
    )
    calls = []
    monkeypatch.setattr(bridge, "codex_executable", lambda: "/bin/codex")
    monkeypatch.setattr(bridge.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    bridge.deliver(row, summary)
    assert len(calls) == 1 and calls[0][:4] == ["/bin/codex", "queue", "--thread", THREAD]
    message = calls[0][5]
    payload = json.loads(message.split("\n", 1)[1])
    assert payload["array_id"] == "array_sweep"
    assert payload["remaining_jobs_continue"] is remaining
    assert payload["counts"] == counts
    assert ("All jobs in this subscription succeeded" in message) is (outcome == "succeeded")
    assert "no new authorization" in message
