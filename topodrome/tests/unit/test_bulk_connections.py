from __future__ import annotations

import httpx
import pytest

from topodrome.config import BulkReconnectSettings
from topodrome.deployment.ssh_tunnel import SSHLocalForwardConfig
from topodrome.storage.base import StorageVerificationError
from topodrome.storage.bulk_connections import AdaptiveBulkConnections
from topodrome.storage.link_probe import LinkProbeDeferred, LinkRates

pytestmark = pytest.mark.unit


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Forward:
    def __init__(self, config):
        self.config = config
        self.starts, self.stops = 0, 0
        self.live = False

    def ensure_running(self):
        if not self.live:
            self.starts += 1
        self.live = True
        return self.config.local_port

    def stop(self):
        if self.live:
            self.stops += 1
        self.live = False
        return True


@pytest.fixture
def connections(tmp_path):
    clock = Clock()
    requests, forwards = [], []
    rates = {9102: LinkRates(100_000, 100_000), 9103: LinkRates(1_000_000, 1_000_000)}

    def forward(config):
        result = Forward(config)
        forwards.append(result)
        return result

    def client(**kwargs):
        def handle(request):
            requests.append(request)
            return httpx.Response(200, content=b"data")

        return httpx.Client(**kwargs, transport=httpx.MockTransport(handle))

    def probe(_client, origin, _site, **_kwargs):
        value = rates[httpx.URL(origin).port]
        if isinstance(value, BaseException):
            raise value
        return value

    config = SSHLocalForwardConfig(
        identity_file=tmp_path / "key",
        known_hosts_file=tmp_path / "hosts",
        remote_host="lab1.example",
        remote_user="site",
        local_port=9102,
        target_port=8042,
        state_directory=tmp_path / "bulk",
    )
    route = AdaptiveBulkConnections(
        config,
        "http://127.0.0.1:9102",
        "sqz",
        "synthetic",
        BulkReconnectSettings(enabled=True, candidate_port=9103),
        clock=clock,
        forward_factory=forward,
        client_factory=client,
        probe=probe,
    )
    route.ensure_running()
    try:
        yield route, clock, forwards, requests, rates
    finally:
        route.close()


def slow(route, clock):
    clock.now = 5
    assert not route.maintain()
    clock.now = 35
    return route.maintain()


def test_swap_drains_old_stream_and_routes_new_requests(connections):
    route, clock, forwards, requests, _ = connections
    endpoint = route._origin + "/file"
    with route.stream("download", "GET", endpoint, transfer_size=1_000_000) as old:
        old_generation = route._current
        assert slow(route, clock)
        assert route._current.slot == 1
        assert forwards[0].live  # In-flight stream is not closed or signaled.
        assert not old_generation.clients["download"].is_closed
        with route.stream("download", "GET", endpoint) as new:
            assert new.read() == b"data"
        assert requests[-1].url.port == 9103
        assert old.read() == b"data"
    assert not forwards[0].live
    assert old_generation.clients["download"].is_closed
    assert forwards[1].live
    assert not route._retired


def test_idle_or_fast_links_never_probe(connections):
    route, clock, forwards, _, _ = connections
    clock.now = 900
    assert not route.maintain()
    assert not forwards[1].live
    with route.stream("download", "GET", route._origin + "/file", transfer_size=10_000_000) as r:
        route.progress(r, 20_000_000)
        clock.now = 905
        assert not route.maintain()
    assert not forwards[1].live


@pytest.mark.parametrize("fresh", [None, LinkRates(110_000, 110_000), LinkRates(10_000, 5_000_000)])
def test_unavailable_insufficient_or_asymmetric_candidate_is_discarded(connections, fresh):
    route, clock, forwards, _, rates = connections
    rates[9103] = fresh
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        assert not slow(route, clock)
        assert route._current.slot == 0
        assert forwards[0].live
        assert not forwards[1].live
        # Failed attempts also honor cooldown rather than reconnect storms.
        clock.now = 100
        assert not route.maintain()
        assert forwards[1].starts == 1


def test_probe_authentication_or_integrity_failure_keeps_original(connections, caplog):
    route, clock, forwards, _, rates = connections
    rates[9103] = StorageVerificationError("synthetic secret must not be logged")
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        assert not slow(route, clock)
    assert forwards[0].live and not forwards[1].live
    assert "must not be logged" not in caplog.text


def test_backpressure_on_old_authority_is_not_treated_as_a_broken_route(connections):
    route, clock, forwards, _, rates = connections
    rates[9102] = LinkProbeDeferred("old authority is busy, not a slow path")
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        assert not slow(route, clock)
        assert route._current.slot == 0
        assert forwards[0].live and not forwards[1].live


def test_network_unavailable_old_route_requires_a_verified_fresh_link(connections):
    route, clock, forwards, _, rates = connections
    rates[9102] = None
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        assert slow(route, clock)
        assert route._current.slot == 1
    assert not forwards[0].live and forwards[1].live


def test_retired_route_bounds_generations_and_does_not_block_new_uploads(connections):
    route, clock, forwards, requests, rates = connections
    endpoint = route._origin + "/upload-prefix"
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        assert slow(route, clock)
        with route.stream("upload", "PUT", endpoint, transfer_size=4, content=b"data") as r:
            r.read()
        assert requests[-1].url.port == 9103
        assert route._current.bytes == 4
        clock.now = 640
        assert not route.maintain()  # Two slots, not an unbounded tunnel pool.
        clock.now = 700
        assert not route.maintain()
        assert forwards[0].starts == 1
    rates[9102], rates[9103] = LinkRates(10_000_000, 10_000_000), LinkRates(100_000, 100_000)
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        clock.now = 705
        assert not route.maintain()
        clock.now = 740
        assert route.maintain()
        assert route._current.slot == 0
    assert forwards[0].starts == 2
    assert not forwards[1].live


def test_close_never_closes_an_inflight_request(connections):
    route, _, forwards, _, _ = connections
    with route.stream("download", "GET", route._origin + "/file") as response:
        route.close()
        assert forwards[0].live
        assert response.read() == b"data"
        with pytest.raises(RuntimeError, match="unavailable"):
            with route.stream("download", "GET", route._origin + "/file"):
                pass
    assert not forwards[0].live


def test_bulk_cannot_redirect_enrollment_to_another_authority(connections):
    route, _, _, requests, _ = connections
    with pytest.raises(ValueError, match="configured storage authority"):
        with route.stream("upload", "PUT", "https://foreign.example/upload-prefix"):
            pass
    assert not requests


def test_candidate_startup_failure_leaves_the_primary_owned_and_usable(connections, monkeypatch):
    route, clock, forwards, requests, _ = connections

    def fail():
        raise OSError("synthetic unowned candidate listener")

    monkeypatch.setattr(forwards[1], "ensure_running", fail)
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000):
        assert not slow(route, clock)
        assert route._current.slot == 0
        assert forwards[0].live
    with route.stream("upload", "PUT", route._origin + "/upload-prefix", content=b"data") as r:
        assert r.read() == b"data"
    assert requests[-1].url.port == 9102


def test_close_during_probe_never_promotes_candidate_or_signals_an_open_stream(connections):
    import threading

    route, clock, forwards, _, _ = connections
    entered, release = threading.Event(), threading.Event()
    result = []

    def blocking_probe(_client, origin, _site, **kwargs):
        if httpx.URL(origin).port == 9102:
            entered.set()
            assert release.wait(3)
            return LinkRates(100_000, 100_000)
        return LinkRates(1_000_000, 1_000_000)

    route._probe = blocking_probe
    with route.stream("download", "GET", route._origin + "/file", transfer_size=100_000) as body:
        clock.now = 5
        assert not route.maintain()
        clock.now = 35
        worker = threading.Thread(target=lambda: result.append(route.maintain()))
        worker.start()
        try:
            assert entered.wait(2)
            # Control supervision can continue while a probe blocks elsewhere.
            route.ensure_running()
            assert forwards[0].starts == 1
            route.close()
            assert forwards[0].live
        finally:
            release.set()
            worker.join(3)
        assert not worker.is_alive() and result == [False]
        assert route._current is None
        assert forwards[0].live and not forwards[1].live
        assert body.read() == b"data"
    assert not forwards[0].live
    assert not route._retired
