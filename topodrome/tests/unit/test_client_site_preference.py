import json

import httpx
import pytest
from typer.testing import CliRunner

from topodrome.client import cli
from topodrome.client.async_client import AsyncClient
from topodrome.client.client import Client
from topodrome.domain.scheduling import SitePreferenceRequest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("status", [200, 409])
async def test_site_preference_sdk_serializes_cas_and_never_retries(async_client, status):
    calls = []
    body = {"preferred_site_ids": ["sqz-128", "lab1"], "expected_generation": 7}

    def respond(request):
        calls.append(request)
        assert request.method == "PUT"
        assert request.url.path == "/api/v1/policy/site-preference"
        assert json.loads(request.content) == body
        return httpx.Response(status, json={"generation": 8, "detail": "conflict"})

    request = SitePreferenceRequest(**body)
    transport = httpx.MockTransport(respond)
    if async_client:
        async with AsyncClient("http://test", transport=transport) as client:
            if status == 200:
                assert (await client.set_site_preference(request))["generation"] == 8
            else:
                with pytest.raises(RuntimeError, match="Topodrome API returned 409"):
                    await client.set_site_preference(request)
    else:
        with Client("http://test", transport=transport) as client:
            if status == 200:
                assert client.set_site_preference(request)["generation"] == 8
            else:
                with pytest.raises(RuntimeError, match="Topodrome API returned 409"):
                    client.set_site_preference(request)
    assert len(calls) == 1


def test_site_preference_cli_inspect_change_reset_and_exclusive_args(monkeypatch):
    calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def system_status(self):
            calls.append("read")
            return {"policy": {"generation": 7, "preferred_site_ids": ["sqz-128"]}}

        def set_site_preference(self, request):
            calls.append(request.model_dump(mode="json"))
            return {"generation": 8, "preferred_site_ids": list(request.preferred_site_ids)}

    monkeypatch.setattr(cli, "Client", FakeClient)
    runner = CliRunner()
    inspected = runner.invoke(cli.app, ["site-preference"])
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["generation"] == 7
    assert calls == ["read"]
    changed = runner.invoke(cli.app, ["site-preference", "sqz-128", "lab1"])
    assert changed.exit_code == 0, changed.output
    assert calls[-1] == {"preferred_site_ids": ["sqz-128", "lab1"], "expected_generation": 7}
    reset = runner.invoke(cli.app, ["site-preference", "--automatic"])
    assert reset.exit_code == 0, reset.output
    assert calls[-1] == {"preferred_site_ids": [], "expected_generation": 7}
    before = len(calls)
    assert runner.invoke(cli.app, ["site-preference", "lab1", "--automatic"]).exit_code == 2
    assert len(calls) == before
