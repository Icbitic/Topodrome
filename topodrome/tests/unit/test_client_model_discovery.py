import json

import httpx
import pytest
from typer.testing import CliRunner

from topodrome.client import cli
from topodrome.client.async_client import AsyncClient
from topodrome.client.client import Client

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("repo_id", [None, "EleutherAI/pythia-160m"])
@pytest.mark.parametrize("status", [200, 422])
async def test_model_discovery_sdk_preserves_refs_and_encodes_filter(asynchronous, repo_id, status):
    calls = []
    rows = [
        {
            "repo_id": "EleutherAI/pythia-160m",
            "revision": "a" * 40,
            "refs": ["step20000"],
            "is_default": False,
            "availability": "cached",
        }
    ]

    def respond(request):
        calls.append(request)
        assert request.method == "GET" and request.url.path == "/api/v1/models"
        assert dict(request.url.params) == ({"repo_id": repo_id} if repo_id is not None else {})
        return httpx.Response(status, json=rows if status == 200 else {"detail": "invalid repo_id"})

    transport = httpx.MockTransport(respond)
    if asynchronous:
        async with AsyncClient("http://test", transport=transport) as client:
            if status == 200:
                assert await client.models(repo_id=repo_id) == rows
            else:
                with pytest.raises(RuntimeError, match="Topodrome API returned 422"):
                    await client.models(repo_id=repo_id)
    else:
        with Client("http://test", transport=transport) as client:
            if status == 200:
                assert client.models(repo_id=repo_id) == rows
            else:
                with pytest.raises(RuntimeError, match="Topodrome API returned 422"):
                    client.models(repo_id=repo_id)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "arguments", [[], ["--repo", "EleutherAI/pythia-160m"], ["--repo-id", "EleutherAI/pythia-160m"]]
)
def test_model_discovery_cli_passes_repo_filter_and_prints_refs(monkeypatch, arguments):
    calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def models(self, *, repo_id=None):
            calls.append(repo_id)
            return [{"repo_id": repo_id, "refs": ["step20000"], "is_default": False}]

    monkeypatch.setattr(cli, "Client", FakeClient)
    result = CliRunner().invoke(cli.app, ["models", *arguments])
    assert result.exit_code == 0, result.output
    assert calls == (["EleutherAI/pythia-160m"] if arguments else [None])
    assert json.loads(result.stdout)[0]["refs"] == ["step20000"]
