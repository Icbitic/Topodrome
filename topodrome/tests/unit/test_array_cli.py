from io import StringIO

import httpx
import pytest
from typer.testing import CliRunner

from topodrome.client.arrays import follow_array, load_array_file
from topodrome.client.cli import app
from topodrome.client.client import Client

pytestmark = pytest.mark.unit


def test_sync_array_sdk_and_safe_manifest(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    manifest = tmp_path / "array.yaml"
    manifest.write_text("""source_dir: src
job:
  entrypoint: experiment.py
  model: test/model
indices: 0-2
max_parallel: 2
""")
    request, directory = load_array_file(manifest)
    assert directory == source and len(request.elements) == 3
    calls = []

    def respond(req):
        calls.append(req)
        return httpx.Response(200, json={"array_id": "array_test", "counts": {"queued": 3}})

    request = request.model_copy(
        update={"job": request.job.model_copy(update={"source_ref": "sha256:" + "a" * 64})}
    )
    with Client("http://test", transport=httpx.MockTransport(respond)) as client:
        assert client.submit_array(request, idempotency_key="safe-key")["array_id"] == "array_test"
        assert calls[-1].headers["Idempotency-Key"] == "safe-key"
        client.array("array_test", after=0, limit=10)
        assert dict(calls[-1].url.params) == {"after": "0", "limit": "10"}
        client.update_array("array_test", max_parallel=1, held=False)
        assert calls[-1].method == "PATCH"
        client.cancel_array("array_test")
        assert calls[-1].url.path.endswith("/cancel")


def test_follow_array_waits_for_all_not_first_failed_element():
    frames = iter(
        [
            {"state": "running", "terminal": 1, "total": 2, "counts": {"failed": 1, "running": 1}},
            {"state": "failed", "terminal": 2, "total": 2, "counts": {"failed": 1, "succeeded": 1}},
        ]
    )

    class Fake:
        def array(self, *args, **kwargs):
            return next(frames)

    output = StringIO()
    result = follow_array(Fake(), "array_test", output=output, poll_seconds=0)
    assert result["terminal"] == 2
    assert "running=1" in output.getvalue() and "succeeded=1" in output.getvalue()


def test_array_cli_records_replay_key_and_exits_nonzero_on_failure(tmp_path, monkeypatch):
    from topodrome.client import cli

    manifest = tmp_path / "array.yaml"
    manifest.write_text("""job:
  entrypoint: experiment.py
  model: test/model
  source_ref: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
indices: 0-1
max_parallel: 1
""")

    class Fake:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def submit_array(self, spec, **kwargs):
            assert kwargs["idempotency_key"] == "stable"
            assert len(spec.elements) == 2
            return {"array_id": "array_test"}

        def array(self, *args, **kwargs):
            return {
                "state": "failed",
                "terminal": 2,
                "total": 2,
                "counts": {"failed": 1, "succeeded": 1},
            }

    monkeypatch.setattr(cli, "Client", Fake)
    result = CliRunner().invoke(
        app, ["submit-array", str(manifest), "--idempotency-key", "stable", "--wait"]
    )
    assert result.exit_code == 1
    assert "array_test" in result.output and "replay key" in result.output
    assert "Traceback" not in result.output
