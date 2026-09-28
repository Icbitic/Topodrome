from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from topodrome.client import cli
from topodrome.domain.specs import AssignmentOffer, JobSpec
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry


def spec(**kwargs):
    return JobSpec(entrypoint="run.py", model="llama", **kwargs)


def test_metadata_optional_and_runtime_neutral():
    original = spec()
    named = spec(
        name="  Lens run  ", description="First pass\nSeed 42", estimated_runtime_seconds=120
    )
    assert original.name is None and original.description is None
    assert named.name == "Lens run"
    assert named.description == "First pass\nSeed 42"
    assert named.runtime_fingerprint == original.runtime_fingerprint
    assert original.estimated_runtime_seconds is None
    assert named.estimated_runtime_seconds == 120
    assert spec(name=" ", description="\n").name is None


@pytest.mark.parametrize(
    "values",
    [
        {"name": "x" * 161},
        {"description": "x" * 4001},
        {"name": "line\nbreak"},
        {"name": "\x1b[31mred"},
        {"description": "\x00"},
        {"name": "\u202eevil"},
        *(
            {"estimated_runtime_seconds": value}
            for value in (0, -1, float("nan"), float("inf"), True, "60", "10 minutes")
        ),
    ],
)
def test_metadata_validation(values):
    with pytest.raises(ValidationError):
        spec(**values)


@pytest.mark.asyncio
async def test_display_metadata_is_not_sent_to_older_agents():
    messages = []

    async def send(message):
        messages.append(message)

    registry = AgentConnectionRegistry()
    await registry.register(AgentConnection("sqz", "session", send))
    offer = AssignmentOffer(
        job_id="job_1",
        attempt_id="attempt_1",
        attempt_number=1,
        site_id="sqz",
        spec=spec(name="named", description="description", estimated_runtime_seconds=60),
        claims=({"gpu_uuid": "gpu_1", "epoch": 1},),
        fencing_token="token",  # noqa: S106 -- synthetic protocol fixture
        expires_at=datetime.now(UTC) + timedelta(seconds=90),
    )
    assert await registry.offer(offer)
    payload = messages[0]["payload"]["spec"]
    assert "name" not in payload and "description" not in payload
    assert "estimated_runtime_seconds" not in payload
    assert payload["entrypoint"] == "run.py"


def test_cli_metadata_overrides_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "job.yaml"
    manifest.write_text("entrypoint: run.py\nmodel: llama\nname: original\n")
    captured = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, value, **kwargs):
            captured.append(value)
            return self

        def status(self):
            return captured[-1]

    monkeypatch.setattr(cli, "Client", Client)
    result = CliRunner().invoke(
        cli.app,
        [
            "submit",
            str(manifest),
            "--name",
            "my run",
            "--description",
            "seed 42",
            "--estimated-runtime-seconds",
            "120",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0].name == "my run"
    assert captured[0].description == "seed 42"
    assert captured[0].estimated_runtime_seconds == 120
