import base64

import httpx
import pytest
import yaml
from sqlalchemy import select

from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.models import JobRow
from topodrome.scheduler.app import create_app

REPO = "research/checkpoints"
FIRST = "a" * 40
SECOND = "b" * 40


@pytest.mark.asyncio
async def test_api_pins_default_ref_and_exact_snapshot_and_arrays(database, tmp_path):
    model_file = tmp_path / "models.yaml"
    model_file.write_text(
        yaml.safe_dump(
            {
                "models": {
                    REPO: {
                        "revision": FIRST,
                        "refs": ["main", "step20000"],
                        "sites": {},
                        "snapshots": [
                            {
                                "revision": SECOND,
                                "refs": ["step40000"],
                                "subfolder": "weights",
                                "sites": {},
                            }
                        ],
                    }
                }
            }
        )
    )
    app = create_app(
        SchedulerSettings(
            global_root=tmp_path / "global",
            fence_signing_key="x" * 32,
            model_config=model_file,
        ),
        database=database,
        run_dispatcher=False,
    )
    async with AsyncClient("http://test", transport=httpx.ASGITransport(app=app)) as client:
        source = (
            await client._http.post(
                "/api/v1/sources",
                json={
                    "files": {
                        "run.py": base64.b64encode(b"def run(context, **kwargs): pass\n").decode(),
                    }
                },
            )
        ).json()["source_ref"]
        default = await client.submit(entrypoint="run.py", source_ref=source, model=REPO)
        exact = await client.submit(
            entrypoint="run.py",
            source_ref=source,
            model={
                "repo_id": REPO,
                "revision": "step40000",
                "subfolder": "weights",
            },
            idempotency_key="explicit-snapshot",
        )
        replay = await client.submit(
            entrypoint="run.py",
            source_ref=source,
            model={
                "repo_id": REPO,
                "revision": "step40000",
                "subfolder": "weights",
            },
            idempotency_key="explicit-snapshot",
        )
        assert replay.job_id == exact.job_id
        assert default.spec.model.revision == FIRST
        assert exact.spec.model.revision == SECOND
        assert exact.spec.model.subfolder == "weights"
        assert (await client.status(exact.job_id)).spec.model == exact.spec.model
        batch = await client.submit_array(
            {
                "job": {
                    "entrypoint": "run.py",
                    "source_ref": source,
                    "model": {"repo_id": REPO, "revision": "step20000"},
                },
                "elements": [{"index": 0}, {"index": 1}],
                "max_parallel": 2,
            },
            idempotency_key="checkpoint-array",
        )
        async with database.session() as session:
            children = list(
                await session.scalars(select(JobRow).where(JobRow.array_id == batch["array_id"]))
            )
        assert len(children) == 2
        assert all(row.spec["model"]["revision"] == FIRST for row in children)
        assert all("alias" not in row.spec["model"] for row in children)
        mixed = await client.submit_array(
            {
                "job": {
                    "entrypoint": "run.py",
                    "source_ref": source,
                    "model": REPO,
                    "resources": {"gpu": 1, "min_vram_gb": 4},
                },
                "elements": [
                    {"index": 0},
                    {
                        "index": 1,
                        "model": {"repo_id": REPO, "revision": "step40000", "subfolder": "weights"},
                    },
                    {"index": 2, "model": None, "resources": {"gpu": 2}},
                ],
                "max_parallel": 2,
            },
            idempotency_key="mixed-checkpoint-array",
        )
        detail = await client.array(mixed["array_id"])
        selected = [(await client.status(row["job_id"])).spec for row in detail["members"]]
        assert [spec.model.revision if spec.model else None for spec in selected] == [
            FIRST,
            SECOND,
            None,
        ]
        assert selected[1].model.subfolder == "weights" and selected[2].resources.gpu == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        {"repo_id": "unknown/repository"},
        {"repo_id": REPO, "revision": "unknown"},
        {"repo_id": REPO, "subfolder": "unknown"},
        {"alias": "meta-llama-3-8b-instruct"},
        "meta-llama-3-8b-instruct",
    ],
)
async def test_api_rejects_old_aliases_and_unknown_selections_without_creating_job(
    database,
    tmp_path,
    model,
):
    model_file = tmp_path / "models.yaml"
    model_file.write_text(
        yaml.safe_dump(
            {
                "models": {
                    REPO: {
                        "revision": FIRST,
                        "sites": {},
                    }
                }
            }
        )
    )
    app = create_app(
        SchedulerSettings(
            global_root=tmp_path / "global",
            fence_signing_key="x" * 32,
            model_config=model_file,
        ),
        database=database,
        run_dispatcher=False,
    )
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.ASGITransport(app=app)
    ) as http:
        source = (
            await http.post(
                "/api/v1/sources",
                json={
                    "files": {
                        "run.py": base64.b64encode(b"def run(context): pass\n").decode(),
                    }
                },
            )
        ).json()["source_ref"]
        response = await http.post(
            "/api/v1/jobs",
            json={
                "entrypoint": "run.py",
                "source_ref": source,
                "model": model,
            },
        )
        assert response.status_code in {400, 422}
        assert "unregistered" in response.text or "repo_id" in response.text
    async with database.session() as session:
        assert (await session.scalar(select(JobRow))) is None
