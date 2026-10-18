"""Model discovery uses registered labels without leaking site-local locations."""

import json

import httpx
import pytest
import yaml

from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import ModelAvailability, SiteKind
from topodrome.domain.specs import ModelCapability, SiteHeartbeat
from topodrome.scheduler.app import create_app

REPO = "research/checkpoints"
FIRST = "a" * 40
SECOND = "b" * 40


@pytest.fixture
async def discovery_app(database, tmp_path):
    model_file = tmp_path / "models.yaml"
    model_file.write_text(
        yaml.safe_dump(
            {
                "models": {
                    REPO: {
                        "revision": FIRST,
                        "refs": ["step20000", "main"],
                        "sites": {"lab1": {"path": "/ssd/private/weights"}},
                        "snapshots": [
                            {
                                "revision": SECOND,
                                "subfolder": "weights",
                                "refs": ["step40000", "stage2/step10"],
                                "sites": {},
                            },
                            {"revision": SECOND, "refs": ["step40000-root"], "sites": {}},
                        ],
                    },
                    "other/checkpoints": {
                        "revision": FIRST,
                        "refs": ["step999"],
                        "sites": {},
                    },
                }
            }
        )
    )

    def capability(repo, revision, subfolder=None):
        return ModelCapability(
            repo_id=repo,
            revision=revision,
            subfolder=subfolder,
            backend="transformers",
            backend_version="configured",
            dtype="bfloat16",
            availability=ModelAvailability.CACHED,
        )

    capabilities = (
        capability(REPO, FIRST),
        capability(REPO, SECOND, "weights"),
        capability(REPO, SECOND),
        capability(REPO, FIRST, "unregistered-subfolder"),
        capability(REPO, "c" * 40),
        capability("other/checkpoints", FIRST),
    )
    repository = SchedulerRepository()
    async with database.transaction() as session:
        for site in ("lab1", "remote"):
            await repository.upsert_heartbeat(
                session,
                SiteHeartbeat(
                    site_id=site,
                    site_kind=SiteKind.LAB,
                    agent_session_id=site,
                    sequence=1,
                    models=capabilities,
                ),
            )
    return create_app(
        SchedulerSettings(
            global_root=tmp_path / "global",
            fence_signing_key="x" * 32,
            model_config=model_file,
        ),
        database=database,
        run_dispatcher=False,
    )


@pytest.mark.asyncio
async def test_model_discovery_returns_refs_defaults_and_site_availability(discovery_app):
    async with AsyncClient(
        "http://test", transport=httpx.ASGITransport(app=discovery_app)
    ) as client:
        rows = await client.models(repo_id=REPO)
        assert len(rows) == 10
        assert {row["repo_id"] for row in rows} == {REPO}
        assert {row["site_id"] for row in rows} == {"lab1", "remote"}
        for row in rows:
            assert row["availability"] == "cached"
            key = row["revision"], row["subfolder"]
            expected = {
                (FIRST, None): (["main", "step20000"], True),
                (SECOND, "weights"): (["stage2/step10", "step40000"], False),
                (SECOND, None): (["step40000-root"], False),
            }.get(key, ([], False))
            assert (row["refs"], row["is_default"]) == expected
        assert "/ssd/private" not in json.dumps(rows)
        assert all("sites" not in row and "path" not in row for row in rows)
        assert await client.models(repo_id="unknown/repository") == []
        response = await client._http.get("/api/v1/models", params={"repo_id": REPO})
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_unfiltered_discovery_keeps_repositories_and_labels_separate(discovery_app):
    async with AsyncClient(
        "http://test", transport=httpx.ASGITransport(app=discovery_app)
    ) as client:
        rows = await client.models()
    assert len(rows) == 12
    other = [row for row in rows if row["repo_id"] == "other/checkpoints"]
    assert len(other) == 2
    assert all(row["refs"] == ["step999"] and row["is_default"] for row in other)


@pytest.mark.asyncio
async def test_uncatalogued_capabilities_do_not_invent_step_labels(discovery_app):
    discovery_app.state.service.model_catalog = None
    async with AsyncClient(
        "http://test", transport=httpx.ASGITransport(app=discovery_app)
    ) as client:
        rows = await client.models(repo_id=REPO)
    assert len(rows) == 10
    assert all(row["refs"] == [] and row["is_default"] is False for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repo_id", ["", "unqualified", "../weights", "research/repo?token=x", "x" * 194]
)
async def test_model_repository_filter_rejects_noncanonical_ids(discovery_app, repo_id):
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.ASGITransport(app=discovery_app)
    ) as client:
        response = await client.get("/api/v1/models", params={"repo_id": repo_id})
    assert response.status_code == 422
