from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from tests.schema_assertions import assert_schema_matches_models
from topodrome.db.session import CURRENT_SCHEMA_REVISION

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_production_readiness_revision_tracks_alembic_head() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    assert ScriptDirectory.from_config(config).get_current_head() == CURRENT_SCHEMA_REVISION


def test_repository_address_conversion_preserves_history_and_user_data(tmp_path: Path) -> None:
    path = tmp_path / "model-addresses.sqlite3"
    _upgrade(path, "0019_remove_db_legacies")
    legacy = {"entrypoint": "run.py", "model": {
        "alias": "meta-llama-3-8b-instruct", "revision": "a" * 40,
    }, "kwargs": {"alias": "this is user data", "model": "do not rewrite"},
        "assets": {"alias": "job://job_other/results/"}}
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO jobs (id,state,spec,priority,cancel_requested,created_at,updated_at) "
            "VALUES ('job_historical','succeeded',?,0,0,'2026-10-01','2026-10-01')",
            (json.dumps(legacy),),
        )
        connection.execute(
            "INSERT INTO model_capabilities (site_id,alias,revision,backend,backend_version,"
            "dtype,quantization,adapter,availability,gpu_uuid,verified_at) "
            "VALUES ('site','meta-llama-3-8b-instruct',?,'transformers','configured',"
            "'bfloat16','','','cached','','2026-10-01')", ("a" * 40,),
        )
    _upgrade(path)
    _assert_sqlite_schema_matches_models(path)
    with sqlite3.connect(path) as connection:
        spec = json.loads(connection.execute("SELECT spec FROM jobs").fetchone()[0])
        assert spec["model"] == {
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct", "revision": "a" * 40,
            "subfolder": None,
        }
        assert spec["kwargs"] == legacy["kwargs"]
        assert spec["assets"] == legacy["assets"]
        capability = connection.execute(
            "SELECT repo_id,subfolder FROM model_capabilities"
        ).fetchone()
        assert capability == (
            "meta-llama/Meta-Llama-3-8B-Instruct", "",
        )


@pytest.mark.parametrize("blocker", ["unknown_alias", "active_lease"])
def test_repository_address_conversion_refuses_unsafe_cutover_before_ddl(tmp_path, blocker):
    path = tmp_path / "model-address-preflight.sqlite3"
    _upgrade(path, "0019_remove_db_legacies")
    with sqlite3.connect(path) as connection:
        if blocker == "unknown_alias":
            connection.execute(
                "INSERT INTO jobs (id,state,spec,priority,cancel_requested,created_at,updated_at) "
                "VALUES ('job','succeeded',?,0,0,'2026-10-01','2026-10-01')",
                (json.dumps({"model": {"alias": "unknown-old-name"}}),),
            )
        else:
            connection.execute(
                "INSERT INTO leases (id,attempt_id,site_id,state,fencing_token_digest,"
                "expires_at,created_at,updated_at) VALUES "
                "('lease','attempt','site','active','token','2026-10-01','2026-10-01','2026-10-01')"
            )
    with pytest.raises(RuntimeError, match=r"explicit repository|idle cluster"):
        _upgrade(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_capabilities)")}
        assert "alias" in columns and "repo_id" not in columns
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0019_remove_db_legacies",
        )


def _run_alembic(path: Path, operation: str, revision: str) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    variable = "TOPODROME_DATABASE_DSN"
    previous = os.environ.get(variable)
    os.environ[variable] = f"sqlite+aiosqlite:///{path}"
    try:
        getattr(command, operation)(config, revision)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def _upgrade(path: Path, revision: str = "head") -> None:
    _run_alembic(path, "upgrade", revision)


def _stamp(path: Path, revision: str) -> None:
    _run_alembic(path, "stamp", revision)


def _assert_sqlite_schema_matches_models(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    try:
        assert_schema_matches_models(engine)
    finally:
        engine.dispose()


def test_incomplete_stamped_0001_schema_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE jobs (id VARCHAR(80) PRIMARY KEY);
            CREATE TABLE attempts (id VARCHAR(80) PRIMARY KEY);
            CREATE TABLE kernel_sessions (id VARCHAR(80) PRIMARY KEY);
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL PRIMARY KEY
            );
            INSERT INTO alembic_version(version_num) VALUES ('0001_initial');
            """
        )

    with pytest.raises(RuntimeError, match="missing original tables"):
        _upgrade(path)

    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    # SQLite DDL is non-transactional, so earlier revisions may have advanced.
    # The parity revision itself must never be stamped onto an incomplete DB.
    assert revision == ("0008_model_replicas",)


def test_initial_revision_is_frozen_original_schema(tmp_path: Path) -> None:
    migration_source = (PROJECT_ROOT / "migrations/versions/0001_initial.py").read_text()
    assert "topodrome.db.models" not in migration_source
    assert "Base.metadata" not in migration_source

    path = tmp_path / "initial.sqlite3"
    _upgrade(path, "0001_initial")

    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        job_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        kernel_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(kernel_sessions)")
        }
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}

    assert revision == ("0001_initial",)
    assert "retry_not_before" not in job_columns
    assert {"attempt_id", "runtime_id", "actor_id", "access_generation"}.isdisjoint(kernel_columns)
    assert "kernel_channel_requests" not in tables


def test_array_upgrade_preserves_existing_jobs_and_marks_unknown_timing(tmp_path):
    path = tmp_path / "pre-arrays.sqlite3"
    _upgrade(path, "0017_adaptive_inputs")
    with sqlite3.connect(path) as connection:
        for state in ("queued", "running", "succeeded"):
            connection.execute(
                "INSERT INTO jobs (id, state, spec, priority, cancel_requested, "
                "created_at, updated_at) VALUES (?, ?, '{}', 0, 0, ?, ?)",
                (f"job_{state}", state, "2026-09-26 00:00:00", "2026-09-26 01:00:00"),
            )
    _upgrade(path)
    _assert_sqlite_schema_matches_models(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT state, array_id, phase, phase_started_at, phase_seconds, eligible_since "
            "FROM jobs ORDER BY state"
        ).fetchall()
    assert len(rows) == 3
    for state, array_id, phase, started, seconds, eligible in rows:
        assert array_id is None and phase == "untracked" and json.loads(seconds) == {}
        assert started == "2026-09-26 01:00:00"
        assert (eligible is not None) is (state == "queued")


def test_falsely_stamped_0008_schema_is_repaired_to_full_parity(tmp_path: Path) -> None:
    path = tmp_path / "stamped.sqlite3"
    _upgrade(path, "0001_initial")
    _stamp(path, "0008_model_replicas")

    _upgrade(path)

    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert revision == (CURRENT_SCHEMA_REVISION,)
    _assert_sqlite_schema_matches_models(path)


def test_schema_parity_rejects_an_incompatible_sql_type(tmp_path: Path) -> None:
    path = tmp_path / "wrong-type.sqlite3"
    _upgrade(path, "0001_initial")
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            ALTER TABLE jobs RENAME COLUMN priority TO legacy_priority;
            ALTER TABLE jobs ADD COLUMN priority TEXT NOT NULL DEFAULT '0';
            """
        )
    _stamp(path, "0008_model_replicas")

    with pytest.raises(RuntimeError, match=r"jobs\.priority.*SQL type"):
        _upgrade(path)

    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert revision == ("0008_model_replicas",)


def test_schema_parity_rejects_an_unconditional_actor_allocation_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-index.sqlite3"
    _upgrade(path, "0001_initial")
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            DROP INDEX uq_active_actor_gpu;
            CREATE UNIQUE INDEX uq_active_actor_gpu
                ON model_actors (site_id, gpu_uuid);
            """
        )
    _stamp(path, "0008_model_replicas")

    with pytest.raises(RuntimeError, match=r"index uq_active_actor_gpu.*incompatible"):
        _upgrade(path)

    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert revision == ("0008_model_replicas",)


def test_fresh_upgrade_has_every_model_table_constraint_and_index(tmp_path: Path) -> None:
    path = tmp_path / "fresh.sqlite3"

    _upgrade(path)

    with sqlite3.connect(path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == (CURRENT_SCHEMA_REVISION,)
    _assert_sqlite_schema_matches_models(path)


def test_multi_user_migration_backfills_legacy_resources_to_operator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-owners.sqlite3"
    _upgrade(path, "0011_timed_gpu_reservations")
    timestamp = "2026-09-12 00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, state, spec, idempotency_key, priority, cancel_requested,
                retry_not_before, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-legacy",
                "queued",
                "{}",
                "legacy-key",
                0,
                0,
                None,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO kernel_sessions (
                id, state, mode, request, job_id, attempt_id, site_id,
                relay_id, runtime_id, actor_id, access_generation, connect_url,
                created_at, last_activity_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "kernel-legacy",
                "requested",
                "service",
                "{}",
                "job-legacy",
                None,
                None,
                "relay-legacy",
                None,
                None,
                0,
                None,
                timestamp,
                timestamp,
                None,
            ),
        )

    # This historical revision backfills ownership, not the later cleanup's
    # no-unscoped-keys prerequisite.
    _upgrade(path, "0012_multi_user_ownership")

    with sqlite3.connect(path) as connection:
        job_owner = connection.execute(
            "SELECT owner_username FROM jobs WHERE id = 'job-legacy'"
        ).fetchone()
        kernel_owner = connection.execute(
            "SELECT owner_username FROM kernel_sessions WHERE id = 'kernel-legacy'"
        ).fetchone()
    assert job_owner == ("@operator",)
    assert kernel_owner == ("@operator",)


def _seed_pre_cleanup_database(path: Path) -> None:
    _upgrade(path, "0018_job_arrays")
    timestamp = "2026-09-27 00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO jobs (id, state, spec, idempotency_key, priority, "
            "cancel_requested, created_at, updated_at) "
            "VALUES ('job-cleanup', 'succeeded', '{}', ?, 7, 0, ?, ?)",
            ("owner-v1:" + "a" * 64, timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO attempts (id, job_id, number, state, created_at, updated_at) "
            "VALUES ('attempt-cleanup', 'job-cleanup', 1, 'succeeded', ?, ?)",
            (timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO kernel_sessions (id, state, mode, request, job_id, attempt_id, "
            "access_generation, created_at, last_activity_at) VALUES "
            "('kernel-cleanup', 'stopped', 'service', '{}', 'job-cleanup', "
            "'attempt-cleanup', 3, ?, ?)",
            (timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO provider_operations (id, provider, action, state, "
            "idempotency_key, request_redacted, created_at, updated_at) "
            "VALUES ('operation-cleanup', 'autodl', 'power_off', 'applied', "
            "'cleanup-audit', '{}', ?, ?)",
            (timestamp, timestamp),
        )


def test_database_cleanup_preserves_records_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "cleanup.sqlite3"
    _seed_pre_cleanup_database(path)
    with sqlite3.connect(path) as connection:
        queries = (
            "SELECT * FROM jobs",
            "SELECT * FROM attempts",
            "SELECT * FROM provider_operations",
            "SELECT id, state, owner_username, job_id, attempt_id, access_generation "
            "FROM kernel_sessions",
        )
        before = [connection.execute(query).fetchall() for query in queries]

    _upgrade(path, "0019_remove_db_legacies")
    _run_alembic(path, "downgrade", "0018_job_arrays")
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM log_indexes").fetchone() == (0,)
        assert connection.execute("SELECT connect_url FROM kernel_sessions").fetchall() == [(None,)]
        assert [connection.execute(query).fetchall() for query in queries] == before

    _upgrade(path)
    _assert_sqlite_schema_matches_models(path)
    with sqlite3.connect(path) as connection:
        assert [connection.execute(query).fetchall() for query in queries] == before


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            "INSERT INTO log_indexes (job_id, attempt_id, logical_path, start_offset, "
            "end_offset) VALUES ('job-cleanup', 'attempt-cleanup', 'attempt.log', 0, 42)",
            "log-index rows",
        ),
        (
            "UPDATE kernel_sessions SET connect_url = 'https://example.invalid/kernel'",
            "stored kernel URLs",
        ),
        ("UPDATE jobs SET idempotency_key = 'unscoped-key'", "unscoped job idempotency keys"),
    ],
)
def test_database_cleanup_refuses_unexpected_data_before_ddl(
    tmp_path: Path, mutation: str, error: str,
) -> None:
    path = tmp_path / "cleanup-refused.sqlite3"
    _seed_pre_cleanup_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
        before = list(connection.iterdump())

    with pytest.raises(RuntimeError, match=error):
        _upgrade(path)

    with sqlite3.connect(path) as connection:
        assert list(connection.iterdump()) == before


def test_global_asset_migration_renames_site_cache_telemetry(tmp_path: Path) -> None:
    path = tmp_path / "asset-cache-telemetry.sqlite3"
    _upgrade(path, "0012_multi_user_ownership")

    with sqlite3.connect(path) as connection:
        old_columns = {row[1] for row in connection.execute("PRAGMA table_info(sites)")}
    assert "cached_datasets" in old_columns
    assert "cached_assets" not in old_columns

    _upgrade(path)

    with sqlite3.connect(path) as connection:
        new_columns = {row[1] for row in connection.execute("PRAGMA table_info(sites)")}
    assert "cached_assets" in new_columns
    assert "cached_datasets" not in new_columns


def test_job_asset_migration_normalizes_empty_legacy_dataset_specs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "job-assets.sqlite3"
    _upgrade(path, "0013_global_assets")
    timestamp = "2026-09-13 00:00:00+00:00"
    specs = {
        "job-list": {"entrypoint": "list.py", "datasets": []},
        "job-object": {"entrypoint": "object.py", "datasets": {}},
        "job-existing-assets": {
            "entrypoint": "assets.py",
            "datasets": [],
            "assets": {"lens": "asset://shared/lens@v1"},
        },
        "job-no-legacy-key": {"entrypoint": "current.py"},
    }
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO jobs (
                id, state, spec, idempotency_key, priority, cancel_requested,
                retry_not_before, created_at, updated_at, owner_username
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    job_id,
                    "queued",
                    json.dumps(spec),
                    None,
                    0,
                    0,
                    None,
                    timestamp,
                    timestamp,
                    "@operator",
                )
                for job_id, spec in specs.items()
            ],
        )

    _upgrade(path)

    with sqlite3.connect(path) as connection:
        migrated = {
            job_id: json.loads(spec)
            for job_id, spec in connection.execute("SELECT id, spec FROM jobs")
        }
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == (CURRENT_SCHEMA_REVISION,)
    assert all("datasets" not in spec for spec in migrated.values())
    assert migrated["job-list"]["assets"] == {}
    assert migrated["job-object"]["assets"] == {}
    assert migrated["job-existing-assets"]["assets"] == {"lens": "asset://shared/lens@v1"}
    assert migrated["job-no-legacy-key"]["assets"] == {}


@pytest.mark.parametrize(
    "legacy",
    [
        ["mmlu@v1"],
        {"mmlu": "v1"},
        None,
        "mmlu@v1",
    ],
)
def test_job_asset_migration_fails_closed_on_ambiguous_legacy_datasets(
    tmp_path: Path,
    legacy: object,
) -> None:
    path = tmp_path / "ambiguous-job-assets.sqlite3"
    _upgrade(path, "0013_global_assets")
    timestamp = "2026-09-13 00:00:00+00:00"
    original = {"entrypoint": "ambiguous.py", "datasets": legacy}
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, state, spec, idempotency_key, priority, cancel_requested,
                retry_not_before, created_at, updated_at, owner_username
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-ambiguous",
                "queued",
                json.dumps(original),
                None,
                0,
                0,
                None,
                timestamp,
                timestamp,
                "@operator",
            ),
        )

    with pytest.raises(RuntimeError, match="cannot convert non-empty legacy datasets"):
        _upgrade(path)

    with sqlite3.connect(path) as connection:
        stored = connection.execute("SELECT spec FROM jobs WHERE id = 'job-ambiguous'").fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert stored is not None
    assert json.loads(stored[0]) == original
    assert revision == ("0013_global_assets",)
