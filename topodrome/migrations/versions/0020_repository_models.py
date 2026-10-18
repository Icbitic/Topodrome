"""Replace model aliases with registered repository/snapshot addresses.

This is a one-time data conversion, not an alias compatibility interface.
Run under the cluster upgrade fence after all owned GPU work has stopped.
Unknown names fail preflight without discarding history or guessing identity.
"""

import hashlib
import json

import sqlalchemy as sa
from alembic import op

revision = "0020_repository_models"
down_revision = "0019_remove_db_legacies"
branch_labels = None
depends_on = None

_REPOSITORIES = {
    "meta-llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "olmo-2-1124-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
}


def _repo_id(value):
    if value in _REPOSITORIES:
        return _REPOSITORIES[value]
    if isinstance(value, str) and len(value.split("/")) == 2:
        return value
    raise RuntimeError(f"model address conversion needs an explicit repository for {value!r}")


def _model(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {"repo_id": _repo_id(value), "revision": None, "subfolder": None}
    result = dict(value)
    if "alias" in result:
        if "repo_id" in result:
            raise RuntimeError("conflicting model alias/repository fields")
        result["repo_id"] = _repo_id(result.pop("alias"))
    if "repo_id" in result:
        result["repo_id"] = _repo_id(result["repo_id"])
        result.setdefault("subfolder", None)
        if result.get("revision") == "configured":
            result["revision"] = None
    return result


def _json_rows(connection, name, column):
    table = sa.table(name, sa.column("id"), sa.column(column, sa.JSON))
    return table, list(connection.execute(sa.select(table.c.id, table.c[column])).all())


def upgrade():
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        op.execute("SET LOCAL lock_timeout = '5s'")
        op.execute("SET LOCAL statement_timeout = '60s'")
        op.execute(
            "LOCK TABLE jobs, kernel_sessions, model_capabilities, model_actors, leases "
            "IN ACCESS EXCLUSIVE MODE"
        )
    if (
        connection.execute(
            sa.text("SELECT 1 FROM leases WHERE state IN ('offered','accepted','active') LIMIT 1")
        ).first()
        is not None
    ):
        raise RuntimeError("model address conversion requires an idle cluster upgrade fence")

    updates = []
    for name, column in (("jobs", "spec"), ("kernel_sessions", "request")):
        table, rows = _json_rows(connection, name, column)
        for identifier, payload in rows:
            if "model" not in payload:
                continue
            converted = {**payload, "model": _model(payload["model"])}
            updates.append((table, identifier, {column: converted}))
    table, rows = _json_rows(connection, "model_actors", "key_data")
    table = sa.table(
        "model_actors", sa.column("id"), sa.column("key_data", sa.JSON), sa.column("key_hash")
    )
    for identifier, payload in rows:
        converted = _model(payload)
        digest = hashlib.sha256(
            json.dumps(converted, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        updates.append((table, identifier, {"key_data": converted, "key_hash": digest}))

    capability = sa.table("model_capabilities", sa.column("id"), sa.column("alias"))
    identities = [
        (identifier, _repo_id(alias))
        for identifier, alias in connection.execute(sa.select(capability.c.id, capability.c.alias))
    ]
    # Preflight every JSON/column value before changing any schema, including
    # SQLite deployments whose DDL is not fully transactional.
    for table, identifier, values in updates:
        connection.execute(table.update().where(table.c.id == identifier).values(**values))
    for identifier, repo_id in identities:
        connection.execute(
            capability.update().where(capability.c.id == identifier).values(alias=repo_id)
        )

    op.drop_index("ix_model_capabilities_alias", table_name="model_capabilities")
    with op.batch_alter_table("model_capabilities") as batch:
        batch.drop_constraint("uq_model_capability_identity", type_="unique")
        batch.alter_column("alias", new_column_name="repo_id", existing_type=sa.String(200))
        batch.add_column(sa.Column("subfolder", sa.String(500), nullable=False, server_default=""))
    # Reflect the renamed columns before constructing the replacement constraint
    # (Alembic's SQLite batch copier otherwise drops a constraint on a new name).
    with op.batch_alter_table("model_capabilities") as batch:
        batch.create_unique_constraint(
            "uq_model_capability_identity",
            [
                "site_id",
                "repo_id",
                "revision",
                "subfolder",
                "backend",
                "backend_version",
                "dtype",
                "quantization",
                "adapter",
                "gpu_uuid",
            ],
        )
    op.create_index("ix_model_capabilities_repo_id", "model_capabilities", ["repo_id"])


def downgrade():
    # The schema/data rename intentionally has no lossy reverse mapping: new
    # registered repositories and snapshots may not have an old alias at all.
    raise RuntimeError(
        "restore the pre-upgrade database/config backup to roll back model addresses"
    )
