from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import Connection, Engine, inspect
from sqlalchemy.schema import ForeignKeyConstraint, UniqueConstraint

from topodrome.db.models import Base


def _normalize_predicate(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[\s()`\"']", "", str(value)).lower()
    # PostgreSQL reflects ``column IN ('a', 'b')`` on a varchar-backed enum as
    # ``column::text = ANY (ARRAY['a'::varchar, 'b'::varchar]::text[])``.
    # Normalize that server rewrite back to the declarative predicate so the
    # parity check compares semantics rather than PostgreSQL pretty-printing.
    normalized = normalized.replace("::charactervarying", "")
    normalized = normalized.replace("::text[]", "").replace("::text", "")
    any_array = re.fullmatch(r"([a-z0-9_]+)=anyarray\[(.*)]", normalized)
    if any_array is not None:
        normalized = f"{any_array.group(1)}in{any_array.group(2)}"
    return normalized


def _foreign_key_signatures(
    constraints: Iterable[Mapping[str, Any]],
) -> set[tuple[tuple[str, ...], str, tuple[str, ...], str]]:
    return {
        (
            tuple(str(column) for column in constraint["constrained_columns"]),
            str(constraint["referred_table"]),
            tuple(str(column) for column in constraint["referred_columns"]),
            str((constraint.get("options") or {}).get("ondelete", "")).upper(),
        )
        for constraint in constraints
    }


def assert_schema_matches_models(connectable: Connection | Engine) -> None:
    """Require every ORM table, column, constraint, and index in a migrated DB."""

    inspector = inspect(connectable)
    assert set(inspector.get_table_names()) - {"alembic_version"} == set(
        Base.metadata.tables
    )

    for name, expected_table in Base.metadata.tables.items():
        actual_columns = {
            str(column["name"]): column for column in inspector.get_columns(name)
        }
        assert set(actual_columns) == set(expected_table.columns.keys()), name
        for column in expected_table.columns:
            actual_column = actual_columns[column.name]
            assert bool(actual_column["nullable"]) is column.nullable, (name, column.name)
            assert actual_column["type"]._type_affinity is column.type._type_affinity, (
                name,
                column.name,
                actual_column["type"],
                column.type,
            )

        actual_primary_key = tuple(
            str(column)
            for column in inspector.get_pk_constraint(name)["constrained_columns"]
        )
        assert actual_primary_key == tuple(column.name for column in expected_table.primary_key)

        expected_unique = {
            tuple(column.name for column in constraint.columns)
            for constraint in expected_table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        actual_unique = {
            tuple(str(column) for column in constraint["column_names"])
            for constraint in inspector.get_unique_constraints(name)
        }
        assert actual_unique == expected_unique, name

        expected_foreign = {
            (
                tuple(element.parent.name for element in constraint.elements),
                constraint.elements[0].column.table.name,
                tuple(element.column.name for element in constraint.elements),
                str(constraint.ondelete or "").upper(),
            )
            for constraint in expected_table.constraints
            if isinstance(constraint, ForeignKeyConstraint)
        }
        assert _foreign_key_signatures(inspector.get_foreign_keys(name)) == expected_foreign, name

        expected_indexes = {
            index.name: (
                tuple(column.name for column in index.columns),
                bool(index.unique),
                _normalize_predicate(
                    index.dialect_options[connectable.dialect.name].get("where")
                ),
            )
            for index in expected_table.indexes
        }
        actual_indexes = {
            str(index["name"]): (
                tuple(str(column) for column in index["column_names"]),
                bool(index["unique"]),
                _normalize_predicate(
                    (index.get("dialect_options") or {}).get(
                        f"{connectable.dialect.name}_where"
                    )
                ),
            )
            for index in inspector.get_indexes(name)
            if index.get("name") is not None and index.get("duplicates_constraint") is None
        }
        assert actual_indexes == expected_indexes, name
