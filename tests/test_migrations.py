"""Smoke tests for database schema migrations."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect

from buurtsense.db import create_engine, init_db


def _list_tables(connection) -> list[str]:
    """Return all table names present in the connected database."""

    inspector = inspect(connection)
    return inspector.get_table_names()


@pytest.mark.asyncio
async def test_alembic_migrations_initialize_schema(tmp_path: Path) -> None:
    """Applying migrations should create the expected tables on an empty database."""

    db_file = tmp_path / "alembic-smoke.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_file}")

    try:
        await init_db(engine)
        async with engine.begin() as conn:
            tables = await conn.run_sync(_list_tables)
    finally:
        await engine.dispose()

    assert {"recording_sessions", "segments", "detections"}.issubset(set(tables))
