"""Async database session helpers."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DEFAULT_DB_URL = "sqlite+aiosqlite:///./buurtsense.db"


def create_engine(db_url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    """Create an :class:`AsyncEngine` configured for SQLite concurrency."""

    runtime_url = db_url or os.getenv("BUURTSENSE_DB_URL") or DEFAULT_DB_URL
    engine = create_async_engine(runtime_url, echo=echo, future=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    return engine


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return an async sessionmaker bound to ``engine``."""

    return async_sessionmaker(engine, expire_on_commit=False)


def _alembic_config(engine: AsyncEngine) -> Config:
    """Return a configured Alembic :class:`Config` for ``engine``."""

    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "alembic.ini"
    if not config_path.exists():
        raise FileNotFoundError(
            "alembic.ini could not be located; ensure migrations are configured."
        )

    config = Config(str(config_path))
    sync_url = _render_sync_url(engine.url)
    config.set_main_option("sqlalchemy.url", sync_url)
    return config


def _render_sync_url(url: URL) -> str:
    """Convert an async database URL to a synchronous driver for Alembic."""

    backend = url.get_backend_name()
    if url.drivername != backend:
        url = url.set(drivername=backend)
    return url.render_as_string(hide_password=False)


async def init_db(engine: AsyncEngine) -> None:
    """Apply all pending Alembic migrations."""

    config = _alembic_config(engine)
    command.upgrade(config, "head")


@asynccontextmanager
async def session_scope(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope around a series of operations."""

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
