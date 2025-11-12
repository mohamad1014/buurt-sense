"""Alembic environment configuration for Buurt Sense."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import URL, make_url

from buurtsense.db.base import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _coerce_to_sync_url(raw_url: str) -> str:
    """Return a synchronous SQLAlchemy URL string for migrations."""

    url: URL = make_url(raw_url)
    backend = url.get_backend_name()
    if url.drivername != backend:
        url = url.set(drivername=backend)
    return url.render_as_string(hide_password=False)


def _configure_sqlalchemy_url() -> None:
    """Propagate the runtime database URL to Alembic's configuration."""

    raw_url = os.getenv("BUURTSENSE_DB_URL") or config.get_main_option("sqlalchemy.url")
    if not raw_url:
        raise RuntimeError("No database URL configured for Alembic migrations.")
    config.set_main_option("sqlalchemy.url", _coerce_to_sync_url(raw_url))


def get_metadata():  # type: ignore[override]
    """Expose the project's metadata to Alembic autogeneration hooks."""

    return Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""

    _configure_sqlalchemy_url()
    target_metadata = get_metadata()

    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    _configure_sqlalchemy_url()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
