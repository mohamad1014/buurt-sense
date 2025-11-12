"""Command-line helper to run pending Alembic migrations."""

import asyncio
from pathlib import Path

from buurtsense.db import create_engine, init_db


async def main() -> None:
    """Apply migrations against the configured database URL."""

    engine = create_engine()
    await init_db(engine)
    db_path = Path(engine.url.database or "buurtsense.db")
    print(f"Initialized database at {db_path.resolve()}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
