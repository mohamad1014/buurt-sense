"""Backend package for the Buurt Sense FastAPI service."""
from __future__ import annotations

import os
from typing import Any

import uvicorn

from .app import create_app
from .session_store import InMemorySessionStore, SessionStore


def main(**uvicorn_kwargs: Any) -> None:
    """Run the Buurt Sense service using ``uvicorn``.

    Parameters
    ----------
    **uvicorn_kwargs: Any
        Optional keyword arguments forwarded to :func:`uvicorn.run`.
    """

    host = os.environ.get("BUURT_SENSE_HOST", "0.0.0.0")
    port = int(os.environ.get("BUURT_SENSE_PORT", "8000"))

    config = {
        "app": "backend.app:create_app",
        "factory": True,
        "host": host,
        "port": port,
        "reload": os.environ.get("BUURT_SENSE_RELOAD", "false").lower() == "true",
    }
    config.update(uvicorn_kwargs)

    # Default to the in-memory session store when running via the CLI helper.
    uvicorn.run(**config)


__all__ = ["create_app", "main", "SessionStore", "InMemorySessionStore"]
