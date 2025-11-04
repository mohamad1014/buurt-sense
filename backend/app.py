"""FastAPI application factory for the Buurt Sense backend."""
from __future__ import annotations

from fastapi import FastAPI

from .api.sessions import router as sessions_router
from .session_store import InMemorySessionStore, SessionStore


def create_app(session_store: SessionStore | None = None) -> FastAPI:
    """Create a FastAPI instance configured with dependencies and routes."""

    app = FastAPI(title="Buurt Sense Backend", version="0.1.0")
    app.state.session_store = session_store or InMemorySessionStore()

    app.include_router(sessions_router)

    return app


__all__ = ["create_app"]
