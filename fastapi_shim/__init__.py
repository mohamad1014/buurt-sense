"""Buurt Sense FastAPI shim.

This lightweight compatibility layer purposefully avoids importing the real
``fastapi`` package so local development and tests do not require the heavy
dependency while still providing the subset of functionality the project needs.
"""
from __future__ import annotations

from . import status  # noqa: F401
from .app import FastAPI, Request  # noqa: F401
from .exceptions import HTTPException  # noqa: F401
from .responses import HTMLResponse, Response  # noqa: F401
from .staticfiles import StaticFiles  # noqa: F401
from .testclient import TestClient  # noqa: F401

__all__ = [
    "FastAPI",
    "HTTPException",
    "HTMLResponse",
    "Request",
    "Response",
    "StaticFiles",
    "status",
    "TestClient",
]
