"""Minimal FastAPI-compatible interface for tests."""
from .app import FastAPI, Request
from .exceptions import HTTPException
from . import status
from .testclient import TestClient
from .responses import HTMLResponse, Response
from .staticfiles import StaticFiles

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
