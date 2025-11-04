"""Minimal FastAPI-compatible interface for tests."""
from .app import FastAPI, Request
from .exceptions import HTTPException
from . import status
from .testclient import TestClient

__all__ = ["FastAPI", "HTTPException", "Request", "status", "TestClient"]
