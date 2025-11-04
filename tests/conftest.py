"""Shared pytest fixtures for API tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import create_app


@pytest.fixture()
def client() -> TestClient:
    """Provide a TestClient backed by a fresh app instance for each test."""

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
