"""Minimal test client compatible with the FastAPI-like app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

from .app import AppResponse, FastAPI


@dataclass(slots=True)
class Response:
    status_code: int
    _body: Any

    def json(self) -> Any:
        return self._body


class TestClient:
    __test__ = False

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    # Context manager support ---------------------------------------------------
    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    # Request helpers -----------------------------------------------------------
    def get(self, path: str) -> Response:
        return self._request("GET", path, None)

    def post(self, path: str, json: Optional[Mapping[str, Any]] = None) -> Response:
        body = dict(json) if json is not None else None
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: Optional[MutableMapping[str, Any]] = None) -> Response:
        app_response: AppResponse = self.app.handle_request(method, path, body)
        return Response(status_code=app_response.status_code, _body=app_response.body)
