from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Iterable

import pytest
from starlette.types import Message

from app import create_app
from buurtsense.storage import SessionStorage


@dataclass(slots=True)
class SimpleResponse:
    """Minimal HTTP response wrapper used for tests."""

    status_code: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")


class SimpleClient:
    """Tiny ASGI client that exercises the FastAPI app without httpx."""

    def __init__(self, app) -> None:
        self._app = app
        self._lifespan = app.router.lifespan_context(app)
        self._started = False
        self._loop = asyncio.new_event_loop()

    def start(self) -> None:
        if not self._started:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._lifespan.__aenter__())
            self._started = True

    def close(self) -> None:
        if self._started:
            self._loop.run_until_complete(self._lifespan.__aexit__(None, None, None))
            self._started = False
        self._loop.run_until_complete(asyncio.sleep(0))
        self._loop.close()
        asyncio.set_event_loop(None)

    def get(self, path: str) -> SimpleResponse:
        return self._request("GET", path)

    def post(self, path: str) -> SimpleResponse:
        return self._request("POST", path)

    def _request(self, method: str, path: str) -> SimpleResponse:
        return self._loop.run_until_complete(self._perform_request(method, path))

    async def _perform_request(self, method: str, path: str) -> SimpleResponse:
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        messages: list[Message] = []

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            messages.append(message)

        await self._app(scope, receive, send)

        status_code = 500
        headers: Iterable[tuple[bytes, bytes]] = []
        body_chunks: list[bytes] = []

        for message in messages:
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                body_chunks.append(message.get("body", b""))

        header_map = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in headers
        }
        body = b"".join(body_chunks)
        return SimpleResponse(status_code=status_code, headers=header_map, body=body)


@pytest.fixture()
def client(tmp_path) -> Iterable[SimpleClient]:
    """Provide a lightweight ASGI client backed by a fresh app instance for each test."""

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    storage = SessionStorage(db_url=db_url)
    app = create_app(session_storage=storage, segment_interval=0.05)
    client = SimpleClient(app)
    client.start()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture()
def anyio_backend() -> str:
    """Restrict ``pytest-anyio`` to the asyncio backend."""

    return "asyncio"
