"""Minimal response primitives used by the test FastAPI shim."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping


@dataclass(slots=True)
class Response:
    """Basic HTTP response representation."""

    content: str | bytes
    status_code: int = 200
    media_type: str = "application/json"
    headers: MutableMapping[str, str] = field(default_factory=dict)

    def with_defaults(self, default_status: int) -> "Response":
        """Return a copy that respects default status when unset."""

        status = self.status_code or default_status
        return Response(
            content=self.content,
            status_code=status,
            media_type=self.media_type,
            headers=dict(self.headers),
        )


class HTMLResponse(Response):
    """Convenience response for HTML documents."""

    def __init__(self, content: str, status_code: int = 200, headers: Mapping[str, str] | None = None) -> None:
        merged_headers = dict(headers or {})
        super().__init__(content=content, status_code=status_code, media_type="text/html; charset=utf-8", headers=merged_headers)
