"""Very small static file handler used by the FastAPI shim."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from .responses import Response


class StaticFiles:
    """Serve files from a directory under a mounted path."""

    def __init__(self, directory: str | Path, html: bool = False) -> None:
        self.directory = Path(directory).resolve()
        self.html = html
        if not self.directory.exists():
            raise RuntimeError(f"Static directory '{self.directory}' does not exist")

    def handle(self, method: str, path: str) -> Response:
        """Return a response for the requested asset path."""

        if method != "GET":
            return Response(content="Method not allowed", status_code=405, media_type="text/plain")

        relative = path.lstrip("/")
        target = (self.directory / relative).resolve()
        if self.html and not relative:
            target = (self.directory / "index.html").resolve()

        if not str(target).startswith(str(self.directory)) or not target.exists():
            return Response(content="Not Found", status_code=404, media_type="text/plain")

        if target.is_dir():
            index_file = target / "index.html"
            if self.html and index_file.exists():
                target = index_file
            else:
                return Response(content="Not Found", status_code=404, media_type="text/plain")

        content = target.read_text(encoding="utf-8")
        media_type = _guess_media_type(target.suffix.lstrip("."))
        return Response(content=content, status_code=200, media_type=media_type)


def _guess_media_type(extension: str) -> str:
    mapping: Dict[str, str] = {
        "html": "text/html; charset=utf-8",
        "css": "text/css; charset=utf-8",
        "js": "application/javascript; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }
    return mapping.get(extension.lower(), "text/plain; charset=utf-8")
