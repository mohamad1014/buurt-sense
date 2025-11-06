"""A lightweight FastAPI-compatible application used for testing."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, get_type_hints
from uuid import UUID

from .exceptions import HTTPException
from .responses import Response as ResponseType


@dataclass(slots=True)
class Route:
    method: str
    path_template: List[tuple[bool, str]]
    handler: Callable[..., Any]
    status_code: int


@dataclass(slots=True)
class AppResponse:
    status_code: int
    body: Any
    media_type: str
    headers: Dict[str, str]


class Request:
    """Simplified request object passed to route handlers."""

    def __init__(self, app: "FastAPI", body: Any | None = None) -> None:
        self.app = app
        self.state = app.state
        self._body = body

    def json(self) -> Any:
        return self._body


class FastAPI:
    """Minimal stand-in for FastAPI supporting routing and testing."""

    def __init__(self, title: str | None = None) -> None:
        self.title = title or "FastAPI"
        self.state = SimpleNamespace()
        self._routes: List[Route] = []
        self._mounts: List[Tuple[str, Any]] = []

    # Route registration helpers -------------------------------------------------
    def get(self, path: str, *, status_code: int = 200, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register(path, "GET", status_code)

    def post(self, path: str, *, status_code: int = 200, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register(path, "POST", status_code)

    def _register(self, path: str, method: str, status_code: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            route = Route(method=method, path_template=self._parse_path(path), handler=func, status_code=status_code)
            self._routes.append(route)
            return func

        return decorator

    def mount(self, path: str, app: Any, name: str | None = None) -> None:  # noqa: ARG002
        prefix = path.rstrip("/") or "/"
        self._mounts.append((prefix, app))

    # Request handling -----------------------------------------------------------
    def handle_request(self, method: str, path: str, body: Any | None = None) -> AppResponse:
        mount_response = self._handle_mount(method, path)
        if mount_response is not None:
            return mount_response
        for route in self._routes:
            if route.method != method:
                continue
            match = self._match_path(route.path_template, path)
            if match is None:
                continue
            request = Request(self, body)
            try:
                result = self._invoke_handler(route.handler, match, request, body)
                return self._build_app_response(result, route.status_code)
            except HTTPException as exc:
                return AppResponse(
                    status_code=exc.status_code,
                    body={"detail": exc.detail},
                    media_type="application/json",
                    headers={},
                )
            except ValueError as exc:
                return AppResponse(
                    status_code=422,
                    body={"detail": str(exc)},
                    media_type="application/json",
                    headers={},
                )
        return AppResponse(
            status_code=404,
            body={"detail": "Not Found"},
            media_type="application/json",
            headers={},
        )

    def _handle_mount(self, method: str, path: str) -> AppResponse | None:
        for prefix, app in self._mounts:
            if prefix == "/":
                match_path = path
            elif path.startswith(prefix + "/") or path == prefix:
                match_path = path[len(prefix) :]
            else:
                continue

            if hasattr(app, "handle"):
                response = app.handle(method, match_path)
            elif hasattr(app, "handle_request"):
                response = app.handle_request(method, match_path, None)
            else:
                continue

            return self._build_app_response(response, 200 if not isinstance(response, ResponseType) else response.status_code)
        return None

    # Internal helpers -----------------------------------------------------------
    def _parse_path(self, path: str) -> List[tuple[bool, str]]:
        segments = [segment for segment in path.strip("/").split("/") if segment]
        if not segments:
            return []
        parsed: List[tuple[bool, str]] = []
        for segment in segments:
            if segment.startswith("{") and segment.endswith("}"):
                parsed.append((True, segment[1:-1]))
            else:
                parsed.append((False, segment))
        return parsed

    def _match_path(self, template: List[tuple[bool, str]], path: str) -> Optional[Dict[str, str]]:
        request_segments = [segment for segment in path.strip("/").split("/") if segment]
        if len(template) != len(request_segments):
            return None
        params: Dict[str, str] = {}
        for (is_param, value), segment in zip(template, request_segments):
            if is_param:
                params[value] = segment
            elif value != segment:
                return None
        return params

    def _invoke_handler(
        self,
        handler: Callable[..., Any],
        path_params: Mapping[str, str],
        request: Request,
        body: Any | None,
    ) -> Any:
        signature = inspect.signature(handler)
        type_hints = get_type_hints(handler)
        kwargs: Dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            annotation = type_hints.get(name, parameter.annotation)
            if name in path_params:
                kwargs[name] = self._convert_parameter(path_params[name], annotation)
            elif annotation is Request or name == "request":
                kwargs[name] = request
            elif parameter.default is not inspect._empty:
                kwargs[name] = parameter.default
            else:
                raise ValueError(f"Unable to resolve argument '{name}' for handler {handler.__name__}")
        return handler(**kwargs)

    def _convert_parameter(self, value: str, annotation: Any) -> Any:
        if isinstance(annotation, str):
            if annotation == "UUID":
                annotation = UUID
            elif annotation == "int":
                annotation = int
            elif annotation == "float":
                annotation = float
            else:
                annotation = str
        if annotation is inspect._empty or annotation is str:
            return value
        if annotation is UUID:
            return UUID(value)
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)
        if getattr(annotation, "__origin__", None) is Optional:
            args = annotation.__args__
            if value is None:
                return None
            return self._convert_parameter(value, args[0])
        return annotation(value)  # type: ignore[call-arg]

    def _serialise(self, value: Any) -> Any:
        if isinstance(value, (str, int, float)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {key: self._serialise(val) for key, val in value.items()}
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            return [self._serialise(item) for item in value]
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        return value

    def _build_app_response(self, result: Any, default_status: int) -> AppResponse:
        if isinstance(result, ResponseType):
            response = result.with_defaults(default_status)
            headers = dict(response.headers)
            headers.setdefault("content-type", response.media_type)
            return AppResponse(
                status_code=response.status_code,
                body=response.content,
                media_type=response.media_type,
                headers=headers,
            )

        serialised = self._serialise(result)
        headers = {"content-type": "application/json"}
        return AppResponse(status_code=default_status, body=serialised, media_type="application/json", headers=headers)
