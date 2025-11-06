"""FastAPI compatibility layer.

This package ships a lightweight fallback implementation tailored for the test
suite so contributors can run the project without the real `fastapi` dependency
installed. When the genuine library is available (for example when running the
development server), we dynamically import and expose it instead of the stub to
guarantee full ASGI behaviour.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys


def _load_real_fastapi() -> ModuleType | None:
    """Return the actual FastAPI package if it exists on sys.path."""

    current_file = Path(__file__).resolve()
    shim_module = sys.modules.get(__name__)
    for path_entry in list(sys.path):
        try:
            candidate = Path(path_entry).resolve() / "fastapi" / "__init__.py"
        except OSError:
            # Some path entries (e.g. invalid encodings) can raise; ignore them.
            continue

        if not candidate.exists() or candidate == current_file:
            continue

        spec = spec_from_file_location(__name__, candidate)
        if spec is None or spec.loader is None:
            continue

        module = module_from_spec(spec)
        # Ensure any relative imports performed by the real module resolve
        # against the object we are about to populate.
        sys.modules[__name__] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - fallback to stub on failure
            # Restore this shim so callers still receive a usable module.
            if shim_module is not None:
                sys.modules[__name__] = shim_module
            else:
                sys.modules.pop(__name__, None)
            continue
        return module

    return None


_real_fastapi = _load_real_fastapi()
if _real_fastapi is not None:
    globals().update(_real_fastapi.__dict__)
else:
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
