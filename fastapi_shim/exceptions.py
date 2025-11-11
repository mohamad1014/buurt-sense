"""Exception definitions compatible with FastAPI usage in tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HTTPException(Exception):
    status_code: int
    detail: Any
