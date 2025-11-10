"""Custom exceptions used by the runtime orchestration layer."""

from __future__ import annotations


class SessionNotFoundError(KeyError):
    """Raised when attempting to access a session that does not exist."""


class SessionAlreadyStoppedError(RuntimeError):
    """Raised when trying to stop a session that has already finished."""
