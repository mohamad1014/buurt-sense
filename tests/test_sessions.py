"""Tests for the session management API."""

from __future__ import annotations

from uuid import uuid4

from app.models import Session
from tests.conftest import SimpleClient


def validate_session_payload(payload: dict) -> Session:
    session = Session.model_validate(payload)
    assert session.started_at.tzinfo is not None, "started_at must be timezone-aware"
    if session.ended_at is not None:
        assert session.ended_at.tzinfo is not None, "ended_at must be timezone-aware"
        assert (
            session.ended_at >= session.started_at
        ), "ended_at cannot precede started_at"
    return session


def test_session_lifecycle(client: SimpleClient) -> None:
    response = client.post("/sessions")
    assert response.status_code == 201
    session = validate_session_payload(response.json())
    assert session.ended_at is None

    get_response = client.get(f"/sessions/{session.id}")
    assert get_response.status_code == 200
    fetched = validate_session_payload(get_response.json())
    assert fetched.ended_at is None
    assert fetched.started_at == session.started_at

    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200
    stopped = validate_session_payload(stop_response.json())
    assert stopped.ended_at is not None

    final_get = client.get(f"/sessions/{session.id}")
    assert final_get.status_code == 200
    final_session = validate_session_payload(final_get.json())
    assert final_session.ended_at == stopped.ended_at

    list_response = client.get("/sessions")
    assert list_response.status_code == 200
    sessions = [validate_session_payload(item) for item in list_response.json()]
    assert session.id in {item.id for item in sessions}


def test_stopping_unknown_session_returns_not_found(client: SimpleClient) -> None:
    unknown_id = uuid4()
    response = client.post(f"/sessions/{unknown_id}/stop")
    assert response.status_code == 404
    assert response.json()["detail"].lower() == "session not found"


def test_double_stop_returns_conflict(client: SimpleClient) -> None:
    session = validate_session_payload(client.post("/sessions").json())

    first_stop = client.post(f"/sessions/{session.id}/stop")
    assert first_stop.status_code == 200
    validate_session_payload(first_stop.json())

    second_stop = client.post(f"/sessions/{session.id}/stop")
    assert second_stop.status_code == 409
    assert second_stop.json()["detail"].lower() == "session already stopped"


def test_retrieving_unknown_session_returns_not_found(client: SimpleClient) -> None:
    response = client.get(f"/sessions/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"].lower() == "session not found"
