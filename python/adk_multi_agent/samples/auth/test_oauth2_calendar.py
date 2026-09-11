from __future__ import annotations

import io
from urllib.error import HTTPError

import pytest
from google.adk.auth import (
    AuthCredential,
    AuthCredentialTypes,
    OAuth2Auth,
)

from samples.auth import oauth2_calendar


class FakeToolContext:
    def __init__(self, exchanged=None):
        self.exchanged = exchanged
        self.requests = []

    def get_auth_response(self, auth_config):
        return self.exchanged

    def request_credential(self, auth_config):
        self.requests.append(auth_config)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@pytest.fixture(autouse=True)
def oauth_env(monkeypatch):
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "client-secret")


def _credential(token: str | None) -> AuthCredential:
    return AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(access_token=token),
    )


def test_requests_authentication_before_calling_calendar(monkeypatch):
    context = FakeToolContext()
    monkeypatch.setattr(
        oauth2_calendar,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("認証前に API を呼んだ"),
    )

    result = oauth2_calendar.get_calendar_events("2026-09-10", context)

    assert result["status"] == "auth_required"
    assert len(context.requests) == 1


def test_uses_exchanged_access_token(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse(b'{"items":[{"id":"event-1"}]}')

    monkeypatch.setattr(oauth2_calendar, "urlopen", fake_urlopen)
    context = FakeToolContext(_credential("access-token"))

    result = oauth2_calendar.get_calendar_events("2026-09-10", context)

    assert result == {
        "status": "success",
        "event_count": 1,
        "events": [{"id": "event-1"}],
    }
    assert captured["authorization"] == "Bearer access-token"
    assert "timeMin=2026-09-10T00%3A00%3A00%2B09%3A00" in captured["url"]
    assert "timeMax=2026-09-11T00%3A00%3A00%2B09%3A00" in captured["url"]
    assert "timeZone=Asia%2FTokyo" in captured["url"]
    assert captured["timeout"] == 10
    assert context.requests == []


def test_http_401_requests_authentication_again(monkeypatch):
    def unauthorized(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(oauth2_calendar, "urlopen", unauthorized)
    context = FakeToolContext(_credential("expired-token"))

    result = oauth2_calendar.get_calendar_events("2026-09-10", context)

    assert result["status"] == "auth_required"
    assert len(context.requests) == 1


def test_http_403_reports_missing_permission_without_reauthentication(monkeypatch):
    def forbidden(request, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(oauth2_calendar, "urlopen", forbidden)
    context = FakeToolContext(_credential("access-token"))

    result = oauth2_calendar.get_calendar_events("2026-09-10", context)

    assert result["status"] == "forbidden"
    assert context.requests == []


def test_rejects_invalid_date_without_calling_calendar(monkeypatch):
    monkeypatch.setattr(
        oauth2_calendar,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("不正な日付で API を呼んだ"),
    )
    context = FakeToolContext(_credential("access-token"))

    result = oauth2_calendar.get_calendar_events("2026/09/10", context)

    assert result == {
        "status": "error",
        "message": "日付は YYYY-MM-DD で指定してください。",
    }


def test_requires_oauth_client_credentials(monkeypatch):
    monkeypatch.delenv("OAUTH_CLIENT_ID")
    monkeypatch.delenv("OAUTH_CLIENT_SECRET")

    with pytest.raises(RuntimeError, match="OAUTH_CLIENT_ID"):
        oauth2_calendar.get_calendar_events("2026-09-10", FakeToolContext())
