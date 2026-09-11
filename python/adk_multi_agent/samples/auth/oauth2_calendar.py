"""Google Calendar をユーザー委譲 OAuth 2.0 で読むツール。

ADK 2.2.0 の認証イベントはクライアントシークレットを含むため、
このサンプルを信頼境界の外へ公開しない。
"""

from __future__ import annotations

import json
import os
from datetime import date as date_type
from datetime import datetime, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi.openapi.models import OAuth2, OAuthFlowAuthorizationCode, OAuthFlows
from google.adk.auth import (
    AuthConfig,
    AuthCredential,
    AuthCredentialTypes,
    OAuth2Auth,
)
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"

AUTH_SCHEME = OAuth2(
    flows=OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl="https://accounts.google.com/o/oauth2/auth",
            tokenUrl="https://oauth2.googleapis.com/token",
            scopes={CALENDAR_READONLY_SCOPE: "Google Calendar の予定を読む"},
        )
    )
)


def _auth_config() -> AuthConfig:
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("OAUTH_CLIENT_ID と OAUTH_CLIENT_SECRET を設定してください")

    return AuthConfig(
        auth_scheme=AUTH_SCHEME,
        credential_key="google_calendar_readonly",
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                client_id=client_id,
                client_secret=client_secret,
            ),
        ),
    )


def _day_range(value: str, time_zone: str) -> tuple[str, str]:
    target = date_type.fromisoformat(value)
    location = ZoneInfo(time_zone)
    start = datetime.combine(target, time.min, tzinfo=location)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _fetch_events(access_token: str, value: str, time_zone: str) -> list[dict]:
    time_min, time_max = _day_range(value, time_zone)
    query = urlencode(
        {
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",
            "orderBy": "startTime",
            "timeZone": time_zone,
        }
    )
    request = Request(
        "https://www.googleapis.com/calendar/v3/calendars/primary/events?" + query,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)
    return payload.get("items", [])


def get_calendar_events(
    date: str,
    tool_context: ToolContext,
    time_zone: str = "Asia/Tokyo",
) -> dict:
    """指定日の Google Calendar の予定を取得する。"""
    auth_config = _auth_config()
    exchanged = tool_context.get_auth_response(auth_config)
    if exchanged is None:
        tool_context.request_credential(auth_config)
        return {
            "status": "auth_required",
            "message": "Google Calendar へのアクセスを許可してください。",
        }

    access_token = exchanged.oauth2.access_token if exchanged.oauth2 else None
    if not access_token:
        tool_context.request_credential(auth_config)
        return {
            "status": "auth_required",
            "message": "アクセストークンを取得できませんでした。再認証してください。",
        }

    try:
        events = _fetch_events(access_token, date, time_zone)
    except ValueError:
        return {"status": "error", "message": "日付は YYYY-MM-DD で指定してください。"}
    except ZoneInfoNotFoundError:
        return {
            "status": "error",
            "message": "IANA 形式のタイムゾーンを指定してください。",
        }
    except HTTPError as error:
        if error.code == 401:
            tool_context.request_credential(auth_config)
            return {
                "status": "auth_required",
                "message": "認証の有効期限が切れました。再認証してください。",
            }
        if error.code == 403:
            return {
                "status": "forbidden",
                "message": "カレンダーを読む権限がありません。",
            }
        return {
            "status": "error",
            "message": f"Google Calendar API が HTTP {error.code} を返しました。",
        }
    except URLError:
        return {
            "status": "error",
            "message": "Google Calendar API へ接続できませんでした。",
        }

    return {"status": "success", "event_count": len(events), "events": events}


calendar_tool = FunctionTool(func=get_calendar_events)
