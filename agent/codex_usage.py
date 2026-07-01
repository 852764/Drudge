"""Read ChatGPT Codex subscription limits with Drudge-owned OAuth credentials."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable

import httpx

from .codex_auth import resolve_runtime_credentials


CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


class CodexUsageError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CodexUsageClient:
    def __init__(
        self,
        *,
        credential_resolver: Callable[[bool], dict[str, Any]] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._credential_resolver = credential_resolver or resolve_runtime_credentials
        self._transport = transport
        self.timeout = timeout

    async def fetch(self) -> dict[str, Any]:
        credentials = await asyncio.to_thread(self._credential_resolver, False)
        try:
            payload = await self._request(credentials)
        except CodexUsageError as exc:
            if exc.status_code != 401:
                raise
            credentials = await asyncio.to_thread(self._credential_resolver, True)
            payload = await self._request(credentials)
        return normalize_codex_usage(payload)

    async def _request(self, credentials: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {credentials['access_token']}",
            "ChatGPT-Account-ID": str(credentials["account_id"]),
            "Accept": "application/json",
            "User-Agent": "Drudge/0.1 CodexUsage",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
        ) as client:
            response = await client.get(CODEX_USAGE_URL, headers=headers)
        if response.status_code >= 400:
            raise CodexUsageError(
                f"Codex usage request failed: HTTP {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CodexUsageError("Codex usage response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise CodexUsageError("Codex usage response is not a JSON object")
        return payload


def normalize_codex_usage(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only display-safe quota fields; omit account identity and email."""
    rate_limit = payload.get("rate_limit") or payload.get("rateLimits") or {}
    primary_raw = rate_limit.get("primary_window") or rate_limit.get("primary")
    secondary_raw = rate_limit.get("secondary_window") or rate_limit.get("secondary")
    credits_raw = payload.get("credits") or rate_limit.get("credits") or {}
    plan_type = payload.get("plan_type") or rate_limit.get("planType")
    reached = payload.get("rate_limit_reached_type")
    if reached is None:
        reached = rate_limit.get("rateLimitReachedType")

    additional = []
    for item in payload.get("additional_rate_limits") or []:
        if not isinstance(item, dict):
            continue
        item_rate = item.get("rate_limit") or item
        additional.append({
            "id": item.get("limit_id") or item.get("id") or item.get("name"),
            "name": item.get("limit_name") or item.get("name") or item.get("limit_id"),
            "primary": _normalize_window(
                item_rate.get("primary_window") or item_rate.get("primary")
            ),
            "secondary": _normalize_window(
                item_rate.get("secondary_window") or item_rate.get("secondary")
            ),
        })

    return {
        "plan_type": str(plan_type) if plan_type is not None else None,
        "primary": _normalize_window(primary_raw),
        "secondary": _normalize_window(secondary_raw),
        "additional": additional,
        "credits": {
            "has_credits": bool(
                credits_raw.get("has_credits", credits_raw.get("hasCredits", False))
            ),
            "unlimited": bool(credits_raw.get("unlimited", False)),
            "balance": credits_raw.get("balance"),
        },
        "rate_limit_reached_type": reached,
        "fetched_at": int(datetime.now().timestamp()),
    }


def _normalize_window(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    used = value.get("used_percent", value.get("usedPercent"))
    seconds = value.get("limit_window_seconds")
    minutes = value.get("windowDurationMins")
    if minutes is None and isinstance(seconds, (int, float)):
        minutes = float(seconds) / 60
    resets_at = value.get("reset_at", value.get("resetsAt"))
    try:
        used_value = max(0.0, min(100.0, float(used)))
    except (TypeError, ValueError):
        used_value = None
    try:
        minute_value = float(minutes)
    except (TypeError, ValueError):
        minute_value = None
    try:
        reset_value = int(resets_at)
    except (TypeError, ValueError):
        reset_value = None
    return {
        "used_percent": used_value,
        "remaining_percent": 100.0 - used_value if used_value is not None else None,
        "window_minutes": minute_value,
        "resets_at": reset_value,
    }


def format_codex_usage(usage: dict[str, Any]) -> list[str]:
    lines = [f"ChatGPT plan: {(usage.get('plan_type') or 'unknown').title()}"]
    for fallback, key in (("Primary", "primary"), ("Secondary", "secondary")):
        window = usage.get(key)
        if window:
            lines.append(_format_window(window, fallback))
    for item in usage.get("additional") or []:
        label = str(item.get("name") or item.get("id") or "Additional")
        for key in ("primary", "secondary"):
            if item.get(key):
                lines.append(_format_window(item[key], label))

    credits = usage.get("credits") or {}
    if credits.get("unlimited"):
        lines.append("Credits: unlimited")
    elif credits.get("has_credits"):
        lines.append(f"Credits balance: {credits.get('balance', 'unknown')}")
    if usage.get("rate_limit_reached_type"):
        lines.append(f"Limit reached: {usage['rate_limit_reached_type']}")
    return lines


def _format_window(window: dict[str, Any], fallback: str) -> str:
    minutes = window.get("window_minutes")
    label = fallback
    if minutes is not None:
        rounded = int(round(float(minutes)))
        if rounded == 300:
            label = "5h limit"
        elif rounded == 10080:
            label = "Weekly limit"
        elif rounded % 1440 == 0:
            label = f"{rounded // 1440}d limit"
        elif rounded % 60 == 0:
            label = f"{rounded // 60}h limit"
        else:
            label = f"{rounded}m limit"
    remaining = window.get("remaining_percent")
    used = window.get("used_percent")
    if remaining is None:
        detail = "usage unavailable"
    else:
        detail = f"{remaining:.0f}% left ({used:.0f}% used)"
    resets_at = window.get("resets_at")
    if resets_at:
        reset = datetime.fromtimestamp(int(resets_at)).astimezone()
        detail += f", resets {reset:%Y-%m-%d %H:%M %Z}"
    return f"{label}: {detail}"
