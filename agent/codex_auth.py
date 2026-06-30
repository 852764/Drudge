"""Experimental ChatGPT Codex device OAuth support.

This follows the device flow used by the open-source Codex CLI and stores
Drudge-owned credentials separately. It never reads Codex's auth.json.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable

import httpx


CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ISSUER = "https://auth.openai.com"
CODEX_TOKEN_URL = f"{CODEX_ISSUER}/oauth/token"
CODEX_DEVICE_URL = f"{CODEX_ISSUER}/codex/device"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
REFRESH_SKEW_SECONDS = 120


class CodexAuthError(RuntimeError):
    pass


def get_drudge_home() -> Path:
    configured = os.getenv("DRUDGE_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".drudge"


def default_auth_path() -> Path:
    return get_drudge_home() / "auth.json"


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        claims = json.loads(decoded.decode("utf-8"))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def extract_account_id(access_token: str) -> str | None:
    claims = _decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth", {})
    if not isinstance(auth_claims, dict):
        return None
    account_id = auth_claims.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def token_expiry(access_token: str) -> float | None:
    value = _decode_jwt_claims(access_token).get("exp")
    return float(value) if isinstance(value, (int, float)) else None


class CodexTokenStore:
    _lock = threading.RLock()

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else default_auth_path()

    def _load_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "providers": {}}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CodexAuthError(f"Cannot read Drudge auth store: {exc}") from exc
        if not isinstance(loaded, dict):
            raise CodexAuthError("Drudge auth store is not a JSON object")
        loaded.setdefault("version", 1)
        loaded.setdefault("providers", {})
        return loaded

    def load(self) -> dict[str, Any] | None:
        with self._lock:
            document = self._load_document()
            value = document.get("providers", {}).get("openai-codex")
            return dict(value) if isinstance(value, dict) else None

    def save(self, credentials: dict[str, Any]) -> None:
        with self._lock:
            document = self._load_document()
            providers = document.setdefault("providers", {})
            providers["openai-codex"] = dict(credentials)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
            temp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temp_path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                try:
                    os.chmod(temp_path, 0o600)
                except OSError:
                    pass
                os.replace(temp_path, self.path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    def clear(self) -> bool:
        with self._lock:
            document = self._load_document()
            providers = document.setdefault("providers", {})
            existed = providers.pop("openai-codex", None) is not None
            if existed:
                self.save_document(document)
            return existed

    def save_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _credentials_from_token_response(
    payload: dict[str, Any],
    *,
    previous_refresh_token: str = "",
) -> dict[str, Any]:
    access_token = str(payload.get("access_token") or "").strip()
    refresh_token = str(payload.get("refresh_token") or previous_refresh_token).strip()
    if not access_token:
        raise CodexAuthError("Codex token response did not include access_token")
    expires_at = token_expiry(access_token)
    if expires_at is None and payload.get("expires_in") is not None:
        expires_at = time.time() + max(1, int(payload["expires_in"]))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "account_id": extract_account_id(access_token),
        "base_url": CODEX_BASE_URL,
        "updated_at": int(time.time()),
    }


def _request_device_user_code(
    client: httpx.Client,
    *,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    retry_after = 1.0
    for attempt in range(max_attempts):
        response = client.post(
            f"{CODEX_ISSUER}/api/accounts/deviceauth/usercode",
            json={"client_id": CODEX_CLIENT_ID},
        )
        if response.status_code != 429:
            return response
        header = response.headers.get("retry-after")
        try:
            retry_after = max(1.0, float(header)) if header else retry_after
        except ValueError:
            pass
        if attempt < max_attempts - 1:
            sleep(retry_after)
            retry_after = min(retry_after * 2, 30.0)
    return response


def login_device_code(
    *,
    store: CodexTokenStore | None = None,
    open_browser: bool = True,
    timeout_seconds: int = 15 * 60,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    store = store or CodexTokenStore()
    owned_client = client is None
    client = client or httpx.Client(timeout=20.0, headers={"Accept": "application/json"})
    try:
        response = _request_device_user_code(client, sleep=sleep)
        if response.status_code != 200:
            raise CodexAuthError(f"Codex device-code request failed: HTTP {response.status_code}")
        device = response.json()
        user_code = str(device.get("user_code") or "").strip()
        device_auth_id = str(device.get("device_auth_id") or "").strip()
        if not user_code or not device_auth_id:
            raise CodexAuthError("Codex device-code response is incomplete")
        interval = max(3, int(device.get("interval") or 5))

        print(f"Open this URL: {CODEX_DEVICE_URL}", flush=True)
        print(f"Enter code: {user_code}", flush=True)
        if open_browser:
            webbrowser.open(CODEX_DEVICE_URL)

        deadline = time.monotonic() + timeout_seconds
        authorization: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            sleep(interval)
            poll = client.post(
                f"{CODEX_ISSUER}/api/accounts/deviceauth/token",
                json={"device_auth_id": device_auth_id, "user_code": user_code},
            )
            if poll.status_code == 200:
                authorization = poll.json()
                break
            if poll.status_code in (403, 404):
                continue
            raise CodexAuthError(f"Codex device authorization failed: HTTP {poll.status_code}")
        if authorization is None:
            raise CodexAuthError("Codex device authorization timed out")

        authorization_code = str(authorization.get("authorization_code") or "").strip()
        code_verifier = str(authorization.get("code_verifier") or "").strip()
        if not authorization_code or not code_verifier:
            raise CodexAuthError("Codex authorization response is incomplete")
        token_response = client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": f"{CODEX_ISSUER}/deviceauth/callback",
                "client_id": CODEX_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token_response.status_code != 200:
            raise CodexAuthError(f"Codex token exchange failed: HTTP {token_response.status_code}")
        credentials = _credentials_from_token_response(token_response.json())
        store.save(credentials)
        return credentials
    finally:
        if owned_client:
            client.close()


def refresh_credentials(
    credentials: dict[str, Any],
    *,
    store: CodexTokenStore | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    refresh_token = str(credentials.get("refresh_token") or "").strip()
    if not refresh_token:
        raise CodexAuthError("Codex refresh token is missing; log in again")
    owned_client = client is None
    client = client or httpx.Client(timeout=20.0, headers={"Accept": "application/json"})
    try:
        response = client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            raise CodexAuthError(f"Codex token refresh failed: HTTP {response.status_code}")
        refreshed = _credentials_from_token_response(
            response.json(),
            previous_refresh_token=refresh_token,
        )
        (store or CodexTokenStore()).save(refreshed)
        return refreshed
    finally:
        if owned_client:
            client.close()


def resolve_runtime_credentials(
    force_refresh: bool = False,
    *,
    store: CodexTokenStore | None = None,
) -> dict[str, Any]:
    store = store or CodexTokenStore()
    credentials = store.load()
    if not credentials:
        raise CodexAuthError("Codex OAuth is not configured; run: drudge auth login")
    expires_at = credentials.get("expires_at") or token_expiry(
        str(credentials.get("access_token") or "")
    )
    if force_refresh or (expires_at and float(expires_at) <= time.time() + REFRESH_SKEW_SECONDS):
        credentials = refresh_credentials(credentials, store=store)
    access_token = str(credentials.get("access_token") or "").strip()
    if not access_token:
        raise CodexAuthError("Codex access token is missing; log in again")
    account_id = credentials.get("account_id") or extract_account_id(access_token)
    if not account_id:
        raise CodexAuthError("Codex access token has no ChatGPT account id")
    return {
        "access_token": access_token,
        "account_id": account_id,
        "base_url": credentials.get("base_url") or CODEX_BASE_URL,
    }


def auth_status(store: CodexTokenStore | None = None) -> dict[str, Any]:
    credentials = (store or CodexTokenStore()).load()
    if not credentials:
        return {"authenticated": False}
    expires_at = credentials.get("expires_at") or token_expiry(
        str(credentials.get("access_token") or "")
    )
    return {
        "authenticated": bool(credentials.get("access_token")),
        "has_refresh_token": bool(credentials.get("refresh_token")),
        "account_id_present": bool(
            credentials.get("account_id")
            or extract_account_id(str(credentials.get("access_token") or ""))
        ),
        "expires_at": expires_at,
        "path": str((store or CodexTokenStore()).path),
    }


def logout(store: CodexTokenStore | None = None) -> bool:
    return (store or CodexTokenStore()).clear()
