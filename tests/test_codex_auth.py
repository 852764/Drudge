from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from agent.codex_auth import (
    CODEX_TOKEN_URL,
    CodexTokenStore,
    auth_status,
    extract_account_id,
    login_device_code,
    logout,
    refresh_credentials,
)


def fake_jwt(*, account_id: str = "acct-test", expires_at: float | None = None) -> str:
    payload = {
        "exp": int(expires_at or (time.time() + 3600)),
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


class CodexAuthTests(unittest.TestCase):
    def test_device_login_and_token_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CodexTokenStore(Path(directory, "auth.json"))
            calls = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(str(request.url))
                if request.url.path.endswith("/deviceauth/usercode"):
                    return httpx.Response(200, json={
                        "user_code": "ABCD-EFGH",
                        "device_auth_id": "device-1",
                        "interval": 3,
                    })
                if request.url.path.endswith("/deviceauth/token"):
                    return httpx.Response(200, json={
                        "authorization_code": "auth-code",
                        "code_verifier": "verifier",
                    })
                if str(request.url) == CODEX_TOKEN_URL:
                    return httpx.Response(200, json={
                        "access_token": fake_jwt(),
                        "refresh_token": "refresh-1",
                    })
                return httpx.Response(404)

            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                credentials = login_device_code(
                    store=store,
                    open_browser=False,
                    client=client,
                    sleep=lambda _: None,
                )

            self.assertEqual(credentials["account_id"], "acct-test")
            self.assertEqual(store.load()["refresh_token"], "refresh-1")
            self.assertEqual(len(calls), 3)
            self.assertTrue(auth_status(store)["authenticated"])

    def test_device_login_retries_user_code_rate_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CodexTokenStore(Path(directory, "auth.json"))
            user_code_attempts = 0

            def handler(request: httpx.Request) -> httpx.Response:
                nonlocal user_code_attempts
                if request.url.path.endswith("/deviceauth/usercode"):
                    user_code_attempts += 1
                    if user_code_attempts == 1:
                        return httpx.Response(429, headers={"retry-after": "1"})
                    return httpx.Response(200, json={
                        "user_code": "ABCD-EFGH",
                        "device_auth_id": "device-1",
                        "interval": 3,
                    })
                if request.url.path.endswith("/deviceauth/token"):
                    return httpx.Response(200, json={
                        "authorization_code": "auth-code",
                        "code_verifier": "verifier",
                    })
                if str(request.url) == CODEX_TOKEN_URL:
                    return httpx.Response(200, json={
                        "access_token": fake_jwt(),
                        "refresh_token": "refresh-1",
                    })
                return httpx.Response(404)

            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                login_device_code(
                    store=store,
                    open_browser=False,
                    client=client,
                    sleep=lambda _: None,
                )

            self.assertEqual(user_code_attempts, 2)
            self.assertTrue(auth_status(store)["authenticated"])

    def test_refresh_preserves_rotating_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CodexTokenStore(Path(directory, "auth.json"))
            original = {
                "access_token": fake_jwt(expires_at=time.time() - 10),
                "refresh_token": "old-refresh",
            }

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={
                    "access_token": fake_jwt(account_id="acct-new"),
                    "refresh_token": "new-refresh",
                })

            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                refreshed = refresh_credentials(original, store=store, client=client)

            self.assertEqual(refreshed["refresh_token"], "new-refresh")
            self.assertEqual(extract_account_id(refreshed["access_token"]), "acct-new")
            self.assertTrue(logout(store))
            self.assertFalse(auth_status(store)["authenticated"])


if __name__ == "__main__":
    unittest.main()
