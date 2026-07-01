from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from agent.codex_usage import CodexUsageClient, format_codex_usage, normalize_codex_usage


def credentials(force_refresh: bool = False) -> dict:
    return {"access_token": "test-token", "account_id": "acct-test"}


class CodexUsageTests(unittest.TestCase):
    def test_direct_usage_payload_is_normalized_without_identity(self):
        payload = {
            "account_id": "acct-secret",
            "email": "private@example.com",
            "user_id": "user-secret",
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25,
                    "limit_window_seconds": 18000,
                    "reset_at": 1779459394,
                },
                "secondary_window": {
                    "used_percent": 18,
                    "limit_window_seconds": 604800,
                    "reset_at": 1779826837,
                },
            },
            "credits": {"has_credits": True, "unlimited": False, "balance": "12.50"},
        }

        normalized = normalize_codex_usage(payload)
        serialized = json.dumps(normalized)

        self.assertEqual(normalized["plan_type"], "plus")
        self.assertEqual(normalized["primary"]["remaining_percent"], 75)
        self.assertEqual(normalized["primary"]["window_minutes"], 300)
        self.assertEqual(normalized["secondary"]["window_minutes"], 10080)
        self.assertNotIn("private@example.com", serialized)
        self.assertNotIn("acct-secret", serialized)
        self.assertNotIn("user-secret", serialized)

    def test_usage_client_refreshes_once_after_401(self):
        attempts = 0
        refresh_calls = []

        def resolver(force_refresh: bool) -> dict:
            refresh_calls.append(force_refresh)
            return credentials(force_refresh)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            self.assertEqual(request.headers["chatgpt-account-id"], "acct-test")
            if attempts == 1:
                return httpx.Response(401, json={"error": "expired"})
            return httpx.Response(200, json={
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 10,
                        "limit_window_seconds": 18000,
                        "reset_at": 1779459394,
                    }
                },
            })

        client = CodexUsageClient(
            credential_resolver=resolver,
            transport=httpx.MockTransport(handler),
        )
        result = asyncio.run(client.fetch())

        self.assertEqual(refresh_calls, [False, True])
        self.assertEqual(result["primary"]["remaining_percent"], 90)

    def test_formatter_uses_remaining_not_used_percent(self):
        usage = normalize_codex_usage({
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 12,
                    "limit_window_seconds": 18000,
                    "reset_at": 1779459394,
                }
            },
        })

        lines = format_codex_usage(usage)

        self.assertTrue(any("5h limit: 88% left (12% used)" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
