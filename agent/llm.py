"""LLM 客户端 — OpenAI-compatible API 抽象"""

import json
import asyncio
import httpx
from typing import Any


class LLMClient:
    """OpenAI-compatible LLM 客户端"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        """发送聊天请求，返回 OpenAI 格式响应"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=body)
                    response.raise_for_status()
                    data = response.json()

                    return {
                        "id": data.get("id", ""),
                        "model": data.get("model", self.model),
                        "choices": data.get("choices", []),
                        "usage": data.get("usage", {}),
                    }

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
                if e.response.status_code in (429, 503):
                    wait = min(2 ** attempt, 30)
                    await asyncio.sleep(wait)
                    continue
                break
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 10))
                    continue
                break

        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def extract_text(response: dict) -> str | None:
        """从响应中提取文本内容"""
        choices = response.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})
        return msg.get("content")

    @staticmethod
    def extract_tool_calls(response: dict) -> list[dict]:
        """从响应中提取工具调用"""
        choices = response.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        return tool_calls

    @staticmethod
    def extract_finish_reason(response: dict) -> str:
        """提取 finish_reason"""
        choices = response.get("choices", [])
        if not choices:
            return "stop"
        return choices[0].get("finish_reason", "stop")

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        """简单 token 估算（4 字符 ≈ 1 token）"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    total += len(str(func)) // 4
        return total


def create_client(config: dict) -> LLMClient:
    """从配置创建 LLM 客户端"""
    return LLMClient(
        base_url=config["base_url"],
        api_key=config.get("api_key", ""),
        model=config["name"],
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
    )
