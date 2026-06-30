"""Web 工具 — HTTP 请求 + 网页内容提取"""

import json
import re
import httpx
from .context import ToolContext
from .registry import registry
from .result import ToolResult


async def web_request_handler(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str | None = None,
    timeout: int = 30,
    context: ToolContext | None = None,
) -> str:
    """发送 HTTP 请求"""
    if not url.startswith(("http://", "https://")):
        return ToolResult.failure("URL must start with http:// or https://")
    if context is None:
        return ToolResult.failure("ToolContext is required", blocked=True)
    allowed, reason = context.network_allowed(f"web_request {url}")
    if not allowed:
        return ToolResult.failure(reason or "Network request blocked", blocked=True)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            req_headers = headers or {}
            req_headers.setdefault("User-Agent", "Drudge-Lite/0.1")

            if method.upper() == "GET":
                response = await client.get(url, headers=req_headers)
            elif method.upper() == "POST":
                response = await client.post(url, headers=req_headers, content=body or "")
            elif method.upper() == "PUT":
                response = await client.put(url, headers=req_headers, content=body or "")
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=req_headers)
            else:
                return json.dumps({"error": f"Unsupported method: {method}"})

            content_type = response.headers.get("content-type", "")
            is_text = any(t in content_type for t in ["text/", "application/json", "application/xml"])

            result = {
                "status": response.status_code,
                "headers": dict(response.headers),
            }

            if is_text:
                text = response.text[:10000]  # 截断
                # 简单 HTML 提取纯文本
                if "text/html" in content_type:
                    text = _strip_html(text)
                result["body"] = text
            else:
                result["body"] = f"(binary, {len(response.content)} bytes)"

            return json.dumps(result, ensure_ascii=False)

    except httpx.TimeoutException:
        return json.dumps({"error": f"Request timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _strip_html(html: str) -> str:
    """简单去除 HTML 标签，提取纯文本"""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:5000]


def web_check() -> bool:
    """Web 工具可注册；实际网络错误由请求时返回。"""
    return True


# 注册 web_request 工具（async handler，Agent 循环中通过 dispatch_async 调用）
registry.register(
    name="web_request",
    description="Make an HTTP request. Use for API calls, web scraping, or fetching data. "
    "Returns status, headers, and body (HTML is stripped to plain text).",
    parameters={
        "url": {"type": str, "description": "The URL to request (must start with http:// or https://)"},
        "method": {"type": str, "description": "HTTP method: GET, POST, PUT, DELETE (default: GET)"},
        "headers": {"type": dict, "description": "Optional HTTP headers"},
        "body": {"type": str, "description": "Optional request body for POST/PUT"},
        "timeout": {"type": int, "description": "Timeout in seconds (default: 30)"},
    },
    handler=web_request_handler,
    toolset="web",
    check_fn=web_check,
    required=["url"],
)
