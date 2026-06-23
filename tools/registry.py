"""工具注册表 — 自注册模式，类似 Hermes 的 tools/registry.py"""

import json
import asyncio
import inspect
from typing import Any, Callable

# JSON Schema property types
JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}  # name -> {schema, handler, toolset, check_fn, is_async}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict],
        handler: Callable,
        toolset: str = "default",
        check_fn: Callable[[], bool] | None = None,
        required: list[str] | None = None,
    ):
        """注册一个工具"""
        properties = {}
        for param_name, param_info in parameters.items():
            prop = {
                "type": JSON_TYPE_MAP.get(param_info.get("type", str), "string"),
                "description": param_info.get("description", ""),
            }
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            if "default" in param_info:
                prop["default"] = param_info["default"]
            properties[param_name] = prop

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or list(parameters.keys()),
                },
            },
        }

        self._tools[name] = {
            "schema": schema,
            "handler": handler,
            "toolset": toolset,
            "check_fn": check_fn,
            "is_async": inspect.iscoroutinefunction(handler),
        }

    def get_schemas(self, toolsets: list[str] | None = None) -> list[dict]:
        """获取工具 schema 列表，可过滤工具集"""
        schemas = []
        for name, info in self._tools.items():
            if toolsets is None or info["toolset"] in toolsets:
                if info["check_fn"] is None or info["check_fn"]():
                    schemas.append(info["schema"])
        return schemas

    def _format_result(self, result: Any) -> str:
        """将 handler 返回值格式化为 JSON 字符串"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def dispatch(self, tool_name: str, args: dict) -> str:
        """执行工具调用（同步），返回 JSON 字符串"""
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        info = self._tools[tool_name]
        try:
            handler = info["handler"]
            if info["is_async"]:
                result = asyncio.run(handler(**args))
            else:
                result = handler(**args)
            return self._format_result(result)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def dispatch_async(self, tool_name: str, args: dict) -> str:
        """执行工具调用（异步），用于 Agent 循环中 await async handler"""
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        info = self._tools[tool_name]
        try:
            handler = info["handler"]
            if info["is_async"]:
                result = await handler(**args)
            else:
                result = handler(**args)
            return self._format_result(result)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def is_async_handler(self, tool_name: str) -> bool:
        """检测 handler 是否为 async function"""
        if tool_name in self._tools:
            return self._tools[tool_name].get("is_async", False)
        return False

    def list_tools(self, toolsets: list[str] | None = None) -> list[str]:
        """列出所有注册的工具名"""
        names = []
        for name, info in self._tools.items():
            if toolsets is None or info["toolset"] in toolsets:
                if info["check_fn"] is None or info["check_fn"]():
                    names.append(name)
        return names


# 全局单例
registry = ToolRegistry()
