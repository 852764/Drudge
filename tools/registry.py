"""工具注册表 — 自注册模式，类似 Drudge 的 tools/registry.py"""

import json
import asyncio
import inspect
from typing import Any, Callable

from .context import ToolContext
from .result import normalize_tool_result
from .risk import RiskLevel, ToolRisk, coerce_risk_level

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

    @staticmethod
    def _handler_args(args: dict, handler: Callable, context: ToolContext) -> dict:
        merged = dict(args)
        if "context" in inspect.signature(handler).parameters:
            merged["context"] = context
        return merged

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict],
        handler: Callable,
        toolset: str = "default",
        check_fn: Callable[[], bool] | None = None,
        required: list[str] | None = None,
        risk: RiskLevel | str = RiskLevel.LOW,
        risk_fn: Callable[[dict, ToolContext], ToolRisk] | None = None,
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
                    "additionalProperties": False,
                },
            },
        }

        self._tools[name] = {
            "schema": schema,
            "handler": handler,
            "toolset": toolset,
            "check_fn": check_fn,
            "is_async": inspect.iscoroutinefunction(handler),
            "parameter_types": {
                param_name: param_info.get("type", str)
                for param_name, param_info in parameters.items()
            },
            "required": set(required or list(parameters.keys())),
            "risk": coerce_risk_level(risk),
            "risk_fn": risk_fn,
        }

    def assess_risk(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
    ) -> ToolRisk:
        """Classify a call using trusted registration metadata, not model input."""
        info = self._tools.get(tool_name)
        if info is None:
            return ToolRisk(RiskLevel.CRITICAL, "Unknown tool", tool_name)
        if info["risk_fn"] is not None:
            return info["risk_fn"](args, context)
        level = info["risk"]
        return ToolRisk(level, f"Registered {level.value}-risk tool", tool_name)

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
        return normalize_tool_result(result)

    def _prepare_call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext | None,
        approved: bool = False,
    ) -> tuple[dict, dict] | str:
        if tool_name not in self._tools:
            return normalize_tool_result({"error": f"Unknown tool: {tool_name}"})
        if context is None:
            return normalize_tool_result({"error": "ToolContext is required", "blocked": True})

        info = self._tools[tool_name]
        if not context.allows_toolset(info["toolset"]):
            return normalize_tool_result({
                "error": f"Tool is disabled for this run: {tool_name}",
                "blocked": True,
            })
        if not isinstance(args, dict):
            return normalize_tool_result({"error": "Tool arguments must be a JSON object"})

        allowed = set(info["parameter_types"])
        unknown = sorted(set(args) - allowed)
        if unknown:
            return normalize_tool_result({
                "error": f"Unknown tool arguments: {', '.join(unknown)}",
                "blocked": True,
            })
        missing = sorted(info["required"] - set(args))
        if missing:
            return normalize_tool_result({"error": f"Missing required arguments: {', '.join(missing)}"})

        for name, value in args.items():
            expected = info["parameter_types"][name]
            valid = isinstance(value, expected)
            if expected in (int, float) and isinstance(value, bool):
                valid = False
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                valid = True
            if not valid:
                return normalize_tool_result({
                    "error": f"Invalid type for '{name}': expected {expected.__name__}"
                })
        risk = self.assess_risk(tool_name, args, context)
        if risk.level is RiskLevel.CRITICAL:
            return normalize_tool_result({
                "error": f"Critical-risk tool call blocked: {risk.reason}",
                "blocked": True,
                "metadata": {
                    "risk": risk.level.value,
                    "action": risk.action,
                },
            })
        if (
            context.approval_mode == "on_request"
            and risk.requires_approval
            and not approved
        ):
            return normalize_tool_result({
                "error": f"Approval required for {tool_name}: {risk.reason}",
                "blocked": True,
                "metadata": {
                    "approval_required": True,
                    "risk": risk.level.value,
                    "action": risk.action,
                },
            })
        return info, self._handler_args(args, info["handler"], context)

    def dispatch(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext | None = None,
        *,
        approved: bool = False,
    ) -> str:
        """执行工具调用（同步），返回 JSON 字符串"""
        prepared = self._prepare_call(tool_name, args, context, approved)
        if isinstance(prepared, str):
            return prepared
        info, call_args = prepared
        try:
            handler = info["handler"]
            if info["is_async"]:
                result = asyncio.run(handler(**call_args))
            else:
                result = handler(**call_args)
            return self._format_result(result)
        except Exception as e:
            return normalize_tool_result({"error": str(e)})

    async def dispatch_async(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext | None = None,
        *,
        approved: bool = False,
    ) -> str:
        """执行工具调用（异步），用于 Agent 循环中 await async handler"""
        prepared = self._prepare_call(tool_name, args, context, approved)
        if isinstance(prepared, str):
            return prepared
        info, call_args = prepared
        try:
            handler = info["handler"]
            if info["is_async"]:
                result = await handler(**call_args)
            else:
                result = await asyncio.to_thread(handler, **call_args)
            return self._format_result(result)
        except Exception as e:
            return normalize_tool_result({"error": str(e)})

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
