"""Composable tool providers, including a dependency-free MCP stdio client."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from .context import ToolContext
from .registry import ToolRegistry
from .result import ToolResult
from .risk import RiskLevel, ToolRisk, coerce_risk_level


class ToolProvider(ABC):
    name: str

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    def schemas(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def tool_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def owns(self, tool_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def assess_risk(self, tool_name: str, args: dict, context: ToolContext) -> ToolRisk:
        raise NotImplementedError

    @abstractmethod
    async def call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
        *,
        approved: bool = False,
    ) -> str:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "connected": True, "tools": self.tool_names()}

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": schema["function"]["name"],
                "description": schema["function"].get("description", ""),
                "category": self.name,
                "provider": self.name,
                "risk": "medium",
            }
            for schema in self.schemas()
        ]


class LocalToolProvider(ToolProvider):
    name = "local"

    def __init__(self, registry: ToolRegistry, toolsets: list[str]) -> None:
        self.registry = registry
        self.toolsets = list(toolsets)

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.get_schemas(self.toolsets)

    def tool_names(self) -> list[str]:
        return self.registry.list_tools(self.toolsets)

    def catalog(self) -> list[dict[str, Any]]:
        return self.registry.get_catalog(self.toolsets)

    def owns(self, tool_name: str) -> bool:
        return tool_name in self.tool_names()

    def assess_risk(self, tool_name: str, args: dict, context: ToolContext) -> ToolRisk:
        return self.registry.assess_risk(tool_name, args, context)

    async def call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
        *,
        approved: bool = False,
    ) -> str:
        return await self.registry.dispatch_async(
            tool_name,
            args,
            context=context,
            approved=approved,
        )


class TaskToolProvider(ToolProvider):
    """Agent-owned persistent task tools."""

    name = "tasks"

    def __init__(
        self,
        list_tasks: Callable[[], list[dict[str, Any]]],
        create_task: Callable[[str, str], dict[str, Any]],
        update_task: Callable[[int, str], dict[str, Any]],
    ) -> None:
        self._list = list_tasks
        self._create = create_task
        self._update = update_task
        self._schemas = [
            _function_schema("task_list", "List persistent tasks for the active session.", {}, []),
            _function_schema(
                "task_create",
                "Create a persistent task for multi-step work.",
                {
                    "title": {"type": "string", "description": "Short actionable task title"},
                    "details": {"type": "string", "description": "Optional task details"},
                },
                ["title"],
            ),
            _function_schema(
                "task_update",
                "Change a persistent task status.",
                {
                    "task_id": {"type": "integer", "description": "Task numeric ID"},
                    "status": {
                        "type": "string",
                        "description": "New task status",
                        "enum": ["pending", "in_progress", "completed", "cancelled"],
                    },
                },
                ["task_id", "status"],
            ),
        ]

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def tool_names(self) -> list[str]:
        return ["task_list", "task_create", "task_update"]

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": schema["function"]["name"],
                "description": schema["function"].get("description", ""),
                "category": "task",
                "provider": self.name,
                "risk": "low",
            }
            for schema in self._schemas
        ]

    def owns(self, tool_name: str) -> bool:
        return tool_name in self.tool_names()

    def assess_risk(self, tool_name: str, args: dict, context: ToolContext) -> ToolRisk:
        return ToolRisk(RiskLevel.LOW, "Agent-local task metadata", tool_name)

    async def call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
        *,
        approved: bool = False,
    ) -> str:
        try:
            if tool_name == "task_list":
                value = self._list()
            elif tool_name == "task_create":
                value = self._create(str(args.get("title", "")), str(args.get("details", "")))
            elif tool_name == "task_update":
                value = self._update(int(args["task_id"]), str(args["status"]))
            else:
                return ToolResult.failure(f"Unknown task tool: {tool_name}").to_json()
            return ToolResult.success(json.dumps(value, ensure_ascii=False)).to_json()
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult.failure(str(exc)).to_json()


class ToolSearchProvider(ToolProvider):
    """Always-small meta tool that activates omitted tools for the current turn."""

    name = "tool_search"

    def __init__(
        self,
        search: Callable[[str, int], list[dict[str, Any]]],
        *,
        default_limit: int = 5,
    ) -> None:
        self._search = search
        self._default_limit = max(1, min(int(default_limit), 20))
        self._schema = _function_schema(
            "tool_search",
            "Search the available tool catalog and activate matching tools for the next model call.",
            {
                "query": {
                    "type": "string",
                    "description": "Capability needed, such as database query or file editing",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches to activate",
                    "default": self._default_limit,
                },
            },
            ["query"],
        )

    def schemas(self) -> list[dict[str, Any]]:
        return [self._schema]

    def tool_names(self) -> list[str]:
        return ["tool_search"]

    def catalog(self) -> list[dict[str, Any]]:
        return [{
            "name": "tool_search",
            "description": self._schema["function"]["description"],
            "category": "core",
            "provider": self.name,
            "risk": "low",
        }]

    def owns(self, tool_name: str) -> bool:
        return tool_name == "tool_search"

    def assess_risk(self, tool_name: str, args: dict, context: ToolContext) -> ToolRisk:
        return ToolRisk(RiskLevel.LOW, "Read-only tool catalog search", tool_name)

    async def call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
        *,
        approved: bool = False,
    ) -> str:
        try:
            query = str(args.get("query") or "").strip()
            limit = max(1, min(int(args.get("limit", self._default_limit)), 20))
            if not query:
                raise ValueError("tool_search query cannot be empty")
            matches = self._search(query, limit)
            return ToolResult.success(
                json.dumps(matches, ensure_ascii=False),
                activated=[item["name"] for item in matches],
            ).to_json()
        except (TypeError, ValueError, RuntimeError) as exc:
            return ToolResult.failure(str(exc)).to_json()


class MCPServerProvider(ToolProvider):
    """One MCP server connected over newline-delimited JSON-RPC stdio."""

    def __init__(self, name: str, config: dict[str, Any], workspace: str | Path) -> None:
        self.name = name
        self.config = dict(config)
        self.workspace = Path(workspace).expanduser().resolve()
        self.namespace = _safe_name(name)
        self.timeout = float(self.config.get("timeout", 30))
        self.risk = coerce_risk_level(self.config.get("risk", "medium"))
        self.process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_lines: list[str] = []
        self._next_id = 1
        self._tools: dict[str, dict[str, Any]] = {}
        self._error: str | None = None

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        command = str(self.config.get("command") or "").strip()
        if not command:
            raise ValueError(f"MCP server '{self.name}' has no command")
        args = [str(value) for value in self.config.get("args", [])]
        cwd_value = self.config.get("cwd")
        cwd = Path(cwd_value).expanduser() if cwd_value else self.workspace
        if not cwd.is_absolute():
            cwd = self.workspace / cwd
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (self.config.get("env") or {}).items()})
        self.process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd.resolve()),
            env=env,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            await self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "drudge", "version": "0.1.0"},
                },
            )
            await self._notify("notifications/initialized", {})
            await self._load_tools()
            self._error = None
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        process = self.process
        self.process = None
        if process:
            if process.stdin:
                process.stdin.close()
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stderr_task = None

    def schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for exposed_name, tool in self._tools.items():
            parameters = tool.get("inputSchema") or {"type": "object", "properties": {}}
            schemas.append({
                "type": "function",
                "function": {
                    "name": exposed_name,
                    "description": str(tool.get("description") or f"MCP tool {tool.get('name')}")[:2000],
                    "parameters": parameters,
                },
            })
        return schemas

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": exposed_name,
                "description": str(tool.get("description") or ""),
                "category": f"mcp:{self.name}",
                "provider": self.name,
                "risk": self.risk.value,
            }
            for exposed_name, tool in self._tools.items()
        ]

    def owns(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def assess_risk(self, tool_name: str, args: dict, context: ToolContext) -> ToolRisk:
        return ToolRisk(
            self.risk,
            f"External MCP tool from server '{self.name}'",
            tool_name,
        )

    async def call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
        *,
        approved: bool = False,
    ) -> str:
        if not self.owns(tool_name):
            return ToolResult.failure(f"Unknown MCP tool: {tool_name}").to_json()
        risk = self.assess_risk(tool_name, args, context)
        if context.approval_mode == "never" and risk.requires_approval:
            return ToolResult.failure(
                f"MCP tool blocked by approval_mode=never: {tool_name}",
                blocked=True,
                risk=risk.level.value,
            ).to_json()
        if context.approval_mode == "on_request" and risk.requires_approval and not approved:
            return ToolResult.failure(
                f"Approval required for MCP tool: {tool_name}",
                blocked=True,
                approval_required=True,
                risk=risk.level.value,
            ).to_json()
        original_name = str(self._tools[tool_name]["name"])
        try:
            result = await self._request("tools/call", {"name": original_name, "arguments": args})
        except Exception as exc:
            return ToolResult.failure(f"MCP call failed: {exc}", server=self.name).to_json()
        return _mcp_result(result, self.name)

    def status(self) -> dict[str, Any]:
        connected = bool(self.process and self.process.returncode is None)
        return {
            "name": self.name,
            "transport": "stdio",
            "connected": connected,
            "tools": self.tool_names(),
            "error": self._error,
            "stderr": self._stderr_lines[-5:],
        }

    async def _load_tools(self) -> None:
        cursor: str | None = None
        tools: list[dict[str, Any]] = []
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self._request("tools/list", params)
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self._tools = {}
        for tool in tools:
            original = str(tool.get("name") or "")
            if not original:
                continue
            exposed = _mcp_tool_name(self.namespace, original)
            if exposed in self._tools:
                digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
                exposed = f"{exposed[:55]}_{digest}"
            self._tools[exposed] = dict(tool)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        request_id = self._next_id
        self._next_id += 1
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.timeout)
            if not line:
                code = await self.process.wait()
                stderr = " | ".join(self._stderr_lines[-3:])
                raise RuntimeError(f"MCP server exited with code {code}: {stderr}")
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("id") != request_id:
                continue
            if payload.get("error"):
                error = payload["error"]
                raise RuntimeError(str(error.get("message") if isinstance(error, dict) else error))
            result = payload.get("result")
            return result if isinstance(result, dict) else {}

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, payload: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        self.process.stdin.write(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        )
        await self.process.stdin.drain()

    async def _drain_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        while True:
            line = await self.process.stderr.readline()
            if not line:
                return
            self._stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())
            self._stderr_lines = self._stderr_lines[-20:]


class CompositeToolProvider(ToolProvider):
    name = "composite"

    def __init__(self, providers: list[ToolProvider]) -> None:
        self.providers = providers
        self.errors: dict[str, str] = {}

    async def start(self) -> None:
        self.errors = {}
        for provider in self.providers:
            try:
                await provider.start()
            except Exception as exc:
                self.errors[provider.name] = str(exc)
                if isinstance(provider, MCPServerProvider):
                    provider._error = str(exc)

    async def close(self) -> None:
        for provider in reversed(self.providers):
            try:
                await provider.close()
            except Exception:
                pass

    def schemas(self) -> list[dict[str, Any]]:
        return [schema for provider in self.providers for schema in provider.schemas()]

    def tool_names(self) -> list[str]:
        return [name for provider in self.providers for name in provider.tool_names()]

    def catalog(self) -> list[dict[str, Any]]:
        return [item for provider in self.providers for item in provider.catalog()]

    def owns(self, tool_name: str) -> bool:
        return any(provider.owns(tool_name) for provider in self.providers)

    def assess_risk(self, tool_name: str, args: dict, context: ToolContext) -> ToolRisk:
        provider = self._provider(tool_name)
        if provider is None:
            return ToolRisk(RiskLevel.CRITICAL, "Unknown tool", tool_name)
        return provider.assess_risk(tool_name, args, context)

    async def call(
        self,
        tool_name: str,
        args: dict,
        context: ToolContext,
        *,
        approved: bool = False,
    ) -> str:
        provider = self._provider(tool_name)
        if provider is None:
            return ToolResult.failure(f"Unknown tool: {tool_name}", blocked=True).to_json()
        return await provider.call(tool_name, args, context, approved=approved)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "errors": dict(self.errors),
            "providers": [provider.status() for provider in self.providers],
        }

    def _provider(self, tool_name: str) -> ToolProvider | None:
        return next((provider for provider in self.providers if provider.owns(tool_name)), None)


def create_tool_provider(
    registry: ToolRegistry,
    toolsets: list[str],
    mcp_servers: dict[str, Any],
    workspace: str | Path,
    *,
    task_provider: ToolProvider | None = None,
    search_provider: ToolProvider | None = None,
) -> CompositeToolProvider:
    providers: list[ToolProvider] = [LocalToolProvider(registry, toolsets)]
    if task_provider is not None:
        providers.append(task_provider)
    if search_provider is not None:
        providers.append(search_provider)
    for name, server_config in (mcp_servers or {}).items():
        if not isinstance(server_config, dict) or not server_config.get("enabled", True):
            continue
        providers.append(MCPServerProvider(str(name), server_config, workspace))
    return CompositeToolProvider(providers)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return cleaned[:64] or "tool"


def _mcp_tool_name(server: str, tool: str) -> str:
    namespace = _safe_name(server)[:20]
    prefix = f"mcp__{namespace}__"
    remaining = max(1, 64 - len(prefix))
    return prefix + _safe_name(tool)[:remaining]


def _function_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _mcp_result(result: dict[str, Any], server: str) -> str:
    blocks = result.get("content") or []
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    structured = result.get("structuredContent")
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False))
    content = "\n".join(part for part in parts if part)
    if result.get("isError"):
        return ToolResult.failure(content or "MCP tool returned an error", server=server).to_json()
    return ToolResult.success(content, server=server).to_json()
