"""Provider capability probing for OpenAI-compatible model endpoints."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .llm import create_client


ClientFactory = Callable[[dict[str, Any]], Any]


@dataclass
class CapabilityResult:
    supported: bool
    latency_ms: int
    status_code: int | None = None
    response_model: str | None = None
    text_received: bool = False
    tool_call_received: bool = False
    streamed: bool = False
    error: str | None = None


@dataclass
class ProviderProbeReport:
    base_url: str
    model: str
    listed: bool | None = None
    model_count: int | None = None
    models_error: str | None = None
    capabilities: dict[str, CapabilityResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


async def probe_provider(
    model_config: dict[str, Any],
    *,
    model: str | None = None,
    include_streaming: bool = True,
    client_factory: ClientFactory = create_client,
) -> ProviderProbeReport:
    config = dict(model_config)
    if model:
        config["name"] = model
    target_model = str(config.get("name") or "")
    report = ProviderProbeReport(
        base_url=str(config.get("base_url") or ""),
        model=target_model,
    )

    list_client = client_factory(_probe_config(config, api=str(config.get("api") or "auto")))
    try:
        models = await list_client.list_models()
        report.model_count = len(models)
        report.listed = target_model in models
    except Exception as error:
        report.models_error = _safe_error(error)

    for api_type in ("chat", "responses"):
        report.capabilities[f"{api_type}.basic"] = await _probe_call(
            config,
            api_type=api_type,
            tools=None,
            streaming=False,
            client_factory=client_factory,
        )
        report.capabilities[f"{api_type}.tools"] = await _probe_call(
            config,
            api_type=api_type,
            tools=[_probe_tool_schema()],
            streaming=False,
            client_factory=client_factory,
        )
        if include_streaming:
            report.capabilities[f"{api_type}.streaming"] = await _probe_call(
                config,
                api_type=api_type,
                tools=None,
                streaming=True,
                client_factory=client_factory,
            )
    return report


async def _probe_call(
    model_config: dict[str, Any],
    *,
    api_type: str,
    tools: list[dict] | None,
    streaming: bool,
    client_factory: ClientFactory,
) -> CapabilityResult:
    client = client_factory(_probe_config(model_config, api=api_type))
    deltas: list[str] = []
    prompt = "Call probe_echo with value OK." if tools else "Reply with exactly OK."
    started = time.perf_counter()
    try:
        response = await client.chat(
            [{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="required" if tools else None,
            stream_callback=deltas.append if streaming else None,
        )
        text = client.extract_text(response) or ""
        tool_calls = client.extract_tool_calls(response)
        supported = bool(tool_calls) if tools else True
        return CapabilityResult(
            supported=supported,
            latency_ms=_elapsed_ms(started),
            response_model=str(response.get("model") or "") or None,
            text_received=bool(text),
            tool_call_received=bool(tool_calls),
            streamed=bool(deltas),
            error=None if supported else "Provider accepted the tool schema but returned no tool call.",
        )
    except Exception as error:
        message = _safe_error(error)
        return CapabilityResult(
            supported=False,
            latency_ms=_elapsed_ms(started),
            status_code=_status_code(message),
            error=message,
        )


def format_probe_report(report: ProviderProbeReport) -> str:
    lines = [
        f"Provider probe: {report.base_url}",
        f"Model: {report.model}",
    ]
    if report.models_error:
        lines.append(f"Model list: error ({report.models_error})")
    else:
        lines.append(f"Model list: {'listed' if report.listed else 'not listed'} ({report.model_count or 0} models)")
    lines.append("Capabilities:")
    for name, result in report.capabilities.items():
        state = "yes" if result.supported else "no"
        detail = f"{result.latency_ms}ms"
        if result.status_code:
            detail += f", HTTP {result.status_code}"
        if result.tool_call_received:
            detail += ", tool_call=yes"
        if result.streamed:
            detail += ", delta=yes"
        lines.append(f"  {name}: {state} ({detail})")
        if result.error:
            lines.append(f"    {result.error}")
    return "\n".join(lines)


def _probe_config(model_config: dict[str, Any], *, api: str) -> dict[str, Any]:
    config = dict(model_config)
    config["api"] = api
    config["temperature"] = 0
    config["max_tokens"] = min(int(config.get("max_tokens", 64)), 64)
    config["max_retries"] = 1
    config["aliases"] = {}
    return config


def _probe_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "probe_echo",
            "description": "Return the supplied value for provider capability testing.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _status_code(message: str) -> int | None:
    match = re.search(r"HTTP\s+(\d{3})", message)
    return int(match.group(1)) if match else None


def _safe_error(error: Exception) -> str:
    return str(error).replace("\r", " ").replace("\n", " ")[:1000]
