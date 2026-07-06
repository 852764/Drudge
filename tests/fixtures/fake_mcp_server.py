"""Tiny JSON-RPC MCP stdio server used by offline tests."""

from __future__ import annotations

import json
import os
import sys


for raw in sys.stdin:
    try:
        request = json.loads(raw)
    except json.JSONDecodeError:
        continue
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {"name": "fake", "version": "1"},
        }
    elif method == "tools/list":
        result = {
            "tools": [{
                "name": "echo",
                "description": "Echo one text value",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            }],
        }
    elif method == "tools/call":
        arguments = request.get("params", {}).get("arguments", {})
        result = {
            "content": [{
                "type": "text",
                "text": f"echo:{arguments.get('text', '')};pid:{os.getpid()}",
            }],
            "isError": False,
        }
    elif method == "resources/list":
        result = {
            "resources": [{
                "uri": "memory://readme",
                "name": "readme",
                "mimeType": "text/plain",
                "description": "demo resource",
            }],
        }
    elif method == "resources/read":
        result = {
            "contents": [{
                "uri": request.get("params", {}).get("uri", "memory://readme"),
                "mimeType": "text/plain",
                "text": "resource-body",
            }],
        }
    elif method == "prompts/list":
        result = {
            "prompts": [{
                "name": "summarize",
                "description": "summarize input",
            }],
        }
    elif method == "prompts/get":
        params = request.get("params", {})
        result = {
            "messages": [{
                "role": "user",
                "content": {"type": "text", "text": f"prompt:{params.get('name')}"},
            }],
        }
    else:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        }
        print(json.dumps(response), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
