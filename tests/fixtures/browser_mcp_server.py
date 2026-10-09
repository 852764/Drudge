"""Offline stdio fixture: large discovery, notifications, stderr and child ownership."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def emit(value):
    print(json.dumps(value), flush=True)


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message or not method:
        continue
    request_id = message["id"]
    params = message.get("params", {})
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "offline", "version": "1"}}
    elif method == "tools/list":
        # Servers may send requests with IDs that coincide with client IDs.
        emit({"jsonrpc": "2.0", "id": request_id, "method": "sampling/createMessage", "params": {}})
        emit(["not-a-jsonrpc-object"])
        result = {"tools": [{"name": name, "description": "x" * 90000,
                             "inputSchema": {"type": "object", "properties": {}}}
                            for name in ("echo", "navigate_page", "not_allowed")]}
        if os.getenv("FIXTURE_REPEAT_CURSOR"):
            result["nextCursor"] = "again"
    elif method == "tools/call":
        args = params.get("arguments", {})
        if args.get("flood"):
            until = time.monotonic() + 3
            while time.monotonic() < until:
                emit({"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "progress"}})
                time.sleep(0.01)
        if args.get("stderr"):
            print("e" * 200000, file=sys.stderr, flush=True)
        if args.get("child_ready"):
            ready = Path(args["child_ready"])
            child = subprocess.Popen([sys.executable, "-u", "-c",
                                      f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); print('ready', flush=True); time.sleep(5)"])
            until = time.monotonic() + 3
            while not ready.exists() and time.monotonic() < until:
                time.sleep(0.01)
        text = {"text": args.get("text"), "has_secret": "FIXTURE_SECRET" in os.environ,
                "visible": os.getenv("FIXTURE_VISIBLE")}
        result = {"content": [{"type": "text", "text": json.dumps(text)}]}
    else:
        result = {}
    emit({"jsonrpc": "2.0", "id": request_id, "result": result})
