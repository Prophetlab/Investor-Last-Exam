"""Standalone stdlib MCP/stdio endpoint copied into the untrusted workspace.

No network, database credentials, evaluator imports or host paths. Harbor's
control channel transports the mailbox to the trusted ToolGateway.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

MAX_MESSAGE = 2 * 1024 * 1024


def serve(root, *, timeout=900):
    manifest = json.loads((root / "tools.json").read_text())
    output_lock = threading.Lock()

    def send(value):
        with output_lock:
            print(json.dumps(value, ensure_ascii=False), flush=True)

    def handle(request):
        ident = request.get("id")
        if ident is None:
            return
        method = request.get("method")
        result = None
        try:
            if method == "initialize":
                version = request.get("params", {}).get("protocolVersion")
                result = {
                    "protocolVersion": version
                    if version in {"2024-11-05", "2025-03-26", "2025-06-18"}
                    else "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "prophet-tools", "version": "1.0"},
                    "instructions": manifest["instructions"],
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": manifest["tools"]}
            elif method == "tools/call":
                params = request.get("params", {})
                token = uuid4().hex
                temporary = root / "requests" / (token + ".tmp")
                destination = temporary.with_suffix(".json")
                temporary.write_text(
                    json.dumps(
                        {
                            "id": token,
                            "name": params.get("name"),
                            "arguments": params.get("arguments", {}),
                        }
                    )
                )
                os.replace(temporary, destination)
                response = root / "responses" / (token + ".json")
                deadline = time.monotonic() + timeout
                while not response.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("tool response timed out")
                    time.sleep(0.05)
                result = json.loads(response.read_text())
                # Host-owned responses stay read-only to the agent. Harbor
                # removes the entire ephemeral workspace at trial teardown.
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": ident,
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                )
                return
            send({"jsonrpc": "2.0", "id": ident, "result": result})
        except Exception as error:  # noqa: BLE001 - JSON-RPC error boundary
            send(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "error": {"code": -32603, "message": str(error)},
                }
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        while line := sys.stdin.buffer.readline(MAX_MESSAGE + 1):
            if len(line) > MAX_MESSAGE:
                raise ValueError("MCP message exceeds size limit")
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("expected a JSON-RPC object")
            except (ValueError, TypeError):
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Invalid JSON"},
                    }
                )
                continue
            executor.submit(handle, request)


def drain(root, limit=4):
    """Bound each control-channel read; only fixed request filenames are read."""
    rows = []
    for path in sorted((root / "requests").glob("*.json"))[:limit]:
        if (
            path.is_symlink()
            or len(path.stem) != 32
            or any(c not in "0123456789abcdef" for c in path.stem)
        ):
            raise ValueError("unsafe mailbox entry")
        if path.stat().st_size > MAX_MESSAGE:
            raise ValueError("tool request exceeds size limit")
        rows.append({"id": path.stem, "request": json.loads(path.read_text())})
        path.unlink()
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "drain"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.mode == "drain":
        drain(args.root, args.limit)
    else:
        serve(args.root, timeout=args.timeout)
