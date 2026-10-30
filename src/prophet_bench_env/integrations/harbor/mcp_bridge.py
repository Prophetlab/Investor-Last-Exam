"""Bounded tool transport over Harbor control I/O, without opening a server port."""

import asyncio
import json
import re
import shlex
from pathlib import Path
from uuid import uuid4

from harbor.models.task.config import MCPServerConfig

from . import mcp_stdio
from .serialization import write_json


class MailboxBridge:
    def __init__(self, environment, session, logs_dir, *, timeout=900):
        self.environment, self.session, self.logs_dir = environment, session, logs_dir
        self.root = "/tmp/prophet-mcp-" + uuid4().hex
        self.script = self.root + "/server.py"
        self.pending = set()
        self.seen = set()
        self.timeout = timeout

    async def setup(self):
        result = await self.environment.exec(
            f"mkdir -p {self.root}/requests {self.root}/responses && chmod 1777 {self.root}/requests && chmod 755 {self.root}/responses",
            user="root",
            timeout_sec=15,
        )
        if result.return_code:
            raise RuntimeError("cannot initialize tool transport")
        await self.environment.upload_file(Path(mcp_stdio.__file__), self.script)
        manifest = self.logs_dir / "mcp-tools.json"
        write_json(
            manifest,
            {
                "tools": self.session.definitions(),
                "instructions": f"Use these tools to access the task's research materials. Submit the final answer with {self.session.task.submit_tool.name}.",
            },
        )
        await self.environment.upload_file(manifest, self.root + "/tools.json")
        readable = await self.environment.exec(
            f"chmod 755 {self.root} && chmod 644 {self.script} {self.root}/tools.json",
            user="root",
            timeout_sec=15,
        )
        if readable.return_code:
            raise RuntimeError("cannot expose public tool schemas to the agent")
        return MCPServerConfig(
            name="prophet",
            transport="stdio",
            command="python",
            args=[self.script, "serve", self.root, "--timeout", str(self.timeout)],
        )

    async def _dispatch(self, token, request):
        # The sandbox is untrusted. Never use its id/name as a host path or import.
        if not isinstance(request, dict) or request.get("id") != token:
            raise ValueError("malformed tool request")
        data = await asyncio.to_thread(
            self.session.call,
            "mcp_" + token,
            request.get("name"),
            request.get("arguments"),
        )
        response = {
            "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
            "isError": not data["ok"],
        }
        local = self.logs_dir / "mcp-responses" / (token + ".json")
        write_json(local, response)
        target = self.root + "/responses/" + token
        await self.environment.upload_file(local, target + ".tmp")
        published = await self.environment.exec(
            f"chmod 644 {target}.tmp && mv -- {target}.tmp {target}.json",
            user="root",
            timeout_sec=15,
        )
        if published.return_code:
            raise RuntimeError("cannot publish tool response")

    async def pump(self):
        while True:
            for task in list(self.pending):
                if task.done():
                    self.pending.remove(task)
                    task.result()
            if len(self.pending) < 4:
                result = await self.environment.exec(
                    f"python {shlex.quote(self.script)} drain {shlex.quote(self.root)} --limit {4 - len(self.pending)}",
                    timeout_sec=15,
                )
                if result.return_code:
                    raise RuntimeError("tool transport read failed")
                rows = json.loads(result.stdout or "[]")
                if not isinstance(rows, list) or len(rows) > 4 - len(self.pending):
                    raise ValueError("invalid tool request batch")
                for row in rows:
                    token = row.get("id") if isinstance(row, dict) else None
                    if (
                        not isinstance(token, str)
                        or not re.fullmatch(r"[0-9a-f]{32}", token)
                        or token in self.seen
                    ):
                        raise ValueError("unsafe or repeated tool request identifier")
                    self.seen.add(token)
                    self.pending.add(
                        asyncio.create_task(self._dispatch(token, row["request"]))
                    )
            await asyncio.sleep(0.2)

    async def finish(self):
        # Let bounded provider calls finish before closing their clients. The
        # backend is already cancelled on abnormal shutdown, blocking new execs.
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
