from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from prophet_bench_env.model_adapters.config import ModelSettings, _read_env_file

ROOT = Path(__file__).resolve().parents[2]
if ROOT.name == "src":
    ROOT = ROOT.parent


def values(path: Path | None = None) -> dict[str, str]:
    return {**_read_env_file(path or ROOT / "local/.env"), **os.environ}


def boolean(value: str) -> bool:
    if value.lower() not in ("true", "false", "1", "0"):
        raise ValueError("Boolean settings must be true or false")
    return value.lower() in ("true", "1")


def url(value: str, name: str) -> str:
    p = urlsplit(value)
    if (
        p.scheme not in ("https", "http")
        or not p.hostname
        or p.username
        or p.password
        or p.query
        or p.fragment
    ):
        raise ValueError(name + " must be an HTTP(S) base URL without credentials/query")
    return value.rstrip("/")


@dataclass(frozen=True)
class Route:
    settings: ModelSettings
    stream: bool = True
    extra_body: dict = field(default_factory=dict, repr=False)
    extra_headers: dict = field(default_factory=dict, repr=False)

    def public(self):
        # Never export connection values or headers into the solver workspace.
        s = self.settings
        return {
            "model": s.model,
            "provider": s.provider,
            "protocol": s.reasoning_protocol,
            "max_output_tokens": s.max_output_tokens,
            "context_window_tokens": s.context_window_tokens,
            "reasoning_effort": s.reasoning_effort,
            "temperature": s.temperature,
            "stream": self.stream,
        }

    def fingerprint(self):
        content = {
            "profile": self.public(),
            "base_url": self.settings.base_url,
            "body": self.extra_body,
            "headers": self.extra_headers,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def route(prefix: str, env: dict[str, str]) -> Route:
    def get(key, default=""):
        return env.get(prefix + "_" + key, default).strip()

    missing = [k for k in ("MODEL", "BASE_URL", "API_KEY") if not get(k)]
    if missing:
        raise ValueError(prefix + ": fill " + ", ".join(missing))
    protocol = get("PROTOCOL", "chat")
    providers = {
        "chat": "openai_compatible",
        "responses": "openai_responses",
        "anthropic": "anthropic",
    }
    if protocol not in providers:
        raise ValueError(prefix + "_PROTOCOL must be chat, responses or anthropic")
    extra = json.loads(get("EXTRA_BODY", "{}") or "{}")
    headers = json.loads(get("EXTRA_HEADERS", "{}") or "{}")
    if not isinstance(extra, dict) or not isinstance(headers, dict):
        raise ValueError("EXTRA_BODY/HEADERS must be JSON objects")
    if set(extra) & {"model", "messages", "input", "instructions", "system", "tools", "stream"}:
        raise ValueError("EXTRA_BODY cannot override model, conversation, tools or stream")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()):
        raise ValueError("Header values must be strings")
    if get("TOP_P"):
        extra["top_p"] = float(get("TOP_P"))
    settings = ModelSettings(
        provider=providers[protocol],
        model=get("MODEL"),
        api_key=get("API_KEY"),
        base_url=url(get("BASE_URL"), prefix + "_BASE_URL"),
        reasoning_protocol=get("REASONING_PROTOCOL", "generic"),
        reasoning_effort=get("REASONING_EFFORT") or None,
        temperature=float(get("TEMPERATURE")) if get("TEMPERATURE") else None,
        max_output_tokens=int(get("MAX_OUTPUT_TOKENS", "32768")),
        context_window_tokens=int(get("CONTEXT_WINDOW_TOKENS", "131072")),
        timeout_seconds=float(get("TIMEOUT_SECONDS", "900")),
        max_retries=int(get("MAX_RETRIES", "3")),
        supports_images=boolean(get("SUPPORTS_IMAGES", "false")),
        store=False,
    )
    return Route(settings, boolean(get("STREAM", "true")), extra, headers)


def judges(env: dict[str, str]) -> list[tuple[str, Route]]:
    count = int(env.get("JUDGE_COUNT", "1"))
    if not 1 <= count <= 3:
        raise ValueError("JUDGE_COUNT must be 1, 2 or 3")
    return [(f"judge_{i}", route(f"JUDGE_{i}", env)) for i in range(1, count + 1)]


def data_environment(env: dict[str, str]) -> None:
    if not env.get("DATALAKE_API_KEY", "").strip():
        raise ValueError("Fill DATALAKE_API_KEY")
    for k in ("DATALAKE_API_KEY", "DATALAKE_PROJECT", "DATALAKE_PURPOSE"):
        if env.get(k):
            os.environ[k] = env[k]
    os.environ["DATALAKE_BASE_URL"] = url(
        env.get("DATALAKE_BASE_URL", "https://datalake.prophet-lab.cn"), "DATALAKE_BASE_URL"
    )
