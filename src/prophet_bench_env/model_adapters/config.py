"""Provider-neutral settings with explicit provider/protocol selection."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("boolean model setting must be true/false")


def _read_env_file(path: Path) -> dict[str, str]:
    """Read the small KEY=value subset used by the project's `.env` files."""

    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Settings for one provider route.

    ``reasoning_protocol`` is intentionally explicit.  Two OpenAI-compatible
    endpoints can require different fields, so the adapter never infers a
    protocol from a hostname or silently changes transport.
    """

    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    temperature: float | None = None
    max_output_tokens: int = 24_000
    context_window_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_protocol: str = "generic"
    timeout_seconds: float = 600.0
    max_retries: int = 5
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    empty_response_retries: int = 2
    empty_response_base_seconds: float = 1.0
    store: bool | None = None
    supports_images: bool = False

    def __post_init__(self) -> None:
        if self.provider not in {"openai_compatible", "openai_responses", "anthropic"}:
            raise ValueError(
                "provider must be openai_compatible, openai_responses, or anthropic"
            )
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if self.provider == "openai_compatible" and not (self.base_url or "").strip():
            raise ValueError("openai-compatible provider requires base_url")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.context_window_tokens is not None and self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0 or self.empty_response_retries < 0:
            raise ValueError("retry counts must not be negative")
        if self.retry_base_seconds < 0 or self.retry_max_seconds < 0:
            raise ValueError("retry delays must not be negative")
        if self.empty_response_base_seconds < 0:
            raise ValueError("empty response delay must not be negative")

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "MODEL_",
        environ: Mapping[str, str] | None = None,
        env_file: str | Path = ".env",
        provider: str | None = None,
    ) -> ModelSettings:
        """Load one adapter from deployment environment variables.

        Required variables are ``<prefix>PROVIDER``, ``<prefix>NAME`` and
        ``<prefix>API_KEY``.  ``<prefix>BASE_URL`` is required for an
        OpenAI-compatible gateway and optional for adapters with a native public
        default.
        """

        file_values = _read_env_file(Path(env_file))
        file_values.update(os.environ)
        if environ is not None:
            file_values.update(environ)
        values = file_values

        def get(name: str, default: str | None = None) -> str | None:
            value = values.get(prefix + name, default)
            return value.strip() if isinstance(value, str) else value

        provider = provider or get("PROVIDER") or "openai_compatible"
        model = get("NAME") or ""
        api_key = get("API_KEY") or ""
        default_base = {
            "openai_responses": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
        }.get(provider)
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=get("BASE_URL", default_base),
            temperature=_optional_float(get("TEMPERATURE")),
            max_output_tokens=int(get("MAX_OUTPUT_TOKENS", "24000") or "24000"),
            context_window_tokens=_optional_int(get("CONTEXT_WINDOW_TOKENS")),
            reasoning_effort=get("REASONING_EFFORT"),
            reasoning_protocol=get("REASONING_PROTOCOL", "generic") or "generic",
            timeout_seconds=float(get("TIMEOUT_SECONDS", "600") or "600"),
            max_retries=int(get("MAX_RETRIES", "5") or "5"),
            retry_base_seconds=float(get("RETRY_BASE_SECONDS", "5") or "5"),
            retry_max_seconds=float(get("RETRY_MAX_SECONDS", "60") or "60"),
            empty_response_retries=int(get("EMPTY_RESPONSE_RETRIES", "2") or "2"),
            empty_response_base_seconds=float(
                get("EMPTY_RESPONSE_BASE_SECONDS", "1") or "1"
            ),
            store=_optional_bool(get("STORE")),
            supports_images=bool(_optional_bool(get("SUPPORTS_IMAGES")) or False),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ModelSettings:
        """Build settings from a manifest/config mapping without exposing secrets."""

        required = {"provider", "model", "api_key"}
        missing = sorted(
            key for key in required if not str(values.get(key, "")).strip()
        )
        if missing:
            raise ValueError(f"missing model settings: {', '.join(missing)}")
        fields = {
            "provider",
            "model",
            "api_key",
            "base_url",
            "temperature",
            "max_output_tokens",
            "context_window_tokens",
            "reasoning_effort",
            "reasoning_protocol",
            "timeout_seconds",
            "max_retries",
            "retry_base_seconds",
            "retry_max_seconds",
            "empty_response_retries",
            "empty_response_base_seconds",
            "store",
            "supports_images",
        }
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"unknown model settings: {', '.join(unknown)}")
        return cls(**{key: values[key] for key in fields if key in values})  # type: ignore[arg-type]

    def with_overrides(self, **updates: object) -> ModelSettings:
        """Apply pipeline-owned behavior settings without copying the API key.

        A deployment can keep connection secrets and model name in `.env` while
        each pipeline fixes its own protocol, reasoning effort, temperature and
        budgets in code or a manifest.
        """

        fields = set(self.__dataclass_fields__)
        unknown = sorted(set(updates) - fields)
        if unknown:
            raise ValueError(f"unknown model settings: {', '.join(unknown)}")
        return replace(self, **updates)
