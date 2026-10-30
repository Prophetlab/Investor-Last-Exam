"""Explicit model adapter factory."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .anthropic import AnthropicMessagesAdapter
from .config import ModelSettings
from .openai_compatible import OpenAICompatibleChatAdapter
from .openai_responses import OpenAIResponsesAdapter


def build_model_adapter(
    settings: ModelSettings | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    prefix: str = "MODEL_",
    client=None,
):
    """Construct exactly one protocol-specific adapter.

    The provider kind is never inferred from a URL.  This is important for
    gateways that expose several wire protocols under the same host.
    """

    resolved = settings or ModelSettings.from_env(prefix=prefix, environ=environ)
    if resolved.provider == "openai_responses":
        return OpenAIResponsesAdapter(resolved, client=client)
    if resolved.provider == "openai_compatible":
        return OpenAICompatibleChatAdapter(resolved, client=client)
    if resolved.provider == "anthropic":
        return AnthropicMessagesAdapter(resolved, client=client)
    # ModelSettings validates this branch; keep a defensive error if a custom
    # dataclass-like object is passed by a caller.
    raise ValueError(f"unsupported model provider: {resolved.provider}")


def build_model_adapter_from_env(*, prefix: str = "MODEL_", environ=None, client=None):
    """Explicit alias for factories that prefer a descriptive name."""

    return build_model_adapter(
        ModelSettings.from_env(prefix=prefix, environ=environ or os.environ),
        client=client,
    )
