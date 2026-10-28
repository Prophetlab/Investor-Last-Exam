"""Reusable provider-specific model adapters.

The environment core speaks the small :class:`~prophet_bench_env.ports.model.ModelAdapter`
protocol.  This package contains optional managed-policy adapters that translate
that protocol to each provider's native API.  Provider packages are imported
lazily, so external-policy users do not need to install them.
"""

from .anthropic import AnthropicMessagesAdapter
from .config import ModelSettings
from .errors import (
    EmptyModelResponseError,
    ModelAdapterError,
    ProviderHTTPError,
    ProviderInvocationError,
)
from .factory import build_model_adapter, build_model_adapter_from_env
from .openai_compatible import OpenAICompatibleChatAdapter
from .openai_responses import OpenAIResponsesAdapter

__all__ = [
    "AnthropicMessagesAdapter",
    "EmptyModelResponseError",
    "ModelAdapterError",
    "ModelSettings",
    "OpenAICompatibleChatAdapter",
    "OpenAIResponsesAdapter",
    "ProviderHTTPError",
    "ProviderInvocationError",
    "build_model_adapter",
    "build_model_adapter_from_env",
]
