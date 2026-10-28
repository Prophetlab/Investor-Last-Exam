"""Runtime loading for a caller-provided pipeline factory."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from ..core.engine import EpisodeEngine
from .server import CreateEpisodeRequest

EngineFactory = Callable[[CreateEpisodeRequest], EpisodeEngine]


def load_engine_factory(spec: str | None) -> EngineFactory | None:
    """Load ``module:function`` without making plugin code a core dependency."""

    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("pipeline factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"pipeline factory {spec!r} is not callable")
    return factory
