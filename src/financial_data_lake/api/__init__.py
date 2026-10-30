"""Canonical candidate factory, imported lazily by the public source entrypoint."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .final_app import create_app as create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .final_app import create_app

        return create_app
    raise AttributeError(name)
