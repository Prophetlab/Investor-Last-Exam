"""Execution-local audit relationships, never model-visible tool arguments."""

from contextlib import contextmanager
from contextvars import ContextVar

_tool_call = ContextVar("audit_tool_call", default=None)
_subagent = ContextVar("audit_subagent", default=None)


def correlation():
    return {
        "parent_tool_call_id": _tool_call.get(),
        "subagent_invocation_id": _subagent.get(),
    }


@contextmanager
def tool_call_scope(call_id):
    token = _tool_call.set(call_id)
    try:
        yield
    finally:
        _tool_call.reset(token)


@contextmanager
def subagent_scope(invocation_id, parent_tool_call_id=None):
    tool_token = _tool_call.set(parent_tool_call_id or _tool_call.get())
    sub_token = _subagent.set(invocation_id)
    try:
        yield
    finally:
        _subagent.reset(sub_token)
        _tool_call.reset(tool_token)
