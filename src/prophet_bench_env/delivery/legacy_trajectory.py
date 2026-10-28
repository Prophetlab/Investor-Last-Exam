"""Decode the historical delivery's content-addressed event storage, offline.

This only expands recorded message/schema references; it never infers subagent
relationships and never changes the source JSONL or its original hash chain.
"""

import copy
import hashlib
import json


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def expand_events(events):
    messages, schemas = {}, {}

    def decode(value):
        if isinstance(value, list):
            return [decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        encoding = value.get("encoding")
        if encoding == "message-delta-v1":
            for entry in value["new_messages"]:
                key, message = entry["message_id"], entry["value"]
                if digest(message) != key or (
                    key in messages and messages[key] != message
                ):
                    raise ValueError("historical message digest mismatch")
                messages[key] = message
            result = [copy.deepcopy(messages[key]) for key in value["message_refs"]]
            if len(result) != value["message_count"]:
                raise ValueError("historical message count mismatch")
            return result
        if encoding == "tool-schema-ref-v1":
            key = value["schema_ref"]
            if value.get("new_schema") is not None:
                schema = value["new_schema"]
                if digest(schema) != key or (key in schemas and schemas[key] != schema):
                    raise ValueError("historical tool schema digest mismatch")
                schemas[key] = schema
            result = copy.deepcopy(schemas[key])
            if len(result) != value["tool_count"]:
                raise ValueError("historical tool schema count mismatch")
            return result
        if encoding == "compaction-request-delta-v1":
            removed = decode(value["removed_messages"])
            return [
                {
                    "role": value["role"],
                    "content": value["instruction"]
                    + "\n\n"
                    + json.dumps(removed, ensure_ascii=False, separators=(",", ":")),
                }
            ]
        return {k: decode(v) for k, v in value.items()}

    for event in events:
        yield {**event, "payload": decode(event["payload"])}
