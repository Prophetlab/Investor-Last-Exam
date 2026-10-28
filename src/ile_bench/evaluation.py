"""Original mechanical rules and independent semantic judges; no judge ensembles."""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from prophet_bench_env.core.contracts import ToolDefinition
from prophet_bench_env.integrations.harbor.serialization import write_json
from prophet_bench_env.ports.model import ModelRequest

from .answer_spans import answer_spans
from .author_checks import numeric as author_numeric
from .checks import CHECKERS_V11, decision_fraction_v14, execute_future_truth_rubrics_v11
from .corrected_checks import custom_decision
from .market import score_answer
from .models import model
from .records import Record


def unique_object(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("Duplicate JSON key: " + key)
        out[key] = value
    return out


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Non-finite JSON number")),
    )


def answer_from_trial(trial: Path):
    """Accepted payload, last actual submit payload, or literal terminal response."""
    path = trial / "agent/submission.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("submission_state", {}).get("accepted"):
            return record["payload"], True, "accepted_submission"
    last_submit = None
    terminal = None
    events = trial / "agent/events.jsonl"
    if events.exists():
        for line in events.open():
            event = json.loads(line)
            p = event.get("payload", {})
            if event["event_type"] == "tool_request" and p.get("name") == "submit_answer":
                last_submit = p.get("arguments")
            if (
                event["event_type"] == "model_response"
                and not p.get("tool_calls")
                and str(p.get("content") or "").strip()
            ):
                terminal = p["content"]
    payload = last_submit if last_submit is not None else terminal
    if isinstance(payload, str):
        try:
            payload = parse_json(payload)
        except (ValueError, TypeError):
            pass
    return (
        payload,
        False,
        "last_submit"
        if last_submit is not None
        else "terminal_answer"
        if terminal
        else "no_answer",
    )


def mechanical(task, payload):
    if task["submission_mode"] == "market":
        try:
            result = score_answer(
                {
                    "public": {"submission_schema": task["submission_schema"]},
                    "private": {"market_evaluation": task["evaluation_policy"]["market"]},
                },
                payload,
            )
            return {"future_outcome_loss": result["rubric_results"][0]}
        except (ValueError, TypeError, KeyError, ValidationError):
            return {
                "future_outcome_loss": {
                    "score": 0,
                    "max_score": 100,
                    "diagnostic": "Missing or invalid forecast",
                }
            }
    obj = (
        payload
        if isinstance(payload, dict)
        else {"answer_summary": str(payload), "structured_output": {}}
    )
    if task["id"] == "TP05":
        result = author_numeric({"code": "TP05", "private": {"rubrics": task["rubrics"]}}, obj)
        return {
            rid: {"score": r["points"], "diagnostic": r["details"]} for rid, r in result.items()
        }
    out = {}
    for r in task["rubrics"]:
        if r["type"] != "programmatic":
            continue
        binding = r.get("deterministic")
        future = r.get("future_truth")
        custom = (binding or {}).get("checker_id", "").startswith("p0_") or (future or {}).get(
            "scoring_method", ""
        ).startswith("p0_")
        if custom:
            d = custom_decision(
                r,
                SimpleNamespace(structured_output=obj.get("structured_output", {})),
                Draft202012Validator,
            )
            fraction = (
                d["score_fraction"] if d.get("score_fraction") is not None else float(d["passed"])
            )
            reason = d["rationale"]
        elif binding:
            # Unknown rules are implementation failures, not model failures.
            fn = CHECKERS_V11[binding["checker_id"]]
            try:
                passed, _, _, reason = fn(obj, binding["parameters"])
                fraction = float(passed)
            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                fraction = 0
                reason = "Missing or invalid submitted value"
        elif future:
            d = execute_future_truth_rubrics_v11(
                [Record(r)], Record(obj), {r["rubric_id"]: future}
            )[0]
            fraction = decision_fraction_v14(d)
            reason = d.rationale
        else:
            raise ValueError("No executable rule for " + r["rubric_id"])
        out[r["rubric_id"]] = {"score": r["max_score"] * fraction, "diagnostic": reason}
    return out


def judge_packet(task, payload):
    packet = deepcopy(task["judge_input_template"])
    schema = deepcopy(task["judge_response_schema"])
    if task["judge_kind"] == "structured":
        packet["anonymous_answer"] = payload
    else:
        text = payload.get("answer") if isinstance(payload, dict) else payload
        if not isinstance(text, str):
            text = json.dumps(payload, ensure_ascii=False)
        packet["submitted_answer"] = text
        packet["answer_spans"] = answer_spans(text)
        allowed = [s["span_id"] for s in packet["answer_spans"]]
        for item in schema["properties"]["decisions_by_id"]["properties"].values():
            item["properties"]["answer_span_ids"]["items"] = (
                {"type": "string", "enum": allowed} if allowed else {"type": "string"}
            )
    return packet, schema


def validate_grade(task, packet, schema, grade):
    Draft202012Validator(schema).validate(grade)
    rubrics = {r["rubric_id"]: r for r in task["rubrics"] if r["type"] == "llm_judge"}
    if "decisions_by_id" in grade:
        items = grade["decisions_by_id"]
        known = {s["span_id"] for s in packet["answer_spans"]}
        if set(items) != set(rubrics):
            raise ValueError("Judge must cover every semantic rubric exactly once")
        for rid, d in items.items():
            spans = d["answer_span_ids"]
            fraction = d["score_fraction"]
            if len(set(spans)) != len(spans) or set(spans) - known:
                raise ValueError("Unknown or duplicate answer span")
            if fraction > 0 and not spans:
                raise ValueError("Positive credit requires actual answer evidence")
            if not d["rationale"].strip():
                raise ValueError("Blank rationale")
            r = rubrics[rid]
            detail = r.get("source_detail") or {}
            allowed = detail.get("allowed_credit")
            if detail.get("component_count"):
                allowed = [
                    i / detail["component_count"] for i in range(detail["component_count"] + 1)
                ]
            if allowed is None and (
                r["verification"] == "judge" or r.get("evaluation_type") == "binary"
            ):
                allowed = [0, 1]
            if allowed is not None and not any(abs(fraction - x) < 1e-8 for x in allowed):
                raise ValueError("Rubric does not allow this fractional credit")
        return {
            rid: {"fraction": d["score_fraction"], "rationale": d["rationale"]}
            for rid, d in items.items()
        }
    items = grade["decisions"]
    ids = [d["rubric_id"] for d in items]
    if len(ids) != len(set(ids)) or set(ids) != set(rubrics):
        raise ValueError("Judge must cover every semantic rubric exactly once")
    if grade.get("task_id") != task["task_id"]:
        raise ValueError("Judge task ID mismatch")
    return {
        d["rubric_id"]: {"fraction": float(d["passed"]), "rationale": d["rationale"]} for d in items
    }


def grade_answer(task, payload, accepted, judge_route, *, json_mode="prompt", adapter=None):
    fixed = mechanical(task, payload)
    semantic = [r for r in task["rubrics"] if r["type"] == "llm_judge"]
    decisions = {}
    attempts = []
    client = adapter
    if semantic and payload is not None:
        packet, schema = judge_packet(task, payload)
        tools = tuple(
            ToolDefinition(t["function"]["name"], t["function"]["description"], schema)
            for t in task["judge_tools"]
        )
        messages = [
            {"role": "system", "content": task["judge_system_prompt"]},
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ]
        if not tools and json_mode == "prompt":
            messages[-1]["content"] += "\n\nRequired JSON output schema:\n" + json.dumps(
                schema, ensure_ascii=False
            )
        client = client or model(judge_route)
        try:
            for attempt in range(3):
                response = client.generate(
                    ModelRequest(
                        tuple(messages),
                        tools,
                        metadata={"output_schema": schema}
                        if not tools and json_mode == "native"
                        else {},
                    )
                )
                try:
                    if tools and response.tool_calls:
                        if (
                            len(response.tool_calls) != 1
                            or response.tool_calls[0]["name"] != tools[0].name
                        ):
                            raise ValueError("Expected exactly one grade submission")
                        grade = response.tool_calls[0]["arguments"]
                        grade = parse_json(grade) if isinstance(grade, str) else grade
                    else:
                        grade = parse_json(response.text)
                    decisions = validate_grade(task, packet, schema, grade)
                    attempts.append(
                        {"attempt": attempt + 1, "valid": True, "usage": response.usage}
                    )
                    break
                except (ValueError, KeyError, TypeError, ValidationError) as error:
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "valid": False,
                            "error_type": type(error).__name__,
                            "usage": response.usage,
                        }
                    )
                    if attempt == 2:
                        raise
                    messages.append(
                        {
                            "role": "user",
                            "content": "Return the complete grade again, correcting only this format or evidence-reference problem: "
                            + str(error)[:2000],
                        }
                    )
        finally:
            if adapter is None:
                client.close()
    rows = []
    for r in task["rubrics"]:
        rid = r["rubric_id"]
        maximum = r["max_score"]
        if payload is None:
            score = 0
            reason = "No identifiable final answer"
        elif rid in fixed:
            score = fixed[rid]["score"]
            reason = fixed[rid]["diagnostic"]
        else:
            score = maximum * decisions[rid]["fraction"]
            reason = decisions[rid]["rationale"]
        rows.append(
            {
                "rubric_id": rid,
                "score": score,
                "max_score": maximum,
                "type": r["type"],
                "diagnostic": reason,
            }
        )
    raw = sum(Fraction(str(r["score"])) for r in rows)
    # Retain unrounded values as well as the original one-decimal display convention.
    total = float(Decimal(str(float(raw))).quantize(Decimal(".1"), rounding=ROUND_HALF_UP))
    return {
        "status": "scored",
        "score": total,
        "score_unrounded": float(raw),
        "strict_score": total if accepted else 0,
        "accepted_submission": accepted,
        "rubric_results": rows,
        "judge_attempts": attempts,
        "judge": judge_route.public() if semantic else None,
        "semantic_judge_called": bool(semantic and payload is not None),
    }


def grade_trial(task, trial, judges, *, json_mode="prompt"):
    payload, accepted, source = answer_from_trial(trial)
    results = {}
    trial_record = (
        json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    )
    failure = (trial_record.get("exception_info") or {}).get("exception_type")
    noncompletion = {
        "SubmissionRetriesExceededError",
        "MissingSubmissionError",
        "AgentTimeoutError",
        "TrialTimeoutError",
        "RepeatedToolCallsError",
    }
    for name, judge_route in judges:
        target = trial / "verifier" / f"{name}.json"
        try:
            if payload is None and failure and failure not in noncompletion:
                result = {
                    "status": "infrastructure_error",
                    "score": None,
                    "strict_score": None,
                    "error_type": failure,
                }
            else:
                result = grade_answer(task, payload, accepted, judge_route, json_mode=json_mode)
        except Exception as error:
            result = {
                "status": "grading_error",
                "score": None,
                "strict_score": None,
                "error_type": type(error).__name__,
            }
        result.update(task_code=task["id"], answer_source=source)
        write_json(target, result)
        results[name] = result
    return results
