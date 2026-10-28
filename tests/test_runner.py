import json
from copy import deepcopy

import pytest

from ile_bench.config import judges, route
from ile_bench.evaluation import grade_answer, judge_packet, mechanical, parse_json, validate_grade
from ile_bench.models import model
from ile_bench.tasks import export_task, load_tasks
from prophet_bench_env.ports.model import ModelRequest, ModelResponse


def env(count=1):
    out = {"JUDGE_COUNT": str(count)}
    for prefix in ["SOLVER", *[f"JUDGE_{i}" for i in range(1, count + 1)]]:
        out.update(
            {
                prefix + "_MODEL": "custom-model",
                prefix + "_BASE_URL": "https://example.invalid/v1",
                prefix + "_API_KEY": "test-only",
                prefix + "_STREAM": "false",
            }
        )
    return out


def test_arbitrary_models_and_independent_judge_count():
    for n in (1, 2, 3):
        assert len(judges(env(n))) == n
    for n in (0, 4):
        with pytest.raises(ValueError):
            judges(env(n))
    e = env()
    e["SOLVER_EXTRA_BODY"] = '{"top_k":32,"thinking":{"type":"enabled"}}'
    r = route("SOLVER", e)
    assert r.settings.model == "custom-model" and r.extra_body["top_k"] == 32
    assert "test-only" not in json.dumps(r.public())
    e["SOLVER_EXTRA_BODY"] = '{"tools":[]}'
    with pytest.raises(ValueError):
        route("SOLVER", e)


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic"])
def test_provider_request_shape_and_no_hardcoded_routes(protocol):
    e = env()
    e["SOLVER_PROTOCOL"] = protocol
    e["SOLVER_TOP_P"] = "0.7"
    r = route("SOLVER", e)
    adapter = model(r)
    try:
        payload = adapter._payload(
            ModelRequest(
                (
                    {"role": "system", "content": "original system"},
                    {"role": "user", "content": "hello"},
                ),
                (),
            )
        )
        assert payload["model"] == "custom-model" and payload["top_p"] == 0.7
        assert not payload.get("stream", False)
        if protocol == "responses":
            assert payload["instructions"] == "original system"
        if protocol == "anthropic":
            assert payload["system"] == "original system"
    finally:
        adapter.close()


def test_only_public_inputs_enter_harbor_task(tmp_path):
    task = load_tasks(["O01"])[0]
    dest = tmp_path / "O01"
    export_task(task, dest, 7200)
    text = "\n".join(p.read_text() for p in dest.rglob("*") if p.is_file())
    assert task["question"] in (dest / "instruction.md").read_text()
    for forbidden in [
        "verified_truth",
        "judge_system_prompt",
        "rubric_id",
        "API_KEY",
        "reference_answer",
    ]:
        assert forbidden not in text
    assert "no-network" in (dest / "task.toml").read_text()
    assert not (dest / "tests").exists()


def test_market_known_outcome_is_exact_and_invalid_is_zero():
    task = load_tasks(["M01"])[0]
    actual = float(task["evaluation_policy"]["market"]["actual_close"])
    assert mechanical(task, {"forecast_close_usd": actual})["future_outcome_loss"]["score"] == 100
    assert mechanical(task, {"forecast_close_usd": -1})["future_outcome_loss"]["score"] == 0


def test_numeric_unit_scale_and_linear_credit():
    task = load_tasks(["O05"])[0]
    r = next(
        r
        for r in task["rubrics"]
        if r.get("future_truth", {}) and r["future_truth"]["scoring_method"] == "numeric_linear"
    )
    truth = r["future_truth"]
    key = r["primary_output_path"].split(".")[-1]
    value = truth["expected_value"] / truth["answer_scale_to_truth"]
    exact = mechanical(task, {"structured_output": {key: value}})[r["rubric_id"]]["score"]
    assert exact == pytest.approx(r["max_score"])
    assert mechanical(task, {"structured_output": {key: value * 2}})[r["rubric_id"]]["score"] == 0


def test_tp05_original_numeric_and_probability_rules():
    task = load_tasks(["TP05"])[0]
    payload = {
        "structured_output": {
            "q4_forecast": {"net_sales_usd_m": 1876.6},
            "revenue_probability": {"p_at_or_above": 0.5, "p_below": 0.5},
            "revenue_bridge": {"implied_q4_revenue_usd_m": 1837},
        }
    }
    scores = mechanical(task, payload)
    assert scores["R04"]["score"] == 10 and scores["R12"]["score"] == 11.25
    payload["structured_output"]["revenue_bridge"]["implied_q4_revenue_usd_m"] = 1700
    assert mechanical(task, payload)["R12"]["score"] == 0


def test_three_judges_never_share_grade_or_average():
    task = load_tasks(["N01"])[0]
    payload = {"answer": "One immutable report. Evidence and reasoning."}
    packet, schema = judge_packet(task, payload)
    calls = []

    class Fake:
        def __init__(self, credit):
            self.credit = credit

        def generate(self, request):
            calls.append(deepcopy(request.messages))
            decisions = {
                rid: {
                    "score_fraction": self.credit,
                    "answer_span_ids": ["A1"],
                    "rationale": "Evidence assessed independently.",
                }
                for rid in schema["properties"]["decisions_by_id"]["properties"]
            }
            return ModelResponse(json.dumps({"decisions_by_id": decisions}))

    routes = judges(env(3))
    scores = []
    for (name, r), credit in zip(routes, [0, 1, 0]):
        scores.append(grade_answer(task, payload, True, r, adapter=Fake(credit))["score"])
    assert scores == [0, 100, 0] and calls[0] == calls[1] == calls[2]


def test_unknown_spans_duplicate_ids_and_fractional_binary_rejected():
    task = load_tasks(["N01"])[0]
    packet, schema = judge_packet(task, {"answer": "Known report."})
    grade = {
        "decisions_by_id": {
            rid: {
                "score_fraction": 1,
                "answer_span_ids": ["A1"],
                "rationale": "A supported conclusion.",
            }
            for rid in schema["properties"]["decisions_by_id"]["properties"]
        }
    }
    validate_grade(task, packet, schema, grade)
    first = next(iter(grade["decisions_by_id"].values()))
    first["answer_span_ids"] = ["A1", "A1"]
    with pytest.raises(ValueError):
        validate_grade(task, packet, schema, grade)
    with pytest.raises(ValueError):
        parse_json('{"a":1,"a":2}')
