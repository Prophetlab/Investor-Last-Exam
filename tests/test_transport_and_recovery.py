import json

import httpx

from ile_bench.config import route
from ile_bench.evaluation import answer_from_trial, grade_trial
from ile_bench.models import model
from ile_bench.tasks import load_tasks
from prophet_bench_env.ports.model import ModelRequest


def test_streamed_chat_retains_tool_arguments_and_usage():
    env = {
        "SOLVER_MODEL": "any-vendor",
        "SOLVER_BASE_URL": "https://example.invalid/v1",
        "SOLVER_API_KEY": "local-fixture",
        "SOLVER_STREAM": "true",
    }
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "submit_answer", "arguments": '{"x":'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
        "[DONE]",
    ]
    content = "".join(
        "data: " + (e if isinstance(e, str) else json.dumps(e)) + "\n\n" for e in events
    )

    def handler(req):
        assert req.url.path == "/v1/chat/completions" and json.loads(req.content)["stream"]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = model(route("SOLVER", env), client=client)
        result = adapter.generate(ModelRequest(({"role": "user", "content": "test"},), ()))
        assert (
            result.tool_calls[0]["arguments"] == {"x": 1} and result.usage["completion_tokens"] == 3
        )


def test_terminal_body_and_last_actual_submit_are_preserved(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    event = {
        "event_type": "model_response",
        "payload": {"content": "Actual terminal report.", "tool_calls": []},
    }
    (agent / "events.jsonl").write_text(json.dumps(event) + "\n")
    assert answer_from_trial(tmp_path) == ("Actual terminal report.", False, "terminal_answer")
    with (agent / "events.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "event_type": "tool_request",
                    "payload": {
                        "name": "submit_answer",
                        "arguments": {"answer": "last actual content"},
                    },
                }
            )
            + "\n"
        )
    assert answer_from_trial(tmp_path)[0] == {"answer": "last actual content"}


def test_infrastructure_failure_does_not_become_zero(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "ConnectionError"}})
    )
    route_ = route(
        "JUDGE_1",
        {
            "JUDGE_1_MODEL": "fake",
            "JUDGE_1_BASE_URL": "https://example.invalid",
            "JUDGE_1_API_KEY": "fixture",
        },
    )
    result = grade_trial(load_tasks(["M01"])[0], tmp_path, [("judge_1", route_)])
    assert (
        result["judge_1"]["status"] == "infrastructure_error" and result["judge_1"]["score"] is None
    )
