import hashlib
import json

from ile_bench.evaluation import judge_packet, validate_grade
from ile_bench.tasks import load_tasks


def test_public_task_identifiers_and_content_fingerprints():
    for task in load_tasks():
        assert task["task_id"] == task["id"]
        content = {k: v for k, v in task.items() if k != "source_sha256"}
        digest = hashlib.sha256(json.dumps(
            content, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        assert task["source_sha256"] == digest
        if task["judge_kind"] == "structured":
            assert task["judge_input_template"]["task_id"] == task["id"]
            assert task["judge_response_schema"]["properties"]["task_id"]["enum"] == [task["id"]]
            for tool in task["judge_tools"]:
                assert tool["function"]["parameters"]["properties"]["task_id"]["enum"] == [task["id"]]


def test_task_records_contain_current_definitions_without_editorial_metadata():
    forbidden = {
        "draft_expected_answer", "reference_correction", "revision", "verification_notes",
        "candidate_environment_hash", "expected_answer_status", "expert_certified",
        "supersedes_criterion_id", "condition_changed", "scoring_revision",
        "semantic_contract_revision", "removed_rubrics", "aggregation_entrypoint",
    }

    def check(value):
        if isinstance(value, dict):
            assert not set(value) & forbidden
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    for task in load_tasks():
        check(task)
        truth = task.get("ground_truth") or {}
        if "question_id" in truth:
            assert truth["question_id"] == task["id"]


def test_structured_grade_accepts_the_public_task_identity():
    task = load_tasks(["O37"])[0]
    packet, schema = judge_packet(task, {"answer_summary": "A financial forecast."})
    grade = {
        "task_id": "O37",
        "decisions": [
            {"rubric_id": r["rubric_id"], "passed": False, "rationale": "No supporting analysis."}
            for r in task["rubrics"] if r["type"] == "llm_judge"
        ],
    }
    # Required explanation fields remain governed by the supplied response schema.
    for key in schema.get("required", []):
        if key not in grade:
            prop = schema["properties"][key]
            grade[key] = [] if prop.get("type") == "array" else "No supporting analysis."
    result = validate_grade(task, packet, schema, grade)
    assert len(result) == len(grade["decisions"])
