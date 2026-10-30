import json
from copy import deepcopy

import pytest

from ile_bench.checks import CHECKERS_V11
from ile_bench.evaluation import judge_packet, mechanical
from ile_bench.judge_references import with_reference_materials
from ile_bench.tasks import export_task, load_tasks


def test_o11_counts_distinct_counterparties():
    task = load_tasks(["O11"])[0]
    rid = "core_counterparty_edge_coverage"

    def score(tickers):
        edges = [{"counterparty": "Company", "ticker": ticker} for ticker in tickers]
        return mechanical(task, {"structured_output": {"edges": edges}})[rid]["score"]

    assert score(["NVDA"] * 5) == 0
    assert score(["NVDA", " nvda ", "NvDa", "NVDA ", " NVDA"]) == 0
    assert score(["NVDA", "AVGO", "TSM", "ANET"]) == 0
    assert score(["NVDA", "AVGO", "TSM", "ANET", ""]) == 0
    maximum = next(r["max_score"] for r in task["rubrics"] if r["rubric_id"] == rid)
    assert score(["NVDA", "AVGO", "TSM", "ANET", "MRVL"]) == maximum
    assert score(["NVDA", "AVGO", "TSM", "ANET", "MRVL", "NVDA"]) == maximum


@pytest.mark.parametrize(
    "rid,key,years",
    [
        ("core_series_years", "series", list(range(2019, 2026))),
        ("core_bridge_coverage", "segment_reconciliation", list(range(2019, 2026))),
        ("core_unavailable_years", "unavailable_years", [2016, 2017, 2018]),
    ],
)
def test_o70_complete_unique_coverage(rid, key, years):
    task = load_tasks(["O70"])[0]

    def score(values):
        records = values if key == "unavailable_years" else [{"fiscal_year": y} for y in values]
        return mechanical(task, {"structured_output": {key: records}})[rid]["score"]

    maximum = next(r["max_score"] for r in task["rubrics"] if r["rubric_id"] == rid)
    assert score(years) == maximum
    assert score(list(reversed(years))) == maximum
    assert score(years[:1]) == 0
    assert score([]) == 0
    assert score(years + [years[0]]) == 0
    assert score(years + [2026]) == 0


def test_coverage_options_do_not_change_other_rules():
    payload = {"items": [{"id": "same"}] * 5, "years": [2019]}
    assert CHECKERS_V11["minimum_item_count"](payload, {"path": "$.items", "minimum": 5})[0]
    assert CHECKERS_V11["all_values_in_set"](
        payload, {"path": "$.years[]", "allowed": [2019, 2020]}
    )[0]


@pytest.mark.parametrize("code", ["TP02", "TP03", "TP04"])
def test_reference_materials_reach_judge_unchanged_and_once(code, tmp_path):
    task = load_tasks([code])[0]
    original = deepcopy(task)
    packet, _ = judge_packet(task, {"answer": "The submitted report."})
    items = {r["rubric_id"]: r for r in packet["rubrics"]}
    for rubric in task["rubrics"]:
        ground = (rubric.get("source_detail") or {}).get("ground_truth") or {}
        if not ground:
            continue
        item = items[rubric["rubric_id"]]
        assert item["criterion"] == rubric["criterion"]
        assert item["pass_condition"] == rubric["pass_condition"]
        assert item["reference_guidance"]["expected_answer"] == ground["expected_answer"]
        for key, kind in (
            ("verified_facts", "source_fact"),
            ("verified_calculations", "derived_calculation"),
            ("source_spans", None),
        ):
            for record in ground.get(key, []):
                expected = {"reference_kind": kind, **record} if kind else record
                assert item["reference_evidence"].count(expected) == 1
    assert with_reference_materials(packet, task["rubrics"]) == packet
    assert task == original
    assert packet["submitted_answer"] == "The submitted report."
    assert packet["answer_spans"][0]["text"] == "The submitted report."

    # No reference material is copied into the solver's task or tool inputs.
    dest = tmp_path / code
    export_task(task, dest, 7200)
    public = (dest / "public-task.json").read_text()
    runtime = (dest / "runtime/config.json").read_text()
    assert "reference_guidance" not in public + runtime
    for rubric in task["rubrics"]:
        ground = (rubric.get("source_detail") or {}).get("ground_truth") or {}
        if expected := ground.get("expected_answer"):
            assert json.dumps(expected, ensure_ascii=False)[1:-1] not in public + runtime


def test_judge_can_fill_existing_empty_reference_templates():
    task = load_tasks(["TP02"])[0]
    for item in task["judge_input_template"]["rubrics"]:
        item["reference_evidence"] = []
        item.pop("reference_guidance", None)
    packet, _ = judge_packet(task, {"answer": "Report."})
    item = next(r for r in packet["rubrics"] if r["rubric_id"] == "CASE-008-R01B")
    fact = next(r for r in item["reference_evidence"] if r.get("fact_id") == "CASE-008-NVDA-F0000")
    assert fact["value"] == 24200000000
    assert fact["filed"] == "2026-05-20"
    assert fact["filing_url"].startswith("https://www.sec.gov/")


def test_o37_disclosure_date():
    task = load_tasks(["O37"])[0]
    assert "July 28, 2026" in task["question"]
    assert "July 21, 2026" not in task["question"]
    assert "July 28, 2026" in task["judge_input_template"]["public_question"]
