"""Include task-authored reference material in report judge packets, offline."""

from copy import deepcopy


REFERENCE_FIELDS = (
    "expected_answer",
    "reference_kind",
    "required_inputs",
    "missing_input_ids",
    "contextual_missing_input_ids",
    "acceptable_alternatives",
    "forbidden_inferences",
    "source_support_scope",
)


def with_reference_materials(packet, rubrics):
    """Keep source facts, calculations and conditional guidance distinct.

    This copies only existing rubric material. It does not fetch evidence,
    infer missing facts, change the scoring conditions, or fill answer gaps.
    """
    result = deepcopy(packet)
    definitions = {r["rubric_id"]: r for r in rubrics}
    for item in result.get("rubrics", []):
        rubric = definitions[item["rubric_id"]]
        ground = (rubric.get("source_detail") or {}).get("ground_truth") or {}
        if not ground:
            continue
        guidance = {key: deepcopy(ground[key]) for key in REFERENCE_FIELDS if key in ground}
        if guidance:
            item["reference_guidance"] = guidance
        evidence = item.setdefault("reference_evidence", [])
        for key, kind in (
            ("verified_facts", "source_fact"),
            ("verified_calculations", "derived_calculation"),
            ("source_spans", None),
        ):
            for source in ground.get(key, []):
                record = deepcopy(source)
                if kind:
                    record = {"reference_kind": kind, **record}
                if record not in evidence:
                    evidence.append(record)
    return result
