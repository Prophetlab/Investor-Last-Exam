from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from pydantic import BaseModel, Field

from .records import FutureTruthScoringMethodV11, VerificationKindV11
from .records import Record as BinaryRubricV11
from .records import Record as FinanceBenchAnswerV11


class MachineRubricDecisionV11(BaseModel):
    rubric_id: str
    passed: bool | None
    locked: bool
    checker_id: str
    observed: Any = None
    expected: Any = None
    rationale: str
    score_fraction: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


def decision_fraction_v14(decision: Any) -> float:
    """Legacy binary decisions are unchanged; continuous machine scores stay locked."""
    fraction = getattr(decision, "score_fraction", None)
    return float(fraction) if fraction is not None else float(decision.passed is True)


def display_score_v14(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def values_at_path_v11(value: Any, path: str) -> list[Any]:
    if not path.startswith("$"):
        raise ValueError(f"invalid JSON path: {path}")
    current = [value]
    for name, array in re.findall(r"\.([a-zA-Z0-9_-]+)(\[\])?", path[1:]):
        following: list[Any] = []
        for item in current:
            if not isinstance(item, dict) or name not in item:
                continue
            child = item[name]
            if array:
                if isinstance(child, list):
                    following.extend(child)
            else:
                following.append(child)
        current = following
    return current


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must resolve to a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _one(payload: dict[str, Any], path: str) -> Any:
    values = values_at_path_v11(payload, path)
    if len(values) != 1:
        raise ValueError(f"{path} must resolve to exactly one value, found {len(values)}")
    return values[0]


def _required_present(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    values = values_at_path_v11(payload, path)
    passed = bool(values) and all(value not in (None, "", [], {}) for value in values)
    return (
        passed,
        values,
        "non-empty",
        f"{path} resolved to {len(values)} non-empty candidate values",
    )


def _numeric_tolerance(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    observed = _number(_one(payload, path), path)
    expected = _number(params["expected"], "expected")
    ratio = float(params.get("tolerance_ratio", 0.0))
    absolute = float(params.get("tolerance_absolute", 0.0))
    tolerance = max(abs(expected) * ratio, absolute)
    passed = abs(Decimal(str(observed)) - Decimal(str(expected))) <= Decimal(str(tolerance))
    relation = "<=" if passed else ">"
    return (
        passed,
        observed,
        expected,
        f"absolute error {abs(observed - expected):.8g} {relation} tolerance {tolerance:.8g}",
    )


def _ratio_relation(payload: dict[str, Any], params: dict[str, Any]) -> tuple[bool, Any, Any, str]:
    numerator = _number(_one(payload, str(params["numerator_path"])), "numerator")
    denominator = _number(_one(payload, str(params["denominator_path"])), "denominator")
    if denominator == 0:
        raise ValueError("ratio denominator must be nonzero")
    scale = _number(params.get("scale", 1.0), "scale")
    expected = _number(numerator / denominator * scale, "computed ratio")
    passed, observed, _, rationale = _numeric_tolerance(
        payload, {**params, "path": params["result_path"], "expected": expected}
    )
    return (
        passed,
        observed,
        expected,
        f"submitted ratio {numerator:.8g}/{denominator:.8g}*{scale:.8g}; {rationale}",
    )


def _array_record_numeric_tolerance(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    records = _one(payload, path)
    if not isinstance(records, list):
        raise TypeError(f"{path} must resolve to an array")
    match_field = str(params["match_field"])
    match_value = params["match_value"]
    # Opt-in, finite equivalence lists. No fuzzy matching or inferred aliases.
    match_aliases = params.get("match_aliases", [])
    value_field_aliases = params.get("value_field_aliases", [])
    if not isinstance(match_aliases, list) or any(
        isinstance(value, bool) or not isinstance(value, (str, int, float))
        for value in match_aliases
    ):
        raise ValueError("match_aliases must be a list of scalar identities")
    if not isinstance(value_field_aliases, list) or any(
        not isinstance(value, str) or not value for value in value_field_aliases
    ):
        raise ValueError("value_field_aliases must be a list of field names")
    match_values = [match_value, *match_aliases]
    matches = [
        item for item in records if isinstance(item, dict) and item.get(match_field) in match_values
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path} must contain exactly one record where {match_field}={match_value!r}"
        )
    value_field = str(params["value_field"])
    value_fields = list(dict.fromkeys([value_field, *value_field_aliases]))
    observed_values = [
        _number(matches[0][field], field) for field in value_fields if field in matches[0]
    ]
    if not observed_values:
        raise ValueError(f"record has none of the declared value fields {value_fields!r}")
    if any(value != observed_values[0] for value in observed_values[1:]):
        raise ValueError("conflicting values in declared equivalent fields")
    observed = observed_values[0]
    expected = _number(params["expected"], "expected")
    ratio = float(params.get("tolerance_ratio", 0.0))
    absolute = float(params.get("tolerance_absolute", 0.0))
    tolerance = max(abs(expected) * ratio, absolute)
    passed = abs(Decimal(str(observed)) - Decimal(str(expected))) <= Decimal(str(tolerance))
    relation = "<=" if passed else ">"
    return (
        passed,
        observed,
        expected,
        (
            f"record {match_field}={match_value!r}; absolute error "
            f"{abs(observed - expected):.8g} {relation} tolerance {tolerance:.8g}"
        ),
    )


def _value_in_set(payload: dict[str, Any], params: dict[str, Any]) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    observed = _one(payload, path)
    allowed = list(params["allowed"])
    return (
        observed in allowed,
        observed,
        allowed,
        f"{observed!r} must be one of the allowed values",
    )


def _all_values_in_set(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    observed = values_at_path_v11(payload, path)
    if not observed:
        raise ValueError(f"{path} must resolve to at least one value")
    allowed = list(params["allowed"])
    unexpected = [value for value in observed if value not in allowed]
    return (
        not unexpected,
        observed,
        allowed,
        f"all {len(observed)} values must belong to the allowed set; unexpected={unexpected!r}",
    )


def _minimum_item_count(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    observed = _one(payload, path)
    if not isinstance(observed, list):
        raise TypeError(f"{path} must resolve to an array")
    minimum = int(params["minimum"])
    return (
        len(observed) >= minimum,
        len(observed),
        minimum,
        f"item count {len(observed)} >= {minimum}",
    )


def _arithmetic_relation(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    terms = params.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("arithmetic_relation requires non-empty terms")
    computed = 0.0
    for term in terms:
        computed += _number(_one(payload, str(term["path"])), str(term["path"])) * float(
            term.get("coefficient", 1.0)
        )
    result_path = str(params["result_path"])
    observed = _number(_one(payload, result_path), result_path)
    tolerance = float(params.get("tolerance_absolute", 1e-6))
    return (
        abs(observed - computed) <= tolerance,
        observed,
        computed,
        f"relation error {abs(observed - computed):.8g} <= {tolerance:.8g}",
    )


def _reconciliation(payload: dict[str, Any], params: dict[str, Any]) -> tuple[bool, Any, Any, str]:
    left = sum(_number(_one(payload, str(path)), str(path)) for path in params["left_paths"])
    right = sum(_number(_one(payload, str(path)), str(path)) for path in params["right_paths"])
    tolerance = float(params.get("tolerance_absolute", 1e-6))
    return (
        abs(left - right) <= tolerance,
        left,
        right,
        f"reconciliation gap {abs(left - right):.8g} <= {tolerance:.8g}",
    )


def _date_not_after_cutoff(
    payload: dict[str, Any], params: dict[str, Any]
) -> tuple[bool, Any, Any, str]:
    path = str(params["path"])
    observed = datetime.fromisoformat(str(_one(payload, path)))
    cutoff = datetime.fromisoformat(str(params["cutoff"]))
    return (
        observed <= cutoff,
        observed.isoformat(),
        cutoff.isoformat(),
        f"{observed.isoformat()} <= {cutoff.isoformat()}",
    )


CHECKERS_V11: dict[str, Callable[[dict[str, Any], dict[str, Any]], tuple[bool, Any, Any, str]]] = {
    "required_present": _required_present,
    "numeric_tolerance": _numeric_tolerance,
    "ratio_relation": _ratio_relation,
    "array_record_numeric_tolerance": _array_record_numeric_tolerance,
    "value_in_set": _value_in_set,
    "all_values_in_set": _all_values_in_set,
    "minimum_item_count": _minimum_item_count,
    "arithmetic_relation": _arithmetic_relation,
    "reconciliation": _reconciliation,
    "date_not_after_cutoff": _date_not_after_cutoff,
}


def execute_future_truth_rubrics_v11(
    rubrics: list[BinaryRubricV11],
    answer: FinanceBenchAnswerV11,
    resolved_truth: dict[str, dict[str, Any]],
) -> list[MachineRubricDecisionV11]:
    """Compare submitted forecasts with frozen post-cutoff outcomes in code.

    A semantic Judge never gets to change these decisions. Analyst estimates,
    management targets and cutoff consensus may be answer inputs, but only the
    rubric's frozen truth target is the expected value here.
    """

    payload = answer.model_dump(mode="json")
    decisions: list[MachineRubricDecisionV11] = []
    for rubric in rubrics:
        if rubric.verification != VerificationKindV11.FUTURE_TRUTH:
            continue
        evidence = resolved_truth.get(rubric.rubric_id)
        if evidence is None:
            continue
        binding = rubric.future_truth
        assert binding is not None
        continuous_brier = bool(
            binding.scoring_method == FutureTruthScoringMethodV11.MULTICLASS_BRIER
            and binding.outcome_bins
        )
        score_fraction = (
            0.0
            if binding.scoring_method == FutureTruthScoringMethodV11.NUMERIC_LINEAR
            or continuous_brier
            else None
        )
        expected = binding.expected_value
        answer_path = rubric.primary_output_path
        values = values_at_path_v11(payload, answer_path)
        # Author-facing output paths are relative to structured_output, while
        # FinanceBenchAnswerV11 wraps that object in the submission envelope.
        # Keep already-qualified/legacy paths working, and only fall back to
        # the wrapped location when the original path resolves to nothing.
        if (
            not values
            and answer_path.startswith("$.")
            and not answer_path.startswith("$.structured_output")
        ):
            wrapped_path = "$.structured_output" + answer_path[1:]
            wrapped_values = values_at_path_v11(payload, wrapped_path)
            if wrapped_values:
                answer_path = wrapped_path
                values = wrapped_values
        observed: Any = values[0] if len(values) == 1 else values
        try:
            method = binding.scoring_method
            if method == FutureTruthScoringMethodV11.NUMERIC_LINEAR:
                _number(observed, answer_path)
                _number(expected, "truth")
            if method == FutureTruthScoringMethodV11.MULTICLASS_BRIER:
                probabilities: dict[str, float] = {}
                for label, probability_path in binding.probability_output_paths.items():
                    path = str(probability_path)
                    if not path.startswith("$.structured_output"):
                        path = "$.structured_output" + path.removeprefix("$")
                    probabilities[label] = _number(_one(payload, path), path)
                if probabilities and max(probabilities.values()) > 1.0:
                    probabilities = {key: value / 100.0 for key, value in probabilities.items()}
                if any(value < 0.0 or value > 1.0 for value in probabilities.values()):
                    raise ValueError("probabilities must be in [0,1] or [0,100]")
                probability_sum = sum(probabilities.values())
                if abs(probability_sum - 1.0) > 0.01:
                    raise ValueError("probabilities must sum to 1 within 0.01")
                realized = (
                    binding.outcome_bins.label_for(expected)
                    if binding.outcome_bins
                    else str(binding.realized_label)
                )
                brier = sum(
                    (value - (1.0 if label == realized else 0.0)) ** 2
                    for label, value in probabilities.items()
                )
                maximum = float(binding.maximum_brier_score)
                passed = brier <= maximum
                if continuous_brier:
                    score_fraction = max(0.0, min(1.0, 1.0 - brier / 2.0))
                observed = {"probabilities": probabilities, "brier": brier}
                rationale = (
                    f"multiclass Brier score {brier:.8g} "
                    f"{'<=' if passed else '>'} maximum {maximum:.8g}"
                )
                if continuous_brier:
                    rationale += f"; proper continuous credit 1-Brier/2 = {score_fraction:.8g}; threshold is diagnostic only"
            elif method in {
                FutureTruthScoringMethodV11.INTERVAL_COVERAGE,
                FutureTruthScoringMethodV11.INTERVAL_SHARPNESS,
            }:
                # Interval bindings locate their own endpoints. The primary
                # output may be an object (or absent), not a scalar forecast.
                lower_path = str(binding.lower_output_path)
                upper_path = str(binding.upper_output_path)
                if not lower_path.startswith("$.structured_output"):
                    lower_path = "$.structured_output" + lower_path.removeprefix("$")
                if not upper_path.startswith("$.structured_output"):
                    upper_path = "$.structured_output" + upper_path.removeprefix("$")
                lower = _number(_one(payload, lower_path), lower_path)
                upper = _number(_one(payload, upper_path), upper_path)
                actual = _number(expected, "truth")
                if lower > upper:
                    raise ValueError("prediction interval lower bound exceeds upper bound")
                width = upper - lower
                if method == FutureTruthScoringMethodV11.INTERVAL_COVERAGE:
                    passed = (
                        lower * binding.answer_scale_to_truth
                        <= actual
                        <= upper * binding.answer_scale_to_truth
                    )
                    observed = {"lower": lower, "upper": upper, "width": width}
                    rationale = f"realized value {actual:.8g} is {'inside' if passed else 'outside'} scaled interval [{lower:.8g}, {upper:.8g}]; answer-to-truth scale={binding.answer_scale_to_truth:.8g}"
                else:
                    maximum = float(binding.maximum_interval_width)
                    passed = width <= maximum
                    observed = width
                    rationale = f"interval width {width:.8g} {'<=' if passed else '>'} maximum {maximum:.8g}"
            elif len(values) != 1:
                raise ValueError(f"{answer_path} resolved to {len(values)} values")
            elif (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and not isinstance(expected, bool)
                and isinstance(expected, (int, float))
            ):
                if method in {
                    FutureTruthScoringMethodV11.NUMERIC_TOLERANCE,
                    FutureTruthScoringMethodV11.NUMERIC_LINEAR,
                }:
                    ratio = float(binding.tolerance_ratio or 0.0)
                    absolute = float(binding.tolerance_absolute or 0.0)
                    tolerance = max(abs(float(expected)) * ratio, absolute)
                    if tolerance <= 0:
                        raise ValueError("numeric future truth requires a positive tolerance")
                    error = abs(
                        _number(observed, answer_path) * binding.answer_scale_to_truth
                        - _number(expected, "truth")
                    )
                    passed = error <= tolerance
                    if method == FutureTruthScoringMethodV11.NUMERIC_LINEAR:
                        score_fraction = max(0.0, min(1.0, 1.0 - error / tolerance))
                        passed = score_fraction == 1.0
                    relation = "<=" if passed else ">"
                    rationale = f"absolute error {error:.8g} {relation} tolerance {tolerance:.8g}"
                    if score_fraction is not None:
                        rationale = f"score=max(0,1-error/T)={score_fraction:.8g}; error={error:.8g}, T={tolerance:.8g}, answer-to-truth scale={binding.answer_scale_to_truth:.8g}"
                else:
                    raise ValueError(f"unsupported numeric scoring method {method.value}")
            else:
                passed = str(observed).strip().casefold() == str(expected).strip().casefold()
                rationale = "normalized submitted value equals frozen realized outcome"
        except Exception as error:  # noqa: BLE001 - locked verifier failure
            passed = False
            if binding.scoring_method == FutureTruthScoringMethodV11.NUMERIC_LINEAR:
                score_fraction = 0.0
            rationale = f"future-truth checker failed: {type(error).__name__}: {error}"
        decisions.append(
            MachineRubricDecisionV11(
                rubric_id=rubric.rubric_id,
                passed=passed,
                locked=True,
                checker_id=f"future_truth_{binding.scoring_method.value}_v11",
                observed=observed,
                expected=expected,
                rationale=rationale,
                score_fraction=score_fraction,
            )
        )
    return decisions
