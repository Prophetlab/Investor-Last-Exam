import math


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("required finite numeric input is missing or invalid")
    return float(value)


def custom_decision(rubric, answer, array_validator):
    data = answer.structured_output
    binding = rubric.get("deterministic")
    expected = None
    observed = None
    fraction = None
    checker = binding["checker_id"] if binding else rubric["future_truth"]["scoring_method"]
    params = binding["parameters"] if binding else rubric["future_truth"]
    try:
        if checker == "p0_distinct_assumptions":
            records = data.get("assumption_labels")
            errors = list(array_validator(params["schema"]).iter_errors(records))
            ids = (
                [record.get("assumption_id") for record in records if isinstance(record, dict)]
                if isinstance(records, list)
                else []
            )
            observed = {"ids": ids, "schema_errors": [error.message for error in errors]}
            expected = params["required_ids"]
            passed = not errors and set(expected).issubset(set(ids))
        elif checker == "p0_numeric_alternatives":
            observed = finite_number(data.get(params["path"]))
            expected = params["values"]
            passed = any(abs(observed - value) <= params["tolerance"] for value in expected)
        elif checker == "p0_fairlife_bridge":
            observed = finite_number(data.get("free_cash_flow_ex_fairlife"))
            exact = finite_number(data.get("reported_free_cash_flow")) + finite_number(
                data.get("fairlife_payment")
            )
            rounded = math.floor(exact / 100 + 0.5) * 100
            expected = {"submitted_input_bridge": exact, "rounded_to_100M": rounded}
            passed = abs(observed - exact) <= 0.5 or (
                abs(observed - 11400) <= 0.5 and observed == rounded
            )
        elif checker == "p0_meta_derived":
            weight = finite_number(data.get("target_weight_pct"))
            if params["metric"] == "risk_headroom":
                target = finite_number(data.get("target_value_usd"))
                loss = finite_number(data.get("loss_at_stop_usd"))
                if (
                    abs(target - weight / 100 * 600_000_000) > 1000
                    or abs(loss - target * 0.2) > 1000
                ):
                    raise ValueError("target value, weight and stop-loss do not reconcile")
                observed = 5_400_000 - loss
            elif params["metric"] == "volatility_use":
                observed = weight * 0.35
            elif params["metric"] == "return_contribution":
                observed = weight * finite_number(data.get("expected_total_return_pct")) / 100
            else:
                observed = 4 - weight
            expected = params["expected"]
            passed = abs(observed - expected) <= params["tolerance"]
        elif checker == "p0_interval_score_80":
            lower = finite_number(data.get(params["lower_output_path"].split(".")[-1]))
            upper = finite_number(data.get(params["upper_output_path"].split(".")[-1]))
            actual = finite_number(params["expected_value"]) / params["answer_scale_to_truth"]
            if lower > upper:
                raise ValueError("interval endpoints are reversed")
            loss = (
                upper
                - lower
                + 2 / params["interval_alpha"] * (max(lower - actual, 0) + max(actual - upper, 0))
            )
            fraction = max(0.0, 1 - loss / params["loss_scale"])
            observed = {"lower": lower, "upper": upper, "raw_interval_loss": loss}
            expected = {
                "actual": actual,
                "loss_scale": params["loss_scale"],
                "alpha": params["interval_alpha"],
            }
            passed = fraction == 1
        elif checker == "p0_revision_direction":
            prior = finite_number(data.get("prior_forecast_bn", {}).get("iphone"))
            updated = finite_number(data.get("updated_forecast_bn", {}).get("iphone"))
            revision = finite_number(data.get("mapping", {}).get("affected_driver_revision"))
            actual = finite_number(params["expected_value"]) / params["answer_scale_to_truth"]
            distance = actual - prior
            tolerance = params["direction_tolerance"]
            observed = {"prior": prior, "updated": updated, "revision": revision}
            expected = {
                "actual": actual,
                "required_direction": "up"
                if distance > tolerance
                else "down"
                if distance < -tolerance
                else "unchanged",
            }
            passed = abs(updated - prior - revision) <= 0.01 and (
                abs(revision) <= tolerance
                if abs(distance) <= tolerance
                else revision * distance > 0
            )
        else:
            raise ValueError("unknown custom checker")
        rationale = f"{checker}: observed={observed}; expected={expected}; " + (
            f"fraction={fraction}" if fraction is not None else f"passed={passed}"
        )
    except (TypeError, ValueError, KeyError) as error:
        passed = False
        fraction = 0.0 if checker == "p0_interval_score_80" else None
        rationale = f"{checker}: invalid or missing input: {error}"
    return {
        "rubric_id": rubric["rubric_id"],
        "passed": passed,
        "locked": True,
        "checker_id": checker,
        "observed": observed,
        "expected": expected,
        "rationale": rationale,
        "score_fraction": fraction,
    }
