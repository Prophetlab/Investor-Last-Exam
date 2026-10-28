import math
from decimal import ROUND_HALF_UP, Decimal

from jsonschema import Draft202012Validator

EVALUATOR_ID = "ile-market-future-loss-v1"


def score_answer(package, payload):
    public = package["public"]
    policy = package["private"]["market_evaluation"]
    Draft202012Validator(public["submission_schema"]).validate(payload)
    if policy["evaluator_id"] != EVALUATOR_ID:
        raise ValueError("Unsupported market evaluation policy")
    if policy["kind"] == "point":
        forecast = payload["forecast_close_usd"]
        if type(forecast) not in (int, float) or not math.isfinite(forecast) or forecast <= 0:
            raise ValueError("Price forecast must be finite and positive")
        anchor = Decimal(policy["anchor_close"])
        actual = Decimal(policy["actual_close"])
        scale = Decimal(policy["error_scale_pp"])
        if min(anchor, actual, scale) <= 0:
            raise ValueError("Invalid frozen point target/scale")
        loss = abs(Decimal(str(forecast)) - actual) / anchor * 100
        normalized = loss / scale
        credit = max(Decimal(0), 1 - normalized)
        diagnostics = {
            "predicted_close": forecast,
            "actual_close": str(actual),
            "anchor_close": str(anchor),
            "absolute_price_error_usd": str(abs(Decimal(str(forecast)) - actual)),
            "loss_return_pp": str(loss),
            "error_scale_pp": str(scale),
            "normalized_loss": str(normalized),
        }
        informative = True
    elif policy["kind"] == "ranking":
        ranking = payload["ranking"]
        values = {key: Decimal(str(v)) for key, v in policy["realized_metrics"].items()}
        if len(ranking) != len(values) or set(ranking) != set(values):
            raise ValueError("Ranking must be a complete permutation")
        if any(not x.is_finite() for x in values.values()):
            raise ValueError("Invalid frozen rank truth")
        positions = {k: i for i, k in enumerate(ranking)}
        epsilon = Decimal(str(policy["deadband_pp"]))
        denominator = Decimal(0)
        numerator = Decimal(0)
        pairs = []
        ids = list(values)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                gap = values[a] - values[b]
                weight = max(abs(gap) - epsilon, Decimal(0))
                inverted = bool(gap and ((gap > 0) != (positions[a] < positions[b])))
                denominator += weight
                if inverted:
                    numerator += weight
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "realized_gap_pp": str(gap),
                        "weight": str(weight),
                        "inverted": inverted,
                    }
                )
        informative = denominator > 0
        normalized = numerator / denominator if informative else Decimal(0)
        credit = 1 - normalized
        diagnostics = {
            "weighted_inversions": str(numerator),
            "total_pair_weight": str(denominator),
            "normalized_loss": str(normalized),
            "deadband_pp": str(epsilon),
            "pairs": pairs,
        }
    else:
        raise ValueError("Unknown objective task kind")
    score = float((100 * credit).quantize(Decimal(".1"), rounding=ROUND_HALF_UP))
    return {
        "evaluator_id": EVALUATOR_ID,
        "total_score": score,
        "reward": score,
        "reward_pending": False,
        "rubric_results": [
            {
                "rubric_id": "future_outcome_loss",
                "score": score,
                "max_score": 100,
                "diagnostic": diagnostics,
            }
        ],
        "informative": informative,
        "metadata": {"only_future_loss_scored": True, "semantic_judge_called": False},
    }
