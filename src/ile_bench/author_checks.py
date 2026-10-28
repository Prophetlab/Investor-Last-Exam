import math
from fractions import Fraction


def number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("not finite numeric")
    if not math.isfinite(v):
        raise ValueError("not finite numeric")
    return Fraction(str(v))


def point(v):
    if not isinstance(v, dict):
        return number(v) if v is not None else None
    x, lo, hi = (v.get(k) for k in ("point", "lower", "upper"))
    bounds = None
    if lo is not None or hi is not None:
        if lo is None or hi is None:
            return None
        lo, hi = number(lo), number(hi)
        if lo > hi:
            return None
        bounds = (lo, hi)
    if x is not None:
        x = number(x)
        return None if bounds and not lo <= x <= hi else x
    return (lo + hi) / 2 if bounds else None


def numeric(package, payload):
    if package["code"] != "TP05":
        return {}
    values = payload.get("structured_output", {})
    fixed = {}
    for r in package["private"]["rubrics"]:
        spec = r["source_detail"]
        check = spec.get("check")
        rid = r["rubric_id"]
        if not check:
            continue
        earned = Fraction(0)
        details = {}
        try:
            kind = check["type"]
            if kind in ("relative_error_band", "inside_announced_guidance"):
                group = "q4_forecast" if kind == "relative_error_band" else "fy2027_guidance"
                x = point(values.get(group, {}).get(check["target"]))
                details = {"extracted_point": None if x is None else float(x), "rule": check}
                if x is not None:
                    if kind == "relative_error_band":
                        a = number(check["actual"])
                        earned = Fraction(
                            int(abs(x - a) / abs(a) <= Fraction(str(check["tolerance"])))
                        )
                    else:
                        earned = Fraction(
                            int(number(check["lower"]) <= x <= number(check["upper"]))
                        )
            elif kind == "brier_reward":
                probs = values.get("revenue_probability", {})
                bridge = values.get("revenue_bridge", {})
                p, q = probs.get("p_at_or_above"), probs.get("p_below")
                benchmark = bridge.get("implied_q4_revenue_usd_m")
                valid = benchmark is not None and abs(number(benchmark) - 1837) <= 1
                if p is None and q is not None:
                    p = float(1 - number(q))
                if p is not None:
                    p = number(p)
                    valid = valid and 0 <= p <= 1
                    if q is not None:
                        q = number(q)
                        valid = valid and 0 <= q <= 1 and abs(p + q - 1) <= Fraction(1, 1000000)
                    if valid:
                        earned = 1 - (p - check["y"]) ** 2
                details = {"benchmark": benchmark, "event_valid": bool(valid), "rule": check}
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            details["invalid_input"] = True
        fixed[rid] = {
            "rubric_id": rid,
            "points": float(earned * r["weight"]),
            "credit": float(earned),
            "source": "mechanical_original_rule",
            "details": details,
        }
    return fixed
