import re
from copy import deepcopy
from types import SimpleNamespace

from .interval_checks import custom_decision as historical_decision


def output_value(data, path):
    """Resolve explicit object/array paths, never search for a matching leaf name."""
    if not isinstance(path, str) or not path.startswith("$."):
        raise ValueError("Expected an absolute output path")
    path = path[2:]
    path = path.removeprefix("structured_output.")
    tokens = re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z_0-9]*)|\[([0-9]+)\]", path)
    rebuilt = "".join(
        (("." if i else "") + key) if key else "[" + index + "]"
        for i, (key, index) in enumerate(tokens)
    )
    if not tokens or rebuilt != path:
        raise ValueError("Unsupported output path")
    try:
        for key, index in tokens:
            data = data[key] if key else data[int(index)]
    except (KeyError, IndexError, TypeError):
        return None  # Original finite-number validator determines zero credit.
    return data


def custom_decision(rubric, answer, array_validator):
    params = rubric.get("future_truth") or {}
    if params.get("scoring_method") != "p0_interval_score_80":
        return historical_decision(rubric, answer, array_validator)
    revised = deepcopy(rubric)
    values = {}
    for side in ("lower", "upper"):
        field = side + "_output_path"
        values[side] = output_value(answer.structured_output, params[field])
        revised["future_truth"][field] = "$." + side
    # Retain the exact frozen formula, tolerance, allocation and invalid-input rule.
    return historical_decision(revised, SimpleNamespace(structured_output=values), array_validator)


def table_receipt(result, data):
    """Accept only tool-confirmed ready reads or verified complete downloads."""
    if not result.ok or not isinstance(result.output, dict):
        return None
    out = result.output
    if result.name == "table_read":
        if out.get("status") != "ready":
            return None
    elif result.name == "table_download":
        if (
            not isinstance(out.get("path"), str)
            or not out["path"]
            or not re.fullmatch(r"[0-9a-f]{64}", out.get("sha256", ""))
            or type(out.get("size_bytes")) is not int
            or out["size_bytes"] < 0
        ):
            raise ValueError("Download has no verified artifact receipt")
    else:
        return None
    ref = out["table_ref"]
    binding = data._table_tools.results[ref]
    receipt = {
        "ref": ref,
        "kind": "table",
        "resolved": True,
        "query_id": binding["query_id"],
        "snapshot": binding["snapshot"],
        "effective_as_of": out.get("effective_as_of") or data._table_tools.cutoff.isoformat(),
        "access_method": result.name,
    }
    if result.name == "table_download":
        receipt.update({k: out[k] for k in ("path", "sha256", "size_bytes")})
    return receipt
