"""Maintainer-only export of the required client/runtime modules, never service code.

Inputs are explicit local source trees. End users run the already-vendored release.
"""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def selected(source, names, imports):
    tree = ast.parse(source)
    found = []
    picked = []
    for node in tree.body:
        keys = {getattr(node, "name", None)}
        if isinstance(node, ast.Assign):
            keys = {getattr(n, "id", None) for n in node.targets}
        if isinstance(node, ast.AnnAssign):
            keys = {getattr(node.target, "id", None)}
        if keys & names:
            found.extend(keys & names)
            picked.append(ast.get_source_segment(source, node))
    if set(found) != names:
        raise ValueError("Missing selected definitions: " + str(names - set(found)))
    return imports + "\n\n" + "\n\n".join(picked) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    a = p.parse_args()
    src = ROOT / "src"
    src.mkdir(exist_ok=True)
    package = src / "ile_bench"
    package.mkdir(exist_ok=True)
    # Runtime source and its neutral harness only. No example pipelines, secrets or logs.
    for relative in ["prophet_bench_env", "plugins/harness/compact_react"]:
        for f in (a.environment / relative).rglob("*"):
            if f.is_file() and f.suffix in {".py", ".md", ".toml"} and "__pycache__" not in f.parts:
                target = src / f.relative_to(a.environment)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
    for relative in ["plugins/__init__.py", "plugins/harness/__init__.py"]:
        (src / relative).write_text("")
    # SDK 1.1.0 client and contract types. Explicitly omit accounts, database and ingestion.
    modules = [
        "__init__",
        "client",
        "sdk",
        "sdk_models",
        "taxonomy",
        "time",
        "errors",
        "api/__init__",
        "api/contract",
        "api/control_contract",
        "api/document_contract",
        "api/session_contract",
        "api/table_contract",
    ]
    wheel = a.environment / "vendor/datacollection/datacollection-1.1.0-py3-none-any.whl"
    with ZipFile(wheel) as z:
        for module in modules:
            name = "financial_data_lake/" + module + ".py"
            target = src / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name) if name in z.namelist() else b"")
    # Public task export already separates the solver projection from evaluation inputs.
    shutil.copytree(a.dataset / "tasks", ROOT / "tasks", dirs_exist_ok=True)
    shutil.copyfile(a.dataset / "prompts/answer_spans.py", package / "answer_spans.py")
    for module in ["harness", "submission", "public_submission"]:
        shutil.copyfile(a.experiment / "ile" / f"{module}.py", package / f"{module}.py")
    verification = (a.experiment / "compatibility/grading/verification_v11.py").read_text()
    names = {
        "MachineRubricDecisionV11",
        "decision_fraction_v14",
        "display_score_v14",
        "values_at_path_v11",
        "_number",
        "_one",
        "_required_present",
        "_numeric_tolerance",
        "_ratio_relation",
        "_array_record_numeric_tolerance",
        "_value_in_set",
        "_all_values_in_set",
        "_minimum_item_count",
        "_arithmetic_relation",
        "_reconciliation",
        "_date_not_after_cutoff",
        "CHECKERS_V11",
        "execute_future_truth_rubrics_v11",
    }
    (package / "checks.py").write_text(
        selected(
            verification,
            names,
            "from __future__ import annotations\nimport re, math\nfrom collections.abc import Callable\nfrom datetime import datetime\nfrom decimal import Decimal, ROUND_HALF_UP\nfrom typing import Any\nfrom pydantic import BaseModel, Field\nfrom .records import FutureTruthScoringMethodV11, VerificationKindV11\nfrom .records import Record as BinaryRubricV11, Record as FinanceBenchAnswerV11",
        )
    )
    p0 = (a.experiment / "compatibility/p0_rules.py").read_text()
    (package / "interval_checks.py").write_text(
        selected(p0, {"finite_number", "custom_decision"}, "import math")
    )
    market = (a.experiment / "ile/market.py").read_text()
    (package / "market.py").write_text(
        selected(
            market,
            {"EVALUATOR_ID", "score_answer"},
            "import math\nfrom decimal import Decimal, ROUND_HALF_UP\nfrom jsonschema import Draft202012Validator",
        )
    )
    stream = (a.experiment / "ile/formal_models.py").read_text()
    (package / "streams.py").write_text(
        selected(
            stream,
            {"StreamAssembly", "AnthropicStreamAssembly"},
            "import json\nfrom copy import deepcopy\nfrom prophet_bench_env.model_adapters.errors import ProviderHTTPError, ProviderInvocationError",
        )
    )
    correction = (a.experiment / "ile/evaluation_repair.py").read_text()
    (package / "corrected_checks.py").write_text(
        selected(
            correction,
            {"output_value", "custom_decision", "table_receipt"},
            "import re\nfrom copy import deepcopy\nfrom types import SimpleNamespace\nfrom .interval_checks import custom_decision as historical_decision",
        )
    )
    author = (a.experiment / "experiments/third-party-20261022/grade.py").read_text()
    (package / "author_checks.py").write_text(
        selected(
            author, {"number", "point", "numeric"}, "import math\nfrom fractions import Fraction"
        )
    )
    files = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for base in [src, ROOT / "tasks"]
        for f in sorted(base.rglob("*"))
        if f.is_file()
    }
    (ROOT / "scripts/source-manifest.json").write_text(
        json.dumps(
            {
                "environment_revision": "341546df1aebe57aee5ccc35e0632b773b8e4ea1",
                "sdk": "1.1.0",
                "dataset": "https://huggingface.co/datasets/HenryJ12/InvestorLastExam",
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"files": len(files), "tasks": len(list((ROOT / "tasks").glob("*.json")))}))


if __name__ == "__main__":
    main()
