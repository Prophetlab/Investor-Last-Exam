"""Single entry point: list, check, run, resume, grade and summarize."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from prophet_bench_env.integrations.harbor.serialization import write_json

from .config import ROOT, boolean, data_environment, judges, route, values
from .evaluation import grade_trial
from .tasks import export_task, load_tasks


def run_dir(name):
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name
    ):
        raise ValueError("Job name must contain only letters, digits, hyphens and underscores")
    return ROOT / "local/runs" / name


def fingerprint(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check(env, *, host=False, credentials=False):
    tasks = load_tasks()
    if credentials:
        route("SOLVER", env)
        judges(env)
        if not boolean(env.get("READER_REUSE_SOLVER", "true")):
            route("READER", env)
        if not env.get("DATALAKE_API_KEY", "").strip():
            raise ValueError("Fill DATALAKE_API_KEY")
    if host:
        subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], check=True)
        subprocess.run(["docker", "compose", "version", "--short"], check=True)
        subprocess.run(["docker", "buildx", "version"], check=True, stdout=subprocess.DEVNULL)
    return {
        "tasks": len(tasks),
        "rubrics": sum(len(t["rubrics"]) for t in tasks),
        "config_checked": credentials,
        "docker_checked": host,
    }


def prepare(codes, name, env, concurrency):
    from prophet_bench_env.integrations.harbor.jobs import job_config, save_config

    if not 1 <= concurrency <= 8:
        raise ValueError("Concurrency must be 1–8; use 1 or 2 initially")
    rows = load_tasks(codes)
    dest = run_dir(name)
    if dest.exists():
        raise ValueError("Job exists; use resume to preserve its results")
    wall = int(env.get("ILE_WALL_CLOCK_SECONDS", "7200"))
    if not 1 <= wall <= 86400:
        raise ValueError("Wall-clock seconds must be 1–86400")
    from .build import ensure_image

    image = ensure_image()
    dest.mkdir(parents=True, mode=0o700)
    for task in rows:
        export_task(task, dest / "tasks" / task["id"], wall, image)
    solver = route("SOLVER", env)
    config = job_config(
        tasks=dest / "tasks",
        jobs=dest / "jobs",
        job_name=name,
        runtime_factory="ile_bench.runtime:create_runtime",
        model_ids=("solver",),
        backend="docker",
        concurrency=concurrency,
        per_model_concurrency=concurrency,
        lease_seconds=wall + 300,
        runtime_kwargs={"public_config": {"$runtime_file": "config.json"}},
    )
    config.verifier.disable = True
    save_config(config, dest / "job.json")
    write_json(
        dest / "plan.json",
        {
            "version": "1.0.0",
            "job_name": name,
            "tasks": {t["id"]: fingerprint(t) for t in rows},
            "solver": solver.public(),
            "judges": {name: r.public() for name, r in judges(env)},
            "score_basis": "final_answer",
            "route_fingerprints": {
                "solver": solver.fingerprint(),
                **{name: r.fingerprint() for name, r in judges(env)},
            },
            "grading": "independent judges; original programmatic checks",
            "files": {
                str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in dest.rglob("*")
                if p.is_file()
            },
        },
    )
    return dest


def validate_plan(dest, env):
    plan = json.loads((dest / "plan.json").read_text())
    tasks = {t["id"]: t for t in load_tasks()}
    if route("SOLVER", env).public() != plan["solver"]:
        raise ValueError("Solver configuration changed; create a new job")
    if {n: r.public() for n, r in judges(env)} != plan["judges"]:
        raise ValueError("Judge configuration changed; create a new job")
    if plan.get("route_fingerprints") is not None and plan["route_fingerprints"] != {
        "solver": route("SOLVER", env).fingerprint(),
        **{n: r.fingerprint() for n, r in judges(env)},
    }:
        raise ValueError("Provider route or parameters changed; create a new job")
    for name, digest in plan["files"].items():
        if hashlib.sha256((dest / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen job input changed: " + name)
    for code, digest in plan["tasks"].items():
        if fingerprint(tasks[code]) != digest:
            raise ValueError("Task changed: " + code)
    return plan


def grade(dest, env):
    plan = validate_plan(dest, env)
    tasks = {t["id"]: t for t in load_tasks()}
    routes = judges(env)
    count = 0
    for result in sorted((dest / "jobs" / plan["job_name"]).glob("*/result.json")):
        public = result.parent / "agent/public-task.json"
        if not public.exists():
            continue
        code = json.loads(public.read_text())["task_id"]
        if code not in plan["tasks"]:
            raise ValueError("Unexpected task in job")
        existing = [result.parent / "verifier" / f"{n}.json" for n, _ in routes]
        pending = [
            (n, r)
            for (n, r), p in zip(routes, existing)
            if not p.exists() or json.loads(p.read_text()).get("status") != "scored"
        ]
        if pending:
            grade_trial(
                tasks[code], result.parent, pending, json_mode=env.get("JUDGE_JSON_MODE", "prompt")
            )
            count += 1
    return {"newly_graded_trials": count, **summary(dest)}


def summary(dest):
    plan = json.loads((dest / "plan.json").read_text())
    by_judge = {name: [] for name in plan["judges"]}
    rows = []
    for file in sorted((dest / "jobs" / plan["job_name"]).glob("*/verifier/judge_*.json")):
        r = json.loads(file.read_text())
        name = file.stem
        if name not in by_judge:
            continue
        rows.append({"judge": name, **r})
        if r["status"] == "scored":
            by_judge[name].append(r)
    result = {"tasks": len(plan["tasks"]), "judges": {}, "results": rows}
    for name, values_ in by_judge.items():
        n = len(values_)
        result["judges"][name] = {
            "model": plan["judges"][name]["model"],
            "scored": n,
            "expected": len(plan["tasks"]),
            "mean_score": sum(r["score"] for r in values_) / n if n else None,
            "mean_strict_score": sum(r["strict_score"] for r in values_) / n if n else None,
        }
    write_json(dest / "summary.json", result)
    return {k: v for k, v in result.items() if k != "results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=ROOT / "local/.env")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("build")
    c = sub.add_parser("check")
    c.add_argument("--host", action="store_true")
    c.add_argument("--config", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--task", nargs="+", default=["O01"])
    r.add_argument("--job-name")
    r.add_argument("--concurrency", type=int)
    r.add_argument("--solve-only", action="store_true")
    for name in ("resume", "grade", "summary"):
        sub.add_parser(name).add_argument("job_name")
    args = parser.parse_args()
    env = values(args.env)
    for key, value in env.items():
        if key.startswith(("SOLVER_", "READER_", "JUDGE_", "ILE_", "DATALAKE_", "HARBOR_")):
            os.environ[key] = value
    if args.command == "list":
        for t in load_tasks():
            print(t["id"], t["ticker"], t["submission_mode"], t["company"])
        return
    if args.command == "check":
        result = check(env, host=args.host, credentials=args.config)
    elif args.command == "build":
        from .build import ensure_image

        result = {"image": ensure_image()}
    elif args.command == "summary":
        result = summary(run_dir(args.job_name))
    elif args.command == "grade":
        result = grade(run_dir(args.job_name), env)
    else:
        from prophet_bench_env.integrations.harbor.jobs import run_config

        check(env, host=True, credentials=True)
        data_environment(env)
        if args.command == "run":
            name = args.job_name or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
            dest = prepare(
                args.task, name, env, args.concurrency or int(env.get("ILE_CONCURRENCY", "1"))
            )
        else:
            dest = run_dir(args.job_name)
            validate_plan(dest, env)
        run_config(dest / "job.json")
        result = summary(dest) if getattr(args, "solve_only", False) else grade(dest, env)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
