"""Harbor-first commands; Gymnasium/HTTP remain optional step-wise surfaces."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    parser = argparse.ArgumentParser(
        description="ProphetBenchEnv: Harbor tasks and delivery"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="optional HTTP episode interface")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", default=8080, type=int)
    run = commands.add_parser("run", help="run/resume a native Harbor JSON job")
    run.add_argument("--config", type=Path, required=True)
    launch_parser = commands.add_parser(
        "launch", help="prepare and run a delivered task set with a native agent"
    )
    launch_parser.add_argument("--package-root", type=Path, default=Path("."))
    launch_parser.add_argument("--config", type=Path)
    launch_parser.add_argument(
        "--agent", help="Prophet adapter or native Harbor agent name"
    )
    launch_parser.add_argument("--agent-mode", choices=("prophet", "harbor"))
    launch_parser.add_argument(
        "--agent-import-path", help="custom Harbor BaseAgent class"
    )
    launch_parser.add_argument("--model")
    launch_parser.add_argument("--backend", choices=("docker", "e2b"))
    launch_parser.add_argument("--tool-factory")
    launch_parser.add_argument("--output", type=Path)
    launch_parser.add_argument("--prepare-only", action="store_true")
    prepare = commands.add_parser(
        "prepare-run",
        help="prepare a new run for an explicit deployment profile; no execution",
    )
    prepare.add_argument("--tasks", type=Path, required=True)
    prepare.add_argument("--template", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--job-name", default="review")
    smoke = commands.add_parser(
        "smoke", help="real sandbox, deterministic finance task; no model charge"
    )
    smoke.add_argument(
        "--backend",
        choices=("docker", "e2b"),
        default=os.getenv("PROPHETBENCH_BACKEND", "docker"),
    )
    smoke.add_argument(
        "--output",
        type=Path,
        default=Path("experiments")
        / ("harbor-smoke-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")),
    )
    smoke.add_argument("--concurrency", type=int, default=1)
    smoke.add_argument("--tasks", type=int, default=1)
    smoke.add_argument("--base-image", default="python:3.12-slim-bookworm")
    export = commands.add_parser(
        "export-tasks", help="invoke a trusted authoring/export plugin"
    )
    export.add_argument(
        "--factory", required=True, help="module:function accepting output=Path"
    )
    export.add_argument("--output", type=Path, required=True)
    delivery = commands.add_parser(
        "deliver", help="customer-selected exporter; no upload"
    )
    delivery.add_argument(
        "--factory", required=True, help="module:function accepting options=dict"
    )
    delivery.add_argument("--options", type=Path, required=True)
    delivery.add_argument("--tasks", type=Path, required=True)
    delivery.add_argument("--jobs", type=Path, nargs="+", default=[])
    delivery.add_argument("--output", type=Path, required=True)
    delivery.add_argument(
        "--draft",
        action="store_true",
        help="export a review draft despite quality/material gaps; never bypass safety",
    )
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        from .registry import load_engine_factory
        from .server import create_app

        uvicorn.run(
            create_app(load_engine_factory(os.getenv("PROPHETBENCH_ENGINE_FACTORY"))),
            host=args.host,
            port=args.port,
        )
    elif args.command == "launch":
        from ..deployment.launch import launch

        print(
            launch(
                args.package_root,
                config_file=args.config,
                overrides={
                    "agent": args.agent,
                    "agent_mode": args.agent_mode,
                    "agent_import_path": args.agent_import_path,
                    "model": args.model,
                    "backend": args.backend,
                    "tool_factory": args.tool_factory,
                },
                output=args.output,
                prepare_only=args.prepare_only,
            )
        )
    elif args.command == "run":
        from ..integrations.harbor.jobs import run_config

        result = run_config(args.config)
        if any(row.exception_info is not None for row in result.trial_results):
            raise SystemExit(2)
    elif args.command == "prepare-run":
        from ..deployment.prepare import prepare_run
        from ..deployment.profile import DeploymentProfile

        profile = DeploymentProfile.model_validate(json.loads(args.profile.read_text()))
        result = prepare_run(
            tasks_dir=args.tasks,
            run_template=json.loads(args.template.read_text()),
            profile=profile,
            output=args.output,
            job_name=args.job_name,
        )
        print(result)
    elif args.command == "smoke":
        from plugins.pipelines.arithmetic_example.pipeline import export_dataset

        from ..integrations.harbor.jobs import job_config, run_job, save_config

        if not 1 <= args.tasks <= 100:
            parser.error("--tasks must be between 1 and 100")
        if args.backend == "e2b" and not os.getenv("E2B_API_KEY"):
            parser.error(
                "E2B_API_KEY is required; use --backend docker for local execution"
            )
        tasks = export_dataset(
            args.output, count=args.tasks, base_image=args.base_image
        )
        config = job_config(
            tasks=tasks,
            jobs=args.output / "jobs",
            job_name="smoke",
            runtime_factory="plugins.pipelines.arithmetic_example.pipeline:create_runtime",
            model_ids=("correct-answer", "wrong-answer"),
            backend=args.backend,
            concurrency=args.concurrency,
            lease_seconds=600,
        )
        save_config(config, args.output / "job.json")
        result = asyncio.run(run_job(config))
        expected = {"correct-answer": 1.0, "wrong-answer": 0.0}
        passed = len(result.trial_results) == 2 * args.tasks and all(
            row.exception_info is None
            and row.verifier_result is not None
            and row.verifier_result.rewards.get("reward")
            == expected[row.config.agent.model_name]
            for row in result.trial_results
        )
        print(
            json.dumps(
                {
                    "passed": passed,
                    "backend": args.backend,
                    "output": str(args.output),
                    "kind": "deterministic integration test; not an LLM benchmark",
                }
            )
        )
        if not passed:
            raise SystemExit(2)
    elif args.command == "export-tasks":
        from ..integrations.harbor.serialization import load_entrypoint

        print(load_entrypoint(args.factory)(output=args.output))
    elif args.command == "deliver":
        from ..integrations.harbor.serialization import load_entrypoint
        from ..ports.delivery import DeliverySource

        options = json.loads(args.options.read_text())
        if args.draft:
            options["draft"] = True
        exporter = load_entrypoint(args.factory)(options=options)
        receipt = exporter.export(
            DeliverySource(args.tasks, tuple(args.jobs)), args.output
        )
        print(
            json.dumps(
                {
                    "path": str(receipt.path),
                    "sha256": receipt.sha256,
                    "task_count": receipt.task_count,
                    "upload_path": receipt.suggested_upload_path,
                }
            )
        )


if __name__ == "__main__":
    main()
