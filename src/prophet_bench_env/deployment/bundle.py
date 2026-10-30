"""Portable launcher/runtime files, reusable by any customer delivery adapter."""

import json
from pathlib import Path

from .handoff import _portable
from .launch import LaunchConfig

RUNTIME_ROOT = Path(__file__).resolve().parents[2]


def launcher_files(options=None, *, runtime_root=RUNTIME_ROOT):
    config = LaunchConfig.model_validate(options or {})
    _portable(config.model_dump())
    files = {}
    for name in (
        "prophet_bench_env",
        "plugins",
        "vendor",
        "pyproject.toml",
        "uv.lock",
        "README.md",
    ):
        root = runtime_root / name
        for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
            if path.is_symlink():
                raise ValueError("runtime bundle cannot contain symlinks")
            if (
                path.is_file()
                and not {".git", ".venv", "__pycache__", "tasks"}.intersection(
                    path.relative_to(runtime_root).parts
                )
                and path.suffix != ".pyc"
                and not any(
                    p == ".env" or p.startswith(".env.")
                    for p in path.relative_to(runtime_root).parts
                )
            ):
                files[
                    "environment/runtime/" + path.relative_to(runtime_root).as_posix()
                ] = path
    files["environment/launch.json"] = json.dumps(
        config.model_dump(exclude_none=True), indent=2
    ).encode()
    files["run.sh"] = b"""#!/bin/sh
set -eu
PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PACKAGE_ROOT"
command -v uv >/dev/null 2>&1 || { echo "Install uv on the receiving machine first." >&2; exit 1; }
exec uv run --project "$PACKAGE_ROOT/environment/runtime" --frozen --no-dev prophetbench-env launch --package-root "$PACKAGE_ROOT" "$@"
"""
    files["docs/RUNNING.md"] = b"""# Run this package

Prerequisites: uv, local Docker (Compose/Buildx), and model credentials.
Remote E2B remains available only when explicitly selected with --backend e2b;
configure E2B_API_KEY in that case. No automatic cloud fallback.
Configure credentials on the receiving machine; never add them to the delivery ZIP.
Claude Code: ANTHROPIC_API_KEY and optional ANTHROPIC_BASE_URL.
Codex: OPENAI_API_KEY and optional OPENAI_BASE_URL.
Financial tools/Reader/verifier use the task's documented host-side configuration.

From the extracted package:
```bash
./run.sh --agent claude-code --model <actual-model-id> --backend docker
./run.sh --agent codex --model <actual-model-id> --backend docker
```
Use --config run.local.json for Reader bindings, image selections and concurrency.
If model and other selections are frozen in environment/launch.json, ./run.sh is enough.
--prepare-only validates and prepares files without creating sandboxes or calling models.
Results are written to a new runs/ directory; task and rubric originals are untouched.

Native Harbor agents perform installation, model execution and ATIF capture. Installation
uses the declared setup allowlist; the agent phase only permits model API hosts by default.
Preinstalled images can use an empty setup_hosts list. General web research is disabled
by default; tool permissions and cutoff remain enforced by the host-side task tools.
MCP stdio requests use Harbor control I/O, not a public listening port. Database and
Reader credentials stay on the host; model credentials follow the native Harbor agent.
Network, image availability, data access and model billing depend on the receiving system.
"""
    files["docs/RUNNING.md"] += """

## 原生 Harbor 模式（不注入 Prophet 工具）

任务已自行提供文件、软件/CLI/MCP 和 verifier 时，可以直接使用 Harbor 原生 agent：

```bash
./run.sh --agent-mode harbor --agent claude-code --model <model-id> --backend docker
./run.sh --agent-mode harbor --agent codex --model <model-id> --backend docker
```

此模式保留题目原有 Compose、网络策略、镜像、MCP、artifacts 及评分入口；
不要求 public-task.json、金融工具或 submit_answer，不自动运行 Prophet 软件 factory。
任务必须已按原生 Harbor 格式自包含；已有 Prophet 工具任务请保留默认 prophet 模式。
其他 Harbor agent 可用 --agent 选择，自定义 BaseAgent 类使用 --agent-import-path。
高级 Harbor JobConfig 仍可通过 prophetbench-env run --config job.json 运行。
模型凭据从接收方环境配置，Harbor 原生 agent 负责校验；不要写入交付文件。
Harbor 0.23.0 的 E2B 后端不支持 Compose/sidecar，这类任务应选择 Docker。
只运行受信任且已审阅的任务和 Compose；框架不替作者审核容器权限及宿主挂载。
缺失网络、镜像或模型配置仍可能导致运行失败，--prepare-only 仅做静态准备。
""".encode()
    return files
