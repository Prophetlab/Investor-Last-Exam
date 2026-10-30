"""Portable runtime instructions shared by customer-specific delivery exporters."""

import json
from pathlib import PurePosixPath, PureWindowsPath

from .profile import example_profiles


def _portable(value):
    if isinstance(value, dict):
        for item in value.values():
            _portable(item)
    elif isinstance(value, list):
        for item in value:
            _portable(item)
    elif isinstance(value, str) and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError(
            "agent configuration contains a machine-local path; provide portable bindings before delivery"
        )


def runtime_handoff_files(agents: list[dict]) -> dict[str, bytes]:
    _portable(agents)
    files = {
        "docs/run.template.json": json.dumps(
            {
                "datasets": [{"path": "tasks"}],
                "agents": agents,
            },
            indent=2,
        ).encode()
    }
    for name, profile in example_profiles().items():
        files["environment/deployment/" + name] = json.dumps(profile, indent=2).encode()
    files["environment/README.md"] = (
        "# 运行环境\n任务的 environment/ 定义 agent 工作区，tests/ 定义独立评分环境。\n"
        "本包不预设客户使用 E2B、本地 Docker、特定镜像仓库、代理或主机路径。\n"
        "选择 environment/deployment/ 中的示例配置，另存为自己的配置。E2B 需要账户密钥且可能计费；"
        "Docker 需要 Linux Docker/Compose/Buildx。docker-none 只适用于所有阶段均离线的单步任务。\n"
        "可设置 agent_image；verifier_images 必须按 task_id 显式选择，不共用隐含的评分镜像。"
        "未提供镜像时优先使用包内 Dockerfile，不继承原试跑机器的本地镜像名。\n"
    ).encode()
    files["docs/README.md"] = (
        "# 样品交付\n在解压根目录安装固定版本：\n\n"
        "```bash\nuv sync --project environment/runtime --frozen --no-dev\n```\n\n"
        "由部署方选择并编辑部署配置，不要把 key 写进任务或 ZIP。准备阶段不执行模型、"
        "创建沙箱或修改主机：\n\n"
        "```bash\nenvironment/runtime/.venv/bin/prophetbench-env prepare-run "
        "--tasks tasks --template docs/run.template.json --profile <你的部署配置.json> "
        "--output runs/prepared --job-name review\n"
        "environment/runtime/.venv/bin/prophetbench-env run --config runs/prepared/job.json\n```\n\n"
        "Harbor 0.23.0；ATIF 与原生审计、收集的工作文件均保留。每条 trajectory 的 "
        "audit-references.json 将审计中的原机器/容器引用映射到相对于解压根目录的文件路径，"
        "并校验捕获文件的 SHA-256；原始事件及其哈希链不改写。tasks 中的私有评分材料只供可信"
        "评分端，不要把整个交付包挂载给 agent。模型和数据库连接取决于选定管线，网络/数据许可"
        "与上线前验收由部署方确认。本机实例成功不等于任意客户机器已验收。\n"
    ).encode()
    return files
