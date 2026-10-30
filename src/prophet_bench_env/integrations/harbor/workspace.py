"""Public task initialization shared by external and installed Harbor agents."""

import json
from pathlib import Path

from .serialization import read_public, write_json


async def prepare_workspace(environment, logs_dir):
    root = Path(environment.environment_dir).parent
    public = json.loads((root / "public-task.json").read_text())
    task = read_public(public)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(logs_dir / "public-task.json", public)
    await environment.upload_file(logs_dir / "public-task.json", "/app/task.json")
    for source in sorted((root / "public-data").glob("*")):
        if source.is_symlink() or not source.is_file():
            raise ValueError("public data must be regular files")
        await environment.upload_file(source, "/app/workspace/" + source.name)
    return root, task
