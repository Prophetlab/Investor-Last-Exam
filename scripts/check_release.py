"""Check the publishable additions without reading private local configuration."""

import json
import re
import subprocess

from ile_bench.config import ROOT
from ile_bench.tasks import load_tasks


def main():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    candidate = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    forbidden = re.compile(
        r"/(?:home/(?:anning|shengkai|xiangzhao)|opt/(?:prophet-data-lake|prophet-table-services|prophet-cutover))|123\.56\.\d+\.\d+|10\.20\.\d+\.\d+|(?<![A-Za-z0-9])(?:ghp_|gho_|dl_live_|sk-)[A-Za-z0-9_-]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"
    )
    count = 0
    for name in set(tracked + candidate):
        if not (
            name.startswith(("src/", "scripts/", "tasks/", "tests/", "environment/", "local/"))
            or name
            in (
                ".env.example",
                "QUICKSTART.md",
                "pyproject.toml",
                "requirements.lock",
                "requirements.txt",
                "run.py",
                "run.sh",
            )
        ):
            continue
        if name.startswith("local/") and name != "local/.gitignore":
            raise ValueError("Private local file would be published")
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("No symlinks in public source")
        if forbidden.search(path.read_text()):
            raise ValueError("Private path or credential-shaped value in " + name)
        count += 1
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.endswith(("_API_KEY", "_BASE_URL")) and key != "DATALAKE_BASE_URL" and value:
            raise ValueError("Connection template must be blank: " + key)
    tasks = load_tasks()
    assert len(tasks) == 86 and sum(len(t["rubrics"]) for t in tasks) == 1472
    changed = subprocess.check_output(
        ["git", "diff", "--name-status", "origin/main"], cwd=ROOT, text=True
    ).splitlines()
    assert all(row.startswith("A\t") for row in changed), "Existing repository files changed"
    print(
        json.dumps(
            {
                "public_files_checked": count,
                "tasks": 86,
                "rubrics": 1472,
                "existing_files_unchanged": True,
                "credential_scan": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
