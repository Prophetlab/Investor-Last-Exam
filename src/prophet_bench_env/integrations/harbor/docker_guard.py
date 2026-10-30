"""Private per-environment lease guard, not a scheduler or Docker replacement.

Harbor owns creation. This standalone stdlib worker can outlive its controller
and remove only resources bearing its unguessable ownership labels. No prune.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path

LABEL = "cn.prophetlab.bench.lease"
OWNER = "cn.prophetlab.bench.uid"
IMAGE_OWNER = "cn.prophetlab.bench.managed-image"


def process_identity(pid):
    try:
        # comm can contain spaces and parentheses; fields after the final ')'.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # starttime, original field 22
    except (OSError, IndexError):
        return None


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


@contextmanager
def locked(path):
    fd = os.open(
        path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def read_state(path):
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("lease state is not owned by this user")
    value = json.loads(path.read_text())
    if value["uid"] != os.getuid() or not re.fullmatch(r"[0-9a-f]{32}", value["token"]):
        raise ValueError("invalid lease identity")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,180}", value["project"]):
        raise ValueError("invalid compose identity")
    return value


def write_state(path, state):
    temporary = path.with_suffix(".new")
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "w") as stream:
        json.dump(state, stream, sort_keys=True)
    temporary.replace(path)


def docker(*args):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=20, check=False
    )
    if result.returncode:
        # Docker stderr can embed caller env/config. Persist only the operation.
        raise RuntimeError("Docker operation failed: " + args[0])
    return result.stdout


def owned(state, resource):
    command = {
        "container": ("ps", "-aq"),
        "network": ("network", "ls", "-q"),
        "volume": ("volume", "ls", "-q"),
    }[resource]
    ids = docker(
        *command,
        "--filter",
        f"label={LABEL}={state['token']}",
        "--filter",
        f"label={OWNER}={state['uid']}",
        "--filter",
        f"label=com.docker.compose.project={state['project']}",
    ).split()
    found = []
    for identifier in ids:
        value = json.loads(docker(resource, "inspect", identifier))[0]
        labels = (
            value.get("Config", {}).get("Labels")
            if resource == "container"
            else value.get("Labels")
        )
        labels = labels or {}
        if (
            labels.get(LABEL) != state["token"]
            or labels.get(OWNER) != str(state["uid"])
            or labels.get("com.docker.compose.project") != state["project"]
        ):
            raise RuntimeError("resource ownership changed during cleanup")
        found.append(identifier)
    return found


def cleanup(state):
    """Verify exact daemon + three labels. Never delete images or external volumes."""
    if docker("info", "--format", "{{.ID}}").strip() != state["daemon_id"]:
        raise RuntimeError("Docker daemon changed; refusing cleanup")
    kill_clients(state)
    remaining = {}
    for resource in ("container", "network", "volume"):
        for identifier in owned(state, resource):
            if resource == "container":
                docker("container", "rm", "--force", identifier)
            else:
                # No force: a resource referenced by anything else must survive.
                docker(resource, "rm", identifier)
        remaining[resource] = owned(state, resource)
    if any(remaining.values()):
        raise RuntimeError("owned resources remain after cleanup")
    return remaining


def kill_clients(state):
    """Kill only registered host CLI trees, never dockerd or another runner."""
    for client in state.get("clients", []):
        pid = client["pid"]
        if process_identity(pid) != client["start"]:
            continue
        parents, identities = {}, {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != state["uid"]:
                    continue
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
                identities[int(entry.name)] = fields[19]
            except (OSError, IndexError, ValueError):
                continue
        selected, frontier = {pid}, {pid}
        while frontier:
            frontier = {
                child
                for child, parent in parents.items()
                if parent in frontier and child not in selected
            }
            selected.update(frontier)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for child in selected:
                if (
                    identities.get(child)
                    and process_identity(child) == identities[child]
                ):
                    with suppress(ProcessLookupError):
                        os.kill(child, sig)
            time.sleep(0.1)


def project_is_owned(state):
    """Do not let compose --remove-orphans touch a foreign same-name project."""
    for resource, command in (
        ("container", ("ps", "-aq")),
        ("network", ("network", "ls", "-q")),
    ):
        all_ids = set(
            docker(
                *command,
                "--filter",
                f"label=com.docker.compose.project={state['project']}",
            ).split()
        )
        if all_ids - set(owned(state, resource)):
            return False
    return True


def image_candidates(min_age_seconds=7 * 86400):
    """Only labeled images built by this adapter; never shared BuildKit records."""
    candidates = []
    ids = docker(
        "image",
        "ls",
        "-q",
        "--no-trunc",
        "--filter",
        f"label={IMAGE_OWNER}=v1",
        "--filter",
        f"label={OWNER}={os.getuid()}",
    ).split()
    from datetime import datetime

    for image_id in sorted(set(ids)):
        record = json.loads(docker("image", "inspect", image_id))[0]
        labels = record.get("Config", {}).get("Labels") or {}
        if labels.get(IMAGE_OWNER) != "v1" or labels.get(OWNER) != str(os.getuid()):
            continue
        tags = record.get("RepoTags") or []
        # Other aliases indicate possible sharing: do not untag them.
        project = labels.get("com.docker.compose.project", "")
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,180}", project)
            or labels.get("com.docker.compose.service") != "main"
        ):
            continue
        if tags != [project + "-main:latest"]:
            continue
        if (
            time.time() - datetime.fromisoformat(record["Created"]).timestamp()
            < min_age_seconds
        ):
            continue
        if docker("ps", "-aq", "--filter", "ancestor=" + image_id).strip():
            continue
        candidates.append({"id": image_id, "tags": tags, "size": record.get("Size", 0)})
    return candidates


def maintain_cache(root):
    """At most daily, remove stale proven-owned image tags without force/prune."""
    marker = root / "cache-check.json"
    with locked(marker):
        if marker.exists() and time.time() - marker.stat().st_mtime < 86400:
            return
        removed, retained, retired = [], [], []
        for candidate in image_candidates():
            for tag in candidate["tags"]:
                # Re-inspect immediately; the daemon also refuses in-use images.
                if (
                    json.loads(docker("image", "inspect", tag))[0]["Id"]
                    != candidate["id"]
                ):
                    continue
                try:
                    docker("image", "rm", tag)
                    removed.append(tag)
                except RuntimeError:
                    retained.append(tag)
        for lease in root.glob("*/lease.json"):
            directory = lease.parent
            if (
                directory.is_symlink()
                or time.time() - lease.stat().st_mtime < 7 * 86400
            ):
                continue
            record = read_state(lease)
            if record["status"] != "clean" or record["token"] != directory.name:
                continue
            files = list(directory.iterdir())
            if any(
                p.is_symlink()
                or p.name not in {"lease.json", "lease.lock", "ownership.json"}
                for p in files
            ):
                continue
            # Only retired control receipts, never trial artifacts or input files.
            for file in files:
                file.unlink()
            directory.rmdir()
            retired.append(record["token"])
        write_state(
            marker,
            {
                "removed_image_tags": removed,
                "retained_image_tags": retained,
                "shared_build_cache_pruned": False,
                "retired_lease_receipts": retired,
            },
        )


def directory_size(root, max_bytes):
    """No symlink traversal. Stop counting as soon as the generous budget is hit."""
    total = 0
    pending = [Path(root)]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for item in entries:
                    if item.is_symlink():
                        continue
                    if item.is_dir(follow_symlinks=False):
                        pending.append(Path(item.path))
                    elif item.is_file(follow_symlinks=False):
                        total += item.stat(follow_symlinks=False).st_size
                        if total > max_bytes:
                            return total
        except FileNotFoundError:
            continue
        except PermissionError:
            return max_bytes + 1  # uninspectable outputs must not disable the guard
    return total


def expired(state):
    return (
        state["boot_id"] != boot_id()
        or process_identity(state["pid"]) != state["process_start"]
        or time.monotonic() >= state["deadline"]
    )


def finish(path, reason):
    with locked(path):
        state = read_state(path)
        if state["status"] == "clean":
            return
        try:
            remaining = cleanup(state)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            state.update(
                status="cleanup_failed", reason=reason, error_type=type(exc).__name__
            )
            write_state(path, state)
            raise
        state.update(status="clean", reason=reason, remaining=remaining)
        state.pop("error_type", None)
        write_state(path, state)


def watch(path, interval=5):
    # The worker has its own session. Controller SIGKILL cannot skip this loop.
    while True:
        try:
            state = read_state(path)
            if state["status"] == "clean":
                return
            stale = expired(state)
            over = not stale and any(
                directory_size(root, state["artifact_bytes"]) > state["artifact_bytes"]
                for root in state["artifact_roots"]
            )
            if stale or over or state["status"] == "cleanup_failed":
                finish(
                    path,
                    "artifact_budget" if over else state.get("reason", "lease_expired"),
                )
                return
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            # Keep trying while Docker is temporarily unavailable. No success
            # receipt is emitted until a fresh zero-resource check succeeds.
            pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    watch(args.state)


if __name__ == "__main__":
    main()
