"""Download public wheels on the controller, then install them offline in Docker."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys

from .config import ROOT


def ensure_image():
    if (
        sys.version_info[:2] != (3, 12)
        or platform.system() != "Linux"
        or platform.machine() not in ("x86_64", "amd64")
    ):
        raise ValueError(
            "The reproducible sandbox build currently requires Linux x86-64 and Python 3.12"
        )
    folder = ROOT / "environment"
    digest = hashlib.sha256(
        (folder / "Dockerfile").read_bytes() + (folder / "requirements.txt").read_bytes()
    ).hexdigest()
    image = "investor-last-exam-scientific:" + digest[:16]
    check = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .Config.Labels}}"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        labels = json.loads(check.stdout) or {}
        if labels.get("org.prophet.ile.environment") != digest:
            raise ValueError("Unexpected image under the release tag")
        return image
    build = ROOT / "local/build" / digest[:16]
    wheels = build / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    for name in ("Dockerfile", "requirements.txt"):
        shutil.copyfile(folder / name, build / name)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--index-url",
            "https://pypi.org/simple",
            "--only-binary=:all:",
            "--no-deps",
            "--timeout",
            "30",
            "--retries",
            "2",
            "--dest",
            str(wheels),
            "-r",
            str(build / "requirements.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "build",
            "--progress=plain",
            "--label",
            "org.prophet.ile.environment=" + digest,
            "-t",
            image,
            str(build),
        ],
        check=True,
    )
    return image
