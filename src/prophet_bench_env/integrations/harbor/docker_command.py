"""Copied into the main container; trusted root supervisor, unprivileged command.

Linux subreaper semantics keep double-fork/setsid descendants attached to this
supervisor. Timeout, cancellation and output overflow reap the entire tree.
"""

import argparse
import base64
import ctypes
import json
import os
import pwd
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


def children(parent):
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            table[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    result = set()
    frontier = {parent}
    while frontier:
        frontier = {
            pid for pid, ppid in table.items() if ppid in frontier and pid not in result
        }
        result.update(frontier)
    return result


def reap():
    # Re-scan after signals: subreaper adoption catches detached descendants.
    for sig in (signal.SIGTERM, signal.SIGKILL, signal.SIGKILL):
        for pid in children(os.getpid()):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(0.1)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
    return not children(os.getpid())


def run(root, token, spec):
    cancelled = False

    def cancel(*_):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    if (
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0
    ):  # PR_SET_CHILD_SUBREAPER
        raise RuntimeError("cannot enable command subreaper")
    (root / (token + ".pid")).write_text(str(os.getpid()))
    receipt = root / (token + ".json")
    if (root / (token + ".cancel")).exists():
        receipt.write_text(json.dumps({"reason": "cancelled", "exit_code": 130}))
        return 130
    user = spec.get("user")
    account = (
        pwd.getpwuid(int(user))
        if user is not None and str(user).isdigit()
        else pwd.getpwnam(str(user or "root"))
    )
    child = subprocess.Popen(
        ["bash", "-c", spec["command"]],
        user=account.pw_uid,
        group=account.pw_gid,
        extra_groups=[],
        cwd=spec.get("cwd") or None,
        env={**os.environ, **spec.get("env", {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    count, reason = 0, "completed"
    poller = selectors.DefaultSelector()
    poller.register(child.stdout, selectors.EVENT_READ)
    try:
        while poller.get_map() or child.poll() is None:
            if cancelled or (root / (token + ".cancel")).exists():
                reason = "cancelled"
                break
            if spec.get("timeout") and time.monotonic() - started >= spec["timeout"]:
                reason = "timeout"
                break
            for key, _ in poller.select(0.1):
                data = os.read(key.fd, 65536)
                if not data:
                    poller.unregister(key.fileobj)
                    continue
                count += len(data)
                if count > spec["max_bytes"]:
                    reason = "output_limit"
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            if reason != "completed":
                break
        if reason != "completed":
            if not reap():
                reason = "reap_failed"
            code = {
                "timeout": 124,
                "cancelled": 130,
                "output_limit": 125,
                "reap_failed": 126,
            }[reason]
        else:
            code = child.wait()
    except BaseException:
        reap()
        raise
    finally:
        poller.close()
        child.stdout.close()
    receipt.write_text(
        json.dumps({"reason": reason, "exit_code": code, "output_bytes": count})
    )
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "cancel", "status"))
    parser.add_argument("root", type=Path)
    parser.add_argument("token")
    parser.add_argument("spec", nargs="?")
    args = parser.parse_args()
    if not args.token.isalnum() or len(args.token) != 32:
        raise ValueError("invalid command token")
    if args.action == "cancel":
        # Root-owned cancellation marker also covers dispatch/start races.
        (args.root / (args.token + ".cancel")).touch()
        return
    if args.action == "status":
        print((args.root / (args.token + ".json")).read_text())
        return
    raise SystemExit(
        run(args.root, args.token, json.loads(base64.b64decode(args.spec)))
    )


if __name__ == "__main__":
    main()
