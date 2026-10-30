"""Real Harbor/Docker infrastructure check with a scripted, local model endpoint.

Requires a data key for the scope preflight. No paid model calls and no ability claim.
"""

import argparse
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from ile_bench.config import ROOT, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=Path, default=ROOT / "local/.env")
    p.add_argument("--job-name", default="infrastructure-smoke")
    args = p.parse_args()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            done = any(m.get("role") == "tool" for m in request["messages"])
            name = "submit_answer" if done else "bash"
            arguments = (
                {"forecast_close_usd": 280.14}
                if done
                else {"command": "python -c 'import numpy, pandas; print(1+1)'"}
            )
            body = json.dumps(
                {
                    "id": "smoke",
                    "model": "scripted-infrastructure-smoke",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "submit" if done else "compute",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = values(args.env)
    base = f"http://127.0.0.1:{server.server_port}/v1"
    env.update(
        JUDGE_COUNT="1",
        ILE_WALL_CLOCK_SECONDS="90",
        READER_REUSE_SOLVER="true",
        HARBOR_TELEMETRY="0",
    )
    for prefix in ("SOLVER", "JUDGE_1"):
        env.update(
            {
                prefix + "_MODEL": "scripted-infrastructure-smoke",
                prefix + "_BASE_URL": base,
                prefix + "_API_KEY": "local-test-only",
                prefix + "_PROTOCOL": "chat",
                prefix + "_STREAM": "false",
                prefix + "_REASONING_EFFORT": "",
                prefix + "_MAX_RETRIES": "0",
            }
        )
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--env",
                str(args.env),
                "run",
                "--task",
                "M01",
                "--job-name",
                args.job_name,
                "--concurrency",
                "1",
            ],
            env=env,
            check=True,
        )
        summary = json.loads((ROOT / "local/runs" / args.job_name / "summary.json").read_text())
        assert (
            summary["judges"]["judge_1"]["scored"] == 1
            and summary["judges"]["judge_1"]["mean_score"] == 100
        )
        traces = list(
            (ROOT / "local/runs" / args.job_name / "jobs").glob("*/*/agent/trajectory.json")
        )
        assert len(traces) == 1 and len(calls) == 2
        print(
            json.dumps(
                {
                    "infrastructure_smoke": "passed",
                    "scripted_model_calls": len(calls),
                    "atif": True,
                    "ability_measurement": False,
                }
            )
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
