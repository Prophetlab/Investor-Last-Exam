"""A trusted per-trial tool session, independent of solver and MCP transport."""

from threading import Lock

from ...core.contracts import ToolCall
from ...core.submission import received
from ...ports.tooling import ToolGateway, ToolRegistration


class ToolSession:
    def __init__(self, task, registration: ToolRegistration, trajectory):
        self.task, self.registration, self.trajectory = task, registration, trajectory
        self.gateway = ToolGateway.from_tools(
            registration.tools, submission_policy=task.submission_policy
        )
        submits = [
            d for d in self.gateway.definitions() if d.side_effect == "submission"
        ]
        if submits != [task.submit_tool]:
            raise ValueError(
                "tool factory must register exactly this task's submission"
            )
        self.submission = None
        self.calls = {}
        self.lock = Lock()

    def definitions(self):
        return [
            {
                "name": d.name,
                "description": d.description,
                "inputSchema": d.input_schema,
            }
            for d in self.gateway.definitions()
        ]

    def call(self, call_id, name, arguments):
        # IDs are issued by the transport, not used as paths or trusted arguments.
        with self.lock:
            if call_id in self.calls:
                raise ValueError("duplicate tool invocation")
            self.calls[call_id] = None
        self.trajectory.append(
            "external_tool_call",
            {"call_id": call_id, "name": name, "arguments": arguments},
        )
        result = self.gateway.execute(ToolCall(call_id, name, arguments))
        data = result.as_dict()
        self.trajectory.append("tool_result", data)
        with self.lock:
            self.calls[call_id] = data
            if name == self.task.submit_tool.name and received(result):
                self.submission = result.output
        return data

    def submission_record(self, *, model_id, harness_id):
        with self.lock:
            return self.gateway.submission.record(
                self.task,
                episode_id=self.trajectory.episode_id,
                model_id=model_id,
                harness_id=harness_id,
            )

    def close(self):
        if self.registration.close:
            self.registration.close()
