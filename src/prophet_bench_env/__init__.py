"""Public lazy imports; mechanical verifiers need no rollout dependencies."""

from importlib import import_module

__version__ = "0.2.0a0"
_GROUPS = {
    ".core.contracts": "Assignment AssistantAction EpisodeContext EvaluationContext EvaluationResult Observation PublicTask RegisteredTool RolloutToolContext RubricBundle RubricCriterion RubricResult StepResult SubmissionArtifact SubmissionPolicy SubmissionReview TaskBundle ToolCall ToolDefinition ToolError ToolResult",
    ".core.validation": "ValidationIssue ANY ANY_INDEX path_matches json_pointer resolve_pointer check_attachment",
    ".core.engine": "EpisodeEngine accepted_report",
    ".core.artifacts": "FilesystemArtifactStore",
    ".core.trajectory": "JsonlTrajectorySink",
    ".core.model_audit": "bind_model_audit",
    ".core.resources": "LimitedModelAdapter ResourceGuards",
    ".ports.artifacts": "ArtifactStore",
    ".ports.plugins": "Agent AgentResult HarnessPlugin PluginManifest ValidationReport",
    ".ports.model": "AsyncModelAdapter ModelAdapter ModelRequest ModelResponse",
    ".ports.tooling": "ToolGateway ToolRegistration default_data_tool_definitions",
    ".ports.sandbox": "LocalSandbox StreamingSandboxAdapter",
    ".adapters.data_lake": "DataLakeClient DataLakeError DownloadReceipt",
    ".adapters.data_tools": "DataLakeTools LocalWorkspaceBackend PassthroughReader",
    ".adapters.model_reader": "ModelBackedReader",
    ".adapters.e2b_backend": "DEFAULT_PYTHON_PACKAGES CommandResult E2BBackend E2BHooks",
    ".model_adapters": "AnthropicMessagesAdapter ModelSettings OpenAICompatibleChatAdapter OpenAIResponsesAdapter build_model_adapter build_model_adapter_from_env",
    ".surfaces.gym_env": "ProphetBenchGymEnv",
    ".surfaces.registry": "load_engine_factory",
}
_EXPORTS = {name: module for module, names in _GROUPS.items() for name in names.split()}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
