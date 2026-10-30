# Released task packages

These 86 JSON files mirror the English task records at
[HenryJ12/InvestorLastExam](https://huggingface.co/datasets/HenryJ12/InvestorLastExam).
They contain 1,472 rubrics, frozen numerical evaluation parameters, tool schemas
and the original solver/judge prompts.

Each record uses its public code (for example, `O37`) for both `id` and `task_id`.
Judge input and response schemas use that same identifier. `source_sha256` is the
SHA-256 of the public record excluding that field, serialized as UTF-8 JSON with
sorted keys, no ASCII escaping, and compact separators.

Use `./run.sh list` to list task codes and companies, then
`./run.sh run --task O01 --job-name first-task` to run one. The full instructions
are in [QUICKSTART.md](../QUICKSTART.md).

`question`, `solver_system_prompt`, public tool definitions and the submission
schema form the solver input. Rubrics, future truth, judge prompts and reference
material remain on the trusted evaluation controller. The runner constructs
native Harbor task directories automatically and never mounts this full directory
into an agent sandbox.

Task content and cited third-party materials retain their original terms;
the runtime code license does not grant redistribution rights to the data lake.
