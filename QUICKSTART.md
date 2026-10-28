# Run Investor’s Last Exam

The runnable release adds to the existing project introduction and published results.
Task definitions match [HenryJ12/InvestorLastExam](https://huggingface.co/datasets/HenryJ12/InvestorLastExam):
86 tasks, 1,472 scoring criteria, the recorded public tool schemas, and the original
solver/judge instructions. The data lake remains an authenticated remote service.

## Requirements

- Linux x86-64, Python 3.12, Docker Engine, Docker Compose v2 and Buildx.
- A data-lake API key authorized for the benchmark’s data categories.
- A solver API and 1–3 independently configured judge APIs.
- Initial sandbox build access to Docker Hub and PyPI. Solver workspaces have no network.

Request data access at <https://datalake.prophet-lab.cn/apply>. Once authorized,
create your own API key in the data portal. No shared or maintainer key is included.

## Configure and run

```bash
cp .env.example local/.env
chmod 600 local/.env
# Edit local/.env: fill SOLVER_*, JUDGE_1_* and DATALAKE_API_KEY.
# Fill JUDGE_2_* / JUDGE_3_* only when JUDGE_COUNT is 2 / 3.
./run.sh check --host --config
./run.sh run --task O01 --job-name first-task --concurrency 1
./run.sh summary first-task
```

The launcher creates `local/.venv`, installs the pinned requirements, and invokes
native Harbor. The controller downloads the scientific wheels once, builds the
Docker image with offline package installation, and reuses it across tasks.
Harbor creates the task’s Docker sandbox, runs the solver, saves
the submission and ATIF trajectory, then the trusted controller evaluates the
same final answer separately with each configured judge. Running a real solver
and semantic judges consumes your provider quota.

```bash
./run.sh list
./run.sh run --task all --job-name full-suite --concurrency 2
./run.sh resume full-suite
./run.sh grade full-suite
```

`resume` reopens the frozen Harbor job, without overwriting its completed trials.
`grade` retries missing/failed judge outputs without solving again. Use a new job
name after changing the task/model configuration. Results and credentials remain
under Git-ignored `local/`; do not copy them into the published task directories.

For a manual environment, use `python3 -m venv local/.venv`, install
`requirements.lock`, then `pip install --no-deps -e .` and run `python run.py ...`.

## Models and independent judges

`*_PROTOCOL` selects `chat` (Chat Completions), `responses`, or `anthropic`
(Messages). Base URLs are API roots: the adapter appends `/chat/completions`,
`/responses`, or `/messages`. There is no fixed allowlist of model names.

All routes expose context/output budgets, reasoning settings, temperature,
top-p, streaming, timeouts and retries. Leave unsupported sampling/reasoning
settings blank. `*_REASONING_PROTOCOL` selects an explicit Chat dialect such as
`generic`, `deepseek`, `glm`, `qwen`, `stepfun` or `bailing`. Vendor-specific
parameters can be supplied as JSON in `*_EXTRA_BODY`; for example, adaptive
Messages thinking can use `{"thinking":{"type":"adaptive"},"output_config":{"effort":"max"}}`.
`*_EXTRA_HEADERS` supports provider-specific headers. Neither option overrides
the task conversation or tool list. Credentials belong only in `local/.env`.

Reader defaults to the solver’s connection, with its own audited invocations.
Set `READER_REUSE_SOLVER=false` to fill a dedicated Reader route. Reading uses
one model call with no research tools, a context-aware budget and source coverage.

Each judge receives the same immutable answer and original rubric/reference
packet in a fresh context. Outputs are `judge_1.json`, `judge_2.json` and
`judge_3.json`; there is no averaging, majority vote or cross-judge feedback.
Mechanical criteria use the original deterministic functions and do not call
a judge. Pure market tasks call no semantic judge.

`JUDGE_JSON_MODE=prompt` works across the three transports; `native` requests
native JSON-schema output for Chat/Responses endpoints that support it.
Structured grading retains the published grade-submission tool.

## Results and scoring

`local/runs/<job>/summary.json` contains separate per-judge means and coverage.
Each native Harbor trial retains:

- `agent/events.jsonl` and `agent/trajectory.json` (ATIF, including Reader links).
- `agent/submission.json`, model/Reader audit records and workspace artifacts.
- `verifier/judge_N.json`, with per-rubric points, final-answer score and strict score.

Main scores assess the accepted answer, otherwise the last actual submission,
otherwise the literal terminal answer. No answer is synthesized. Strict scores
additionally require successful submission. Original numerical rules, units,
forecast allocation and display rounding are preserved; unrounded totals are saved.
Infrastructure and judge failures are reported separately from scored answers.

Provider settings and a changing authorized data corpus can affect model outputs.
The release reproduces the task/tool/evaluation pipeline; it does not promise
identical stochastic answers from different endpoints or judges. Historical
published tables are preserved independently of new runs.

## Verification

```bash
local/.venv/bin/python -m pip install pytest
local/.venv/bin/python -m pytest -q tests
local/.venv/bin/python scripts/smoke.py --env local/.env
```

The smoke check uses a scripted local model, real Harbor/Docker and the data
scope preflight. It makes no paid model calls and is an infrastructure check,
not an ability measurement. Its output is kept separate from benchmark results.

## Added directories

```text
src/ile_bench/          Public configuration, runtime, scoring and CLI
src/prophet_bench_env/  Pinned environment/tool/Harbor implementation
src/financial_data_lake/ SDK 1.1.0 client and API contract types only
src/plugins/harness/    Compact ReAct harness
tasks/                 86 task packages; evaluator inputs stay on the controller
environment/           Rebuildable scientific Docker workspace
scripts/               Smoke check and maintainer source-export tooling
tests/                 Protocol, scoring, isolation and configuration regressions
local/                 Ignored private configuration, environments and new runs
```

The bundled client omits database services, account administration, ingestion,
provider credentials and raw data archives. Future truth, rubrics and judge inputs
are never copied into the solver container. The standalone environment remains
decoupled from ILE task definitions and from the website.
