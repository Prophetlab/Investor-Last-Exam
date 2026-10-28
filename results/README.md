# Reported results

These UTF-8 CSV files transcribe tables from the supplied October 27, 2026 revision of *Investor’s Last Exam: Can AI Agents Do Equity Investment Without Seeing the Future?* They are reported aggregate results, not raw predictions or newly executed experiments.

| File | Paper source | Units / contents |
|---|---|---|
| [answer_quality.csv](answer_quality.csv) | Table 3(a) | Overall and seven answer-quality dimensions, 0–100 |
| [task_categories.csv](task_categories.csv) | Table 3(b) | Seven task-category scores, 0–100; mean API cost in USD/task |
| [forecast_outcomes.csv](forecast_outcomes.csv) | Table 13 | Combined and four realized-outcome components, 0–100 |
| [investment_returns.csv](investment_returns.csv) | Table 4 | Cumulative return and maximum drawdown in percent; excess in percentage points |
| [task_taxonomy.csv](task_taxonomy.csv) | Table 2 | Task counts; total 86 |

The main score tables use answer-based evaluation with GPT-6 Astra for semantic criteria and deterministic numerical scoring. Forecast outcomes are normalized rubric credit, not hit rates. Category, dimension, and outcome scores cover different criteria and sometimes different task sets.

The investment CSV includes ten models and two controls. Step 5 Preview was excluded because it had only three usable forecasts, not because its return was zero. The `kind` column distinguishes models from controls.

Values preserve the paper’s precision and rounding. In particular, do not overwrite published excess returns by subtracting already rounded display values.

See [evaluation.md](../docs/evaluation.md) for aggregation, strict submission scoring, simulation assumptions, and limitations. See the [project page](https://prophet-lab.cn/benchmarks/investor-last-exam-v1/) for project information.
