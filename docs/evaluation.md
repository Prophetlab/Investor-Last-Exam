# Evaluation protocol

This document summarizes the supplied October 27, 2026 manuscript revision. It describes the reported experiment; the evaluation runner and data-access infrastructure are not included in this repository.

## Common research environment

- 86 assignments: 78 company-research assignments and 8 Market Trading assignments.
- Shared tool families: `search`, `read`, `download`, `table_operations`, `bash`, and `submit_answer`.
- Each task allows 7,200 seconds and up to three submissions.
- Tool-visible sources must satisfy the task cutoff and authorization limits. Retrieval and computation operate on eligible source versions; later outcomes and private evaluation materials are isolated from the agent.
- Corpus snapshot: October 15, 2026; 3,430,336 document versions; nine source categories; 1.52 TB. Structured market data supplements the document corpus.

Temporal access control does not remove information from model weights. Source availability and model pretraining contamination are distinct issues.

## Answer-based and strict submission evaluation

Each task has weighted criteria on a 0–100 scale. Its score is the weighted fraction of credit earned, multiplied by 100. The overall score is the equal-weight mean across all 86 tasks.

**Answer-based evaluation**, used for the main results, grades the final available answer using the original task criteria. It takes an accepted submission when available, otherwise the last submission payload, or the terminal assistant answer if no submission call was made. No identifiable final answer means zero credit.

**Strict submission evaluation** additionally requires an accepted `submit_answer` call. Omitted or rejected submissions receive zero credit. These scores must not be mixed with the main leaderboard.

| System | Answer-based overall | Strict overall |
|---|---:|---:|
| GPT-6 Astra | 68.1 | 68.1 |
| Claude Opus 5.5 | 66.9 | 62.7 |
| DeepSeek V4.1 Flash | 63.0 | 53.0 |
| GLM 5.3 | 62.3 | 62.3 |
| Qwen 3.8 Max | 62.1 | 62.1 |
| Hunyuan 4 Preview | 60.4 | 47.7 |
| DeepSeek V4 Pro | 59.3 | 59.3 |
| Kimi K3 | 58.2 | 58.2 |
| Step 5 Preview | 55.1 | 34.5 |
| Qwen 3.8 27B | 54.6 | 54.6 |
| Ling 3.0 Flash | 42.6 | 40.9 |

Sources: Tables 3 and 11; Appendix E. Numerical criteria use deterministic checks. Semantic criteria in the main results use GPT-6 Astra. The paper also regrades fixed answers using GPT-OSS-120B; absolute scores and some rankings change with the judge (Table 17).

## Three different meanings of performance

1. **Answer-quality dimensions:** Result, Evidence, Financial logic, Calculation, Reconciliation, Decision consistency, and Uncertainty. Each dimension is averaged over applicable tasks after within-task normalization. Different dimensions can cover different task sets.
2. **Task-category scores:** complete-task scores averaged within a professional category, such as Forecast & Model. These include all criteria in those tasks, not just prediction accuracy.
3. **Realized forecast-outcome scores:** credit on predictions compared with realized values. The Combined column pools point, interval, probability, and market-outcome criteria within each applicable task, normalizes by available outcome points, then averages over applicable tasks.

All are 0–100 scores. None should be labeled “percentage of correct predictions.” The Overall score is not the unweighted mean of category or dimension columns.

API costs in `results/task_categories.csv` are the paper’s mean answering-model API expenditure in USD per task. They are reported experimental estimates, not current model pricing.

## Forecast-driven investment simulation

This experiment reuses numerical fundamental forecasts from the time-constrained research tasks. It is separate from the eight Market Trading tasks and their scores.

### Sample

- March–August 2026 retrospective sample.
- Twenty opportunities from eleven companies and thirteen disclosures.
- Eligibility requires an archived pre-cutoff market-participant estimate with matching metric, period, and units, plus a verified disclosure date and price window.
- References include market consensus and identified individual analysts. Management guidance and hypothetical task priors are excluded.
- Assignments sharing an issuer or disclosure remain separate equally weighted opportunities.

| Company | Assignment IDs | Disclosure dates (2026) |
|---|---|---|
| Alphabet | O19, O21, O22 | Apr 29; Jul 22 |
| Apple | O05, O07, O09 | Jul 30 |
| Amazon | O12 | Jul 30 |
| Clean Harbors | N40 | Jul 29 |
| JPMorgan | O26 | Jul 14 |
| McDonald’s | O43, O44, O45, O46 | Aug 4 |
| Meta | O52 | Jul 29 |
| Microsoft | O58 | Jul 29 |
| Netflix | O66 | Jul 16 |
| Tapestry | TP05 | Aug 13 |
| Tesla | O73, O74, O80 | Jul 2; Jul 22 |

Source: Table 16. The paper reports thirteen disclosures; the grouped date list does not map every assignment to an individual event.

### Signal and capital allocation

For each metric, take the sign of the model forecast minus the reference estimate; reverse the sign for cost metrics. Sum equally weighted metric votes:

- Positive sum: long.
- Negative sum: short.
- Tie: cash.

Missing numerical forecasts hold cash at zero interest. Initial capital assigns 1/20 to each opportunity, without leverage or reallocation of unused capital. Overlapping positions retain their original allocations. Step 5 Preview has only three usable forecasts and is omitted from the portfolio comparison.

### Execution, costs, and measurement

- Entry: first observed trading-day close strictly after the task cutoff instant.
- Exit: first trading-day close after the metric’s disclosure date.
- Costs: 10 basis points on entry and exit notionals, plus 3% annualized short-borrowing cost on initial notional.
- Prices: unadjusted daily closes; cash dividends excluded.
- Maximum drawdown: daily marked-to-market portfolio, including idle cash.
- Always-long and always-short controls use the same allocations, windows, and costs.

Return and maximum drawdown are percentages; excess return is in percentage points relative to always long. Cumulative returns are **not annualized**. Published values retain their original rounding, so subtraction of displayed values can differ by 0.01 percentage point.

### Interpretation

The results measure the payoff of forecast-implied positions over these disclosure windows. They depend on forecast availability, the direction relative to expectations, and subsequent stock-price movements. They are not live investment results, an unrestricted agent trading test, or evidence that a return will persist out of sample.

Sources: Table 4 and Appendix I.2. Exact event-level forecasts, price series, and executable evaluation infrastructure are not shipped in this initial repository; the CSV tables alone do not reproduce the simulation.
