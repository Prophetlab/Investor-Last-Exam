<div align="center">

# Investor’s Last Exam

### Can AI Agents Do Equity Investment Without Seeing the Future?

**Make the call. Before the reveal.**

[Paper (PDF)](docs/paper/Investors-Last-Exam.pdf) · [Hugging Face Space](https://huggingface.co/spaces/HenryJ12/InvestorLastExam) · [Project page](https://prophet-lab.cn/benchmarks/investor-last-exam-v1/) · [中文介绍](README.zh-CN.md) · [Results](#results) · [Evaluation protocol](docs/evaluation.md) · [Citation](#citation)

**86 assignments · 11 evaluated systems · 3.43 million document versions**

</div>

## Paper

[Read the full paper](docs/paper/Investors-Last-Exam.pdf) · [Download PDF](https://github.com/Prophetlab/Investor-Last-Exam/raw/refs/heads/main/docs/paper/Investors-Last-Exam.pdf)

Preprint; submitted to arXiv, announcement pending. This is the same author-provided manuscript available on the [Hugging Face Space](https://huggingface.co/spaces/HenryJ12/InvestorLastExam).

## 78-second video overview

https://github.com/user-attachments/assets/333ac228-db23-4110-86fe-9602423cded3

[Open standalone player](https://prophetlab.github.io/Investor-Last-Exam/) · 1080p · English narration with Chinese and English subtitles · [Download MP4](https://github.com/Prophetlab/Investor-Last-Exam/raw/refs/heads/main/docs/assets/investors-last-exam-78s.mp4)

A short introduction to the time-frozen environment, Tesla and Nebius research workflows, and the evaluation of eleven models.

Investor’s Last Exam (**ILE**) is a buy-side-expert-led benchmark for equity investment research. Agents investigate evidence, build forecasts, form investment judgments, and update their views using information available at a specified historical cutoff. Their answers are evaluated against expert-defined financial requirements and, where applicable, subsequent realized outcomes.

A convincing research report is only part of the test. ILE examines whether evidence, assumptions, calculations, and decisions fit together—and whether forecasts hold up when the results arrive.

Led by **Prophetlab**, jointly launched with **Foresight Fund Management**, **IvyRock Asset Management**, the **School of Artificial Intelligence, Shanghai Jiao Tong University (ReThinkLab)**, and **Antai College of Economics and Management, Shanghai Jiao Tong University**.

## What makes ILE different?

Existing financial benchmarks cover document question answering, agentic search, research workflows, forecasting, or trading. ILE connects these concerns through three components:

1. **A time-frozen research environment.** A versioned corpus enforces source availability across retrieval and computation. A date in a prompt or search query is not treated as sufficient temporal control.
2. **Professional research assignments.** Experienced buy-side investors and fund managers design tasks around analyst deliverables, from evidence reconciliation and forecast construction to investment decisions.
3. **Traceable, multidimensional evaluation.** Expert-led rubrics connect scoring requirements to sources, calculations, assumptions, and financial dependencies. Research quality, realized forecast performance, and simulated investment returns are reported separately.

The corpus snapshot used in the paper contains **3,430,336 document versions across nine source categories**, totaling **1.52 TB** as of **October 15, 2026**. These include financial reports, structured statements, earnings calls, investor materials, issuer events, analyst estimates, research reports, institutional holdings, and news. Structured market data supports price, return, and volatility analysis.

Search, reads, downloads, cached results, and structured-table access share the task’s authorization and availability boundary. Reference analyses, rubrics, and later outcomes remain in a separate evaluation environment. This controls tool-visible evidence; it does **not** erase knowledge potentially acquired during model pretraining.

## The task suite

The current paper evaluates **86 assignments: 78 company-research assignments and 8 market assignments**. The first five categories contain twenty professional subtypes; Integrated Research combines these capabilities into full reports.

| Category | Example deliverable | Tasks |
|---|---|---:|
| Discover & Frame | Opportunity screen, catalyst map, or testable investment thesis | 10 |
| Research & Evidence | Comparable historical series or reconciled financial facts | 13 |
| Forecast & Model | Operating forecast, earnings bridge, or peer read-through | 13 |
| Evaluate & Decide | Valuation, implied expectations, or investment action | 14 |
| Monitor & Update | Event assessment, revised forecast, or thesis update | 17 |
| Integrated Research | Report linking forecasts, stance, risks, and thesis conditions | 11 |
| Market Trading | Price forecast or forward return/volatility ranking | 8 |
| **Total** | | **86** |

## Two examples

### Tesla: coherent analysis can still miss the future

**Task O80** asks for Tesla’s Q2 2026 deliveries, GAAP revenue, and GAAP automotive gross margin using information available as of **June 22, 2026**.

GPT-6 Astra, Claude Opus 5.5, and Qwen 3.8 Max each earned **50/50 supporting-analysis points**, but **0/50 forecast-outcome points**. All three underestimated deliveries and revenue and overestimated automotive gross margin; each error exceeded the corresponding zero-credit threshold.

Calculation bridges and coherent recommendations do not establish that assumptions about future demand, pricing, and cost are accurate. ILE evaluates both supporting analysis and realized predictions. These figures describe this particular task, not each model’s benchmark-wide performance.

### AEP: more operating cash, yet a larger funding gap

**Task N27** asks how American Electric Power’s funding position and fuel-cost recovery changed between H1 2025 and H1 2026, supported by its Q2 2026 report.

| USD millions | H1 2025 | H1 2026 | Change |
|---|---:|---:|---:|
| Operating cash inflow | 2,671 | 3,421 | +750 |
| Investing cash use | 5,347 | 6,602 | +1,255 |
| Pre-financing funding shortfall | 2,676 | 3,181 | **+505** |

All eleven systems recovered the eight reported inputs and reconciled both cash-flow identities. Only **five** computed both funding shortfalls, and **five** earned full credit on the integrated funding-conclusion criteria. Only **three** distinguished the net deferred-fuel adjustment from gross fuel collections.

The test is whether an agent understands the source and use of cash, including temporary movements, rather than simply reporting a rise in operating cash flow.

## Results

**Reading the scores:** the main results use the paper’s *answer-based* protocol, which grades the final available answer independently of submission acceptance. Every assignment has equal weight in the overall score. GPT-6 Astra grades semantic criteria; numerical criteria use deterministic checks. Strict submission results are documented separately in the [evaluation protocol](docs/evaluation.md#answer-based-and-strict-submission-evaluation).

### Overall performance and realized forecasts

All columns below are scores on a **0–100 scale**. They evaluate different task sets or criteria and are **not percentages of correct forecasts**.

| System | Overall | Forecast & Model | Realized forecast outcomes |
|---|---:|---:|---:|
| GPT-6 Astra | **68.1** | 52.5 | 37.7 |
| Claude Opus 5.5 | 66.9 | **52.6** | 37.6 |
| DeepSeek V4.1 Flash | 63.0 | 45.8 | **40.3** |
| GLM 5.3 | 62.3 | 46.6 | 34.1 |
| Qwen 3.8 Max | 62.1 | 49.9 | 34.5 |
| Hunyuan 4 Preview | 60.4 | 50.5 | 37.9 |
| DeepSeek V4 Pro | 59.3 | 45.6 | 35.0 |
| Kimi K3 | 58.2 | 41.0 | 37.2 |
| Step 5 Preview | 55.1 | 43.9 | 31.2 |
| Qwen 3.8 27B | 54.6 | 44.5 | 32.8 |
| Ling 3.0 Flash | 42.6 | 38.5 | 33.4 |

*Source: paper Tables 3 and 13. Forecast & Model is the complete-task score for that category. Realized forecast outcomes is the Combined column of Table 13: task-normalized credit on point, interval, probability, and market-outcome criteria, averaged over applicable tasks.*

- **GPT-6 Astra leads overall**, but its Research & Evidence score of 78.9 contrasts with 52.5 on Forecast & Model.
- **DeepSeek V4.1 Flash leads realized forecast outcomes** at 40.3.
- **DeepSeek V4 Pro leads Market Trading** at 64.6, despite an overall score of 59.3.

Full numerical breakdowns are available as CSV: [answer-quality dimensions](results/answer_quality.csv), [task categories and API costs](results/task_categories.csv), and [realized forecast outcomes](results/forecast_outcomes.csv).

### Forecast-driven simulated investment returns

The paper also converts fundamental forecasts into trading signals by comparing them with archived, pre-cutoff analyst estimates. This simulation is **separate from the eight Market Trading assignments**.

The retrospective sample contains **20 opportunities across 11 companies and 13 disclosures**, spanning **March–August 2026**. Forecast-implied signals produce long, short, or cash positions. Each opportunity receives 1/20 of initial capital, with no leverage or reallocation of unused capital.

| System / control | Cumulative return | Excess vs. always long (pp) | Maximum drawdown |
|---|---:|---:|---:|
| Hunyuan 4 Preview | **+8.50%** | **+7.32** | 2.72% |
| DeepSeek V4.1 Flash | +7.98% | +6.80 | 3.65% |
| GPT-6 Astra | +7.18% | +6.00 | **1.50%** |
| Kimi K3 | +7.15% | +5.97 | 4.02% |
| Claude Opus 5.5 | +6.39% | +5.21 | 2.94% |
| GLM 5.3 | +6.19% | +5.01 | 3.53% |
| Qwen 3.8 27B | +5.20% | +4.02 | 1.90% |
| DeepSeek V4 Pro | +3.64% | +2.46 | 2.03% |
| Ling 3.0 Flash | +2.51% | +1.33 | 2.64% |
| Qwen 3.8 Max | +1.67% | +0.48 | 2.05% |
| Always long | +1.18% | +0.00 | 7.63% |
| Always short | −2.27% | −3.45 | 8.86% |

*Source: paper Table 4 and Appendix I.2. Values retain the paper’s rounding. Returns include 10 bps per entry/exit leg and 3% annualized short-borrowing costs. They are not annualized and exclude cash dividends. Step 5 Preview has only three usable forecasts and is omitted from the portfolio comparison; its return is not reported as zero.*

Hunyuan leads this sample, while Kimi’s 7.15% exceeds Opus’s 6.39% despite its lower overall research score. **Overall scores, realized forecast scores, and simulated returns capture different strengths.** These are limited-sample retrospective results, not live trading performance or evidence of persistent future returns.

See the [simulation specification](docs/evaluation.md#forecast-driven-investment-simulation) and [downloadable return table](results/investment_returns.csv).

## Repository contents and availability

This repository currently provides the project overview, evaluation documentation, and machine-readable tables transcribed from the supplied paper revision (October 27, 2026).

| Resource | Availability in this repository |
|---|---|
| Paper PDF | [docs/paper/Investors-Last-Exam.pdf](docs/paper/Investors-Last-Exam.pdf) |
| English / Chinese overview | [README.md](README.md) / [README.zh-CN.md](README.zh-CN.md) |
| Evaluation and simulation protocol | [docs/evaluation.md](docs/evaluation.md) |
| Reported model results and task counts | [results/](results/README.md) |
| Evaluation runner and agent tools | Not included |
| Executable task suite and data-access service | Not included |
| Full versioned financial corpus | Not distributed here |
| Private references and scoring rubrics | Not distributed here |

The CSV files reproduce reported results; they are not a replacement for running the evaluation. There is no runnable benchmark installation in this initial repository. Visit the [project page](https://prophet-lab.cn/benchmarks/investor-last-exam-v1/) for project information, or [open an issue](https://github.com/Prophetlab/Investor-s-Last-Exam/issues) with questions about access or reproduction.

## Citation

```bibtex
@misc{jiang2026investorslastexam,
  title        = {Investor's Last Exam: Can AI Agents Do Equity Investment Without Seeing the Future?},
  author       = {Huangfei Jiang and Anning Hu and Kai Sheng and Yuchen Cai and Rui Ge and Xiangyu Kong and Jianghao Lin and Shuzi Chen and Jiyi Weng and Junchi Yan},
  year         = {2026},
  howpublished = {Project manuscript},
  url          = {https://prophet-lab.cn/benchmarks/investor-last-exam-v1/}
}
```

## Acknowledgments

Led by **Prophetlab**, with investment and academic partners:

- **Foresight Fund Management** · 睿远基金
- **IvyRock Asset Management** · 常春藤基金
- **School of Artificial Intelligence, Shanghai Jiao Tong University (ReThinkLab)** · 上海交通大学人工智能学院（ReThinkLab）
- **Antai College of Economics and Management, Shanghai Jiao Tong University** · 上海交通大学安泰经济与管理学院

## Licensing

No open-source license is specified in this initial repository. The paper, financial source materials, future code releases, and datasets may have different terms; this repository does not grant redistribution rights to the underlying financial corpus.
