"""Typed formal-delivery findings, with a backwards-compatible list interface."""

from contextlib import contextmanager


class DeliveryReport(list):
    """Keep preflight's list API and render findings in one summary section."""

    def __init__(self):
        super().__init__()
        self.sections = []
        self.findings = []
        self._advisory_depth = 0

    def append(self, value):
        super().append(value)
        self.findings.append(
            ("advisory" if self._advisory_depth else "blocking", value)
        )

    def extend(self, values):
        for value in values:
            self.append(value)

    @contextmanager
    def advisory(self):
        self._advisory_depth += 1
        try:
            yield
        finally:
            self._advisory_depth -= 1

    def warn(self, value):
        with self.advisory():
            self.append(value)

    @property
    def blockers(self):
        return list(
            dict.fromkeys(
                value for level, value in self.findings if level == "blocking"
            )
        )

    @property
    def warnings(self):
        return list(
            dict.fromkeys(
                value for level, value in self.findings if level == "advisory"
            )
        )

    def as_dict(self):
        return {
            "formal_ready": not self.blockers,
            "blockers": self.blockers,
            "warnings": self.warnings,
        }

    @contextmanager
    def section(self, title):
        start = len(self)
        yield self
        self.sections.append((title, list(self[start:])))

    def check(self, title, function, *args):
        from .tencent_v3 import DeliveryNotReady

        with self.section(title):
            try:
                return function(*args, self)
            except DeliveryNotReady:
                # Unsafe paths/credentials are not customer acceptance findings.
                raise
            except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
                self.append(
                    f"Cannot verify supplied evidence: {type(error).__name__}: {error}"
                )

    def markdown(self, *, draft=False):
        def text(value):
            return str(value).replace("|", "\\|").replace("\n", " ")

        lines = [
            "# 交付摘要 / Delivery summary",
            "",
            "交付用途：审阅稿（draft），供交接与材料核对。"
            if draft
            else "交付用途：正式导出（通过本导出器静态准入，非腾讯验收声明）。",
            "正式导出存在阻断项时不生成 ZIP；draft 可保留全部缺项材料，但不能绕过安全检查。",
            "",
            "## 需求满足情况 / Requirements observations",
            "",
            (
                "依据腾讯 2026-09-26 文档。以下仅核对提供的文件和运行记录，"
                "不是腾讯验收结论；未重新运行模型、镜像或评分器。"
            ),
            f"静态准入核对：{len(self.blockers)} 个阻断项，{len(self.warnings)} 个提醒。两种模式均不自动筛除题目或运行。",
            "‘未发现静态不一致’不代表已完成专家审查、镜像加载或人工因果验证。",
            "原始 reward 不改写、不因量纲排除统计；由明确满分及评分记录或 reward_scales 计算 normalized_reward，并保留依据。量纲未知时只记原始统计，不猜分母、不截断。",
            "pass@4 暂以 normalized_reward == 1.0 为成功口径（单位区间 reward 等价于原口径），未声称腾讯已确认该定义。",
            "指定配置：Hy4 preview-high-1M 与 Claude Opus 5.5-xhigh-1M，均使用 Claude Code。",
            "26Sep 难度门槛：混元 pass@4 < 50%；Claude 每题 mean reward < 60%；全零需额外可解性与 hint 证据。",
            "Oracle 应为 1，NOP 应为 0。混元至少 4 次才能计算 pass@4，目录示例的 8 次作为提醒；Claude 暂按目录示例保守要求至少 5 次，不声称正文给出了唯一次数。",
            "",
            "| 核对项 | 记录结果 |",
            "| --- | --- |",
        ]
        sections = {}
        for title, findings in self.sections:
            sections.setdefault(title, []).extend(findings)
        covered = set()
        for title, findings in sections.items():
            covered.update(findings)
            result = "有缺项、不符合或待核实事项" if findings else "未发现静态不一致"
            lines.append(f"| {text(title)} | {result} |")
        for title, findings in [
            *sections.items(),
            ("其他记录", [x for x in self if x not in covered]),
        ]:
            if findings:
                lines.extend(["", f"### {text(title)}", ""])
                lines.extend(
                    f"- {'[阻断]' if finding in self.blockers else '[提醒]'} {text(finding)}"
                    for finding in dict.fromkeys(findings)
                )
        return ("\n".join(lines) + "\n").encode()


def warn(issues, message):
    """Older checker callers can still pass an ordinary list."""
    if isinstance(issues, DeliveryReport):
        issues.warn(message)
    else:
        issues.append(message)
