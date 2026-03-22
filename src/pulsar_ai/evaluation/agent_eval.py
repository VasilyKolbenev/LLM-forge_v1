"""Agent evaluation framework with scenario-based task suites.

Provides structured evaluation of ReAct agents using configurable
test cases, multiple scoring strategies (exact, contains, LLM judge),
and aggregate reporting with per-tag breakdowns.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import yaml

from pulsar_ai.evaluation.llm_judge import (
    JudgeCriterion,
    LLMJudge,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskCase:
    """Single test case for an agent.

    Args:
        id: Unique case identifier.
        query: User input to send to the agent.
        expected_answer: Expected final answer for exact/contains match.
        expected_tools: Expected tool calls (order-insensitive).
        rubric: LLM judge rubric for quality scoring.
        tags: Category tags (e.g. "math", "retrieval").
        max_time_s: Timeout per case in seconds.
    """

    id: str
    query: str
    expected_answer: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    rubric: str = ""
    tags: list[str] = field(default_factory=list)
    max_time_s: float = 30.0


@dataclass
class TaskSuite:
    """Collection of test cases for evaluating an agent.

    Args:
        name: Suite display name.
        description: What this suite tests.
        cases: List of individual test cases.
        version: Suite version string.
    """

    name: str
    description: str
    cases: list[TaskCase]
    version: str = "1.0"


@dataclass
class CaseResult:
    """Result of running a single test case.

    Args:
        case_id: Matching TaskCase.id.
        query: The input query that was sent.
        response: Agent's final response text.
        trace: Agent execution trace entries.
        success: Whether the case passed.
        score: Quality score from 0.0 to 1.0.
        latency_ms: Wall-clock time in milliseconds.
        tokens_used: Total tokens consumed.
        cost: Estimated cost in USD.
        tools_used: Tool names the agent actually called.
        tools_match: Whether expected tools were used.
        error: Error message if the case failed.
    """

    case_id: str
    query: str
    response: str
    trace: list[dict[str, Any]]
    success: bool
    score: float
    latency_ms: int
    tokens_used: int
    cost: float
    tools_used: list[str]
    tools_match: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict."""
        return {
            "case_id": self.case_id,
            "query": self.query,
            "response": self.response,
            "trace": self.trace,
            "success": self.success,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "tools_used": self.tools_used,
            "tools_match": self.tools_match,
            "error": self.error,
        }


@dataclass
class EvalReport:
    """Aggregate evaluation report.

    Args:
        suite_name: Name of the task suite that was run.
        model_name: Model identifier used for evaluation.
        timestamp: ISO-format timestamp of the run.
        results: Per-case results.
    """

    suite_name: str
    model_name: str
    timestamp: str
    results: list[CaseResult]

    @property
    def success_rate(self) -> float:
        """Fraction of cases that passed (0.0-1.0)."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.success)
        return round(passed / len(self.results), 4)

    @property
    def avg_score(self) -> float:
        """Mean quality score across all cases."""
        if not self.results:
            return 0.0
        return round(
            sum(r.score for r in self.results) / len(self.results),
            4,
        )

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency in milliseconds."""
        if not self.results:
            return 0.0
        return round(
            sum(r.latency_ms for r in self.results) / len(self.results),
            2,
        )

    @property
    def total_tokens(self) -> int:
        """Sum of tokens used across all cases."""
        return sum(r.tokens_used for r in self.results)

    @property
    def total_cost(self) -> float:
        """Sum of cost across all cases."""
        return round(sum(r.cost for r in self.results), 6)

    @property
    def tools_accuracy(self) -> float:
        """Fraction of cases with correct tool usage."""
        if not self.results:
            return 0.0
        matched = sum(1 for r in self.results if r.tools_match)
        return round(matched / len(self.results), 4)

    @property
    def by_tag(self) -> dict[str, dict[str, Any]]:
        """Per-tag aggregate metrics.

        Returns:
            Dict mapping tag name to metrics dict with keys:
            count, success_rate, avg_score, avg_latency_ms.
        """
        tag_results: dict[str, list[CaseResult]] = defaultdict(list)
        for result in self.results:
            # Recover tags from case_id convention or stored data
            for tag in getattr(result, "_tags", []):
                tag_results[tag].append(result)

        # Build from results that carry tag info
        return self._compute_by_tag(tag_results)

    def compute_by_tag(
        self,
        cases: list[TaskCase],
    ) -> dict[str, dict[str, Any]]:
        """Compute per-tag metrics using original case definitions.

        Args:
            cases: The TaskCase list from the suite.

        Returns:
            Dict mapping tag to metrics.
        """
        case_map = {c.id: c for c in cases}
        tag_results: dict[str, list[CaseResult]] = defaultdict(list)

        for result in self.results:
            case_def = case_map.get(result.case_id)
            if not case_def:
                continue
            for tag in case_def.tags:
                tag_results[tag].append(result)

        return self._compute_by_tag(tag_results)

    @staticmethod
    def _compute_by_tag(
        tag_results: dict[str, list[CaseResult]],
    ) -> dict[str, dict[str, Any]]:
        """Build per-tag metrics from grouped results."""
        by_tag: dict[str, dict[str, Any]] = {}
        for tag, results in sorted(tag_results.items()):
            count = len(results)
            passed = sum(1 for r in results if r.success)
            by_tag[tag] = {
                "count": count,
                "success_rate": round(passed / count, 4),
                "avg_score": round(sum(r.score for r in results) / count, 4),
                "avg_latency_ms": round(sum(r.latency_ms for r in results) / count, 2),
            }
        return by_tag


class AgentEvaluator:
    """Runs agent evaluation task suites.

    Args:
        agent_config: Config dict for BaseAgent.from_config().
    """

    def __init__(self, agent_config: dict[str, Any]) -> None:
        self._agent_config = agent_config
        self._judge = LLMJudge(
            criteria=[
                JudgeCriterion(
                    "quality",
                    "Overall quality of the response",
                    scale_min=1,
                    scale_max=5,
                ),
            ],
            model="judge",
        )

    def run_suite(
        self,
        suite: TaskSuite,
        scoring: str = "judge",
    ) -> EvalReport:
        """Run all cases in a suite against the agent.

        Args:
            suite: Task suite to run.
            scoring: Scoring strategy — "judge", "exact", or
                "contains".

        Returns:
            Aggregate report with per-case results.
        """
        logger.info(
            "Running suite '%s' (%d cases, scoring=%s)",
            suite.name,
            len(suite.cases),
            scoring,
        )
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        results: list[CaseResult] = []

        for case in suite.cases:
            result = self._run_case(case, scoring)
            results.append(result)
            logger.info(
                "Case '%s': success=%s score=%.2f latency=%dms",
                case.id,
                result.success,
                result.score,
                result.latency_ms,
            )

        model_name = self._agent_config.get("model", {}).get("name", "unknown")

        report = EvalReport(
            suite_name=suite.name,
            model_name=model_name,
            timestamp=timestamp,
            results=results,
        )

        logger.info(
            "Suite '%s' complete: success_rate=%.1f%% " "avg_score=%.2f avg_latency=%dms",
            suite.name,
            report.success_rate * 100,
            report.avg_score,
            report.avg_latency_ms,
        )
        return report

    def _run_case(
        self,
        case: TaskCase,
        scoring: str,
    ) -> CaseResult:
        """Run a single test case.

        Args:
            case: The test case definition.
            scoring: Scoring strategy.

        Returns:
            CaseResult with all fields populated.
        """
        from pulsar_ai.agent.base import BaseAgent
        from pulsar_ai.agent.tool import ToolRegistry

        start = time.time()
        response = ""
        trace: list[dict[str, Any]] = []
        error: str | None = None
        tokens_used = 0

        try:
            agent = BaseAgent.from_config(
                self._agent_config,
                tools=ToolRegistry(),
            )
            response = agent.run(user_input=case.query)
            trace = agent.trace
            tokens_used = getattr(agent, "_total_tokens", 0)
        except Exception as exc:
            error = str(exc)
            logger.warning("Case '%s' failed: %s", case.id, error)

        latency_ms = int((time.time() - start) * 1000)

        # Extract tool names from trace
        tools_used = [entry["tool"] for entry in trace if entry.get("type") == "tool_call"]
        tools_match = _check_tools_match(tools_used, case.expected_tools)

        # Score response
        score = 0.0
        if error is None:
            score = self._score(scoring, case.query, response, case)

        success = error is None and score >= 0.5 and (tools_match or not case.expected_tools)

        return CaseResult(
            case_id=case.id,
            query=case.query,
            response=response,
            trace=trace,
            success=success,
            score=score,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost=0.0,
            tools_used=tools_used,
            tools_match=tools_match,
            error=error,
        )

    def _score(
        self,
        scoring: str,
        query: str,
        response: str,
        case: TaskCase,
    ) -> float:
        """Dispatch to the appropriate scoring method.

        Args:
            scoring: Strategy name.
            query: Original user query.
            response: Agent response text.
            case: Full case definition.

        Returns:
            Score from 0.0 to 1.0.
        """
        if scoring == "exact":
            return self._score_exact(response, case.expected_answer or "")
        elif scoring == "contains":
            return self._score_contains(response, case.expected_answer or "")
        elif scoring == "judge":
            return self._score_judge(query, response, case.rubric)
        else:
            logger.warning("Unknown scoring '%s', defaulting to 0", scoring)
            return 0.0

    @staticmethod
    def _score_exact(response: str, expected: str) -> float:
        """Exact match scoring (case-insensitive, stripped).

        Args:
            response: Agent response.
            expected: Expected answer.

        Returns:
            1.0 if match, 0.0 otherwise.
        """
        if not expected:
            return 0.0
        return 1.0 if response.strip().lower() == expected.strip().lower() else 0.0

    @staticmethod
    def _score_contains(response: str, expected: str) -> float:
        """Substring match scoring (case-insensitive).

        Args:
            response: Agent response.
            expected: Expected substring.

        Returns:
            1.0 if expected is found in response, 0.0 otherwise.
        """
        if not expected:
            return 0.0
        return 1.0 if expected.strip().lower() in response.strip().lower() else 0.0

    def _score_judge(
        self,
        query: str,
        response: str,
        rubric: str,
    ) -> float:
        """LLM-as-Judge scoring using existing llm_judge module.

        Builds a judge prompt incorporating the rubric and
        delegates to ``LLMJudge.build_prompt``. Since we cannot
        call an LLM directly, we return 0.0 as a placeholder
        that callers should override with actual judge output.

        Args:
            query: Original user query.
            response: Agent response.
            rubric: Evaluation rubric text.

        Returns:
            Normalized score from 0.0 to 1.0.
        """
        instruction = f"{query}\n\nRubric: {rubric}" if rubric else query
        _prompt = self._judge.build_prompt(instruction, response)

        # In a real deployment the prompt would be sent to an LLM
        # and the output parsed via self._judge.parse_scores().
        # Here we return a neutral score; integration tests should
        # mock or inject the judge output.
        logger.debug(
            "Judge scoring requested for case (rubric=%s)",
            rubric[:60] if rubric else "none",
        )
        return 0.0

    def score_with_judge_output(
        self,
        query: str,
        response: str,
        rubric: str,
        judge_output: str,
    ) -> float:
        """Score using actual LLM judge output.

        Args:
            query: Original user query.
            response: Agent response.
            rubric: Evaluation rubric text.
            judge_output: Raw text from the judge LLM.

        Returns:
            Normalized score from 0.0 to 1.0.
        """
        instruction = f"{query}\n\nRubric: {rubric}" if rubric else query
        result = self._judge.evaluate(
            instruction=instruction,
            response=response,
            judge_output=judge_output,
        )
        # Normalize overall_score (1-5 scale) to 0-1
        if result.overall_score <= 0:
            return 0.0
        return round((result.overall_score - 1.0) / 4.0, 4)


def _check_tools_match(
    used: list[str],
    expected: list[str],
) -> bool:
    """Check if expected tools were used (order-insensitive).

    Args:
        used: Tools the agent actually called.
        expected: Tools that should have been called.

    Returns:
        True if all expected tools appear in used list.
    """
    if not expected:
        return True
    return set(expected).issubset(set(used))


def load_suite_from_yaml(path: str) -> TaskSuite:
    """Load a task suite from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed TaskSuite instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level, got {type(data)}")

    cases: list[TaskCase] = []
    for raw in data.get("cases", []):
        cases.append(
            TaskCase(
                id=raw["id"],
                query=raw["query"],
                expected_answer=raw.get("expected_answer"),
                expected_tools=raw.get("expected_tools", []),
                rubric=raw.get("rubric", ""),
                tags=raw.get("tags", []),
                max_time_s=raw.get("max_time_s", 30.0),
            )
        )

    return TaskSuite(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        cases=cases,
        version=data.get("version", "1.0"),
    )


def compare_reports(
    report_a: EvalReport,
    report_b: EvalReport,
) -> dict[str, Any]:
    """Compare two eval reports (e.g. model v1 vs v2).

    Args:
        report_a: First (baseline) report.
        report_b: Second (candidate) report.

    Returns:
        Dict with winner, deltas, and per-tag comparison.
    """
    success_delta = report_b.success_rate - report_a.success_rate
    score_delta = report_b.avg_score - report_a.avg_score
    latency_delta = report_b.avg_latency_ms - report_a.avg_latency_ms
    cost_delta = report_b.total_cost - report_a.total_cost

    # Determine winner by score, then success rate
    if score_delta > 0.01:
        winner = "B"
    elif score_delta < -0.01:
        winner = "A"
    elif success_delta > 0:
        winner = "B"
    elif success_delta < 0:
        winner = "A"
    else:
        winner = "tie"

    return {
        "winner": winner,
        "model_a": report_a.model_name,
        "model_b": report_b.model_name,
        "success_delta": round(success_delta, 4),
        "score_delta": round(score_delta, 4),
        "latency_delta": round(latency_delta, 2),
        "cost_delta": round(cost_delta, 6),
    }
