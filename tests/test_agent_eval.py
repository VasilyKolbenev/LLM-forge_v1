"""Tests for the agent evaluation framework."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pulsar_ai.evaluation.agent_eval import (
    AgentEvaluator,
    CaseResult,
    EvalReport,
    TaskCase,
    compare_reports,
    load_suite_from_yaml,
)
from pulsar_ai.evaluation.agent_eval_store import AgentEvalStore
from pulsar_ai.storage.database import Database

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Database:
    """Isolated DB for store tests."""
    db = Database(tmp_path / "test_eval.db")
    yield db
    db.close()


@pytest.fixture()
def sample_cases() -> list[TaskCase]:
    """A small set of test cases for unit tests."""
    return [
        TaskCase(
            id="math_1",
            query="What is 2 + 2?",
            expected_answer="4",
            expected_tools=["calculator"],
            rubric="Must return 4",
            tags=["math", "single-step"],
        ),
        TaskCase(
            id="search_1",
            query="Capital of France?",
            expected_answer="Paris",
            expected_tools=["search"],
            rubric="Must return Paris",
            tags=["retrieval", "factual"],
        ),
        TaskCase(
            id="multi_1",
            query="Complex multi-step task",
            expected_tools=["search", "calculator"],
            rubric="Must combine tools",
            tags=["multi-step", "math"],
        ),
    ]


@pytest.fixture()
def sample_results() -> list[CaseResult]:
    """Pre-built CaseResult list for report tests."""
    return [
        CaseResult(
            case_id="math_1",
            query="What is 2 + 2?",
            response="4",
            trace=[{"type": "tool_call", "tool": "calculator"}],
            success=True,
            score=1.0,
            latency_ms=150,
            tokens_used=50,
            cost=0.001,
            tools_used=["calculator"],
            tools_match=True,
        ),
        CaseResult(
            case_id="search_1",
            query="Capital of France?",
            response="Paris",
            trace=[{"type": "tool_call", "tool": "search"}],
            success=True,
            score=0.8,
            latency_ms=300,
            tokens_used=80,
            cost=0.002,
            tools_used=["search"],
            tools_match=True,
        ),
        CaseResult(
            case_id="multi_1",
            query="Complex multi-step task",
            response="Some answer",
            trace=[
                {"type": "tool_call", "tool": "search"},
                {"type": "tool_call", "tool": "calculator"},
            ],
            success=False,
            score=0.3,
            latency_ms=500,
            tokens_used=200,
            cost=0.005,
            tools_used=["search", "calculator"],
            tools_match=True,
            error=None,
        ),
    ]


@pytest.fixture()
def sample_report(sample_results: list[CaseResult]) -> EvalReport:
    """Pre-built EvalReport for property tests."""
    return EvalReport(
        suite_name="test-suite",
        model_name="test-model",
        timestamp="2026-03-22T12:00:00",
        results=sample_results,
    )


# ── TaskCase creation ────────────────────────────────────────


class TestTaskCase:
    def test_task_case_creation(self) -> None:
        case = TaskCase(
            id="test_1",
            query="Hello",
            expected_answer="Hi",
            expected_tools=["tool_a"],
            rubric="Be polite",
            tags=["greeting"],
        )
        assert case.id == "test_1"
        assert case.query == "Hello"
        assert case.expected_answer == "Hi"
        assert case.expected_tools == ["tool_a"]
        assert case.rubric == "Be polite"
        assert case.tags == ["greeting"]
        assert case.max_time_s == 30.0

    def test_task_case_defaults(self) -> None:
        case = TaskCase(id="minimal", query="test")
        assert case.expected_answer is None
        assert case.expected_tools == []
        assert case.rubric == ""
        assert case.tags == []
        assert case.max_time_s == 30.0


# ── TaskSuite from YAML ─────────────────────────────────────


class TestTaskSuiteYaml:
    def test_task_suite_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
name: "Test Suite"
description: "A test suite"
version: "2.0"
cases:
  - id: case_a
    query: "What is 1+1?"
    expected_answer: "2"
    expected_tools: [calculator]
    rubric: "Must be correct"
    tags: [math]
    max_time_s: 10.0
  - id: case_b
    query: "Hello world"
    rubric: "Be friendly"
    tags: [greeting]
"""
        yaml_path = tmp_path / "suite.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        suite = load_suite_from_yaml(str(yaml_path))

        assert suite.name == "Test Suite"
        assert suite.description == "A test suite"
        assert suite.version == "2.0"
        assert len(suite.cases) == 2

        assert suite.cases[0].id == "case_a"
        assert suite.cases[0].expected_answer == "2"
        assert suite.cases[0].expected_tools == ["calculator"]
        assert suite.cases[0].max_time_s == 10.0

        assert suite.cases[1].id == "case_b"
        assert suite.cases[1].expected_answer is None
        assert suite.cases[1].expected_tools == []

    def test_load_real_suite(self) -> None:
        suite_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "eval-suites",
            "basic-agent-suite.yaml",
        )
        if not os.path.exists(suite_path):
            pytest.skip("Suite YAML not found")

        suite = load_suite_from_yaml(suite_path)
        assert suite.name == "Basic Agent Evaluation"
        assert len(suite.cases) == 5

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("- just a list", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected dict"):
            load_suite_from_yaml(str(bad_path))


# ── Scoring ──────────────────────────────────────────────────


class TestScoring:
    def test_score_exact_match(self) -> None:
        assert AgentEvaluator._score_exact("345", "345") == 1.0

    def test_score_exact_case_insensitive(self) -> None:
        assert AgentEvaluator._score_exact("Paris", "paris") == 1.0

    def test_score_exact_mismatch(self) -> None:
        assert AgentEvaluator._score_exact("London", "Paris") == 0.0

    def test_score_exact_empty_expected(self) -> None:
        assert AgentEvaluator._score_exact("something", "") == 0.0

    def test_score_contains_match(self) -> None:
        assert AgentEvaluator._score_contains("The answer is 345 tokens", "345") == 1.0

    def test_score_contains_case_insensitive(self) -> None:
        assert AgentEvaluator._score_contains("The capital is PARIS.", "paris") == 1.0

    def test_score_contains_mismatch(self) -> None:
        assert AgentEvaluator._score_contains("The capital is London.", "Paris") == 0.0

    def test_score_contains_empty_expected(self) -> None:
        assert AgentEvaluator._score_contains("text", "") == 0.0


# ── CaseResult ───────────────────────────────────────────────


class TestCaseResult:
    def test_case_result_success(self) -> None:
        result = CaseResult(
            case_id="t1",
            query="test",
            response="answer",
            trace=[],
            success=True,
            score=0.9,
            latency_ms=100,
            tokens_used=50,
            cost=0.001,
            tools_used=["calc"],
            tools_match=True,
        )
        assert result.success is True
        assert result.error is None

    def test_case_result_failure(self) -> None:
        result = CaseResult(
            case_id="t2",
            query="test",
            response="",
            trace=[],
            success=False,
            score=0.0,
            latency_ms=50,
            tokens_used=10,
            cost=0.0,
            tools_used=[],
            tools_match=False,
            error="Timeout",
        )
        assert result.success is False
        assert result.error == "Timeout"

    def test_case_result_to_dict(self) -> None:
        result = CaseResult(
            case_id="t3",
            query="q",
            response="r",
            trace=[{"type": "answer"}],
            success=True,
            score=0.75,
            latency_ms=200,
            tokens_used=100,
            cost=0.002,
            tools_used=["search"],
            tools_match=True,
        )
        d = result.to_dict()
        assert d["case_id"] == "t3"
        assert d["score"] == 0.75
        assert d["tools_used"] == ["search"]
        assert d["error"] is None


# ── EvalReport computed properties ───────────────────────────


class TestEvalReport:
    def test_success_rate(self, sample_report: EvalReport) -> None:
        # 2 out of 3 succeeded
        assert sample_report.success_rate == pytest.approx(2 / 3, abs=0.001)

    def test_avg_score(self, sample_report: EvalReport) -> None:
        expected = (1.0 + 0.8 + 0.3) / 3
        assert sample_report.avg_score == pytest.approx(expected, abs=0.001)

    def test_avg_latency(self, sample_report: EvalReport) -> None:
        expected = (150 + 300 + 500) / 3
        assert sample_report.avg_latency_ms == pytest.approx(expected, abs=0.1)

    def test_total_tokens(self, sample_report: EvalReport) -> None:
        assert sample_report.total_tokens == 330

    def test_total_cost(self, sample_report: EvalReport) -> None:
        assert sample_report.total_cost == pytest.approx(0.008, abs=0.0001)

    def test_tools_accuracy(self, sample_report: EvalReport) -> None:
        # All 3 have tools_match=True
        assert sample_report.tools_accuracy == 1.0

    def test_by_tag_with_cases(
        self,
        sample_report: EvalReport,
        sample_cases: list[TaskCase],
    ) -> None:
        by_tag = sample_report.compute_by_tag(sample_cases)
        assert "math" in by_tag
        assert by_tag["math"]["count"] == 2
        assert "retrieval" in by_tag
        assert by_tag["retrieval"]["count"] == 1
        assert "single-step" in by_tag

    def test_empty_report(self) -> None:
        report = EvalReport(
            suite_name="empty",
            model_name="none",
            timestamp="2026-01-01T00:00:00",
            results=[],
        )
        assert report.success_rate == 0.0
        assert report.avg_score == 0.0
        assert report.avg_latency_ms == 0.0
        assert report.total_tokens == 0
        assert report.total_cost == 0.0
        assert report.tools_accuracy == 0.0


# ── compare_reports ──────────────────────────────────────────


class TestCompareReports:
    def test_compare_b_wins(self, sample_results: list[CaseResult]) -> None:
        report_a = EvalReport(
            suite_name="s",
            model_name="model-v1",
            timestamp="2026-01-01T00:00:00",
            results=sample_results,
        )
        # Make B better: all scores 1.0
        better_results = [
            CaseResult(
                case_id=r.case_id,
                query=r.query,
                response=r.response,
                trace=r.trace,
                success=True,
                score=1.0,
                latency_ms=r.latency_ms,
                tokens_used=r.tokens_used,
                cost=r.cost,
                tools_used=r.tools_used,
                tools_match=r.tools_match,
            )
            for r in sample_results
        ]
        report_b = EvalReport(
            suite_name="s",
            model_name="model-v2",
            timestamp="2026-01-02T00:00:00",
            results=better_results,
        )

        comparison = compare_reports(report_a, report_b)
        assert comparison["winner"] == "B"
        assert comparison["score_delta"] > 0
        assert comparison["model_a"] == "model-v1"
        assert comparison["model_b"] == "model-v2"

    def test_compare_a_wins(self) -> None:
        result_a = [
            CaseResult(
                case_id="x",
                query="q",
                response="r",
                trace=[],
                success=True,
                score=0.9,
                latency_ms=100,
                tokens_used=50,
                cost=0.001,
                tools_used=[],
                tools_match=True,
            )
        ]
        result_b = [
            CaseResult(
                case_id="x",
                query="q",
                response="r",
                trace=[],
                success=False,
                score=0.2,
                latency_ms=100,
                tokens_used=50,
                cost=0.001,
                tools_used=[],
                tools_match=True,
            )
        ]
        report_a = EvalReport(
            suite_name="s",
            model_name="a",
            timestamp="t",
            results=result_a,
        )
        report_b = EvalReport(
            suite_name="s",
            model_name="b",
            timestamp="t",
            results=result_b,
        )
        comparison = compare_reports(report_a, report_b)
        assert comparison["winner"] == "A"

    def test_compare_tie(self) -> None:
        result = [
            CaseResult(
                case_id="x",
                query="q",
                response="r",
                trace=[],
                success=True,
                score=0.5,
                latency_ms=100,
                tokens_used=50,
                cost=0.001,
                tools_used=[],
                tools_match=True,
            )
        ]
        report_a = EvalReport(
            suite_name="s",
            model_name="same",
            timestamp="t",
            results=list(result),
        )
        report_b = EvalReport(
            suite_name="s",
            model_name="same",
            timestamp="t",
            results=list(result),
        )
        comparison = compare_reports(report_a, report_b)
        assert comparison["winner"] == "tie"


# ── AgentEvalStore ───────────────────────────────────────────


class TestAgentEvalStore:
    def test_save_and_get(
        self,
        tmp_db: Database,
        sample_report: EvalReport,
    ) -> None:
        store = AgentEvalStore(db=tmp_db)
        report_id = store.save_report(sample_report)
        assert len(report_id) == 12

        loaded = store.get_report(report_id)
        assert loaded is not None
        assert loaded["suite_name"] == "test-suite"
        assert loaded["model_name"] == "test-model"
        assert loaded["success_rate"] == pytest.approx(sample_report.success_rate, abs=0.001)
        assert isinstance(loaded["results_json"], list)
        assert len(loaded["results_json"]) == 3

    def test_get_nonexistent(self, tmp_db: Database) -> None:
        store = AgentEvalStore(db=tmp_db)
        assert store.get_report("nonexistent") is None

    def test_list_reports(
        self,
        tmp_db: Database,
        sample_report: EvalReport,
    ) -> None:
        store = AgentEvalStore(db=tmp_db)
        store.save_report(sample_report)

        # Second report with different model
        report_b = EvalReport(
            suite_name="other-suite",
            model_name="other-model",
            timestamp="2026-03-22T13:00:00",
            results=[],
        )
        store.save_report(report_b)

        all_reports = store.list_reports()
        assert len(all_reports) == 2

        filtered = store.list_reports(model_name="test-model")
        assert len(filtered) == 1
        assert filtered[0]["model_name"] == "test-model"

    def test_list_reports_limit(
        self,
        tmp_db: Database,
        sample_report: EvalReport,
    ) -> None:
        store = AgentEvalStore(db=tmp_db)
        for _ in range(5):
            store.save_report(sample_report)

        limited = store.list_reports(limit=3)
        assert len(limited) == 3

    def test_get_comparison(
        self,
        tmp_db: Database,
        sample_results: list[CaseResult],
    ) -> None:
        store = AgentEvalStore(db=tmp_db)

        report_a = EvalReport(
            suite_name="suite",
            model_name="v1",
            timestamp="2026-01-01T00:00:00",
            results=sample_results,
        )
        id_a = store.save_report(report_a)

        better = [
            CaseResult(
                case_id=r.case_id,
                query=r.query,
                response=r.response,
                trace=r.trace,
                success=True,
                score=1.0,
                latency_ms=100,
                tokens_used=r.tokens_used,
                cost=r.cost,
                tools_used=r.tools_used,
                tools_match=True,
            )
            for r in sample_results
        ]
        report_b = EvalReport(
            suite_name="suite",
            model_name="v2",
            timestamp="2026-01-02T00:00:00",
            results=better,
        )
        id_b = store.save_report(report_b)

        comparison = store.get_comparison(id_a, id_b)
        assert comparison["winner"] == "B"
        assert comparison["model_a"] == "v1"
        assert comparison["model_b"] == "v2"

    def test_get_comparison_not_found(self, tmp_db: Database) -> None:
        store = AgentEvalStore(db=tmp_db)
        with pytest.raises(ValueError, match="Report not found"):
            store.get_comparison("aaa", "bbb")
