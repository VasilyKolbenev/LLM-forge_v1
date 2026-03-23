"""API routes for agent evaluation operations.

Provides endpoints for running eval suites, listing reports,
inspecting individual reports, and comparing two reports.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pulsar_ai.evaluation.agent_eval import (
    AgentEvaluator,
    load_suite_from_yaml,
)
from pulsar_ai.evaluation.agent_eval_store import AgentEvalStore
from pulsar_ai.ui.auth import get_current_user, get_scoped_user_id, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent-eval"])

_store = AgentEvalStore()

SUITES_DIR = Path("configs/eval-suites")


class RunEvalRequest(BaseModel):
    """Body for running an agent evaluation suite."""

    suite_path: str
    agent_config: dict[str, Any]
    scoring: str = "exact"


@router.get("/agent-eval/reports")
async def list_reports(
    request: Request,
    model_name: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List eval reports.

    Args:
        model_name: Optional filter by model name.
        limit: Maximum number of reports to return.

    Returns:
        Dict with reports list.
    """
    user_id = get_scoped_user_id(request)
    reports = _store.list_reports(model_name=model_name, limit=limit, user_id=user_id)
    return {"reports": reports, "total": len(reports)}


@router.get("/agent-eval/reports/{report_id}")
async def get_report(report_id: str) -> dict[str, Any]:
    """Get full report with case results.

    Args:
        report_id: The report identifier.

    Returns:
        Full report dict with parsed results.

    Raises:
        HTTPException: 404 if report not found.
    """
    report = _store.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.post("/agent-eval/run")
async def run_eval(body: RunEvalRequest, request: Request) -> dict[str, Any]:
    """Run an eval suite.

    Loads suite from YAML, creates evaluator, runs suite,
    saves report and returns summary.

    Args:
        body: Run parameters including suite path, agent config, scoring.

    Returns:
        Dict with report_id and summary metrics.

    Raises:
        HTTPException: 400 if suite file not found or invalid.
    """
    suite_path = Path(body.suite_path)
    if not suite_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Suite file not found: {body.suite_path}",
        )

    try:
        suite = load_suite_from_yaml(str(suite_path))
    except (ValueError, yaml.YAMLError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid suite YAML: {exc}",
        )

    if body.scoring not in ("judge", "exact", "contains"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scoring mode: {body.scoring}. Use 'judge', 'exact', or 'contains'.",
        )

    user_id = get_user_id(request)
    evaluator = AgentEvaluator(body.agent_config)
    report = evaluator.run_suite(suite, scoring=body.scoring)
    report_id = _store.save_report(report, user_id=user_id)

    logger.info(
        "Eval run complete: report=%s suite=%s success_rate=%.1f%%",
        report_id,
        suite.name,
        report.success_rate * 100,
    )

    return {
        "report_id": report_id,
        "suite_name": report.suite_name,
        "model_name": report.model_name,
        "success_rate": report.success_rate,
        "avg_score": report.avg_score,
        "avg_latency_ms": report.avg_latency_ms,
        "total_tokens": report.total_tokens,
        "total_cost": report.total_cost,
        "num_cases": len(report.results),
    }


@router.get("/agent-eval/compare/{report_a}/{report_b}")
async def compare_reports_endpoint(
    report_a: str,
    report_b: str,
    request: Request,
) -> dict[str, Any]:
    """Compare two reports.

    Args:
        report_a: ID of the baseline report.
        report_b: ID of the candidate report.

    Returns:
        Comparison dict with deltas and winner.

    Raises:
        HTTPException: 404 if either report not found.
    """
    user_id = get_scoped_user_id(request)
    try:
        comparison = _store.get_comparison(report_a, report_b, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return comparison


@router.get("/agent-eval/suites")
async def list_suites() -> list[dict[str, Any]]:
    """List available eval suite YAML files from configs/eval-suites/.

    Returns:
        List of suite info dicts with name, path, description, and case count.
    """
    suites: list[dict[str, Any]] = []
    if not SUITES_DIR.exists():
        return suites

    for yaml_file in sorted(SUITES_DIR.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                suites.append({
                    "path": str(yaml_file),
                    "name": data.get("name", yaml_file.stem),
                    "description": data.get("description", ""),
                    "num_cases": len(data.get("cases", [])),
                    "version": data.get("version", "1.0"),
                })
        except Exception as exc:
            logger.warning("Failed to read suite %s: %s", yaml_file, exc)

    return suites
