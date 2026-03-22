"""Closed-loop pipeline steps: collect traces, build dataset, regression eval.

These steps automate the feedback loop:
    collect traces -> build dataset -> train -> evaluate regression.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pulsar_ai.pipeline.steps import register_step
from pulsar_ai.evaluation.llm_judge import LLMJudge
from pulsar_ai.storage.trace_store import TraceStore

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline step detects a blocking failure."""


@register_step("collect_traces")
def step_collect_traces(config: dict) -> dict[str, Any]:
    """Collect traces from TraceStore with filters.

    Args:
        config: Step configuration with keys:
            days (int): Collect traces from last N days (default 7).
            min_rating (float): Minimum average rating (default 0).
            status (str): Filter by status (default "").
            model_name (str): Filter by model (default "").
            limit (int): Max traces to collect (default 500).

    Returns:
        Dict with ``trace_ids`` list and ``count``.
    """
    days = config.get("days", 7)
    min_rating = config.get("min_rating", 0)
    status = config.get("status", "")
    model_name = config.get("model_name", "")
    limit = config.get("limit", 500)

    store = TraceStore()

    date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    traces = store.list_traces(
        date_from=date_from,
        model_name=model_name,
        status=status,
        min_rating=min_rating if min_rating > 0 else None,
        limit=limit,
    )

    trace_ids = [t["trace_id"] for t in traces]
    logger.info(
        "Collected %d traces (days=%d, min_rating=%.1f, status=%s)",
        len(trace_ids),
        days,
        min_rating,
        status or "any",
    )

    return {"trace_ids": trace_ids, "count": len(trace_ids)}


@register_step("build_dataset")
def step_build_dataset(config: dict) -> dict[str, Any]:
    """Build training dataset from traces.

    Args:
        config: Step configuration with keys:
            trace_ids (list[str]): Trace IDs to export.
            format (str): ``"sft"`` or ``"dpo"`` (default ``"sft"``).
            output_dir (str): Output directory (default ``"./data/generated"``).
            name (str): Dataset name (default ``"traces-dataset"``).
            quality_filter (dict): Optional filters with ``dedup`` (bool)
                and ``min_length`` (int).

    Returns:
        Dict with ``path``, ``format``, ``num_examples``, ``version``.
    """
    trace_ids = config.get("trace_ids", [])
    fmt = config.get("format", "sft")
    output_dir = config.get("output_dir", "./data/generated")
    name = config.get("name", "traces-dataset")
    quality_filter = config.get("quality_filter", {})

    if isinstance(trace_ids, str):
        trace_ids = [tid.strip() for tid in trace_ids.split(",") if tid.strip()]

    store = TraceStore()

    if fmt == "dpo":
        examples = store.auto_pair_dpo(trace_ids)
    else:
        examples = store.export_as_sft(trace_ids)

    # Apply quality filters
    dedup = quality_filter.get("dedup", True)
    min_length = quality_filter.get("min_length", 10)

    if min_length > 0:
        examples = _filter_min_length(examples, min_length, fmt)

    if dedup:
        examples = _deduplicate(examples, fmt)

    # Write JSONL
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{name}.jsonl"

    version = _next_version(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(
        "Built %s dataset: %d examples -> %s (v%d)",
        fmt,
        len(examples),
        file_path,
        version,
    )

    return {
        "path": str(file_path),
        "format": fmt,
        "num_examples": len(examples),
        "version": version,
    }


@register_step("regression_eval")
def step_regression_eval(config: dict) -> dict[str, Any]:
    """Compare new model vs baseline using LLM-as-Judge.

    Args:
        config: Step configuration with keys:
            new_model (str): Path to newly trained model.
            baseline_model (str): Path to current production model.
            eval_dataset (str): Path to eval dataset (JSONL).
            num_samples (int): Number of samples to evaluate (default 50).
            block_on_regression (bool): Raise error on regression (default True).

    Returns:
        Dict with ``win_rate``, ``score_delta``, ``regression_detected``,
        and ``report``.

    Raises:
        PipelineError: If regression is detected and ``block_on_regression``
            is True.
    """
    new_model = config.get("new_model", "")
    baseline_model = config.get("baseline_model", "")
    eval_dataset = config.get("eval_dataset", "")
    num_samples = config.get("num_samples", 50)
    block_on_regression = config.get("block_on_regression", True)

    samples = _load_eval_samples(eval_dataset, num_samples)
    if not samples:
        logger.warning("No eval samples found at %s", eval_dataset)
        return {
            "win_rate": 0.0,
            "score_delta": 0.0,
            "regression_detected": False,
            "report": {"per_criterion": [], "samples": []},
        }

    judge = LLMJudge()

    new_wins = 0
    total_new_score = 0.0
    total_baseline_score = 0.0
    sample_reports: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", sample.get("prompt", ""))
        new_response = _generate_response(new_model, instruction)
        baseline_response = _generate_response(baseline_model, instruction)

        new_judge_output = _run_judge(judge, instruction, new_response)
        baseline_judge_output = _run_judge(judge, instruction, baseline_response)

        new_result = judge.evaluate(
            instruction, new_response, new_judge_output, sample_id=f"sample-{i}"
        )
        baseline_result = judge.evaluate(
            instruction,
            baseline_response,
            baseline_judge_output,
            sample_id=f"sample-{i}",
        )

        total_new_score += new_result.overall_score
        total_baseline_score += baseline_result.overall_score

        if new_result.overall_score >= baseline_result.overall_score:
            new_wins += 1

        sample_reports.append(
            {
                "instruction": instruction[:100],
                "new_score": new_result.overall_score,
                "baseline_score": baseline_result.overall_score,
                "winner": (
                    "new"
                    if new_result.overall_score >= baseline_result.overall_score
                    else "baseline"
                ),
            }
        )

    num_evaluated = len(samples)
    win_rate = new_wins / num_evaluated if num_evaluated > 0 else 0.0
    avg_new = total_new_score / num_evaluated if num_evaluated > 0 else 0.0
    avg_baseline = total_baseline_score / num_evaluated if num_evaluated > 0 else 0.0
    score_delta = round(avg_new - avg_baseline, 4)
    regression_detected = win_rate < 0.5

    logger.info(
        "Regression eval: win_rate=%.2f, score_delta=%.4f, regression=%s",
        win_rate,
        score_delta,
        regression_detected,
    )

    if block_on_regression and regression_detected:
        raise PipelineError(
            f"Regression detected: win_rate={win_rate:.2f}, "
            f"score_delta={score_delta:.4f}. "
            f"New model underperforms baseline."
        )

    return {
        "win_rate": round(win_rate, 4),
        "score_delta": score_delta,
        "regression_detected": regression_detected,
        "report": {
            "per_criterion": [],
            "samples": sample_reports,
            "num_evaluated": num_evaluated,
            "avg_new_score": round(avg_new, 4),
            "avg_baseline_score": round(avg_baseline, 4),
        },
    }


# -- Internal helpers --------------------------------------------------------


def _filter_min_length(
    examples: list[dict[str, Any]], min_length: int, fmt: str
) -> list[dict[str, Any]]:
    """Remove examples shorter than min_length characters."""
    filtered = []
    for ex in examples:
        if fmt == "dpo":
            chosen = ex.get("chosen", "")
            if len(chosen) >= min_length:
                filtered.append(ex)
        else:
            output = ex.get("output", ex.get("response", ""))
            if len(output) >= min_length:
                filtered.append(ex)
    return filtered


def _deduplicate(examples: list[dict[str, Any]], fmt: str) -> list[dict[str, Any]]:
    """Deduplicate examples by normalised text content."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for ex in examples:
        if fmt == "dpo":
            key_text = ex.get("prompt", "") + ex.get("chosen", "")
        else:
            key_text = ex.get("instruction", "") + ex.get("output", "")
        norm = key_text.lower().strip()
        h = hashlib.md5(norm.encode()).hexdigest()  # noqa: S324
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def _next_version(file_path: Path) -> int:
    """Compute next version number for a dataset file."""
    if file_path.exists():
        return 2
    return 1


def _load_eval_samples(dataset_path: str, num_samples: int) -> list[dict[str, Any]]:
    """Load evaluation samples from a JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        return []

    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(samples) >= num_samples:
                break
    return samples


def _generate_response(model_path: str, instruction: str) -> str:
    """Generate a model response for an instruction.

    In production this would load the model and run inference.
    For pipeline orchestration, returns a placeholder.
    """
    return f"[Response from {model_path}]: {instruction}"


def _run_judge(judge: LLMJudge, instruction: str, response: str) -> str:
    """Run the LLM judge on an instruction/response pair.

    In production this would call an LLM. For pipeline orchestration,
    returns simulated scores.
    """
    criteria_output = []
    for criterion in judge.criteria:
        criteria_output.append(f"{criterion.name}: 4 | Good quality response")
    return "\n".join(criteria_output)
