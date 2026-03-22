"""WebSocket-based pipeline execution with live progress.

Allows one-click run from the workflow builder with
real-time step status updates.
"""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pulsar_ai.pipeline.executor import PipelineExecutor
from pulsar_ai.pipeline.tracker import PipelineTracker
from pulsar_ai.ui.workflow_policy import (
    format_governance_error,
    validate_pipeline_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pipeline-run"])

# In-memory store of recent pipeline runs
_recent_runs: list[dict[str, Any]] = []
_MAX_RUNS = 50


@router.websocket("/api/v1/pipeline/run")
async def pipeline_run_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for running a pipeline with live progress.

    Client sends: {"pipeline_config": {...}}
    Server sends status updates per step:
        {"type": "step_update", "step": name, "status": "running|completed|failed|skipped", ...}
        {"type": "pipeline_complete", "outputs": {...}}
        {"type": "pipeline_error", "error": "..."}
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        pipeline_config = data.get("pipeline_config", {})

        if not pipeline_config.get("steps"):
            await websocket.send_json({"type": "error", "error": "No steps in pipeline config"})
            return

        violations = validate_pipeline_config(pipeline_config)
        if violations:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": format_governance_error(violations),
                }
            )
            return

        tracker = PipelineTracker(
            pipeline_config.get("pipeline", {}).get("name", "ws-run"),
        )
        executor = PipelineExecutor(pipeline_config, tracker=tracker)

        execution_order = executor._resolve_order()

        await websocket.send_json(
            {
                "type": "pipeline_start",
                "name": executor.name,
                "steps": execution_order,
                "total": len(execution_order),
            }
        )

        # Execute steps one by one with progress updates
        outputs: dict[str, Any] = {}
        for i, step_name in enumerate(execution_order):
            step = executor._get_step(step_name)

            # Check condition
            from pulsar_ai.pipeline.steps import check_condition

            condition = step.get("condition")
            if condition and not check_condition(condition, outputs):
                outputs[step_name] = {"_skipped": True}
                await websocket.send_json(
                    {
                        "type": "step_update",
                        "step": step_name,
                        "index": i,
                        "status": "skipped",
                        "reason": "condition not met",
                    }
                )
                continue

            await websocket.send_json(
                {
                    "type": "step_update",
                    "step": step_name,
                    "index": i,
                    "status": "running",
                    "step_type": step.get("type", "unknown"),
                }
            )

            start = time.time()
            try:
                resolved_config = executor._resolve_vars(step.get("config", {}))

                # Run step in thread pool to not block WebSocket
                from pulsar_ai.pipeline.steps import dispatch_step

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    dispatch_step,
                    step["type"],
                    resolved_config,
                )

                elapsed = time.time() - start
                outputs[step_name] = result
                executor._outputs[step_name] = result

                await websocket.send_json(
                    {
                        "type": "step_update",
                        "step": step_name,
                        "index": i,
                        "status": "completed",
                        "duration_s": round(elapsed, 1),
                        "result_keys": list(result.keys()) if isinstance(result, dict) else [],
                    }
                )

            except Exception as e:
                elapsed = time.time() - start
                await websocket.send_json(
                    {
                        "type": "step_update",
                        "step": step_name,
                        "index": i,
                        "status": "failed",
                        "error": str(e),
                        "duration_s": round(elapsed, 1),
                    }
                )
                await websocket.send_json(
                    {
                        "type": "pipeline_error",
                        "step": step_name,
                        "error": str(e),
                    }
                )

                _store_run(executor.name, execution_order, outputs, error=str(e))
                return

        await websocket.send_json(
            {
                "type": "pipeline_complete",
                "name": executor.name,
                "steps_completed": len(outputs),
                "output_keys": {
                    k: list(v.keys()) if isinstance(v, dict) else [] for k, v in outputs.items()
                },
            }
        )

        _store_run(executor.name, execution_order, outputs)

    except WebSocketDisconnect:
        logger.info("Pipeline WebSocket client disconnected")
    except Exception as e:
        logger.exception("Pipeline run error")
        try:
            await websocket.send_json({"type": "pipeline_error", "error": str(e)})
        except Exception:
            pass


@router.post("/api/v1/pipeline/run/sync")
async def pipeline_run_sync(body: dict) -> dict:
    """Synchronous pipeline execution endpoint (non-WebSocket).

    Args:
        body: Request body with pipeline_config.

    Returns:
        Pipeline execution results.
    """
    pipeline_config = body.get("pipeline_config", {})
    if not pipeline_config.get("steps"):
        return {"error": "No steps in pipeline config"}

    violations = validate_pipeline_config(pipeline_config)
    if violations:
        return {
            "status": "blocked",
            "error": format_governance_error(violations),
        }

    executor = PipelineExecutor(pipeline_config)
    try:
        outputs = executor.run()
        result = {
            "status": "completed",
            "name": executor.name,
            "outputs": {
                k: (v if isinstance(v, dict) else {"result": str(v)}) for k, v in outputs.items()
            },
        }
        _store_run(executor.name, list(outputs.keys()), outputs)
        return result
    except RuntimeError as e:
        return {"status": "failed", "error": str(e)}


@router.get("/api/v1/pipeline/trace/{workflow_id}")
async def get_pipeline_trace(workflow_id: str) -> dict[str, Any]:
    """Get execution trace for replay.

    For MVP this returns mock trace data generated from recent runs.
    Later this will read from actual execution logs.

    Args:
        workflow_id: The workflow identifier.

    Returns:
        Dictionary with trace events for replay visualization.
    """
    # Find the most recent run for this workflow (by name match or any run)
    matching_run = None
    for run in reversed(_recent_runs):
        if run.get("name", "") == workflow_id or matching_run is None:
            matching_run = run
            if run.get("name", "") == workflow_id:
                break

    if not matching_run:
        return {"workflow_id": workflow_id, "events": []}

    # Generate mock trace events from the stored run data
    events: list[dict[str, Any]] = []
    steps = matching_run.get("steps", [])
    cumulative_ms = 0

    working_messages = [
        ["Analyzing input data...", "Processing records...", "Validating schema..."],
        ["Running inference...", "Computing embeddings...", "Generating response..."],
        ["Checking compliance rules...", "Applying policies...", "Verifying constraints..."],
        ["Routing request...", "Evaluating conditions...", "Selecting pathway..."],
        ["Aggregating results...", "Building report...", "Finalizing output..."],
    ]

    for i, step_name in enumerate(steps):
        duration_ms = 2000 + (hash(step_name) % 4000)
        msgs = working_messages[i % len(working_messages)]

        events.append({
            "timestamp": cumulative_ms,
            "nodeId": step_name,
            "type": "start",
            "message": f"{step_name}: Starting...",
        })

        for p in range(1, 3):
            progress_time = cumulative_ms + (duration_ms * p) // 3
            events.append({
                "timestamp": progress_time,
                "nodeId": step_name,
                "type": "progress",
                "progress": p / 3,
                "message": msgs[(p - 1) % len(msgs)],
            })

        final_type = "error" if (
            matching_run.get("status") == "failed"
            and i == len(steps) - 1
        ) else "complete"

        events.append({
            "timestamp": cumulative_ms + duration_ms,
            "nodeId": step_name,
            "type": final_type,
            "message": f"{step_name}: {'Error!' if final_type == 'error' else 'Done!'}",
            "duration": duration_ms / 1000,
        })

        cumulative_ms += duration_ms + 300

    events.sort(key=lambda e: e["timestamp"])

    return {
        "workflow_id": workflow_id,
        "events": events,
        "total_time_ms": cumulative_ms,
        "source": "mock",
    }


@router.get("/api/v1/pipeline/runs")
def list_pipeline_runs() -> list[dict]:
    """List recent pipeline runs."""
    return list(reversed(_recent_runs))


def _store_run(
    name: str,
    steps: list[str],
    outputs: dict[str, Any],
    error: str = "",
) -> None:
    """Store a run result in the recent runs list."""
    _recent_runs.append(
        {
            "name": name,
            "steps": steps,
            "steps_completed": sum(
                1 for v in outputs.values() if isinstance(v, dict) and not v.get("_skipped")
            ),
            "status": "failed" if error else "completed",
            "error": error,
            "timestamp": time.time(),
        }
    )
    while len(_recent_runs) > _MAX_RUNS:
        _recent_runs.pop(0)
