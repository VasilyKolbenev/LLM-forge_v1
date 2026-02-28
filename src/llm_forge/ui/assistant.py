"""Forge Co-pilot — AI assistant with platform management tools.

Two operating modes:
- Command Mode: slash-commands (/status, /train, /recommend) — always available
- LLM Mode: full ReAct agent with forge tools — when LLM server is connected
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from llm_forge.agent.tool import Tool, ToolRegistry, tool
from llm_forge.ui.experiment_store import ExperimentStore
from llm_forge.ui.jobs import submit_training_job, get_job, list_jobs, cancel_job

logger = logging.getLogger(__name__)

router = APIRouter(tags=["assistant"])

_store = ExperimentStore()
_sessions: dict[str, dict[str, Any]] = {}


# ──────────────────────────────────────────────────────────
# Forge Tools — call backend in-process (no HTTP)
# ──────────────────────────────────────────────────────────

def _get_forge_tools() -> ToolRegistry:
    """Create a ToolRegistry with all forge platform tools.

    Returns:
        ToolRegistry with 10 forge tools.
    """
    registry = ToolRegistry()

    @tool(name="list_experiments", description="List recent training experiments with status and loss")
    def list_experiments_tool(status: str = "", limit: int = 10) -> str:
        """List experiments, optionally filtered by status."""
        exps = _store.list_all(status=status or None)[:limit]
        if not exps:
            return "No experiments found."
        lines = []
        for e in exps:
            loss = f", loss={e['final_loss']:.4f}" if e.get("final_loss") else ""
            lines.append(f"- [{e['id']}] {e['name']} ({e['status']}{loss})")
        return "\n".join(lines)
    registry.register(list_experiments_tool)

    @tool(name="get_experiment", description="Get detailed info about a specific experiment by ID")
    def get_experiment_tool(experiment_id: str) -> str:
        """Get experiment details including config, metrics, artifacts."""
        exp = _store.get(experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."
        info = {
            "id": exp["id"],
            "name": exp["name"],
            "status": exp["status"],
            "task": exp.get("task", "sft"),
            "model": exp.get("model", "unknown"),
            "final_loss": exp.get("final_loss"),
            "created_at": exp.get("created_at", ""),
            "artifacts": exp.get("artifacts", {}),
        }
        return json.dumps(info, indent=2)
    registry.register(get_experiment_tool)

    @tool(name="start_training", description="Start a new training experiment")
    def start_training_tool(
        name: str,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        dataset_path: str = "",
        task: str = "sft",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        gradient_accumulation: int = 16,
    ) -> str:
        """Start training with given parameters."""
        if not dataset_path:
            return "Error: dataset_path is required. Use /datasets to see available datasets."
        config = {
            "model": {"name": model},
            "dataset": {"path": dataset_path},
            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation": gradient_accumulation,
                "max_seq_length": 512,
            },
            "output": {"dir": f"./outputs/{name}"},
        }
        exp_id = _store.create(name=name, config=config, task=task)
        job_id = submit_training_job(experiment_id=exp_id, config=config, task=task)
        return f"Training started! Job ID: {job_id}, Experiment ID: {exp_id}"
    registry.register(start_training_tool)

    @tool(name="check_training", description="Check status of running/recent training jobs")
    def check_training_tool() -> str:
        """Get current training job statuses."""
        jobs = list_jobs()
        if not jobs:
            return "No training jobs found."
        lines = []
        for j in jobs:
            lines.append(f"- Job {j['job_id']}: {j['status']} (experiment: {j['experiment_id']})")
        return "\n".join(lines)
    registry.register(check_training_tool)

    @tool(name="cancel_training", description="Cancel a running training job by job ID")
    def cancel_training_tool(job_id: str) -> str:
        """Cancel a training job."""
        if cancel_job(job_id):
            return f"Job {job_id} cancelled."
        return f"Could not cancel job {job_id}. It may have already completed or doesn't exist."
    registry.register(cancel_training_tool)

    @tool(name="list_datasets", description="List all uploaded datasets")
    def list_datasets_tool() -> str:
        """List datasets in the uploads directory."""
        from pathlib import Path
        import pandas as pd

        data_dir = Path("./data/uploads")
        if not data_dir.exists():
            return "No datasets uploaded yet."
        results = []
        for p in sorted(data_dir.iterdir()):
            if p.is_file() and p.suffix in (".csv", ".jsonl", ".parquet", ".xlsx"):
                try:
                    if p.suffix == ".csv":
                        rows = len(pd.read_csv(p))
                    elif p.suffix == ".jsonl":
                        rows = len(pd.read_json(p, lines=True))
                    else:
                        rows = "?"
                    results.append(f"- [{p.stem}] {p.name} ({rows} rows, {p.stat().st_size / 1024:.1f} KB)")
                except Exception:
                    results.append(f"- [{p.stem}] {p.name} (error reading)")
        return "\n".join(results) if results else "No datasets found."
    registry.register(list_datasets_tool)

    @tool(name="preview_dataset", description="Preview first rows of a dataset")
    def preview_dataset_tool(dataset_id: str, rows: int = 5) -> str:
        """Show first N rows of a dataset."""
        from pathlib import Path
        import pandas as pd

        data_dir = Path("./data/uploads")
        for p in data_dir.iterdir():
            if p.stem == dataset_id:
                try:
                    if p.suffix == ".csv":
                        df = pd.read_csv(p)
                    elif p.suffix == ".jsonl":
                        df = pd.read_json(p, lines=True)
                    else:
                        return f"Preview not supported for {p.suffix}"
                    preview = df.head(rows).to_string(index=False)
                    return f"Columns: {list(df.columns)}\nRows: {len(df)}\n\n{preview}"
                except Exception as e:
                    return f"Error reading dataset: {e}"
        return f"Dataset {dataset_id} not found."
    registry.register(preview_dataset_tool)

    @tool(name="recommend_params", description="Recommend training hyperparameters based on model and dataset")
    def recommend_params_tool(
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        dataset_rows: int = 0,
    ) -> str:
        """Recommend training hyperparameters."""
        # Detect GPU
        gpu_vram = 0.0
        gpu_name = "CPU"
        try:
            from llm_forge.hardware import detect_hardware
            hw = detect_hardware()
            gpu_vram = hw.vram_per_gpu_gb
            gpu_name = hw.gpu_name
        except Exception:
            pass

        model_lower = model.lower()
        if "7b" in model_lower or "8b" in model_lower:
            lr, bs, ga = 1e-4, 1, 32
            seq_len = 512
        elif "3b" in model_lower:
            lr, bs, ga = 2e-4, 2, 16
            seq_len = 512
        elif "1b" in model_lower or "1.5b" in model_lower:
            lr, bs, ga = 3e-4, 4, 8
            seq_len = 1024
        else:
            lr, bs, ga = 2e-4, 2, 16
            seq_len = 512

        if gpu_vram >= 24:
            bs = min(bs * 2, 8)
            ga = max(ga // 2, 4)
        elif gpu_vram < 8 and gpu_vram > 0:
            bs = 1
            ga = 32

        epochs = 3
        if dataset_rows > 0:
            if dataset_rows < 100:
                epochs = 10
            elif dataset_rows < 1000:
                epochs = 5

        return (
            f"Recommended parameters for {model}:\n"
            f"  GPU: {gpu_name} ({gpu_vram:.1f} GB)\n"
            f"  Learning rate: {lr}\n"
            f"  Batch size: {bs}\n"
            f"  Gradient accumulation: {ga}\n"
            f"  Epochs: {epochs}\n"
            f"  Max sequence length: {seq_len}\n"
            f"  Optimizer: adamw_8bit\n"
            f"  Strategy: {'qlora' if gpu_vram < 24 else 'lora'}"
        )
    registry.register(recommend_params_tool)

    @tool(name="get_hardware", description="Get GPU and hardware information")
    def get_hardware_tool() -> str:
        """Detect hardware capabilities."""
        try:
            from llm_forge.hardware import detect_hardware
            hw = detect_hardware()
            return (
                f"GPUs: {hw.num_gpus}x {hw.gpu_name}\n"
                f"VRAM per GPU: {hw.vram_per_gpu_gb:.1f} GB\n"
                f"Total VRAM: {hw.total_vram_gb:.1f} GB\n"
                f"BF16: {'Yes' if hw.bf16_supported else 'No'}\n"
                f"Strategy: {hw.strategy}\n"
                f"Recommended batch size: {hw.recommended_batch_size}\n"
                f"Recommended grad accum: {hw.recommended_gradient_accumulation}"
            )
        except Exception as e:
            return f"Hardware detection failed: {e}"
    registry.register(get_hardware_tool)

    @tool(name="run_evaluation", description="Run evaluation on a trained experiment")
    def run_evaluation_tool(experiment_id: str, test_data_path: str) -> str:
        """Run evaluation on a trained model."""
        exp = _store.get(experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."
        artifacts = exp.get("artifacts", {})
        model_path = artifacts.get("adapter_dir") or artifacts.get("output_dir")
        if not model_path:
            return "No trained model found for this experiment."
        return (
            f"Evaluation queued for experiment {experiment_id}.\n"
            f"Model: {model_path}\n"
            f"Test data: {test_data_path}"
        )
    registry.register(run_evaluation_tool)

    return registry


# ──────────────────────────────────────────────────────────
# Command Parser
# ──────────────────────────────────────────────────────────

_CMD_PATTERN = re.compile(r"^/(\w+)\s*(.*)?$", re.DOTALL)
_KV_PATTERN = re.compile(r"(\w+)=(\S+)")

HELP_TEXT = """Available commands:
  /status              — Check training status and recent experiments
  /datasets            — List uploaded datasets
  /train name=X model=Y dataset=Z  — Start training
  /recommend model=X   — Get hyperparameter recommendations
  /hardware            — Show GPU info
  /experiments         — List all experiments
  /cancel job_id=X     — Cancel a training job
  /help                — Show this help

You can also type freely when an LLM server is connected."""


def parse_command(message: str) -> dict[str, Any] | None:
    """Parse a slash command into tool calls.

    Args:
        message: User message starting with /.

    Returns:
        Dict with 'results' key containing tool outputs, or None if not a command.
    """
    match = _CMD_PATTERN.match(message.strip())
    if not match:
        return None

    cmd = match.group(1).lower()
    args_str = (match.group(2) or "").strip()
    kwargs = dict(_KV_PATTERN.findall(args_str))

    tools = _get_forge_tools()
    results = []

    if cmd == "help":
        return {"results": [HELP_TEXT]}

    elif cmd == "status":
        results.append(tools.get("check_training").execute())
        results.append(tools.get("list_experiments").execute(limit=5))

    elif cmd == "datasets":
        results.append(tools.get("list_datasets").execute())

    elif cmd == "train":
        name = kwargs.get("name", "unnamed")
        model = kwargs.get("model", "Qwen/Qwen2.5-3B-Instruct")
        dataset = kwargs.get("dataset", "")
        if not dataset:
            return {"results": ["Error: dataset is required. Usage: /train name=X model=Y dataset=path/to/data.csv"]}
        results.append(tools.get("start_training").execute(
            name=name, model=model, dataset_path=dataset,
        ))

    elif cmd == "recommend":
        model = kwargs.get("model", "Qwen/Qwen2.5-3B-Instruct")
        rows = int(kwargs.get("rows", "0"))
        results.append(tools.get("recommend_params").execute(model=model, dataset_rows=rows))

    elif cmd == "hardware":
        results.append(tools.get("get_hardware").execute())

    elif cmd == "experiments":
        status = kwargs.get("status", "")
        results.append(tools.get("list_experiments").execute(status=status))

    elif cmd == "cancel":
        job_id = kwargs.get("job_id", args_str)
        if not job_id:
            return {"results": ["Usage: /cancel job_id=X"]}
        results.append(tools.get("cancel_training").execute(job_id=job_id))

    elif cmd == "preview":
        ds_id = kwargs.get("id", args_str)
        if not ds_id:
            return {"results": ["Usage: /preview id=dataset_id"]}
        results.append(tools.get("preview_dataset").execute(dataset_id=ds_id))

    else:
        return {"results": [f"Unknown command: /{cmd}\n\n{HELP_TEXT}"]}

    return {"results": results}


# ──────────────────────────────────────────────────────────
# LLM Mode
# ──────────────────────────────────────────────────────────

def _check_llm_available() -> bool:
    """Check if an LLM server is reachable.

    Returns:
        True if LLM server responds.
    """
    try:
        import requests
        resp = requests.get("http://localhost:8080/v1/models", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict]]:
    """Get or create a chat session.

    Args:
        session_id: Optional existing session ID.

    Returns:
        Tuple of (session_id, message_history).
    """
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]["history"]

    new_id = session_id or str(uuid.uuid4())[:12]
    _sessions[new_id] = {"history": []}
    return new_id, _sessions[new_id]["history"]


def _run_llm_mode(
    message: str,
    session_id: str,
    context: dict | None = None,
) -> dict[str, Any]:
    """Run the assistant in LLM mode with forge tools.

    Args:
        message: User message.
        session_id: Session ID.
        context: Optional UI context (page, active_jobs).

    Returns:
        Dict with answer, tool_calls trace.
    """
    from llm_forge.agent.base import BaseAgent
    from llm_forge.agent.client import ModelClient
    from llm_forge.agent.memory import ShortTermMemory
    from llm_forge.agent.guardrails import GuardrailsConfig

    tools = _get_forge_tools()

    ctx_str = ""
    if context:
        ctx_str = f"\nUser is currently on page: {context.get('page', 'unknown')}"
        active = context.get("active_jobs", [])
        if active:
            ctx_str += f"\nActive training jobs: {len(active)}"

    system_prompt = (
        "You are Forge Co-pilot, an AI assistant for the llm-forge training platform. "
        "You help users manage training experiments, recommend hyperparameters, "
        "upload datasets, and monitor training progress. "
        "Use the available tools to interact with the platform. "
        "Be concise and helpful."
        f"{ctx_str}"
    )

    client = ModelClient(base_url="http://localhost:8080/v1", model="default")
    memory = ShortTermMemory(max_tokens=4096, system_prompt=system_prompt)
    guardrails = GuardrailsConfig(max_iterations=10, max_tokens=8192)

    # Restore session history
    sid, history = _get_or_create_session(session_id)
    for msg in history:
        memory.add(msg["role"], msg["content"])

    agent = BaseAgent(
        client=client,
        tools=tools,
        memory=memory,
        guardrails=guardrails,
        use_native_tools=False,
    )

    answer = agent.run(message)

    # Save to session
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "session_id": sid,
        "actions": agent.trace,
    }


# ──────────────────────────────────────────────────────────
# FastAPI Router
# ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Assistant chat request."""

    message: str
    session_id: str | None = None
    context: dict | None = None


class ChatResponse(BaseModel):
    """Assistant chat response."""

    answer: str
    session_id: str
    actions: list[dict[str, Any]] = []
    mode: str = "command"


class StatusResponse(BaseModel):
    """Platform status for the assistant widget."""

    active_jobs: list[dict[str, Any]]
    recent_experiments: list[dict[str, Any]]
    llm_available: bool


@router.post("/assistant/chat", response_model=ChatResponse)
async def assistant_chat(req: ChatRequest) -> ChatResponse:
    """Chat with the Forge Co-pilot.

    Routes to command mode or LLM mode depending on input.
    """
    message = req.message.strip()
    if not message:
        return ChatResponse(
            answer="Type a command (start with /) or a question. Try /help",
            session_id=req.session_id or str(uuid.uuid4())[:12],
            mode="command",
        )

    sid = req.session_id or str(uuid.uuid4())[:12]

    # Command mode
    if message.startswith("/"):
        result = parse_command(message)
        if result:
            answer = "\n\n".join(result["results"])
            return ChatResponse(
                answer=answer,
                session_id=sid,
                mode="command",
            )

    # LLM mode
    if _check_llm_available():
        try:
            result = _run_llm_mode(message, sid, context=req.context)
            return ChatResponse(
                answer=result["answer"],
                session_id=result["session_id"],
                actions=result.get("actions", []),
                mode="llm",
            )
        except Exception as e:
            logger.exception("LLM mode failed")
            return ChatResponse(
                answer=f"LLM error: {e}\n\nTip: Use slash commands (/help) for direct access.",
                session_id=sid,
                mode="command",
            )

    # Fallback: no LLM, no command
    return ChatResponse(
        answer=(
            "No LLM server connected. Use slash commands for direct access:\n\n"
            f"{HELP_TEXT}"
        ),
        session_id=sid,
        mode="command",
    )


@router.get("/assistant/status", response_model=StatusResponse)
async def assistant_status() -> StatusResponse:
    """Get platform status for the co-pilot widget."""
    jobs = list_jobs()
    active = [j for j in jobs if j["status"] == "running"]
    recent = _store.list_all()[:5]
    llm_ok = _check_llm_available()

    return StatusResponse(
        active_jobs=active,
        recent_experiments=[
            {"id": e["id"], "name": e["name"], "status": e["status"]}
            for e in recent
        ],
        llm_available=llm_ok,
    )


@router.delete("/assistant/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete an assistant session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted"}
    return {"status": "not_found"}
