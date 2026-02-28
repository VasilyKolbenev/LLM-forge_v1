"""CLI entrypoint for llm-forge: forge train/eval/export/serve."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger("llm_forge")


def setup_logging(verbose: bool = False) -> None:
    """Configure rich logging handler.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _parse_overrides(overrides: tuple[str, ...]) -> dict[str, str]:
    """Parse key=value CLI overrides into a dict.

    Args:
        overrides: Tuple of "key=value" strings from CLI.

    Returns:
        Dict of parsed overrides.
    """
    result = {}
    for item in overrides:
        if "=" not in item:
            console.print(f"[red]Invalid override (expected key=value): {item}[/red]")
            sys.exit(1)
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.version_option(package_name="llm-forge")
def main(verbose: bool) -> None:
    """llm-forge: Universal LLM fine-tuning pipeline.

    Supports SFT, DPO, evaluation, GGUF export, and serving.
    Auto-detects hardware and selects optimal training strategy.
    """
    setup_logging(verbose)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("overrides", nargs=-1)
@click.option(
    "--task",
    type=click.Choice(["sft", "dpo", "auto"]),
    default="auto",
    help="Training task (default: auto-detect from config).",
)
@click.option(
    "--base-model",
    type=click.Path(),
    default=None,
    help="Path to SFT adapter for DPO training.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume training from checkpoint directory.",
)
def train(
    config_path: str,
    overrides: tuple[str, ...],
    task: str,
    base_model: Optional[str],
    resume: Optional[str],
) -> None:
    """Run training (SFT or DPO).

    \b
    Examples:
        forge train experiments/cam-sft.yaml
        forge train experiments/cam-dpo.yaml --task dpo --base-model ./outputs/cam-sft
        forge train experiments/cam-sft.yaml learning_rate=1e-4 epochs=5
    """
    from llm_forge.config import load_config

    cli_overrides = _parse_overrides(overrides)
    if base_model:
        cli_overrides["sft_adapter_path"] = base_model

    config = load_config(config_path, cli_overrides=cli_overrides)

    if resume:
        config["resume_from_checkpoint"] = resume
        logger.info("Resuming from checkpoint: %s", resume)

    # Determine task
    if task == "auto":
        task = config.get("task", "sft")

    # Validate config
    from llm_forge.validation import validate_config

    errors = validate_config(config, task=task)
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        sys.exit(1)

    _show_config_summary(config, task)

    if task == "sft":
        from llm_forge.training.sft import train_sft

        results = train_sft(config)
    elif task == "dpo":
        from llm_forge.training.dpo import train_dpo

        results = train_dpo(config)
    else:
        console.print(f"[red]Unknown task: {task}[/red]")
        sys.exit(1)

    _show_training_results(results)


@main.command(name="eval")
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model or adapter directory.",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to test dataset.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Config file for eval settings.",
)
@click.option("--batch-size", type=int, default=8, help="Inference batch size.")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output directory for eval report.",
)
def evaluate(
    model: str,
    test_data: str,
    config_path: Optional[str],
    batch_size: int,
    output: Optional[str],
) -> None:
    """Evaluate a trained model on test data.

    \b
    Examples:
        forge eval --model ./outputs/cam-sft/lora --test-data data/test.csv
        forge eval --model ./outputs/cam-sft/lora --test-data data/test.csv --output reports/
    """
    from llm_forge.config import load_config

    if config_path:
        config = load_config(config_path, auto_hardware=False)
    else:
        config = {}

    config["model_path"] = model
    config["test_data_path"] = test_data
    config.setdefault("evaluation", {})["batch_size"] = batch_size
    if output:
        config.setdefault("output", {})["eval_dir"] = output

    from llm_forge.evaluation.runner import run_evaluation

    results = run_evaluation(config)
    _show_eval_results(results)


@main.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model or adapter directory.",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["gguf", "merged", "hub"]),
    default="gguf",
    help="Export format.",
)
@click.option(
    "--quant",
    type=click.Choice(["q4_k_m", "q8_0", "f16"]),
    default="q4_k_m",
    help="Quantization level for GGUF.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for exported model.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Config file for export settings.",
)
def export(
    model: str,
    export_format: str,
    quant: str,
    output: Optional[str],
    config_path: Optional[str],
) -> None:
    """Export model to production format.

    \b
    Examples:
        forge export --model ./outputs/cam-sft/lora --format gguf --quant q4_k_m
        forge export --model ./outputs/cam-sft/lora --format merged --output ./merged/
        forge export --model ./outputs/cam-sft/lora --format hub
    """
    from llm_forge.config import load_config

    if config_path:
        config = load_config(config_path, auto_hardware=False)
    else:
        config = {}

    config["model_path"] = model
    config.setdefault("export", {}).update({
        "format": export_format,
        "quantization": quant,
    })
    if output:
        config["export"]["output_path"] = output

    if export_format == "gguf":
        from llm_forge.export.gguf import export_gguf

        result = export_gguf(config)
    elif export_format == "merged":
        from llm_forge.export.merged import export_merged

        result = export_merged(config)
    elif export_format == "hub":
        from llm_forge.export.hub import push_to_hub

        result = push_to_hub(config)
    else:
        console.print(f"[red]Unknown format: {export_format}[/red]")
        sys.exit(1)

    console.print(Panel(f"[green]Export complete:[/green] {result.get('output_path', 'done')}"))


@main.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file (GGUF) or directory.",
)
@click.option("--port", type=int, default=8080, help="Server port.")
@click.option(
    "--backend",
    type=click.Choice(["llamacpp", "vllm"]),
    default="llamacpp",
    help="Serving backend.",
)
@click.option("--host", default="0.0.0.0", help="Server host.")
def serve(model: str, port: int, backend: str, host: str) -> None:
    """Start model serving.

    \b
    Examples:
        forge serve --model ./outputs/model.gguf --port 8080
        forge serve --model ./outputs/cam-sft --backend vllm --port 8000
    """
    console.print(
        Panel(f"Starting [bold]{backend}[/bold] server on {host}:{port}")
    )

    if backend == "llamacpp":
        from llm_forge.serving.llamacpp import start_server

        start_server(model_path=model, host=host, port=port)
    elif backend == "vllm":
        from llm_forge.serving.vllm import start_server

        start_server(model_path=model, host=host, port=port)


@main.command()
@click.argument("name")
@click.option(
    "--task",
    type=click.Choice(["sft", "dpo"]),
    default="sft",
    help="Training task.",
)
@click.option(
    "--model",
    type=click.Choice(["qwen2.5-3b", "llama3.2-1b", "mistral-7b"]),
    default="qwen2.5-3b",
    help="Base model.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./outputs/<name>).",
)
def init(name: str, task: str, model: str, output_dir: Optional[str]) -> None:
    """Create a new experiment config.

    \b
    Examples:
        forge init my-classifier
        forge init my-classifier --task dpo --model llama3.2-1b
    """
    import yaml

    if output_dir is None:
        output_dir = f"./outputs/{name}"

    config: dict = {
        "inherit": ["base", f"models/{model}"],
        "task": task,
        "dataset": {
            "path": f"data/{name}.csv",
            "format": "csv",
            "text_column": "text",
            "label_columns": ["label"],
            "test_size": 0.15,
        },
        "training": {
            "epochs": 3,
            "learning_rate": 2e-4,
        },
        "output": {
            "dir": output_dir,
        },
    }

    if task == "dpo":
        config["inherit"].append("tasks/dpo")
        config["sft_adapter_path"] = f"./outputs/{name}-sft/lora"
        config["dpo"] = {
            "pairs_path": f"./outputs/{name}-sft/dpo_pairs.jsonl",
            "beta": 0.1,
        }
        config["training"]["epochs"] = 2
        config["training"]["learning_rate"] = 5e-5

    config_dir = Path("configs/experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(Panel(
        f"[green]Created:[/green] {config_path}\n"
        f"[dim]Edit dataset.path and label_columns, then run:[/dim]\n"
        f"  forge train {config_path}",
        title=f"New Experiment: {name}",
    ))


@main.command()
def info() -> None:
    """Show detected hardware and recommended strategy.

    \b
    Examples:
        forge info
    """
    from llm_forge.hardware import detect_hardware

    hw = detect_hardware()

    table = Table(title="Hardware Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("GPUs", str(hw.num_gpus))
    table.add_row("GPU Name", hw.gpu_name)
    table.add_row("VRAM per GPU", f"{hw.vram_per_gpu_gb:.1f} GB")
    table.add_row("Total VRAM", f"{hw.total_vram_gb:.1f} GB")
    table.add_row("Compute Capability", f"{hw.compute_capability[0]}.{hw.compute_capability[1]}")
    table.add_row("BF16 Supported", str(hw.bf16_supported))
    table.add_row("Recommended Strategy", f"[bold]{hw.strategy}[/bold]")
    table.add_row("Recommended Batch Size", str(hw.recommended_batch_size))
    table.add_row("Recommended Grad Accum", str(hw.recommended_gradient_accumulation))

    console.print(table)


# ──────────────────────────────────────────────────────────
# Agent subgroup
# ──────────────────────────────────────────────────────────

@main.group()
def agent() -> None:
    """Agent system commands: init, test, serve."""


@agent.command(name="init")
@click.argument("name")
@click.option(
    "--model-url",
    default="http://localhost:8080/v1",
    help="Model server URL.",
)
@click.option(
    "--model-name",
    default="default",
    help="Model name on the server.",
)
def agent_init(name: str, model_url: str, model_name: str) -> None:
    """Create a new agent config.

    \b
    Examples:
        forge agent init my-assistant
        forge agent init code-helper --model-url http://localhost:11434/v1
    """
    import yaml

    config = {
        "inherit": ["agents/base"],
        "agent": {
            "name": name,
            "system_prompt": (
                f"You are {name}, a helpful AI assistant. "
                "Use the available tools when needed."
            ),
        },
        "model": {
            "base_url": model_url,
            "name": model_name,
            "timeout": 120,
        },
        "tools": [
            {"name": "search_files", "module": "llm_forge.agent.builtin_tools"},
            {"name": "read_file", "module": "llm_forge.agent.builtin_tools"},
            {"name": "calculate", "module": "llm_forge.agent.builtin_tools"},
        ],
        "memory": {
            "max_tokens": 4096,
            "strategy": "sliding_window",
        },
        "guardrails": {
            "max_iterations": 15,
            "max_tokens": 8192,
        },
    }

    config_dir = Path("configs/agents")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(Panel(
        f"[green]Created:[/green] {config_path}\n"
        f"[dim]Edit the config, then run:[/dim]\n"
        f"  forge agent test {config_path}",
        title=f"New Agent: {name}",
    ))


@agent.command(name="test")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--native-tools", is_flag=True, help="Use native tool calling instead of ReAct.")
def agent_test(config_path: str, native_tools: bool) -> None:
    """Interactive REPL to test an agent.

    \b
    Examples:
        forge agent test configs/agents/my-assistant.yaml
        forge agent test configs/agents/my-assistant.yaml --native-tools
    """
    from llm_forge.agent.loader import load_agent_config
    from llm_forge.agent.base import BaseAgent
    from llm_forge.agent.builtin_tools import get_default_registry
    from llm_forge.validation import validate_agent_config

    config = load_agent_config(config_path)

    errors = validate_agent_config(config)
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        sys.exit(1)

    if native_tools:
        config.setdefault("agent", {})["native_tools"] = True

    tools = get_default_registry()
    agent_instance = BaseAgent.from_config(config, tools=tools)

    agent_name = config.get("agent", {}).get("name", "agent")
    console.print(Panel(
        f"Agent [bold]{agent_name}[/bold] loaded with "
        f"{len(tools)} tools.\n"
        f"Type your message and press Enter. Type 'quit' to exit.",
        title="Agent REPL",
    ))

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        try:
            answer = agent_instance.run(user_input)
            console.print(f"[bold green]{agent_name}:[/bold green] {answer}\n")
        except ConnectionError as e:
            console.print(f"[red]Connection error:[/red] {e}")
            console.print("[dim]Is your model server running?[/dim]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            logger.exception("Agent error")


@agent.command(name="serve")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=8081, help="Server port.")
def agent_serve(config_path: str, host: str, port: int) -> None:
    """Start agent REST API server.

    \b
    Examples:
        forge agent serve configs/agents/my-assistant.yaml
        forge agent serve configs/agents/my-assistant.yaml --port 9000
    """
    from llm_forge.agent.loader import load_agent_config
    from llm_forge.agent.server import start_agent_server
    from llm_forge.validation import validate_agent_config

    config = load_agent_config(config_path)

    errors = validate_agent_config(config)
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        sys.exit(1)

    agent_name = config.get("agent", {}).get("name", "agent")
    console.print(Panel(
        f"Starting agent [bold]{agent_name}[/bold] server\n"
        f"Endpoint: http://{host}:{port}/v1/agent/chat\n"
        f"Health:   http://{host}:{port}/v1/agent/health",
        title="Agent Server",
    ))

    start_agent_server(config, host=host, port=port)


# ──────────────────────────────────────────────────────────
# Web UI command
# ──────────────────────────────────────────────────────────

@main.command(name="ui")
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=8888, help="Server port.")
def ui(host: str, port: int) -> None:
    """Start the Web UI dashboard.

    \b
    Examples:
        forge ui
        forge ui --port 9000
    """
    from llm_forge.ui.app import start_ui_server

    console.print(Panel(
        f"Starting [bold]llm-forge UI[/bold]\n"
        f"Dashboard: http://{host}:{port}\n"
        f"API docs:  http://{host}:{port}/docs",
        title="Web UI",
    ))

    start_ui_server(host=host, port=port)


# ──────────────────────────────────────────────────────────
# Pipeline subgroup
# ──────────────────────────────────────────────────────────

@main.group()
def pipeline() -> None:
    """Pipeline orchestrator: run multi-step training pipelines."""


@pipeline.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
def pipeline_run(config_path: str) -> None:
    """Run a pipeline from YAML config.

    \b
    Examples:
        forge pipeline run configs/pipelines/example.yaml
    """
    import yaml

    with open(config_path, encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)

    name = pipeline_config.get("pipeline", {}).get("name", "unnamed")
    steps = pipeline_config.get("steps", [])

    console.print(Panel(
        f"Pipeline: [bold]{name}[/bold]\n"
        f"Steps: {len(steps)}",
        title="Pipeline Run",
    ))

    from llm_forge.pipeline.executor import PipelineExecutor

    executor = PipelineExecutor(pipeline_config)

    try:
        outputs = executor.run()
        console.print(f"\n[green]Pipeline '{name}' completed successfully![/green]")

        table = Table(title="Step Results")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Key Outputs")

        for step_name, result in outputs.items():
            keys = ", ".join(
                f"{k}={v}" for k, v in result.items()
                if isinstance(v, (str, int, float)) and k != "output_dir"
            )[:80]
            table.add_row(step_name, "completed", keys or "—")

        console.print(table)
    except RuntimeError as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        sys.exit(1)


@pipeline.command(name="list")
@click.option("--name", default=None, help="Filter by pipeline name.")
def pipeline_list(name: str | None) -> None:
    """List past pipeline runs.

    \b
    Examples:
        forge pipeline list
        forge pipeline list --name full-pipeline
    """
    from llm_forge.pipeline.tracker import PipelineTracker

    runs = PipelineTracker.list_runs(pipeline_name=name)

    if not runs:
        console.print("[dim]No pipeline runs found.[/dim]")
        return

    table = Table(title="Pipeline Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Pipeline", style="green")
    table.add_column("Status")
    table.add_column("Started", style="dim")
    table.add_column("Steps")

    for run in runs:
        status = run.get("status", "unknown")
        style = "green" if status == "completed" else "red" if status == "failed" else "yellow"
        steps = run.get("steps", {})
        step_summary = f"{sum(1 for s in steps.values() if s.get('status') == 'completed')}/{len(steps)}"

        table.add_row(
            run.get("run_id", "?"),
            run.get("pipeline", "?"),
            f"[{style}]{status}[/{style}]",
            run.get("started_at", "?")[:19],
            step_summary,
        )

    console.print(table)


def _show_config_summary(config: dict, task: str) -> None:
    """Display config summary panel.

    Args:
        config: Resolved config dict.
        task: Training task name.
    """
    table = Table(title=f"Training Config — {task.upper()}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    model_name = config.get("model", {}).get("name", "unknown")
    strategy = config.get("_detected_strategy", config.get("strategy", "unknown"))
    training = config.get("training", {})

    table.add_row("Model", model_name)
    table.add_row("Strategy", strategy)
    table.add_row("Learning Rate", str(training.get("learning_rate", "—")))
    table.add_row("Epochs", str(training.get("epochs", "—")))
    table.add_row("Batch Size", str(training.get("batch_size", "—")))
    table.add_row(
        "Gradient Accum",
        str(training.get("gradient_accumulation", "—")),
    )
    table.add_row(
        "Max Seq Length",
        str(training.get("max_seq_length", "—")),
    )

    hw = config.get("_hardware", {})
    if hw:
        table.add_row(
            "GPU",
            f"{hw.get('num_gpus', '?')}× {hw.get('gpu_name', '?')} "
            f"({hw.get('vram_per_gpu_gb', '?')} GB)",
        )

    console.print(table)


def _show_training_results(results: dict) -> None:
    """Display training results panel.

    Args:
        results: Dict with training results.
    """
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def _show_eval_results(results: dict) -> None:
    """Display evaluation results panel.

    Args:
        results: Dict with evaluation results.
    """
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if key == "per_class" and isinstance(value, dict):
            continue
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)

    # Per-class breakdown if available
    per_class = results.get("per_class")
    if per_class and isinstance(per_class, dict):
        cls_table = Table(title="Per-Class Accuracy")
        cls_table.add_column("Class", style="cyan")
        cls_table.add_column("Accuracy", style="green")
        cls_table.add_column("Count", style="yellow")

        for cls_name, cls_data in sorted(per_class.items()):
            if isinstance(cls_data, dict):
                acc = cls_data.get("accuracy", 0)
                count = cls_data.get("count", 0)
                cls_table.add_row(cls_name, f"{acc:.2%}", str(count))

        console.print(cls_table)
